# Revised Kano wand implementation.
# Key reliability changes:
# - No deepcopy in BLE notification callbacks.
# - Completed gestures are transferred to the queue without copying.
# - BLE packet lengths are validated.
# - BLE disconnect state is tracked.
# - Reconnect creates a fresh BleakClient.
# - Notification subscriptions are staggered slightly.
# - Signed 16-bit decoding uses int.from_bytes directly.
# - Dataclass timestamps use default_factory.
# - ZeroMQ is closed only during final shutdown.
# - BLE disconnect callback remains lightweight.

import asyncio
import json
import os
from bleak import BleakClient, BleakScanner
import numpy as np
import zmq
import time
import queue
import random
import threading
from dataclasses import dataclass, field
import joblib
from dtw_classifier import DTWNearestNeighborClassifier

CLASSIFIER_TYPE = os.getenv("SPELL_CLASSIFIER", "extra_trees").lower()
MODEL_FILE = (
    "dtw_spell_classifier.joblib"
    if CLASSIFIER_TYPE == "dtw"
    else "spell_classifier.joblib"
)
NUM_SAMPLES = 100
UNKNOWN_CONFIDENCE_THRESHOLD = 0.40
UNKNOWN_MARGIN_THRESHOLD = 0.15

try:
    classifier = joblib.load(MODEL_FILE)
    print(f"Loaded {CLASSIFIER_TYPE} classifier: {MODEL_FILE}")
except FileNotFoundError:
    classifier = None
    print(f"WARNING: Classifier file not found: {MODEL_FILE}")

BUTTON_UUID = "64A7000D-F691-4B93-A6F4-0968F5B648F8"
SERVICE_UUID = "64A70011-F691-4B93-A6F4-0968F5B648F8"
QUATERNIONS_UUID = "64A70002-F691-4B93-A6F4-0968F5B648F8"
MOTION_UUID = "64A7000C-F691-4B93-A6F4-0968F5B648F8"
MAGN_CALIBRATE_UUID = "64A70021-F691-4B93-A6F4-0968F5B648F8"
QUATERNIONS_RESET_UUID = "64A70004-F691-4B93-A6F4-0968F5B648F8"
# WAND_ADDRESS = "D0:1F:65:71:51:32"  # OG
WAND_ADDRESS = os.getenv("WAND_ADDRESS", "DA:94:FD:35:20:15")

ZMQ_BIND_HOST = os.getenv("ZMQ_BIND_HOST", "0.0.0.0")
ZMQ_PORT = int(os.getenv("ZMQ_PORT", "5555"))
NOTIFICATION_TIMEOUT = float(
    os.getenv("WAND_NOTIFICATION_TIMEOUT", "5.0")
)
RECONNECT_DELAY = float(os.getenv("WAND_RECONNECT_DELAY", "2.0"))
RECONNECT_MAX_DELAY = float(
    os.getenv("WAND_RECONNECT_MAX_DELAY", "30.0")
)


def interpolate_stream(samples, timestamps, num_samples):
    if len(samples) == 0:
        return np.zeros((num_samples, 1))

    samples = np.asarray(samples, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)

    order = np.argsort(timestamps)
    timestamps = timestamps[order]
    samples = samples[order]

    _, indices = np.unique(timestamps, return_index=True)
    timestamps = timestamps[indices]
    samples = samples[indices]

    if len(timestamps) == 1:
        return np.repeat(samples, num_samples, axis=0)

    old_time = np.linspace(0, 1, len(timestamps))
    new_time = np.linspace(0, 1, num_samples)

    result = np.zeros((num_samples, samples.shape[1]))
    for column in range(samples.shape[1]):
        result[:, column] = np.interp(
            new_time, old_time, samples[:, column]
        )

    return result


def gesture_to_features(gesture):
    if gesture.motion:
        timestamps = [s.timestamp for s in gesture.motion]
        values = [[
            s.mag_x, s.mag_y, s.mag_z,
            s.acc_x, s.acc_y, s.acc_z,
            s.pitch, s.roll, s.yaw
        ] for s in gesture.motion]
        motion = interpolate_stream(values, timestamps, NUM_SAMPLES)
    else:
        motion = np.zeros((NUM_SAMPLES, 9))

    if gesture.orientation:
        timestamps = [s.timestamp for s in gesture.orientation]
        values = [[s.x, s.y, s.z, s.w] for s in gesture.orientation]
        orientation = interpolate_stream(
            values, timestamps, NUM_SAMPLES
        )
    else:
        orientation = np.zeros((NUM_SAMPLES, 4))

    return np.hstack([motion, orientation]).flatten()


def classify_spell(gesture):
    if classifier is None:
        return "Unknown", 0.0, []

    features = gesture_to_features(gesture).reshape(1, -1)
    probabilities = classifier.predict_proba(features)[0]
    classes = classifier.classes_
    sorted_indices = np.argsort(probabilities)[::-1]

    top_predictions = [
        (str(classes[i]), float(probabilities[i]))
        for i in sorted_indices
    ]

    best = sorted_indices[0]
    confidence = float(probabilities[best])
    prediction = str(classes[best])

    if len(sorted_indices) > 1:
        margin = confidence - float(probabilities[sorted_indices[1]])
    else:
        margin = confidence

    if (
        confidence < UNKNOWN_CONFIDENCE_THRESHOLD
        or margin < UNKNOWN_MARGIN_THRESHOLD
    ):
        prediction = "Unknown"

    return prediction, confidence, top_predictions


@dataclass
class WandOrientationState:
    timestamp: float = field(default_factory=time.monotonic)
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 0.0


@dataclass
class WandMotionState:
    timestamp: float = field(default_factory=time.monotonic)
    mag_x: float = 0.0
    mag_y: float = 0.0
    mag_z: float = 0.0
    acc_x: float = 0.0
    acc_y: float = 0.0
    acc_z: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    yaw: float = 0.0


@dataclass
class Gesture:
    motion: list[WandMotionState]
    orientation: list[WandOrientationState]



class KanoWand:
    """
    Kano Wand BLE interface.

    Reliability design:
      * Every BLE connection gets a unique generation number.
      * Notification callbacks are bound to that generation, so callbacks
        from an old BleakClient are ignored after reconnect.
      * Reconnect always retires the old client before creating a new one.
      * Notification subscriptions are installed one at a time.
      * Motion/orientation notification delivery is verified before the
        connection is declared ready.
      * BLE callbacks do only lightweight state updates and queue transfers.
      * Disconnect callbacks never perform BLE operations themselves.
    """

    def __init__(self, address):
        self.address = address

        self.client = None
        self._client_generation = 0
        self._client_lock = asyncio.Lock()

        self.latest_motion = WandMotionState()
        self.latest_orientation = WandOrientationState()

        self.bluetooth_disconnected = threading.Event()
        self.bluetooth_disconnected.clear()
        self.shutdown_event = threading.Event()

        self.last_notification_time = 0.0
        self.last_motion_time = 0.0
        self.last_orientation_time = 0.0
        self.last_button_time = 0.0

        self.button_pressed = False
        self.recording = False
        self.current_gesture = Gesture([], [])
        self.spell_queue = queue.Queue()

        self._disconnecting = False
        self._connecting = False
        self._connected_ready = False
        self._reconnect_lock = asyncio.Lock()
        self._notifications_started = set()

        # Set when the current connection has actually delivered motion and
        # orientation notifications. Starting notify() alone is not enough.
        self._sensor_notifications_ready = asyncio.Event()
        self._motion_received_generation = None
        self._orientation_received_generation = None

        # Notification diagnostics.
        self.notification_counts = {
            "motion": 0,
            "orientation": 0,
            "button": 0,
        }
        self.invalid_packet_counts = {
            "motion": 0,
            "orientation": 0,
            "button": 0,
        }
        self.button_down_count = 0
        self.button_up_count = 0

        # Per-characteristic timing diagnostics.
        self._last_notification_times = {
            "motion": None,
            "orientation": None,
            "button": None,
        }
        self.notification_gap_stats = {
            name: {
                "count": 0,
                "max_gap": 0.0,
                "sum_gap": 0.0,
                "gaps_over_0_1": 0,
                "gaps_over_0_2": 0,
                "gaps_over_0_5": 0,
            }
            for name in ("motion", "orientation", "button")
        }
        self.button_transition_log = []
        self.gesture_start_time = None
        self.gesture_end_time = None

        # Counts of connection/recovery causes. These are deliberately
        # separate from notification counts so a later diagnosis can tell
        # whether BlueZ reported a disconnect or we detected a notification
        # stall.
        self.connection_generation = 0
        self.disconnect_callback_count = 0
        self.notification_stall_count = 0
        self.last_disconnect_reason = None
        self.last_reconnect_reason = None

        self.context = zmq.Context()
        self.publisher = self.context.socket(zmq.PUB)
        self.publisher.setsockopt(zmq.LINGER, 0)
        self.publisher.bind(f"tcp://{ZMQ_BIND_HOST}:{ZMQ_PORT}")

    def publish_spell(self, spell, confidence, top_predictions):
        """Publish a classified spell as a JSON message over ZeroMQ."""
        message = {
            "spell": spell,
            "confidence": confidence,
            "top_predictions": [
                {
                    "spell": prediction,
                    "confidence": probability,
                }
                for prediction, probability in top_predictions[:3]
            ],
            "timestamp": time.time(),
        }
        self.publisher.send_string(json.dumps(message))

    def _new_generation(self):
        self._client_generation += 1
        self.connection_generation = self._client_generation
        return self._client_generation

    def _callback_is_current(self, generation):
        return (
            generation == self._client_generation
            and self.client is not None
            and (self._connected_ready or self._connecting)
            and not self._disconnecting
        )

    def _make_disconnect_callback(self, generation):
        def callback(client):
            self.disconnected_callback(client, generation)
        return callback

    def _make_notification_callback(self, handler, generation):
        def callback(sender, data):
            # This check is intentionally before any packet processing.
            # Old callbacks can therefore become harmless immediately after
            # a reconnect starts.
            if generation != self._client_generation:
                return
            handler(sender, data, generation)
        return callback

    async def _retire_client(self, client, started_notifications):
        """
        Best-effort shutdown of one specific BleakClient.

        This method never changes self.client and never changes the current
        generation. That is important: a reconnect can retire an old client
        without accidentally affecting the new client.
        """
        if client is None:
            return

        if client.is_connected:
            for uuid in tuple(started_notifications):
                try:
                    await client.stop_notify(uuid)
                except Exception:
                    # A disconnected BlueZ client can reject stop_notify().
                    # The important cleanup operation is disconnect().
                    pass

            try:
                await client.disconnect()
            except Exception:
                pass

    async def _wait_for_sensor_notifications(self, generation, timeout):
        """Verify that the new connection is actually delivering sensors."""
        try:
            await asyncio.wait_for(
                self._sensor_notifications_ready.wait(),
                timeout=timeout,
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                "BLE connection established and notifications were subscribed, "
                "but motion/orientation notifications were not received."
            ) from exc

        if generation != self._client_generation:
            raise RuntimeError("BLE client became stale during notification verification.")

    async def connect(
        self,
        scan_timeout=10.0,
        connect_timeout=20.0,
        notification_verify_timeout=3.0,
    ):
        """
        Scan, create a fresh BleakClient, connect, subscribe, and verify data.

        The method does not declare the connection ready until actual sensor
        notifications have arrived.
        """
        async with self._client_lock:
            self._connecting = True
            self._connected_ready = False
            self._disconnecting = False
            self.bluetooth_disconnected.clear()
            self._sensor_notifications_ready.clear()
            self._motion_received_generation = None
            self._orientation_received_generation = None
            self._notifications_started.clear()

            # Invalidate every callback belonging to an older client before
            # scanning/connecting. Old callbacks will return immediately.
            generation = self._new_generation()

            old_client = self.client
            self.client = None

            print("Scanning for wand...")

            try:
                # Retire the previous client before creating another one.
                if old_client is not None:
                    await self._retire_client(old_client, ())

                device = await BleakScanner.find_device_by_address(
                    self.address,
                    timeout=scan_timeout,
                )
                if device is None:
                    raise RuntimeError(
                        f"Wand {self.address} was not found during Bluetooth scan."
                    )

                print(
                    f"Found wand: {device.name or 'Unknown'} "
                    f"({device.address})"
                )

                client = BleakClient(
                    device,
                    disconnected_callback=self._make_disconnect_callback(generation),
                )
                self.client = client

                print("Connecting to wand...")
                await client.connect(timeout=connect_timeout)

                if generation != self._client_generation:
                    raise RuntimeError("BLE client became stale during connect.")

                if not client.is_connected:
                    raise RuntimeError("Wand did not report as connected.")

                print("Connected:", client.is_connected)

                # Give BlueZ/the wand a short settling period before enabling
                # notifications.
                await asyncio.sleep(0.5)

                # Button first: once it is subscribed, a press/release can be
                # observed even while the higher-rate sensors are being added.
                await client.start_notify(
                    BUTTON_UUID,
                    self._make_notification_callback(
                        self._button_handler,
                        generation,
                    ),
                )
                self._notifications_started.add(BUTTON_UUID)
                await asyncio.sleep(0.15)

                await client.start_notify(
                    MOTION_UUID,
                    self._make_notification_callback(
                        self._motion_handler,
                        generation,
                    ),
                )
                self._notifications_started.add(MOTION_UUID)
                await asyncio.sleep(0.15)

                await client.start_notify(
                    QUATERNIONS_UUID,
                    self._make_notification_callback(
                        self._orientation_handler,
                        generation,
                    ),
                )
                self._notifications_started.add(QUATERNIONS_UUID)

                # Do not fake notification timestamps here. Wait for real
                # packets from the wand.
                await self._wait_for_sensor_notifications(
                    generation,
                    notification_verify_timeout,
                )

                if not client.is_connected:
                    raise RuntimeError(
                        "Wand disconnected while verifying notifications."
                    )

                self._connecting = False
                self._connected_ready = True
                self.bluetooth_disconnected.clear()
                self.shutdown_event.clear()

                print(
                    "Notifications verified "
                    f"(motion={self.notification_counts['motion']}, "
                    f"orientation={self.notification_counts['orientation']})"
                )

            except Exception:
                self._connecting = False
                self._connected_ready = False

                # Invalidate callbacks before retiring this failed client.
                self._client_generation += 1
                failed_client = self.client
                self.client = None
                started = tuple(self._notifications_started)
                self._notifications_started.clear()

                self._disconnecting = True
                try:
                    await self._retire_client(failed_client, started)
                finally:
                    self._disconnecting = False

                raise

    async def reconnect(self, attempts=5, delay=2.0, reason=None):
        """
        Recover using a completely fresh BLE client.

        The old client's callbacks are invalidated before cleanup. This is the
        key protection against stale notification/disconnect callbacks
        accumulating across reconnects.
        """
        async with self._reconnect_lock:
            self.last_reconnect_reason = reason or "unspecified"
            print("Attempting to reconnect to wand...")
            if reason:
                print(f"Reconnect reason: {reason}")

            self.recording = False
            self.button_pressed = False
            self._connected_ready = False
            self._sensor_notifications_ready.clear()
            self._motion_received_generation = None
            self._orientation_received_generation = None

            for attempt in range(1, attempts + 1):
                print(f"Reconnect attempt {attempt} of {attempts}...")

                try:
                    self._disconnecting = True
                    self._connecting = False

                    # Invalidate callbacks FIRST.
                    self._client_generation += 1
                    old_client = self.client
                    old_started = tuple(self._notifications_started)

                    self.client = None
                    self._notifications_started.clear()
                    self.bluetooth_disconnected.clear()

                    # Now it is safe to retire the old client. Any callback
                    # generated during this cleanup is stale and ignored.
                    await self._retire_client(old_client, old_started)

                    self._disconnecting = False

                    if delay > 0:
                        await asyncio.sleep(delay)

                    await self.connect(
                        scan_timeout=12.0,
                        connect_timeout=25.0,
                        notification_verify_timeout=4.0,
                    )

                    print("Reconnected successfully.")
                    return True

                except Exception as e:
                    self._disconnecting = True
                    self._connecting = False
                    self._connected_ready = False
                    self.recording = False
                    self.button_pressed = False
                    self._sensor_notifications_ready.clear()
                    self._motion_received_generation = None
                    self._orientation_received_generation = None

                    # Invalidate anything from the failed attempt.
                    self._client_generation += 1
                    failed_client = self.client
                    failed_started = tuple(self._notifications_started)
                    self.client = None
                    self._notifications_started.clear()

                    try:
                        await self._retire_client(
                            failed_client,
                            failed_started,
                        )
                    finally:
                        self._disconnecting = False

                    print(
                        f"Reconnect attempt {attempt} failed: "
                        f"{type(e).__name__}: {e}"
                    )

                    if attempt < attempts and delay > 0:
                        await asyncio.sleep(delay)

            print("Unable to reconnect to wand after this retry batch.")
            return False

    async def disconnect(self):
        """Final shutdown. No reconnect should be attempted afterward."""
        if self._disconnecting:
            return

        self._disconnecting = True
        self._connecting = False
        self._connected_ready = False
        self.recording = False
        self.button_pressed = False
        self._sensor_notifications_ready.clear()
        self._motion_received_generation = None
        self._orientation_received_generation = None

        # Invalidate callbacks before touching the old client.
        self._client_generation += 1
        client = self.client
        started = tuple(self._notifications_started)
        self.client = None
        self._notifications_started.clear()

        print("Disconnecting wand...")

        try:
            await self._retire_client(client, started)
        finally:
            try:
                self.publisher.close(linger=0)
            except Exception:
                pass

            try:
                self.context.term()
            except Exception:
                pass

            print("Wand disconnected")

    def disconnected_callback(self, client, generation=None):
        """
        Lightweight disconnect callback.

        It performs no BLE operations. The main asyncio loop sees the event
        and performs the actual recovery.
        """
        if generation is not None and generation != self._client_generation:
            return

        if client is not self.client:
            return

        self.disconnect_callback_count += 1

        if self._disconnecting:
            return

        # A disconnect while connect() is still setting things up is a failed
        # connection attempt, not a user-visible "connection lost" event.
        if self._connecting:
            self.last_disconnect_reason = "disconnect_during_connection_setup"
            print("BLE disconnect callback occurred during connection setup.")
            self.bluetooth_disconnected.set()
            return

        if not self._connected_ready:
            return

        self.last_disconnect_reason = "bluez_disconnect_callback"

        print()
        print("WARNING: Wand Bluetooth connection was lost!")
        print(
            "Notifications before disconnect: "
            f"motion={self.notification_counts['motion']}, "
            f"orientation={self.notification_counts['orientation']}, "
            f"button={self.notification_counts['button']}"
        )
        print(
            "Button events before disconnect: "
            f"down={self.button_down_count}, "
            f"up={self.button_up_count}"
        )

        # Do not set shutdown_event. The recorder/application can recover.
        self.bluetooth_disconnected.set()
        self.recording = False
        self.button_pressed = False

    def mark_notification_stall(self):
        """Record that the application, rather than BlueZ, detected a stall."""
        self.notification_stall_count += 1
        self.last_disconnect_reason = "notification_stall"

    def _record_notification_timing(self, kind, now):
        """Record notification inter-arrival timing with O(1) callback work."""
        previous = self._last_notification_times[kind]
        self._last_notification_times[kind] = now
        if previous is None:
            return

        gap = now - previous
        stats = self.notification_gap_stats[kind]
        stats["count"] += 1
        stats["sum_gap"] += gap
        if gap > stats["max_gap"]:
            stats["max_gap"] = gap
        if gap > 0.1:
            stats["gaps_over_0_1"] += 1
        if gap > 0.2:
            stats["gaps_over_0_2"] += 1
        if gap > 0.5:
            stats["gaps_over_0_5"] += 1

    def reset_gap_diagnostics(self):
        """Reset only inter-arrival gap statistics before a new gesture."""
        self._last_notification_times = {
            "motion": None,
            "orientation": None,
            "button": None,
        }
        for stats in self.notification_gap_stats.values():
            for key in stats:
                stats[key] = (
                    0
                    if key not in ("max_gap", "sum_gap")
                    else 0.0
                )

    def get_notification_diagnostics(self):
        """Return a snapshot suitable for the recorder's per-gesture report."""
        result = {}

        for kind, stats in self.notification_gap_stats.items():
            count = stats["count"]
            result[kind] = {
                "count": self.notification_counts[kind],
                "mean_gap": (stats["sum_gap"] / count) if count else 0.0,
                "max_gap": stats["max_gap"],
                "gaps_over_0_1": stats["gaps_over_0_1"],
                "gaps_over_0_2": stats["gaps_over_0_2"],
                "gaps_over_0_5": stats["gaps_over_0_5"],
            }

        result["button_down"] = self.button_down_count
        result["button_up"] = self.button_up_count
        result["connected"] = bool(
            self.client is not None and self.client.is_connected
        )
        result["connected_ready"] = self._connected_ready
        result["connection_generation"] = self._client_generation
        result["disconnect_callback_count"] = self.disconnect_callback_count
        result["notification_stall_count"] = self.notification_stall_count
        result["last_disconnect_reason"] = self.last_disconnect_reason
        result["last_reconnect_reason"] = self.last_reconnect_reason
        result["invalid_packets"] = dict(self.invalid_packet_counts)

        return result

    def _orientation_handler(self, sender, data, generation):
        if not self._callback_is_current(generation):
            return

        try:
            if len(data) < 8:
                self.invalid_packet_counts["orientation"] += 1
                return

            self.notification_counts["orientation"] += 1
            orientation = self.decode_orientation(data)
            self.latest_orientation = orientation

            now = time.monotonic()
            self.last_orientation_time = now
            self.last_notification_time = now
            self._record_notification_timing("orientation", now)

            if self.recording:
                self.current_gesture.orientation.append(orientation)

            self._orientation_received_generation = generation
            if self._motion_received_generation == generation:
                self._sensor_notifications_ready.set()

        except Exception as e:
            print(
                f"WARNING: Orientation callback error: "
                f"{type(e).__name__}: {e}"
            )

    def _motion_handler(self, sender, data, generation):
        if not self._callback_is_current(generation):
            return

        try:
            if len(data) < 18:
                self.invalid_packet_counts["motion"] += 1
                return

            self.notification_counts["motion"] += 1
            motion = self.decode_motion(data)
            self.latest_motion = motion

            now = time.monotonic()
            self.last_motion_time = now
            self.last_notification_time = now
            self._record_notification_timing("motion", now)

            if self.recording:
                self.current_gesture.motion.append(motion)

            self._motion_received_generation = generation
            if self._orientation_received_generation == generation:
                self._sensor_notifications_ready.set()

        except Exception as e:
            print(
                f"WARNING: Motion callback error: "
                f"{type(e).__name__}: {e}"
            )

    def _button_handler(self, sender, data, generation):
        if not self._callback_is_current(generation):
            return

        try:
            if not data:
                self.invalid_packet_counts["button"] += 1
                return

            self.notification_counts["button"] += 1
            pressed = self.decode_button(data)
            now = time.monotonic()

            self.last_button_time = now
            self.last_notification_time = now
            self._record_notification_timing("button", now)

            self.button_transition_log.append((now, pressed))
            if len(self.button_transition_log) > 100:
                del self.button_transition_log[:-100]

            if pressed and not self.button_pressed:
                self.button_down_count += 1
                print("BUTTON DOWN")

                self.button_pressed = True
                self.recording = True
                self.gesture_start_time = now
                self.gesture_end_time = None
                self.current_gesture = Gesture([], [])

            elif not pressed and self.button_pressed:
                self.button_up_count += 1
                print(
                    f"BUTTON UP - "
                    f"{len(self.current_gesture.motion)} motion samples, "
                    f"{len(self.current_gesture.orientation)} "
                    f"orientation samples"
                )

                self.button_pressed = False
                self.recording = False
                self.gesture_end_time = now

                if (
                    self.current_gesture.motion
                    or self.current_gesture.orientation
                ):
                    # O(1) transfer. No deepcopy in the BLE callback.
                    completed_gesture = self.current_gesture
                    self.current_gesture = Gesture([], [])
                    self.spell_queue.put(completed_gesture)
                else:
                    self.current_gesture = Gesture([], [])

        except Exception as e:
            print(
                f"WARNING: Button callback error: "
                f"{type(e).__name__}: {e}"
            )

    # Keep the public handler names for compatibility with any other code
    # that may import KanoWand and call them directly.
    def orientation_handler(self, sender, data):
        self._orientation_handler(
            sender,
            data,
            self._client_generation,
        )

    def motion_handler(self, sender, data):
        self._motion_handler(
            sender,
            data,
            self._client_generation,
        )

    def button_handler(self, sender, data):
        self._button_handler(
            sender,
            data,
            self._client_generation,
        )

    @staticmethod
    def decode_button(data):
        return data[0] == 1

    @staticmethod
    def _signed_int16(data):
        return int.from_bytes(data, byteorder="little", signed=True)

    def decode_orientation(self, data):
        return WandOrientationState(
            timestamp=time.monotonic(),
            x=self._signed_int16(data[2:4]) / 1024.0,
            y=self._signed_int16(data[4:6]) / 1024.0,
            z=self._signed_int16(data[6:8]) / 1024.0,
            w=self._signed_int16(data[0:2]) / 1024.0,
        )

    def decode_motion(self, data):
        return WandMotionState(
            timestamp=time.monotonic(),
            acc_x=self._signed_int16(data[0:2]),
            acc_y=self._signed_int16(data[2:4]),
            acc_z=self._signed_int16(data[4:6]),
            mag_x=self._signed_int16(data[6:8]),
            mag_y=self._signed_int16(data[8:10]),
            mag_z=self._signed_int16(data[10:12]),
            yaw=self._signed_int16(data[12:14]),
            pitch=self._signed_int16(data[14:16]),
            roll=self._signed_int16(data[16:18]),
        )


def print_ble_diagnostics(wand):
    now = time.monotonic()
    diagnostics = wand.get_notification_diagnostics()
    print(
        "BLE diagnostics: "
        f"connected={diagnostics['connected']}, "
        f"motion={diagnostics['motion']['count']}, "
        f"orientation={diagnostics['orientation']['count']}, "
        f"button={diagnostics['button']['count']}, "
        f"button_down={diagnostics['button_down']}, "
        f"button_up={diagnostics['button_up']}, "
        f"last_any={now - wand.last_notification_time:.2f}s ago"
    )
    for kind in ("motion", "orientation", "button"):
        d = diagnostics[kind]
        print(
            f"  {kind:11s}: mean_gap={d['mean_gap']:.4f}s, "
            f"max_gap={d['max_gap']:.4f}s, "
            f">0.1s={d['gaps_over_0_1']}, "
            f">0.2s={d['gaps_over_0_2']}, "
            f">0.5s={d['gaps_over_0_5']}"
        )


def classify_worker(wand):
    print("Classifier thread started")

    while not wand.shutdown_event.is_set():
        try:
            gesture = wand.spell_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        try:
            prediction, confidence, top_predictions = classify_spell(gesture)

            print()
            print("=" * 60)
            print(f"Prediction: {prediction}")
            print(f"Confidence: {confidence:.1%}")
            print()
            print("Top predictions:")

            for spell, probability in top_predictions[:3]:
                print(f"  {spell:25s} {probability:.1%}")

            if prediction == "Unknown":
                print()
                print(
                    "The gesture was not classified with "
                    "enough confidence."
                )
            else:
                wand.publish_spell(
                    prediction,
                    confidence,
                    top_predictions,
                )
                print(
                    f"Published {prediction} to "
                    f"tcp://{ZMQ_BIND_HOST}:{ZMQ_PORT}"
                )

            print("=" * 60)

        except Exception as e:
            print(
                f"Classification error: "
                f"{type(e).__name__}: {e}"
            )
        finally:
            wand.spell_queue.task_done()

    print("Classifier thread stopped")


async def main():
    wand = KanoWand(WAND_ADDRESS)

    classifier_thread = threading.Thread(
        target=classify_worker,
        args=(wand,),
        daemon=True
    )

    try:
        await wand.connect()
        classifier_thread.start()

        print()
        print("Running.")
        print("Press and hold the wand button to perform a spell.")
        print("Release the button when the spell is complete.")
        print("Press Ctrl+C to stop.")
        print()

        reconnect_delay = RECONNECT_DELAY
        while not wand.shutdown_event.is_set():
            await asyncio.sleep(0.5)

            notification_age = (
                time.monotonic() - wand.last_notification_time
            )
            connection_lost = wand.bluetooth_disconnected.is_set()
            notifications_stalled = (
                wand._connected_ready
                and notification_age > NOTIFICATION_TIMEOUT
            )

            if not connection_lost and not notifications_stalled:
                reconnect_delay = RECONNECT_DELAY
                continue

            if notifications_stalled and not connection_lost:
                print(
                    "WARNING: No BLE notifications received for "
                    f"{notification_age:.1f}s; reconnecting."
                )
                wand.mark_notification_stall()
                reconnect_reason = "notification_stall"
            else:
                reconnect_reason = wand.last_disconnect_reason or "bluez_disconnect"

            print_ble_diagnostics(wand)

            reconnected = await wand.reconnect(
                attempts=5,
                delay=0,
                reason=reconnect_reason,
            )
            if reconnected:
                reconnect_delay = RECONNECT_DELAY
                print("Wand connection restored.")
                continue

            print(
                "Reconnect batch failed; retrying in "
                f"{reconnect_delay:.1f}s."
            )
            await asyncio.sleep(
                reconnect_delay + random.uniform(0.0, 0.5)
            )
            reconnect_delay = min(
                reconnect_delay * 2,
                RECONNECT_MAX_DELAY,
            )

    except asyncio.CancelledError:
        raise

    except Exception as e:
        print()
        print(f"ERROR: {type(e).__name__}: {e}")

    finally:
        print()
        print("Shutting down...")

        wand.shutdown_event.set()
        classifier_thread.join(timeout=2)

        if classifier_thread.is_alive():
            print(
                "WARNING: Classifier thread did not stop "
                "within the timeout."
            )

        await wand.disconnect()
        print("Shutdown complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print()
        print("Ctrl+C received")
