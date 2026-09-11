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
from bleak import BleakClient, BleakScanner
import numpy as np
import zmq
import time
import queue
import threading
from dataclasses import dataclass, field
import joblib

MODEL_FILE = "spell_classifier.joblib"
NUM_SAMPLES = 100
UNKNOWN_CONFIDENCE_THRESHOLD = 0.65
UNKNOWN_MARGIN_THRESHOLD = 0.15

try:
    classifier = joblib.load(MODEL_FILE)
    print(f"Loaded classifier: {MODEL_FILE}")
except FileNotFoundError:
    classifier = None
    print(f"WARNING: Classifier file not found: {MODEL_FILE}")

BUTTON_UUID = "64A7000D-F691-4B93-A6F4-0968F5B648F8"
SERVICE_UUID = "64A70011-F691-4B93-A6F4-0968F5B648F8"
QUATERNIONS_UUID = "64A70002-F691-4B93-A6F4-0968F5B648F8"
MOTION_UUID = "64A7000C-F691-4B93-A6F4-0968F5B648F8"
MAGN_CALIBRATE_UUID = "64A70021-F691-4B93-A6F4-0968F5B648F8"
QUATERNIONS_RESET_UUID = "64A70004-F691-4B93-A6F4-0968F5B648F8"
WAND_ADDRESS = "D0:1F:65:71:51:32"

NERF_IP = "192.168.0.26"
NERF_PORT = 5555


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
    def __init__(self, address):
        self.address = address
        self.client = BleakClient(
            address,
            disconnected_callback=self.disconnected_callback
        )

        self.latest_motion = WandMotionState()
        self.latest_orientation = WandOrientationState()

        self.bluetooth_disconnected = threading.Event()
        self.bluetooth_disconnected.clear()
        self.shutdown_event = threading.Event()

        self.last_notification_time = time.monotonic()
        self.last_motion_time = time.monotonic()
        self.last_orientation_time = time.monotonic()
        self.last_button_time = time.monotonic()

        self.button_pressed = False
        self.recording = False
        self.current_gesture = Gesture([], [])
        self.spell_queue = queue.Queue()

        self._disconnecting = False
        self._connecting = False
        self._connected_ready = False
        self._reconnect_lock = asyncio.Lock()
        self._notifications_started = set()

        # Notification diagnostics.  These let us distinguish a real BLE
        # disconnect from a connection that remains "connected" but stops
        # delivering notifications.
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

        self.context = zmq.Context()
        self.publisher = self.context.socket(zmq.PUB)
        self.publisher.setsockopt(zmq.LINGER, 0)
        self.publisher.connect(f"tcp://{NERF_IP}:{NERF_PORT}")

    async def connect(self, scan_timeout=10.0, connect_timeout=20.0):
        """Scan for the wand, then connect using the discovered BLEDevice.

        Using a discovered BLEDevice is more reliable on Linux/BlueZ than
        repeatedly constructing BleakClient from the raw MAC address.
        """
        self._disconnecting = False
        self._connecting = True
        self._connected_ready = False
        self.bluetooth_disconnected.clear()
        self._notifications_started.clear()

        print("Scanning for wand...")

        try:
            device = await BleakScanner.find_device_by_address(
                self.address,
                timeout=scan_timeout,
            )
            if device is None:
                raise RuntimeError(
                    f"Wand {self.address} was not found during Bluetooth scan."
                )

            print(f"Found wand: {device.name or 'Unknown'} ({device.address})")

            # The scanner returns a BLEDevice with the BlueZ discovery state
            # already established.  Bind the disconnect callback to this
            # exact client instance so stale callbacks can be ignored.
            self.client = BleakClient(
                device,
                disconnected_callback=self.disconnected_callback,
            )

            print("Connecting to wand...")
            await self.client.connect(timeout=connect_timeout)

            if not self.client.is_connected:
                raise RuntimeError("Wand did not report as connected.")

            print("Connected:", self.client.is_connected)

            # Give BlueZ/the wand a short settling period before enabling the
            # notification characteristics.
            await asyncio.sleep(0.5)

            await self.client.start_notify(MOTION_UUID, self.motion_handler)
            self._notifications_started.add(MOTION_UUID)
            await asyncio.sleep(0.15)

            await self.client.start_notify(QUATERNIONS_UUID, self.orientation_handler)
            self._notifications_started.add(QUATERNIONS_UUID)
            await asyncio.sleep(0.15)

            await self.client.start_notify(BUTTON_UUID, self.button_handler)
            self._notifications_started.add(BUTTON_UUID)

            self._connecting = False
            self._connected_ready = True
            self.bluetooth_disconnected.clear()
            self.shutdown_event.clear()
            now = time.monotonic()
            self.last_notification_time = now
            self.last_motion_time = now
            self.last_orientation_time = now
            self.last_button_time = now
            print("Notifications started")

        except Exception:
            self._connecting = False
            self._connected_ready = False
            self._disconnecting = True
            try:
                for uuid in tuple(self._notifications_started):
                    try:
                        await self.client.stop_notify(uuid)
                    except Exception:
                        pass
                self._notifications_started.clear()
                if self.client.is_connected:
                    try:
                        await self.client.disconnect()
                    except Exception:
                        pass
            finally:
                self._disconnecting = False
            raise

    async def reconnect(self, attempts=5, delay=2.0):
        """Recover a lost connection using a fresh scan and BLE client."""
        async with self._reconnect_lock:
            print("Attempting to reconnect to wand...")

            self.recording = False
            self.button_pressed = False
            self._connected_ready = False

            for attempt in range(1, attempts + 1):
                print(f"Reconnect attempt {attempt} of {attempts}...")
                try:
                    self._disconnecting = True
                    self._connecting = False
                    self._connected_ready = False

                    for uuid in tuple(self._notifications_started):
                        try:
                            await self.client.stop_notify(uuid)
                        except Exception:
                            pass
                    self._notifications_started.clear()

                    try:
                        if self.client.is_connected:
                            await self.client.disconnect()
                    except Exception:
                        pass

                    self._disconnecting = False
                    self.bluetooth_disconnected.clear()
                    await asyncio.sleep(delay)

                    await self.connect(scan_timeout=12.0, connect_timeout=25.0)
                    print("Reconnected successfully.")
                    return True

                except Exception as e:
                    self._disconnecting = True
                    self._connecting = False
                    self._connected_ready = False
                    try:
                        if self.client.is_connected:
                            await self.client.disconnect()
                    except Exception:
                        pass
                    self._disconnecting = False
                    print(
                        f"Reconnect attempt {attempt} failed: "
                        f"{type(e).__name__}: {e}"
                    )
                    if attempt < attempts:
                        await asyncio.sleep(delay)

            print("Unable to reconnect to wand.")
            return False

    async def disconnect(self):
        if self._disconnecting:
            return

        self._disconnecting = True
        self._connecting = False
        self._connected_ready = False
        self.recording = False
        self.button_pressed = False

        print("Disconnecting wand...")

        try:
            for uuid in tuple(self._notifications_started):
                try:
                    if self.client.is_connected:
                        await self.client.stop_notify(uuid)
                except Exception as e:
                    # Shutdown should remain best-effort.  In particular,
                    # don't report service-discovery errors as a new BLE fault.
                    if "Service Discovery has not been performed" not in str(e):
                        print(
                            f"Could not stop notification {uuid}: "
                            f"{type(e).__name__}: {e}"
                        )
            self._notifications_started.clear()

            try:
                if self.client.is_connected:
                    await self.client.disconnect()
            except Exception as e:
                print(
                    f"Error disconnecting Bluetooth: "
                    f"{type(e).__name__}: {e}"
                )
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

    def disconnected_callback(self, client):
        # Ignore callbacks from a previous BleakClient after a reconnect.
        if client is not self.client:
            print("Ignoring disconnect callback from stale BLE client.")
            return

        if self._disconnecting:
            return

        if self._connecting:
            print("BLE disconnect callback occurred during connection setup.")
            return

        if not self._connected_ready:
            return

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

        # IMPORTANT: do not set shutdown_event here.  The training recorder
        # needs to be able to recover the BLE connection instead of terminating
        # the entire data-collection session.
        self.bluetooth_disconnected.set()
        self.recording = False
        self.button_pressed = False

    def orientation_handler(self, sender, data):
        try:
            if len(data) < 8:
                self.invalid_packet_counts["orientation"] += 1
                print(
                    f"WARNING: Invalid orientation packet: "
                    f"{len(data)} bytes"
                )
                return

            self.notification_counts["orientation"] += 1
            orientation = self.decode_orientation(data)
            self.latest_orientation = orientation

            now = time.monotonic()
            self.last_orientation_time = now
            self.last_notification_time = now

            if self.recording:
                self.current_gesture.orientation.append(orientation)

        except Exception as e:
            print(
                f"WARNING: Orientation callback error: "
                f"{type(e).__name__}: {e}"
            )

    def motion_handler(self, sender, data):
        try:
            if len(data) < 18:
                self.invalid_packet_counts["motion"] += 1
                print(
                    f"WARNING: Invalid motion packet: "
                    f"{len(data)} bytes"
                )
                return

            self.notification_counts["motion"] += 1
            motion = self.decode_motion(data)
            self.latest_motion = motion

            now = time.monotonic()
            self.last_motion_time = now
            self.last_notification_time = now

            if self.recording:
                self.current_gesture.motion.append(motion)

        except Exception as e:
            print(
                f"WARNING: Motion callback error: "
                f"{type(e).__name__}: {e}"
            )

    def button_handler(self, sender, data):
        try:
            if not data:
                self.invalid_packet_counts["button"] += 1
                print("WARNING: Empty button packet")
                return

            self.notification_counts["button"] += 1
            pressed = self.decode_button(data)
            now = time.monotonic()
            self.last_button_time = now
            self.last_notification_time = now

            if pressed and not self.button_pressed:
                self.button_down_count += 1
                print("BUTTON DOWN")
                self.button_pressed = True
                self.recording = True
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

                if (
                    self.current_gesture.motion
                    or self.current_gesture.orientation
                ):
                    # O(1) transfer; do NOT deepcopy a long gesture here.
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
            w=self._signed_int16(data[0:2]) / 1024.0
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
            roll=self._signed_int16(data[16:18])
        )


def print_ble_diagnostics(wand):
    now = time.monotonic()
    print(
        "BLE diagnostics: "
        f"connected={wand.client.is_connected}, "
        f"motion={wand.notification_counts['motion']}, "
        f"orientation={wand.notification_counts['orientation']}, "
        f"button={wand.notification_counts['button']}, "
        f"button_down={wand.button_down_count}, "
        f"button_up={wand.button_up_count}, "
        f"last_any={now - wand.last_notification_time:.2f}s ago"
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

        while not wand.shutdown_event.is_set():
            if wand.bluetooth_disconnected.is_set():
                break
            await asyncio.sleep(0.2)

        if wand.bluetooth_disconnected.is_set():
            print("Bluetooth connection was lost.")
            print("Application will shut down.")
            print_ble_diagnostics(wand)

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
