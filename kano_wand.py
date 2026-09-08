import asyncio
from bleak import BleakClient
import numpy as np
import zmq
import time
import queue
import threading
from dataclasses import dataclass
from copy import deepcopy
import joblib


# ============================================================
# Classification parameters
# ============================================================

MODEL_FILE = "spell_classifier.joblib"
NUM_SAMPLES = 100

# A prediction below this probability is treated as Unknown.
UNKNOWN_CONFIDENCE_THRESHOLD = 0.65

# Also require the winning class to beat the runner-up by this margin.
# This helps reject ambiguous gestures.
UNKNOWN_MARGIN_THRESHOLD = 0.15

try:
    classifier = joblib.load(MODEL_FILE)
    print(f"Loaded classifier: {MODEL_FILE}")
except FileNotFoundError:
    classifier = None
    print(f"WARNING: Classifier file not found: {MODEL_FILE}")
    print("Spell classification will be unavailable until a model is trained.")


# ============================================================
# Bluetooth UUIDs
# ============================================================

BUTTON_UUID = "64A7000D-F691-4B93-A6F4-0968F5B648F8"
SERVICE_UUID = "64A70011-F691-4B93-A6F4-0968F5B648F8"
QUATERNIONS_UUID = "64A70002-F691-4B93-A6F4-0968F5B648F8"
MOTION_UUID = "64A7000C-F691-4B93-A6F4-0968F5B648F8"
MAGN_CALIBRATE_UUID = "64A70021-F691-4B93-A6F4-0968F5B648F8"
QUATERNIONS_RESET_UUID = "64A70004-F691-4B93-A6F4-0968F5B648F8"

WAND_ADDRESS = "D0:1F:65:71:51:32"


# ============================================================
# ZeroMQ
# ============================================================

NERF_IP = "192.168.0.26"
NERF_PORT = 5555


# ============================================================
# Classification helpers
# ============================================================

def interpolate_stream(samples, timestamps, num_samples):
    """
    Resample a time series to a fixed number of samples.
    """

    if len(samples) == 0:
        return np.zeros((num_samples, 1))

    samples = np.asarray(samples, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)

    # Remove duplicate timestamps
    _, indices = np.unique(
        timestamps,
        return_index=True
    )

    indices = np.sort(indices)

    timestamps = timestamps[indices]
    samples = samples[indices]

    if len(timestamps) == 1:
        return np.repeat(
            samples,
            num_samples,
            axis=0
        )

    # We are interested in the shape of the gesture over its duration,
    # so normalize each stream's time axis to 0..1.
    old_time = np.linspace(
        0,
        1,
        len(timestamps)
    )

    new_time = np.linspace(
        0,
        1,
        num_samples
    )

    result = np.zeros(
        (num_samples, samples.shape[1])
    )

    for column in range(samples.shape[1]):
        result[:, column] = np.interp(
            new_time,
            old_time,
            samples[:, column]
        )

    return result


def gesture_to_features(gesture):
    """
    Convert a Gesture object into the same feature vector
    used during training.
    """

    # --------------------------------------------------------------
    # Motion
    # --------------------------------------------------------------

    if len(gesture.motion) > 0:

        timestamps = [
            sample.timestamp
            for sample in gesture.motion
        ]

        values = [
            [
                sample.mag_x,
                sample.mag_y,
                sample.mag_z,
                sample.acc_x,
                sample.acc_y,
                sample.acc_z,
                sample.pitch,
                sample.roll,
                sample.yaw
            ]
            for sample in gesture.motion
        ]

        motion = interpolate_stream(
            values,
            timestamps,
            NUM_SAMPLES
        )

    else:
        motion = np.zeros(
            (NUM_SAMPLES, 9)
        )

    # --------------------------------------------------------------
    # Orientation
    # --------------------------------------------------------------

    if len(gesture.orientation) > 0:

        timestamps = [
            sample.timestamp
            for sample in gesture.orientation
        ]

        values = [
            [
                sample.x,
                sample.y,
                sample.z,
                sample.w
            ]
            for sample in gesture.orientation
        ]

        orientation = interpolate_stream(
            values,
            timestamps,
            NUM_SAMPLES
        )

    else:
        orientation = np.zeros(
            (NUM_SAMPLES, 4)
        )

    # --------------------------------------------------------------
    # Combine motion + orientation
    # --------------------------------------------------------------

    combined = np.hstack([
        motion,
        orientation
    ])

    return combined.flatten()


def classify_spell(gesture):
    """
    Classify a completed Gesture.

    Returns:
        prediction: predicted spell or "Unknown"
        confidence: probability of the top prediction
        top_predictions: list of (spell, probability), highest first
    """

    if classifier is None:
        return "Unknown", 0.0, []

    features = gesture_to_features(gesture)
    features = features.reshape(1, -1)

    probabilities = classifier.predict_proba(features)[0]
    classes = classifier.classes_

    # Sort all classes from highest to lowest probability
    sorted_indices = np.argsort(probabilities)[::-1]

    top_predictions = [
        (
            str(classes[index]),
            float(probabilities[index])
        )
        for index in sorted_indices
    ]

    best_index = sorted_indices[0]
    prediction = str(classes[best_index])
    confidence = float(probabilities[best_index])

    # Probability gap between first and second choice.
    if len(sorted_indices) > 1:
        second_confidence = float(
            probabilities[sorted_indices[1]]
        )
        margin = confidence - second_confidence
    else:
        margin = confidence

    # Reject weak or ambiguous predictions.
    if (
        confidence < UNKNOWN_CONFIDENCE_THRESHOLD
        or margin < UNKNOWN_MARGIN_THRESHOLD
    ):
        prediction = "Unknown"

    return prediction, confidence, top_predictions


# ============================================================
# Data classes
# ============================================================

@dataclass
class WandOrientationState:
    timestamp: float = 0.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 0.0


@dataclass
class WandMotionState:
    timestamp: float = 0.0

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


# ============================================================
# Kano Wand
# ============================================================

class KanoWand:

    def __init__(self, address):

        self.address = address

        self.client = BleakClient(
            address,
            disconnected_callback=self.disconnected_callback
        )

        # Current state
        self.latest_motion = WandMotionState()
        self.latest_orientation = WandOrientationState()
        self.bluetooth_disconnected = threading.Event()
        
        # Button / recording state
        self.button_pressed = False
        self.recording = False

        # Current gesture being recorded
        self.current_gesture = Gesture(
            motion=[],
            orientation=[]
        )

        # Completed gestures waiting for classification
        self.spell_queue = queue.Queue()

        # Used to stop the classifier thread and the main loop
        self.shutdown_event = threading.Event()

        # Prevent the disconnect callback from reporting our own
        # intentional shutdown as an unexpected Bluetooth failure.
        self._disconnecting = False
        self._connecting = False
        self._connected_ready = False

        # ----------------------------------------------------
        # ZeroMQ
        # ----------------------------------------------------

        self.context = zmq.Context()
        self.publisher = self.context.socket(zmq.PUB)

        self.publisher.connect(
            f"tcp://{NERF_IP}:{NERF_PORT}"
        )

    # ========================================================
    # Bluetooth
    # ========================================================

    async def connect(self):

        self._disconnecting = False
        self._connecting = True
        self._connected_ready = False
        self.shutdown_event.clear()

        print("Connecting to wand...")

        await self.client.connect(timeout=30)

        print("Connected:", self.client.is_connected)

        await self.client.start_notify(
            MOTION_UUID,
            self.motion_handler
        )

        await self.client.start_notify(
            QUATERNIONS_UUID,
            self.orientation_handler
        )

        await self.client.start_notify(
            BUTTON_UUID,
            self.button_handler
        )
        self._connecting = False
        self._connected_ready = True
        self.bluetooth_disconnected.clear()
        self.shutdown_event.clear()
        print("Notifications started")

    async def disconnect(self):

        # Prevent the disconnected callback from looking like an
        # unexpected Bluetooth failure.
        self._disconnecting = True

        self.recording = False
        self.button_pressed = False

        print("Disconnecting wand...")

        try:

            if self.client.is_connected:

                # Stop notifications first.
                for uuid in (
                    MOTION_UUID,
                    QUATERNIONS_UUID,
                    BUTTON_UUID
                ):
                    try:
                        await self.client.stop_notify(uuid)
                    except Exception:
                        pass

                try:
                    await self.client.disconnect()
                except Exception as e:
                    print(f"Error disconnecting Bluetooth: {e}")

        except Exception as e:
            print(f"Error during wand shutdown: {e}")

        # Close ZeroMQ.
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
        if self._disconnecting:
            return

        if self._connecting:
            print("BLE disconnect callback occurred during connection setup.")
            return

        if not self._connected_ready:
            return
        print()
        print("WARNING: Wand Bluetooth connection was lost!")
        print("Stopping application...")
        self.recording = False
        self.button_pressed = False
        self.shutdown_event.set()

    # ========================================================
    # Notification handlers
    # ========================================================

    def orientation_handler(self, sender, data):

        orientation = self.decode_orientation(data)

        self.latest_orientation = orientation

        if self.recording:
            self.current_gesture.orientation.append(
                orientation
            )

    def motion_handler(self, sender, data):

        motion = self.decode_motion(data)

        self.latest_motion = motion

        if self.recording:
            self.current_gesture.motion.append(
                motion
        )

    def button_handler(self, sender, data):

        pressed = self.decode_button(data)

        if pressed and not self.button_pressed:

            print("BUTTON DOWN")

            self.button_pressed = True
            self.recording = True

            self.current_gesture = Gesture(
                motion=[],
                orientation=[]
            )

        elif not pressed and self.button_pressed:

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

                completed_gesture = self.current_gesture
                self.spell_queue.put(completed_gesture)

            self.current_gesture = Gesture(
                motion=[],
                orientation=[]
            )

    # ========================================================
    # Decoders
    # ========================================================

    def decode_button(self, data):

        return data[0] == 1

    def decode_orientation(self, data):

        w = int(np.int16(
            np.uint16(
                int.from_bytes(
                    data[0:2],
                    byteorder="little"
                )
            )
        ))

        x = int(np.int16(
            np.uint16(
                int.from_bytes(
                    data[2:4],
                    byteorder="little"
                )
            )
        ))

        y = int(np.int16(
            np.uint16(
                int.from_bytes(
                    data[4:6],
                    byteorder="little"
                )
            )
        ))

        z = int(np.int16(
            np.uint16(
                int.from_bytes(
                    data[6:8],
                    byteorder="little"
                )
            )
        ))

        w /= 1024
        x /= 1024
        y /= 1024
        z /= 1024

        return WandOrientationState(
            timestamp=time.monotonic(),
            x=x,
            y=y,
            z=z,
            w=w
        )

    def decode_motion(self, data):

        acc_x = int(np.int16(
            np.uint16(
                int.from_bytes(
                    data[0:2],
                    byteorder="little"
                )
            )
        ))

        acc_y = int(np.int16(
            np.uint16(
                int.from_bytes(
                    data[2:4],
                    byteorder="little"
                )
            )
        ))

        acc_z = int(np.int16(
            np.uint16(
                int.from_bytes(
                    data[4:6],
                    byteorder="little"
                )
            )
        ))

        mag_x = int(np.int16(
            np.uint16(
                int.from_bytes(
                    data[6:8],
                    byteorder="little"
                )
            )
        ))

        mag_y = int(np.int16(
            np.uint16(
                int.from_bytes(
                    data[8:10],
                    byteorder="little"
                )
            )
        ))

        mag_z = int(np.int16(
            np.uint16(
                int.from_bytes(
                    data[10:12],
                    byteorder="little"
                )
            )
        ))

        yaw = int(np.int16(
            np.uint16(
                int.from_bytes(
                    data[12:14],
                    byteorder="little"
                )
            )
        ))

        pitch = int(np.int16(
            np.uint16(
                int.from_bytes(
                    data[14:16],
                    byteorder="little"
                )
            )
        ))

        roll = int(np.int16(
            np.uint16(
                int.from_bytes(
                    data[16:18],
                    byteorder="little"
                )
            )
        ))

        return WandMotionState(
            timestamp=time.monotonic(),
            mag_x=mag_x,
            mag_y=mag_y,
            mag_z=mag_z,
            acc_x=acc_x,
            acc_y=acc_y,
            acc_z=acc_z,
            pitch=pitch,
            roll=roll,
            yaw=yaw
        )


# ============================================================
# Classification worker
# ============================================================

def classify_worker(wand):

    print("Classifier thread started")

    while not wand.shutdown_event.is_set():

        try:
            gesture = wand.spell_queue.get(
                timeout=0.5
            )

        except queue.Empty:
            continue

        try:

            prediction, confidence, top_predictions = (
                classify_spell(gesture)
            )

            print()
            print("=" * 60)
            print(f"Prediction: {prediction}")
            print(f"Confidence: {confidence:.1%}")

            print()
            print("Top predictions:")

            for spell, probability in top_predictions[:3]:
                print(
                    f"  {spell:25s} "
                    f"{probability:.1%}"
                )

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


# ============================================================
# Main
# ============================================================

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

        # Keep the asyncio event loop alive so Bleak can continue
        # processing Bluetooth notifications.
        while not wand.shutdown_event.is_set():

            await asyncio.sleep(0.2)

    except Exception as e:

        print()
        print(
            f"ERROR: {type(e).__name__}: {e}"
        )

    finally:

        print()
        print("Shutting down...")

        # Tell classifier thread to stop.
        wand.shutdown_event.set()

        # Give it time to finish anything currently being classified.
        classifier_thread.join(timeout=2)

        if classifier_thread.is_alive():
            print(
                "WARNING: Classifier thread did not stop "
                "within the timeout."
            )

        # Safely disconnect Bluetooth and ZeroMQ.
        await wand.disconnect()

        print("Shutdown complete")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    try:
        asyncio.run(main())

    except KeyboardInterrupt:
        # asyncio normally propagates Ctrl+C into main(), where the
        # finally block performs the actual cleanup.
        print()
        print("Ctrl+C received")
