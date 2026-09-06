import asyncio
from bleak import BleakClient
import numpy
import zmq
import time
import queue
import threading
from dataclasses import dataclass
from copy import deepcopy


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
# Data classes
# ============================================================

spells = [
    "Stupefy",
    "Wingardium Leviosa",
    "Reducio",
    "Flipendo",
    "Expelliarmus",
    "Incendio",
    "Lumos",
    "Locomotor",
    "Engorgio",
    "Aguamenti",
    "Avis",
    "Reducto"
]

@dataclass
class WandOrientationState:
    timestamp = time.monotonic()
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 0.0


@dataclass
class WandMotionState:
    timestamp = time.monotonic()

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

        # Used to shut down the classifier thread
        self.shutdown_event = threading.Event()

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

        print("Notifications started")

    async def disconnect(self):

        print("Disconnecting wand...")

        try:

            if self.client.is_connected:

                # Stop notifications first
                try:
                    await self.client.stop_notify(MOTION_UUID)
                except Exception:
                    pass

                try:
                    await self.client.stop_notify(QUATERNIONS_UUID)
                except Exception:
                    pass

                try:
                    await self.client.stop_notify(BUTTON_UUID)
                except Exception:
                    pass

                await self.client.disconnect()

        except Exception as e:

            print(f"Error disconnecting wand: {e}")

        print("Wand disconnected")

        # Close ZeroMQ
        try:
            self.publisher.close()
            self.context.term()
        except Exception:
            pass

    def disconnected_callback(self, client):

        print("Wand disconnected!")

    # ========================================================
    # Notification handlers
    # ========================================================

    def orientation_handler(self, sender, data):

        orientation = self.decode_orientation(data)

        self.latest_orientation = orientation

        if self.recording:
            self.current_gesture.orientation.append(
                deepcopy(orientation)
            )

    def motion_handler(self, sender, data):

        motion = self.decode_motion(data)

        self.latest_motion = motion

        if self.recording:
            self.current_gesture.motion.append(
                deepcopy(motion)
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

                self.spell_queue.put(
                    deepcopy(self.current_gesture)
                )

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

        w = numpy.int16(
            numpy.uint16(
                int.from_bytes(
                    data[0:2],
                    byteorder="little"
                )
            )
        )

        x = numpy.int16(
            numpy.uint16(
                int.from_bytes(
                    data[2:4],
                    byteorder="little"
                )
            )
        )

        y = numpy.int16(
            numpy.uint16(
                int.from_bytes(
                    data[4:6],
                    byteorder="little"
                )
            )
        )

        z = numpy.int16(
            numpy.uint16(
                int.from_bytes(
                    data[6:8],
                    byteorder="little"
                )
            )
        )

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

        acc_x = numpy.int16(
            numpy.uint16(
                int.from_bytes(
                    data[0:2],
                    byteorder="little"
                )
            )
        )

        acc_y = numpy.int16(
            numpy.uint16(
                int.from_bytes(
                    data[2:4],
                    byteorder="little"
                )
            )
        )

        acc_z = numpy.int16(
            numpy.uint16(
                int.from_bytes(
                    data[4:6],
                    byteorder="little"
                )
            )
        )

        mag_x = numpy.int16(
            numpy.uint16(
                int.from_bytes(
                    data[6:8],
                    byteorder="little"
                )
            )
        )

        mag_y = numpy.int16(
            numpy.uint16(
                int.from_bytes(
                    data[8:10],
                    byteorder="little"
                )
            )
        )

        mag_z = numpy.int16(
            numpy.uint16(
                int.from_bytes(
                    data[10:12],
                    byteorder="little"
                )
            )
        )

        yaw = numpy.int16(
            numpy.uint16(
                int.from_bytes(
                    data[12:14],
                    byteorder="little"
                )
            )
        )

        pitch = numpy.int16(
            numpy.uint16(
                int.from_bytes(
                    data[14:16],
                    byteorder="little"
                )
            )
        )

        roll = numpy.int16(
            numpy.uint16(
                int.from_bytes(
                    data[16:18],
                    byteorder="little"
                )
            )
        )

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
# Spell classification
# ============================================================
def classify_spell(gesture):

    motion = gesture.motion
    orientation = gesture.orientation

    print(
        f"Motion samples: {len(motion)}"
    )

    print(
        f"Orientation samples: {len(orientation)}"
    )

    # Classification...

    return "Not detected"


# ============================================================
# Classification worker
# ============================================================

def classify_worker(wand):

    print("Classifier thread started")

    while not wand.shutdown_event.is_set():

        try:

            # Wait up to 0.5 seconds so we can check
            # the shutdown event.
            gesture = wand.spell_queue.get(
                timeout=0.5
            )

        except queue.Empty:

            continue

        try:

            spell = classify_spell(gesture)

            print(f"Detected: {spell}")

        except Exception as e:

            print(f"Classification error: {e}")

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

        print("Running. Press Ctrl+C to stop.")

        while True:

            # Don't use .get() here.
            # latest_motion is just the latest state.
            motion = wand.latest_motion

            print(
                f"X={motion.acc_x} "
                f"Y={motion.acc_y} "
                f"Z={motion.acc_z}"
            )

            await asyncio.sleep(0.2)

    finally:

        print("Shutting down...")

        # Tell classifier thread to stop
        wand.shutdown_event.set()

        # Wait for classifier thread
        classifier_thread.join(timeout=2)

        # Safely disconnect Bluetooth
        await wand.disconnect()

        print("Shutdown complete")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    try:

        asyncio.run(main())

    except KeyboardInterrupt:

        print("\nCtrl+C received")
