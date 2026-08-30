import asyncio
from bleak import BleakClient, BleakScanner
import numpy
import zmq
import pandas as pd
import json
from copy import deepcopy
import time
import queue
import threading
from dataclasses import dataclass

@dataclass
class WandState:
    timestamp: float = 0.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    button: bool = False

# BUTTON_UUID = "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
BUTTON_UUID = '64A7000D-F691-4B93-A6F4-0968F5B648F8'
SERVICE_UUID = '64A70011-F691-4B93-A6F4-0968F5B648F8'
QUATERNIONS_UUID = '64A70002-F691-4B93-A6F4-0968F5B648F8'
MOTION_UUID = '64A7000C-F691-4B93-A6F4-0968F5B648F8'
MAGN_CALIBRATE_UUID = '64A70021-F691-4B93-A6F4-0968F5B648F8'
QUATERNIONS_RESET_UUID = '64A70004-F691-4B93-A6F4-0968F5B648F8'

WAND_ADDRESS = "D0:1F:65:71:51:32"

# ZeroMQ
NERF_IP = "192.168.0.26"  # IP of NERF raspberry pi
NERF_PORT = 5555
# ----------------------------------------------------
# ZeroMQ
# ----------------------------------------------------


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

class KanoWand:

    def __init__(self, address):
        self.address = address
        self.client = BleakClient(address)

        self.latest_motion = WandMotionState()
        self.latest_orientation = WandOrientationState()
        self.button_pressed = False
        self.recording = False

        self.current_motion = []
        self.spell_queue = queue.Queue()
        # ZeroMQ publisher
        context = zmq.Context()
        self.publisher = context.socket(zmq.PUB)
        self.publisher.connect(f"tcp://{NERF_IP}:{NERF_PORT}")

    async def connect(self):
        await self.client.connect(timeout=20)

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

    async def disconnect(self):
        await self.client.disconnect()

    def orientation_handler(self, sender, data):
        self.latest_orientation = self.decode_orientation(data)

    def motion_handler(self, sender, data):
        self.latest_motion = self.decode_motion(data)

    def button_handler(self, sender, data):
        pressed = self.decode_button(data)
        # Start recording
        if pressed and not self.recording:
            self.recording = True
            self.current_gesture = []
        # Stop recording
        elif not pressed and self.recording:
            self.recording = False
            if self.latest_motion:
                self.spell_queue.put(deepcopy(self.latest_motion))

    def decode_button(self, data):
        return data[0] == 1

    def decode_orientation(self, data):
        w = numpy.int16(numpy.uint16(int.from_bytes(data[0:2], byteorder='little')))
        x = numpy.int16(numpy.uint16(int.from_bytes(data[2:4], byteorder='little')))
        y = numpy.int16(numpy.uint16(int.from_bytes(data[4:6], byteorder='little')))
        z = numpy.int16(numpy.uint16(int.from_bytes(data[6:8], byteorder='little')))
        w = w / 1024
        x = x / 1024
        y = y / 1024
        z = z / 1024
        return WandOrientationState(
            time.time(),
            x,
            y,
            z,
            w
        )

    def decode_motion(self, data):
        acc_x = numpy.int16(numpy.uint16(int.from_bytes(data[0:2], byteorder='little')))
        acc_y = numpy.int16(numpy.uint16(int.from_bytes(data[2:4], byteorder='little')))
        acc_z = numpy.int16(numpy.uint16(int.from_bytes(data[4:6], byteorder='little')))
        mag_x = numpy.int16(numpy.uint16(int.from_bytes(data[6:8], byteorder='little')))
        mag_y = numpy.int16(numpy.uint16(int.from_bytes(data[8:10], byteorder='little')))
        mag_z = numpy.int16(numpy.uint16(int.from_bytes(data[10:12], byteorder='little')))
        yaw = numpy.int16(numpy.uint16(int.from_bytes(data[12:14], byteorder='little')))
        pitch = numpy.int16(numpy.uint16(int.from_bytes(data[14:16], byteorder='little')))
        roll = numpy.int16(numpy.uint16(int.from_bytes(data[16:18], byteorder='little')))
        return WandMotionState(
            time.time(),
            mag_x,
            mag_y,
            mag_z,
            acc_x,
            acc_y,
            acc_z,
            pitch,
            roll,
            yaw            
        )

def classify_spell(gesture):
    print(gesture)
    return "None"

def classify_worker(wand):
    while True:
        gesture = wand.spell_queue.get()
        spell = classify_spell(gesture)
        print(f"Detected: {spell}")
        wand.spell_queue.task_done()
        time.sleep(0.1)

async def main():

    wand = KanoWand(WAND_ADDRESS)
    await wand.connect()

    threading.Thread(
        target=classify_worker,
        args=(wand,),
        daemon=True
    ).start()

    while True:
        print(wand.recording)
        time.sleep(0.5)

if __name__ == "__main__":
    asyncio.run(main())
