import asyncio
from bleak import BleakClient, BleakScanner
import numpy
import zmq
import pandas as pd
import json
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

context = zmq.Context()

publisher = context.socket(zmq.PUB)
publisher.connect(f"tcp://{NERF_IP}:{NERF_PORT}")

position_queue = queue.Queue()
latest_position = None
is_pressed = False
lock = threading.Lock()
pos_data = {
    "mag_x": [],
    "mag_y": [],
    "mag_z": [],
    "acc_x": [],
    "acc_y": [],
    "acc_z": [],
    "pitch": [],
    "roll": [],
    "yaw": [],
    "time": []
}
@dataclass
class WandOrientationState:
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
class WandMotionState:
    timestamp: float = 0.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

def button_handler(sender, data):
    global is_pressed
    print("Button data:", list(data))
    is_pressed = data[0] == 1
    if data[0] == 1:
        if latest_position is not None:
            position_queue.put(latest_position)
        publisher.send_string(json.dumps({
            "spell": "nerf",
            "timestamp": time.time()
        }))

def position_handler(sender, data):
    global pos_data
    print("position data:", list(data))
    acc_x = numpy.int16(numpy.uint16(int.from_bytes(data[0:2], byteorder='little')))
    acc_y = numpy.int16(numpy.uint16(int.from_bytes(data[2:4], byteorder='little')))
    acc_z = numpy.int16(numpy.uint16(int.from_bytes(data[4:6], byteorder='little')))
    mag_x = numpy.int16(numpy.uint16(int.from_bytes(data[6:8], byteorder='little')))
    mag_y = numpy.int16(numpy.uint16(int.from_bytes(data[8:10], byteorder='little')))
    mag_z = numpy.int16(numpy.uint16(int.from_bytes(data[10:12], byteorder='little')))
    yaw = numpy.int16(numpy.uint16(int.from_bytes(data[12:14], byteorder='little')))
    pitch = numpy.int16(numpy.uint16(int.from_bytes(data[14:16], byteorder='little')))
    roll = numpy.int16(numpy.uint16(int.from_bytes(data[16:18], byteorder='little')))
    pos_data["mag_x"].append(mag_x)
    pos_data["mag_y"].append(mag_y)
    pos_data["mag_z"].append(mag_z)
    pos_data["acc_x"].append(acc_x)
    pos_data["acc_y"].append(acc_y)
    pos_data["acc_z"].append(acc_z)
    pos_data["pitch"].append(pitch)
    pos_data["roll"].append(roll)
    pos_data["yaw"].append(yaw)
    pos_data["time"].append(time.time())

    if pos_data["time"]:
        df = pd.DataFrame.from_dict(
            self.pos_data,
            orient='index'
        ).transpose()
        print("Calculating spell")
        performed_spell = classify_spell(df, self.spell_data)
        print(f"performed spell: {performed_spell}")
    with lock:
        latest_position = data

def disconnected(client):
    print("Disconnected")


class KanoWand:

    def __init__(self, address):
        self.address = address
        self.client = BleakClient(address)

        self.latest_position = None
        self.button_pressed = False

        self.position_queue = queue.Queue()

    async def connect(self):
        await self.client.connect(timeout=30)

        await self.client.start_notify(
            MOTION_UUID,
            self.position_handler
        )

        await self.client.start_notify(
            BUTTON_UUID,
            self.button_handler
        )

    async def disconnect(self):
        await self.client.disconnect()

    def position_handler(self, sender, data):
        self.latest_position = self.decode_position(data)

    def button_handler(self, sender, data):
        pressed = self.decode_button(data)

        if pressed and self.latest_position is not None:
            self.position_queue.put(self.latest_position)

    def decode_position(self, data):
        # decode bytes into x,y,z...
        return data

    def decode_button(self, data):
        return data[0] == 1

async def main():
    # lient = BleakClient(WAND_ADDRESS)
    
    # async with BleakClient(WAND_ADDRESS) as client:
    #     print("Connected: ", client.is_connected)
    #     await client.start_notify(BUTTON_UUID, button_handler)
    #     # await client.start_notify(MOTION_UUID, position_handler)
    #     # await client.start_notify(QUATERNIONS_UUID, orientation_handler)
    #     print("Listening...")
    #     await asyncio.sleep(300)

    wand = KanoWand(WAND_ADDRESS)
    await wand.connect()

    while True:
        position = wand.position_queue.get()
        print(position)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        publisher.close()
        context.term()