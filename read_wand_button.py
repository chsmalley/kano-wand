import asyncio
from bleak import BleakClient, BleakScanner
import zmq
import json
import time
import subprocess

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


def button_handler(sender, data):
    print("Button data:", list(data))
    if data[0] == 1:
        publisher.send_string(json.dumps({
            "spell": "nerf",
            "timestamp": time.time()
        }))

def orientation_handler(sender, data):
    print("quaternion data:", list(data))

def disconnected(client):
    print("Disconnected")

async def main():
    client = BleakClient(WAND_ADDRESS)
    
    async with BleakClient(WAND_ADDRESS) as client:
        print("Connected: ", client.is_connected)
        await client.start_notify(BUTTON_UUID, button_handler)
        # await client.start_notify(QUATERNIONS_UUID, orientation_handler)
        print("Listening...")
        await asyncio.sleep(300)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        publisher.close()
        context.term()
