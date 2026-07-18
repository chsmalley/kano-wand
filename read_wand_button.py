import asyncio
from bleak import BleakClient, BleakScanner

BUTTON_UUID = "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"

def button_handler(sender, data):
    print("Button data:", list(data))

async def main():
    device = await BleakScanner.find_device_by_filter(
        lambda d, ad: d.name and "Kano" in d.name
    )

    if device is None:
        print("Wand not found")
        return

    async with BleakClient(device) as client:
        print("Connected")

        await client.start_notify(BUTTON_UUID, button_handler)

        print("Listening...")
        await asyncio.sleep(300)

asyncio.run(main())