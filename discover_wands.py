import asyncio
from bleak import BleakScanner, BleakClient

async def main():
    print("Scanning...")

    devices = await BleakScanner.discover(timeout=5)

    wand = None
    for d in devices:
        if d.name and "Kano" in d.name:
            wand = d
            print("Found:", d)
            break

    if wand is None:
        print("No Kano Wand found.")
        return

    async with BleakClient(wand.address) as client:
        print("\nConnected\n")

        for service in client.services:
            print(f"Service: {service.uuid}")

            for char in service.characteristics:
                print(
                    f"  Characteristic: {char.uuid} "
                    f"Properties: {char.properties}"
                )

asyncio.run(main())