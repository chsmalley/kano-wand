from bluepy import btle
from enum import Enum

class _IO(Enum):
    """Enum containing _IO UUIDs"""
    SERVICE = '64A70012-F691-4B93-A6F4-0968F5B648F8'
    BATTERY_CHAR = '64A70007-F691-4B93-A6F4-0968F5B648F8'
    USER_BUTTON_CHAR = '64A7000D-F691-4B93-A6F4-0968F5B648F8'
    VIBRATOR_CHAR = '64A70008-F691-4B93-A6F4-0968F5B648F8'
    LED_CHAR = '64A70009-F691-4B93-A6F4-0968F5B648F8'
    KEEP_ALIVE_CHAR = '64A7000F-F691-4B93-A6F4-0968F5B648F8'

class BluetoothLEConnector:
    def __init__(self, target_device_name, target_device_address=None):
        self.target_device_name = target_device_name
        self.target_device_address = target_device_address
        self.peripheral = None

    def discover_device(self):
        scanner = btle.Scanner()
        devices = scanner.scan(2)  # Scan for 2 seconds

        for device in devices:
            print(f"device: {device}")
            print(f"device name: {device.getValueText(9)}")
            if self.target_device_name in device.getValueText(9):  # Check the local name
                self.target_device_address = device.addr
                return True
        return False

    def connect(self):
        if not self.target_device_address:
            if not self.discover_device():
                print(f"Device '{self.target_device_name}' not found.")
                return

        try:
            self.peripheral = btle.Peripheral(self.target_device_address)
            print(f"Connected to {self.target_device_name} at address {self.target_device_address}.")
        except btle.BTLEException as e:
            print(f"Failed to connect: {e}")
            self.peripheral = None

    def read_characteristic(self, service_uuid, char_uuid):
        if self.peripheral:
            service = self.peripheral.getServiceByUUID(service_uuid)
            characteristic = service.getCharacteristics(forUUID=char_uuid)[0]
            value = characteristic.read()
            return value
        else:
            print("Not connected to any device.")
            return None

    def disconnect(self):
        if self.peripheral:
            self.peripheral.disconnect()
            print(f"Disconnected from {self.target_device_name}.")
        else:
            print("Not connected to any device.")


if __name__ == "__main__":
    device_name = "Kano-Wand-71-51-32"  # Replace with the name of your Bluetooth LE device
    bluetooth_le_connector = BluetoothLEConnector(device_name)
    bluetooth_le_connector.connect()

    if bluetooth_le_connector.peripheral:
        battery_level = bluetooth_le_connector.read_characteristic(_IO.SERVICE.value,
                                                                   _IO.BATTERY_CHAR.value)
        print(f"Battery Level: {int.from_bytes(battery_level, byteorder='little')}%")

    bluetooth_le_connector.disconnect()
