from bluepy import btle
from bluepy.btle import DefaultDelegate, Peripheral
from enum import Enum
import time


SCAN_TIME = 2

class _IO(Enum):
    """Enum containing _IO UUIDs"""
    SERVICE = '64A70012-F691-4B93-A6F4-0968F5B648F8'
    BATTERY_CHAR = '64A70007-F691-4B93-A6F4-0968F5B648F8'
    USER_BUTTON_CHAR = '64A7000D-F691-4B93-A6F4-0968F5B648F8'
    VIBRATOR_CHAR = '64A70008-F691-4B93-A6F4-0968F5B648F8'
    LED_CHAR = '64A70009-F691-4B93-A6F4-0968F5B648F8'
    KEEP_ALIVE_CHAR = '64A7000F-F691-4B93-A6F4-0968F5B648F8'

class WandDelegate(DefaultDelegate):
    '''
    '''

    def __init__(self):
        DefaultDelegate.__init__(self)

    def clear_notification(self):
        self.notification_ack = "DEFAULT ACK"
        self.notification_seq = -1

    def bits_to_num(self, bits):
        '''
        This helper function decodes bytes from sensor packets
        into single precision floats. Encoding follows the
        the IEEE-754 standard.
        '''
        num = int(bits, 2).to_bytes(len(bits) // 8, byteorder='little')
        num = struct.unpack('f', num)[0]
        return num

    def handleNotification(self, cHandle, data):
        '''
        '''
        # parse each byte separately (sometimes they arrive simultaneously)
        for data_byte in data:
            print(f"data_bype: {data}")
            self.notificationPacket.append(data_byte) # Add new byte to packet list


class Wand:
    def __init__(self, device_name, device_address=None):
        self.device_name = device_name
        self.device_address = device_address
        self.peripheral = None
        self.delegate = WandDelegate()

    def discover_device(self):
        device_found = False
        scanner = btle.Scanner()
        devices = scanner.scan(SCAN_TIME)
        for device in devices:
            dev_name = device.getValueText(9)
            print(f"device: {device}")
            print(f"device name: {dev_name}")
            if dev_name is not None and \
                    self.target_device_name in dev_name:
                print("name passed")
                self.target_device_address = device.addr
                device_found = True
                break
        print("end of discover device")
        return device_found

    def connect(self):
        if not self.target_device_address:
            if not self.discover_device():
                print(f"Device '{self.target_device_name}' not found.")
                return False
        print(f"connecting to {self.target_device_address}")
        while self.peripheral is None:
            try:
                self.peripheral = /
                    btle.Peripheral(self.target_device_address,
                                    btle.ADDR_TYPE_RANDOM)
            except btle.BTLEException as e:
                p = None
                print(f"Failed to connect: {e}")

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
    """
    mac_address = "D0:1F:65:71:51:32"
    """
    
    device_name = "Kano-Wand-71-51-32"
    print("initializing connector")
    wand = Wand(device_name)
    print("starting connect")
    wand.connect()
    print("after connect")
    if wand.peripheral:
        print("read char")
        battery_level = /
            wand.read_characteristic(_IO.SERVICE.value,
                                     _IO.BATTERY_CHAR.value)
        print(f"Battery Level: {int.from_bytes(battery_level, byteorder='little')}%")
    print("disconnect")
    wand.disconnect()
