from .sphero_mini import sphero_mini
import sys
import queue
from .kano_wand import Wand
import moosegesture
from typing import Any, List
from bluepy.btle import DefaultDelegate, Scanner


class SpheroWand(Wand):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pressed = False
        # self.sphero_mac = "eb:b6:31:82:7c:f0"
        # if "sphero_mac" in kwargs.keys():
        #     self.sphero_mac = kwargs["sphero_mac"]
        # else:
        #     print("NO SPHERO MAC ADDRESS")
        self.sphero = None
        self.positions = queue.Queue()

    def post_connect(self):
        self.subscribe_button()
        self.subscribe_position()
        self.sphero_mac = "EB:B6:31:82:7C:F0"
        self.sphero = init_sphero(self.sphero_mac)

    def on_position(self, x, y, pitch, roll):
        if self.pressed:
            print("reading position: {} {}".format(x, y))
            # Add the mouse's position to the positions array
            self.positions.put(tuple([x, -1 * y]))
            if self.positions.qsize() > 10:
                tmp_p = []
                for i in range(self.positions.qsize()):
                    tmp_p.append(self.positions.get())
                print("temp p: {}".format(tmp_p))
                gesture = moosegesture.getGesture(tmp_p)
                print("gesture: {}".format(gesture))
                # self.sphero.setLEDColor(red=1, green=255, blue=0)
                control_sphero_gesture(self.sphero, gesture)
    def on_button(self, pressed):
        if self.sphero is None:
            return
        self.pressed = pressed
        # # print("position sizea: {}".format(self.positions.qsize()))
        # tmp_p = []
        # # while self.positions.not_empty():
        # for i in range(self.positions.qsize()):
        #     tmp_p.append(self.positions.get())
        #     if (i + 1) % 10 == 0:
        #         print("temp p: {}".format(tmp_p))
        #         gesture = moosegesture.getGesture(tmp_p)
        #         print("gesture: {}".format(gesture))
        #         # self.sphero.setLEDColor(red=1, green=255, blue=0)
        #         control_sphero_gesture(self.sphero, gesture)
        #         tmp_p = []


    def disconnect(self):
        super().disconnect()
        if self.sphero is not None:
            self.sphero.sleep()
            self.sphero.disconnect()

def control_sphero_gesture(
    sphero: sphero_mini,
    gesture: List[str]
) -> None:
    heading = 0
    speed = 0
    for g in gesture:
        if "R" in gesture:
            heading += 50
        if "L" in gesture:
            heading -+ 50
        if "U" in gesture:
            speed += 50
        if "D" in gesture:
            speed -= 50
    sphero.roll(speed, heading)
    sphero.wait(0.5)  # Keep rolling for two seconds


def init_sphero(MAC: str) -> sphero_mini:
    # Connect:
    sphero = sphero_mini(MAC, verbosity = 1)

    # battery voltage
    sphero.getBatteryVoltage()
    print(f"Battery voltage: {sphero.v_batt}v")

    # firmware version number
    sphero.returnMainApplicationVersion()
    print(f"Firmware version: {'.'.join(str(x) for x in sphero.firmware_version)}")

    #Configure sensors to make IMU_yaw values available
    sphero.configureSensorMask(
        # sample_rate_divisor = 0x25, # Must be > 0
        # packet_count = 0,
        IMU_pitch = True,
        IMU_roll = True,
        IMU_yaw = True,
        IMU_acc_x = True,
        IMU_acc_y = True,
        IMU_acc_z = True,
        IMU_gyro_x = True,
        IMU_gyro_y = True,
        IMU_gyro_z = True
    )
    sphero.configureSensorStream()
    return sphero


class WandSpheroScanner(DefaultDelegate):
    """A scanner class to connect to wands
    """
    def __init__(
        self,
        kano_mac: str,
        sphero_mac: str,
        debug: bool=False
    ):
        """Create a new scanner

        Keyword Arguments:
            wand_class {class} -- Class to use when connecting to wand (default: {Wand})
            debug {bool} -- Print debug messages (default: {False})
        """
        super().__init__()
        self.wand_class = SpheroWand
        self.debug = debug
        self._kano_mac = kano_mac
        self._sphero_mac = sphero_mac
        self.kano_device = None
        self.sphero_device = None
        self.wand = None
        print("kano mac: {}".format(self._kano_mac))
        self._scanner = Scanner().withDelegate(self)

    def scan(
        self,
        timeout: float=1.0,
        connect: bool=False
    ):
        """Scan for devices

        Keyword Arguments:
            timeout {float} -- Timeout before returning from scan (default: {1.0})
            connect {bool} -- Connect to the wands automatically (default: {False})

        Returns {Wand} -- wand objects
        """

        if self.debug:
            print("Scanning for {} seconds...".format(timeout))

        self._scanner.scan(timeout)
        if connect:
            self.wand.connect()
        return self.wand

    def handleDiscovery(self, device, isNewDev, isNewData):
        """Check if the device matches

        Arguments:
            device {bluepy.ScanEntry} -- Device data
            isNewDev {bool} -- Whether the device is new
            isNewData {bool} -- Whether the device has already been seen
        """
        print("device: {}".format(device.addr))
        if isNewDev:
            # Perform initial detection attempt
            print("device.addr: {}".format(device.addr))
            if device.addr == self._kano_mac:
                self.kano_device = device
                if self.debug:
                    print("found kano wand")
            # if device.addr == self._sphero_mac:
            #     self.sphero_device = device
            #     if self.debug:
            #         print("found sphero device")
            # if self.kano_device is not None and self.sphero_device is not None:
            if self.kano_device is not None:
                if self.debug:
                    print("creating sphero wand")
                self.wand = SpheroWand(self.kano_device,
                                       # sphero_mac=self._sphero_mac,
                                       debug=self.debug)


if __name__ == '__main__':
    print("On Linux, use 'sudo hcitool lescan' to find your Sphero Mini's MAC address")
    kano_mac = sys.argv[1]
    sphero_mac = sys.argv[2]
    # Create a new wand scanner
    shop = WandSpheroScanner(kano_mac, sphero_mac)
    wand = None
    try:
        # While we don't have any wands
        while wand is None:
            # print("Scanning...")
            # Scan for wands and automatically connect
            wand = shop.scan(connect=True)
            # print("after scan")
        print("out of while loop")
    # Detect keyboard interrupt and disconnect wands
    except KeyboardInterrupt as e:
        # print("keyboard interrupt")
        wand.disconnect()
