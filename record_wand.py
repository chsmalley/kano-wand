import sys
from .kano_wand.kano_wand import Wand
from bluepy.btle import DefaultDelegate, Scanner
import time
import pandas as pd


class RecordWand(Wand):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pressed = False
        self.positions = []
        self.data = {
            "time": [],
            "x": [],
            "y": [],
            "z": [],
            "w": []
        }

    def post_connect(self):
        self.subscribe_button()
        self.subscribe_position()

    def on_position(self, x, y, z, w):
        print("on_position")
        if self.pressed:
            print("saving position: {} {} {} {}".format(x, y, z, w))
            self.data["time"].append(time.time())
            self.data["x"].append(x)
            self.data["y"].append(y)
            self.data["z"].append(z)
            self.data["w"].append(w)

    def on_button(self, pressed):
        self.pressed = pressed
        print("on_button: {}".format(pressed))
        if not pressed:
            print("current pressed status: {self.pressed}")

    def post_disconnect(self):
        pd.DataFrame.from_dict(self.data,
                               orient='index').transpose().to_csv("tmp.csv",
                                                                  index=False)


class WandScanner(DefaultDelegate):
    """A scanner class to connect to wands
    """
    def __init__(
        self,
        kano_mac: str,
        debug: bool=False
    ):
        """Create a new scanner

        Keyword Arguments:
            wand_class {class} -- Class to use when connecting to wand (default: {Wand})
            debug {bool} -- Print debug messages (default: {False})
        """
        super().__init__()
        self.wand_class = RecordWand
        self.debug = debug
        self._kano_mac = kano_mac
        self.kano_device = None
        self.wand = None
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

        if isNewDev:
            # Perform initial detection attempt
            if device.addr == self._kano_mac:
                self.kano_device = device
                if self.debug:
                    print("found kano wand")
            if device.addr == self._sphero_mac:
                self.sphero_device = device
                if self.debug:
                    print("found sphero device")
            if self.kano_device is not None and self.sphero_device is not None:
                if self.debug:
                    print("creating sphero wand")
                self.wand = RecordWand(device,
                                       debug=self.debug)


if __name__ == '__main__':
    kano_mac = sys.argv[1]
    # Create a new wand scanner
    shop = WandScanner(kano_mac)
    wand = None
    try:
        # While we don't have any wands
        while wand is None:
            print("Scanning...")
            # Scan for wands and automatically connect
            wand = shop.scan(connect=True)
            print("after scan")
        print("out of while loop")
    # Detect keyboard interrupt and disconnect wands
    except KeyboardInterrupt:
        wand.disconnect()
