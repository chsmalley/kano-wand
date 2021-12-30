import sys
from .kano_wand import Wand
from bluepy.btle import DefaultDelegate, Scanner
import time
import pandas as pd

SPELLS = [
    "STUPEFY",
    "WINGARDIUM_LEVIOSA",
    "REDUCIO",
    "FLIPENDO",
    "EXPELLIARMUS",
    "INCENDIO",
    "LUMOS",
    "LOCOMOTOR",
    "ENGORGIO",
    "AGUAMENTI",
    "AVIS",
    "REDUCTO"
]

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
        self.spells = iter(SPELLS)
        self.current_spell = next(self.spells)
        print(f"press button and perform spell {self.current_spell}")
        print("release button when finished")

    def post_connect(self):
        self.subscribe_button()
        self.subscribe_position()

    def on_position(self, x, y, z, w):
        if self.pressed:
            print("saving position: {} {} {} {}".format(x, y, z, w))
            self.data["time"].append(time.time())
            self.data["x"].append(x)
            self.data["y"].append(y)
            self.data["z"].append(z)
            self.data["w"].append(w)

    def on_button(self, pressed):
        print("on_button: {}".format(pressed))
        print("self pressed: {}".format(self.pressed))
        # When button is released
        if self.pressed and not pressed:
            # Save data from previous spell 
            pd.DataFrame.from_dict(
                self.data,
                orient='index'
            ).transpose().to_csv(self.current_spell + ".csv",
                                index=False)

            # Tell user which spell to perform next
            self.current_spell = next(self.spells)
            print(f"current pressed status: {self.pressed}")
            print(f"press button and perform spell {self.current_spell}")
            print("release button when finished")
        self.pressed = pressed
        print(f"slef pressed after: {self.pressed}")

    # def post_disconnect(self):
    #     pd.DataFrame.from_dict(self.data,
    #                            orient='index').transpose().to_csv("tmp.csv",
    #                                                               index=False)


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
        timeout: float=5.0,
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
        print("after scan")
        if connect:
            print("before connect")
            self.wand.connect()
            print("after connect")
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
            if device.addr == self._kano_mac.lower():
                self.kano_device = device
                if self.debug:
                    print("found kano wand")
            if self.kano_device is not None:
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
