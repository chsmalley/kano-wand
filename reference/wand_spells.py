import sys
import os
from .kano_wand import Wand
from bluepy.btle import DefaultDelegate, Scanner
import time
import pandas as pd
from .utils import classify_spell
from typing import List, Dict
import glob


class SpellWand(Wand):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pressed = False
        self.positions = []
        self.pos_data = {
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
        spell_dirname = os.path.join(os.path.abspath(''), 'test_data', 'position_data')
        self.spell_data = {}
        for filename in glob.glob(spell_dirname + "*.csv"):
            name = os.path.splitext(os.path.basename(filename))[0]
            self.spell_data[name] = pd.read_csv(filename)

    def post_connect(self):
        print("Post connect")
        self.subscribe_button()
        print("subscribed button")
        self.subscribe_position()
        print("subscribed position")
        print("Post connect complete")

    def on_position(
        self,
        mag_x,
        mag_y,
        mag_z,
        acc_x,
        acc_y,
        acc_z,
        pitch,
        roll,
        yaw
    ):
        if self.pressed:
            print("on position and pressed")
            print(f"{mag_x}\t{mag_y}\t{mag_z}")
            print(f"{acc_x}\t{acc_y}\t{acc_z}")
            print(f"{yaw}\t{pitch}\t{roll}")
            self.pos_data["mag_x"].append(mag_x)
            self.pos_data["mag_y"].append(mag_y)
            self.pos_data["mag_z"].append(mag_z)
            self.pos_data["acc_x"].append(acc_x)
            self.pos_data["acc_y"].append(acc_y)
            self.pos_data["acc_z"].append(acc_z)
            self.pos_data["pitch"].append(pitch)
            self.pos_data["roll"].append(roll)
            self.pos_data["yaw"].append(yaw)
            self.pos_data["time"].append(time.time())

    def on_button(self, pressed):
        # self.reset_position()
        print("on_button: {}".format(pressed))
        print("self pressed: {}".format(self.pressed))
        # When button is released
        if self.pressed and not pressed:
            if self.pos_data["time"]:
                df = pd.DataFrame.from_dict(
                    self.pos_data,
                    orient='index'
                ).transpose()
                print("Calculating spell")
                performed_spell = classify_spell(df, self.spell_data)
                print(f"performed spell: {performed_spell}")
                # Reset data
                self.pos_data = {
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
        self.pressed = pressed

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
        self.wand_class = SpellWand
        self.debug = debug
        self._kano_mac = kano_mac
        self.kano_device = None
        self.wand = None
        self._scanner = Scanner().withDelegate(self)

    def scan(
        self,
        timeout: float=5.0,
    ) -> SpellWand:
        """Scan for devices

        Keyword Arguments:
            timeout {float} -- Timeout before returning from scan (default: {1.0})

        Returns {Wand} -- wand objects
        """

        if self.debug:
            print("Scanning for {} seconds...".format(timeout))

        self._scanner.scan(timeout)
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
                self.wand = SpellWand(self.kano_device,
                                       debug=self.debug)


if __name__ == '__main__':
    kano_mac = sys.argv[1]
    # Create a new wand scanner
    shop = WandScanner(kano_mac)
    wand = None
    try:
        # While we don't have any wands
        while wand is None:
            # Scan for wands and automatically connect
            wand = shop.scan(connect=False)
    # Detect keyboard interrupt and disconnect wands
    except KeyboardInterrupt:
        if wand is not None:
            wand.disconnect()
    