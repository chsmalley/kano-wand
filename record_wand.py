import sys
import os
import shutil
from .kano_wand import Wand
from bluepy.btle import DefaultDelegate, Scanner
import time
import pandas as pd
from scipy.spatial.transform import Rotation as R
import numpy as np

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

def make_euler(q):
    q_norm = q / np.linalg.norm(q)
    r = R.from_quat(q_norm)
    euler = r.as_euler('zyx', degrees=True)
    print(f"roll: {euler[0]:.2f}\tyaw: {euler[1]:.2f}\tpitch: {euler[2]:.2f}")
    return euler

class RecordWand(Wand):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pressed = False
        self.pos_save_dir = os.path.expanduser("~/wand_position_data")
        self.quat_save_dir = os.path.expanduser("~/wand_quaternion_data")
        if os.path.exists(self.pos_save_dir) and os.path.isdir(self.pos_save_dir):
           shutil.rmtree(self.pos_save_dir)
        os.mkdir(self.pos_save_dir)
        if os.path.exists(self.quat_save_dir) and os.path.isdir(self.quat_save_dir):
           shutil.rmtree(self.quat_save_dir)
        os.mkdir(self.quat_save_dir)
        self.positions = []
        self.quat_data = {
            "time": [],
            "x": [],
            "y": [],
            "z": [],
            "w": []
        }
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
        self.spells = iter(SPELLS)
        self.current_spell = next(self.spells)
        time.sleep(10)
        print(f"init: press button and perform spell {self.current_spell}")
        print("init: release button when finished")

    def post_connect(self):
        print("Post connect")
        self.subscribe_button()
        print("subscribed button")
        self.subscribe_position()
        print("subscribed position")
        # self.subscribe_orientation()
        print("Post connect complete")

    def on_orientation(self, w, x, y, z):
        if self.pressed:
            # euler = make_euler(np.array((w, x, y, z)))
            self.quat_data["time"].append(time.time())
            self.quat_data["x"].append(x)
            self.quat_data["y"].append(y)
            self.quat_data["z"].append(z)
            self.quat_data["w"].append(w)

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
            # print("on position and pressed")
            # print(f"{mag_x}\t{mag_y}\t{mag_z}")
            # print(f"{acc_x}\t{acc_y}\t{acc_z}")
            # print(f"{yaw}\t{pitch}\t{roll}")
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
            print("saving data")
            # Save data from previous spell 
            # if self.quat_data["time"]:
            #     pd.DataFrame.from_dict(
            #         self.quat_data,
            #         orient='index'
            #     ).transpose().to_csv(
            #         os.path.join(self.quat_save_dir, self.current_spell + "_quat.csv"),
            #         index=False)
            #     self.quat_data = {
            #         "time": [],
            #         "x": [],
            #         "y": [],
            #         "z": [],
            #         "w": []
            #     }
            if self.pos_data["time"]:
                # print(f"pos data: {self.pos_data}")
                pd.DataFrame.from_dict(
                    self.pos_data,
                    orient='index'
                ).transpose().to_csv(
                    os.path.join(self.pos_save_dir, self.current_spell + "_pos.csv"),
                    index=False)
                print("saved data")
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

            # Tell user which spell to perform next
            self.current_spell = next(self.spells)
            print(f"press button and perform spell {self.current_spell}")
            print("release button when finished")
        self.pressed = pressed

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
        if connect:
            self.wand.connect()
            print("wand connected")
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
                self.wand = RecordWand(self.kano_device,
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
