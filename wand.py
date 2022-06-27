from bluepy.btle import DefaultDelegate, Peripheral, Scanner
from typing import NamedTuple
import threading

KANO_PREFIX = "Kano-Wand"
SPHERO_PREFIX = "TODO"

class _INFO(NamedTuple):
    """NamedTuple containing info UUIDs"""
    SERVICE = '64A70010-F691-4B93-A6F4-0968F5B648F8'
    ORGANIZATION_CHAR = '64A7000B-F691-4B93-A6F4-0968F5B648F8'
    SOFTWARE_CHAR = '64A70013-F691-4B93-A6F4-0968F5B648F8'
    HARDWARE_CHAR = '64A70001-F691-4B93-A6F4-0968F5B648F8'

class _IO(NamedTuple):
    """NamedTuple containing _IO UUIDs"""
    SERVICE = '64A70012-F691-4B93-A6F4-0968F5B648F8'
    BATTERY_CHAR = '64A70007-F691-4B93-A6F4-0968F5B648F8'
    USER_BUTTON_CHAR = '64A7000D-F691-4B93-A6F4-0968F5B648F8'
    VIBRATOR_CHAR = '64A70008-F691-4B93-A6F4-0968F5B648F8'
    LED_CHAR = '64A70009-F691-4B93-A6F4-0968F5B648F8'
    KEEP_ALIVE_CHAR = '64A7000F-F691-4B93-A6F4-0968F5B648F8'

class _SENSOR(NamedTuple):
    """NamedTuple containing sensor UUIDs"""
    SERVICE = '64A70011-F691-4B93-A6F4-0968F5B648F8'
    TEMP_CHAR = '64A70014-F691-4B93-A6F4-0968F5B648F8'
    QUATERNIONS_CHAR = '64A70002-F691-4B93-A6F4-0968F5B648F8'
    # RAW_CHAR = '64A7000A-F691-4B93-A6F4-0968F5B648F8'
    MOTION_CHAR = '64A7000C-F691-4B93-A6F4-0968F5B648F8'
    MAGN_CALIBRATE_CHAR = '64A70021-F691-4B93-A6F4-0968F5B648F8'
    QUATERNIONS_RESET_CHAR = '64A70004-F691-4B93-A6F4-0968F5B648F8'

class PATTERN(NamedTuple):
    """NamedTuple for wand vibration patterns"""
    REGULAR = 1
    SHORT = 2
    BURST = 3
    LONG = 4
    SHORT_LONG = 5
    SHORT_SHORT = 6
    BIG_PAUSE = 7


class KanoWand():
    def __init__(self, MACAddr):
        '''
        Class to handle bluetooth connection with wand device
        and methods to interface with device
        '''
        self.p = Peripheral(MACAddr)

        # Subscribe to notifications
        self.wand_delegate = WandDelegate(self) # Pass a reference to this instance when initializing
        self.p.setDelegate(WandDelegate)

    def connect(self):
        """
        FIXME: Originally from overwriting Peripheral connect
        """
        if self.debug:
            print("Connecting to {}...".format(self.name))

        super(WandDelegate, self).connect(self._dev)
        self._lock = threading.Lock()
        self.connected = True
        self.setDelegate(self)
        self._info_service = self.getServiceByUUID(_INFO.SERVICE.value)
        self._io_service = self.getServiceByUUID(_IO.SERVICE.value)
        self._sensor_service = self.getServiceByUUID(_SENSOR.SERVICE.value)

        self.post_connect()

        if self.debug:
            print("Connected to {}".format(self.name))

    def disconnect(self):
        self.connected = False
        self._orientation_subscribed = False
        self._position_subscribed = False
        self._button_subscribed = False
        self._temperature_subscribed = False
        self._battery_subscribed = False


class WandDelegate(DefaultDelegate):
    """A wand class to interact with the Kano wand
    """

    def __init__(self, device):
        """Create a new wand

        Arguments:
            device {bluepy.ScanEntry} -- Device information

        Keyword Arguments:
        """
        super().__init__(None)
        # Meta stuff
        self._dev = device
        self.name = device.getValueText(9)

        # Notification stuff
        self.connected = False
        self._orientation_callbacks = {}
        self._orientation_subscribed = False
        self._position_callbacks = {}
        self._position_subscribed = False
        self._button_callbacks = {}
        self._button_subscribed = False
        self._temperature_callbacks = {}
        self._temperature_subscribed = False
        self._battery_callbacks = {}
        self._battery_subscribed = False
        self._notification_thread = None
        # TODO: Is this just a unique number?
        self._orientation_notification_handle = 41
        self._position_notification_handle = 52
        self._button_notification_handle = 33
        self._temp_notification_handle = 56
        self._battery_notification_handle = 23

    def get_organization(self):
        """Get organization of device

        Returns {str} -- Organization name
        """
        with self._lock:
            if not hasattr(self, "_organization_handle"):
                handle = self._info_service.getCharacteristics(_INFO.ORGANIZATION_CHAR.value)[0]
                self._organization_handle = handle.getHandle()
            return self.readCharacteristic(self._organization_handle).decode("utf-8")

    def get_software_version(self):
        """Get software version

        Returns {str} -- Version number
        """
        with self._lock:
            if not hasattr(self, "_software_handle"):
                handle = self._info_service.getCharacteristics(_INFO.SOFTWARE_CHAR.value)[0]
                self._software_handle = handle.getHandle()
            return self.readCharacteristic(self._software_handle).decode("utf-8")

    def get_hardware_version(self):
        """Get hardware version

        Returns {str} -- Hardware version
        """
        with self._lock:
            if not hasattr(self, "_hardware_handle"):
                handle = self._info_service.getCharacteristics(_INFO.HARDWARE_CHAR.value)[0]
                self._hardware_handle = handle.getHandle()
            return self.readCharacteristic(self._hardware_handle).decode("utf-8")

    def get_battery(self):
        """Get battery level (currently only returns 0)

        Returns {str} -- Battery level
        """
        with self._lock:
            if not hasattr(self, "_battery_handle"):
                handle = self._io_service.getCharacteristics(_IO.BATTERY_CHAR.value)[0]
                self._battery_handle = handle.getHandle()
            return self.readCharacteristic(self._battery_handle).decode("utf-8")

    def get_button(self):
        """Get current button status

        Returns {bool} -- Button pressed status
        """
        with self._lock:
            if not hasattr(self, "_button_handle"):
                handle = self._io_service.getCharacteristics(_IO.USER_BUTTON_CHAR.value)[0]
                self._button_handle = handle.getHandle()

            data = self.readCharacteristic(self._button_handle)
            return data[0] == 1

    def get_temperature(self):
        """Get temperature

        Returns {str} -- Battery level
        """
        with self._lock:
            if not hasattr(self, "_temperature_handle"):
                handle = self._sensor_service.getCharacteristics(_SENSOR.TEMP_CHAR.value)[0]
                self._temperature_handle = handle.getHandle()
            return self.readCharacteristic(self._temperature_handle).decode("utf-8")

    def keep_alive(self):
        """Keep the wand's connection active

        Returns {bytes} -- Status
        """
        # Is not documented because it doesn't seem to work?
        if self.debug:
            print("Keeping wand alive.")

        with self._lock:
            if not hasattr(self, "_alive_handle"):
                handle = self._io_service.getCharacteristics(_IO.KEEP_ALIVE_CHAR.value)[0]
                self._alive_handle = handle.getHandle()
            return self.writeCharacteristic(self._alive_handle, bytes([1]), withResponse=True)

    def vibrate(self, pattern=PATTERN.REGULAR):
        """Vibrate wand with pattern

        Keyword Arguments:
            pattern {kano_wand.PATTERN} -- Vibration pattern (default: {PATTERN.REGULAR})

        Returns {bytes} -- Status
        """
        with self._lock:
            if isinstance(pattern, PATTERN):
                message = [pattern.value]
            else:
                message = [pattern]

            if self.debug:
                print("Setting LED to {}".format(message))

            if not hasattr(self, "_vibrator_handle"):
                handle = self._io_service.getCharacteristics(_IO.VIBRATOR_CHAR.value)[0]
                self._vibrator_handle = handle.getHandle()
            return self.writeCharacteristic(self._vibrator_handle, bytes(message), withResponse=True)

    def set_led(self, color="0x2185d0", on=True):
        """Set the LED's color

        Keyword Arguments:
            color {str} -- Color hex code (default: {"0x2185d0"})
            on {bool} -- Whether light is on or off (default: {True})

        Returns {bytes} -- Status
        """
        message = []
        if on:
            message.append(1)
        else:
            message.append(0)

        # I got this from Kano's node module
        color = int(color.replace("#", ""), 16)
        r = (color >> 16) & 255
        g = (color >> 8) & 255
        b = color & 255
        rgb = (((r & 248) << 8) + ((g & 252) << 3) + ((b & 248) >> 3))
        message.append(rgb >> 8)
        message.append(rgb & 0xff)

        if self.debug:
            print("Setting LED to {}".format(message))

        with self._lock:
            if not hasattr(self, "_led_handle"):
                handle = self._io_service.getCharacteristics(_IO.LED_CHAR.value)[0]
                self._led_handle = handle.getHandle()
            return self.writeCharacteristic(self._led_handle, bytes(message), withResponse=True)

    # SENSORS
    def on(self, event, callback):
        """Add an event listener

        Arguments:
            event {str} -- Event type, "position", "button", "temp", or "battery"
            callback {function} -- Callback function

        Returns {str} -- ID of the callback for removal later
        """
        if self.debug:
            print("Adding callback for {} notification...".format(event))

        id = None
        print(f"event: {event}")
        if event == "position":
            id = uuid.uuid4()
            self._position_callbacks[id] = callback
            self.subscribe_position()
        elif event == "orientation":
            id = uuid.uuid4()
            self._orientation_callbacks[id] = callback
            self.subscribe_orientation()
        elif event == "button":
            id = uuid.uuid4()
            self._button_callbacks[id] = callback
            self.subscribe_button()
        elif event == "temp":
            id = uuid.uuid4()
            self._temperature_callbacks[id] = callback
            self.subscribe_temperature()
        elif event == "battery":
            id = uuid.uuid4()
            self._battery_callbacks[id] = callback
            self.subscribe_battery()

        return id

    def off(self, uuid, continue_notifications=False):
        """Remove a callback

        Arguments:
            uuid {str} -- Remove a callback with its id

        Keyword Arguments:
            continue_notifications {bool} -- Keep notification thread running (default: {False})

        Returns {bool} -- If removal was successful or not
        """
        removed = False
        if self._position_callbacks.get(uuid) != None:
            removed = True
            self._position_callbacks.pop(uuid)
            if len(self._position_callbacks.values()) == 0:
                self.unsubscribe_position(continue_notifications=continue_notifications)
        elif self._orientation_callbacks.get(uuid) != None:
            removed = True
            self._orientation_callbacks.pop(uuid)
            if len(self._orientation_callbacks.values()) == 0:
                self.unsubscribe_orientation(continue_notifications=continue_notifications)
        elif self._button_callbacks.get(uuid) != None:
            removed = True
            self._button_callbacks.pop(uuid)
            if len(self._button_callbacks.values()) == 0:
                self.unsubscribe_button(continue_notifications=continue_notifications)
        elif self._temperature_callbacks.get(uuid) != None:
            removed = True
            self._temperature_callbacks.pop(uuid)
            if len(self._temperature_callbacks.values()) == 0:
                self.unsubscribe_temperature(continue_notifications=continue_notifications)
        elif self._battery_callbacks.get(uuid) != None:
            removed = True
            self._battery_callbacks.pop(uuid)
            if len(self._battery_callbacks.values()) == 0:
                self.unsubscribe_battery(continue_notifications=continue_notifications)

        if self.debug:
            if removed:
                print("Removed callback {}".format(uuid))
            else:
                print("Could not remove callback {}".format(uuid))

        return removed

    def subscribe_position(self):
        """Subscribe to position notifications and start thread if necessary
        """
        if self.debug:
            print("Subscribing to position notification")

        self._position_subscribed = True
        with self._lock:
            if not hasattr(self, "_position_handle"):
                handle = self._sensor_service.getCharacteristics(_SENSOR.MOTION_CHAR.value)[0]
                self._position_handle = handle.getHandle()

            self.writeCharacteristic(self._position_handle + 1, bytes([1, 0]))
        self._start_notification_thread()

    def unsubscribe_position(self, continue_notifications=False):
        """Unsubscribe to position notifications

        Keyword Arguments:
            continue_notifications {bool} -- Keep notification thread running (default: {False})
        """
        if self.debug:
            print("Unsubscribing from position notification")

        self._position_subscribed = continue_notifications
        with self._lock:
            if not hasattr(self, "_position_handle"):
                handle = self._sensor_service.getCharacteristics(_SENSOR.MOTION_CHAR.value)[0]
                self._position_handle = handle.getHandle()

            self.writeCharacteristic(self._position_handle + 1, bytes([0, 0]))

    def subscribe_orientation(self):
        """Subscribe to orientation notifications and start thread if necessary
        """
        if self.debug:
            print("Subscribing to orientation notification")
        print("subscribe orientation")
        self._orientation_subscribed = True
        with self._lock:
            print("subscribe orientation: with lock")
            if not hasattr(self, "_orientation_handle"):
                handle = self._sensor_service.getCharacteristics(_SENSOR.QUATERNIONS_CHAR.value)[0]
                self._orientation_handle = handle.getHandle()
            print("subscribe orientation: after set attribute")

            self.writeCharacteristic(self._orientation_handle + 1, bytes([1, 0]))
            print("subscribe orientation: after write characteristics")
        self._start_notification_thread()
        print("subscribe orientation: after start notification thread")

    def unsubscribe_orientation(self, continue_notifications=False):
        """Unsubscribe to orientation notifications

        Keyword Arguments:
            continue_notifications {bool} -- Keep notification thread running (default: {False})
        """
        if self.debug:
            print("Unsubscribing from orientation notification")

        self._orientation_subscribed = continue_notifications
        with self._lock:
            if not hasattr(self, "_orientation_handle"):
                handle = self._sensor_service.getCharacteristics(_SENSOR.QUATERNIONS_CHAR.value)[0]
                self._orientation_handle = handle.getHandle()

            self.writeCharacteristic(self._orientation_handle + 1, bytes([0, 0]))

    def subscribe_button(self):
        """Subscribe to button notifications and start thread if necessary
        """
        if self.debug:
            print("Subscribing to button notification")

        self._button_subscribed = True
        with self._lock:
            if not hasattr(self, "_button_handle"):
                handle = self._io_service.getCharacteristics(_IO.USER_BUTTON_CHAR.value)[0]
                self._button_handle = handle.getHandle()

            self.writeCharacteristic(self._button_handle + 1, bytes([1, 0]))
        self._start_notification_thread()

    def unsubscribe_button(self, continue_notifications=False):
        """Unsubscribe to button notifications

        Keyword Arguments:
            continue_notifications {bool} -- Keep notification thread running (default: {False})
        """
        if self.debug:
            print("Unsubscribing from button notification")

        self._button_subscribed = continue_notifications
        with self._lock:
            if not hasattr(self, "_button_handle"):
                handle = self._io_service.getCharacteristics(_IO.USER_BUTTON_CHAR.value)[0]
                self._button_handle = handle.getHandle()

            self.writeCharacteristic(self._button_handle + 1, bytes([0, 0]))

    def subscribe_temperature(self):
        """Subscribe to temperature notifications and start thread if necessary
        """
        if self.debug:
            print("Subscribing to temperature notification")

        self._temperature_subscribed = True
        with self._lock:
            if not hasattr(self, "_temp_handle"):
                handle = self._sensor_service.getCharacteristics(_SENSOR.TEMP_CHAR.value)[0]
                self._temp_handle = handle.getHandle()

            self.writeCharacteristic(self._temp_handle + 1, bytes([1, 0]))
        self._start_notification_thread()

    def unsubscribe_temperature(self, continue_notifications=False):
        """Unsubscribe to temperature notifications

        Keyword Arguments:
            continue_notifications {bool} -- Keep notification thread running (default: {False})
        """
        if self.debug:
            print("Unsubscribing from temperature notification")

        self._temperature_subscribed = continue_notifications
        with self._lock:
            if not hasattr(self, "_temp_handle"):
                handle = self._sensor_service.getCharacteristics(_SENSOR.TEMP_CHAR.value)[0]
                self._temp_handle = handle.getHandle()

            self.writeCharacteristic(self._temp_handle + 1, bytes([0, 0]))

    def subscribe_battery(self):
        """Subscribe to battery notifications and start thread if necessary
        """
        if self.debug:
            print("Subscribing to battery notification")

        self._battery_subscribed = True
        with self._lock:
            if not hasattr(self, "_battery_handle"):
                handle = self._io_service.getCharacteristics(_IO.BATTERY_CHAR .value)[0]
                self._battery_handle = handle.getHandle()

            self.writeCharacteristic(self._battery_handle + 1, bytes([1, 0]))
        self._start_notification_thread()

    def unsubscribe_battery(self, continue_notifications=False):
        """Unsubscribe to battery notifications

        Keyword Arguments:
            continue_notifications {bool} -- Keep notification thread running (default: {False})
        """
        if self.debug:
            print("Unsubscribing from battery notification")

        self._battery_subscribed = continue_notifications
        with self._lock:
            if not hasattr(self, "_battery_handle"):
                handle = self._io_service.getCharacteristics(_IO.BATTERY_CHAR .value)[0]
                self._battery_handle = handle.getHandle()

            self.writeCharacteristic(self._battery_handle + 1, bytes([0, 0]))

    def _start_notification_thread(self):
        try:
            if self._notification_thread == None:
                self.reset_position()
                self._notification_thread = threading.Thread(target=self._notification_wait)
                self._notification_thread.start()
        except:
            pass

    def _notification_wait(self):
        if self.debug:
            print("Notification thread started")

        while (self.connected and
            (self._position_subscribed or
             self._orientation_subscribed or
             self._button_subscribed or
             self._temperature_subscribed or
             self._battery_subscribed)):
            try:
                if super().waitForNotifications(1):
                    continue
            except:
                continue

        if self.debug:
            print("Notification thread stopped")

    def _on_orientation(self, data):
        """Private function for orientation notification

        Arguments:
            data {bytes} -- Data from device
        """
        # print("on_orienation")
        # print(f"data: {data.hex()}")
        w = numpy.int16(numpy.uint16(int.from_bytes(data[0:2], byteorder='little')))
        x = numpy.int16(numpy.uint16(int.from_bytes(data[2:4], byteorder='little')))
        y = numpy.int16(numpy.uint16(int.from_bytes(data[4:6], byteorder='little')))
        z = numpy.int16(numpy.uint16(int.from_bytes(data[6:8], byteorder='little')))
        w = w / 1024
        x = x / 1024
        y = y / 1024
        z = z / 1024
        if self.debug:
            print(f"w: {w}\nx: {x}\ny: {y}\nz: {z}")
        self.on_orientation(w, x, y, z)
        for callback in self._orientation_callbacks.values():
            callback(w, x, y, z)

    def on_orientation(self, w, x, y, z):
        """Function called on position notification

        Arguments:
            w {int} -- Quaternion of wand
            x {int} -- Quaternion of wand
            y {int} -- Quaternion of wand
            z {int} -- Quaternion of wand
        """
        pass

    def _on_position(self, data):
        """Private function for position notification

        Arguments:
            data {bytes} -- Data from device
        """
        # print("\non_position")
        # print(f"data: {data.hex()}")
        # mag_x = int.from_bytes(data[0:2], byteorder='little')
        # mag_y = int.from_bytes(data[2:4], byteorder='little')
        # mag_z = int.from_bytes(data[4:6], byteorder='little')
        # acc_x = int.from_bytes(data[6:8], byteorder='little')
        # acc_y = int.from_bytes(data[8:10], byteorder='little')
        # acc_z = int.from_bytes(data[10:12], byteorder='little')
        # pitch = int.from_bytes(data[12:14], byteorder='little')
        # roll = int.from_bytes(data[14:16], byteorder='little')
        # yaw = int.from_bytes(data[16:18], byteorder='little')
        acc_x = numpy.int16(numpy.uint16(int.from_bytes(data[0:2], byteorder='little')))
        acc_y = numpy.int16(numpy.uint16(int.from_bytes(data[2:4], byteorder='little')))
        acc_z = numpy.int16(numpy.uint16(int.from_bytes(data[4:6], byteorder='little')))
        mag_x = numpy.int16(numpy.uint16(int.from_bytes(data[6:8], byteorder='little')))
        mag_y = numpy.int16(numpy.uint16(int.from_bytes(data[8:10], byteorder='little')))
        mag_z = numpy.int16(numpy.uint16(int.from_bytes(data[10:12], byteorder='little')))
        yaw = numpy.int16(numpy.uint16(int.from_bytes(data[12:14], byteorder='little')))
        pitch = numpy.int16(numpy.uint16(int.from_bytes(data[14:16], byteorder='little')))
        roll = numpy.int16(numpy.uint16(int.from_bytes(data[16:18], byteorder='little')))
        # if self.debug:
        # print(f"{mag_x}\t{mag_y}\t{mag_z}")
        # print(f"{acc_x}\t{acc_y}\t{acc_z}")
        # print(f"{yaw}\t{pitch}\t{roll}")
        self.on_position(mag_x, mag_y, mag_z, acc_x, acc_y, acc_z, pitch, roll, yaw)
        for callback in self._position_callbacks.values():
            callback(mag_x, mag_y, mag_z, acc_x, acc_y, acc_z, pitch, roll, yaw)

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
        """Function called on position notification

            
        Arguments:
            mag_x
            mag_y
            mag_z
            acc_x
            acc_y
            acc_z
            pitch
            roll 
            yaw
        """
        pass

    def reset_position(self):
        """Reset the quaternains of the wand
        """
        handle = self._sensor_service.getCharacteristics(_SENSOR.QUATERNIONS_RESET_CHAR.value)[0].getHandle()
        with self._lock:
            self.writeCharacteristic(handle, bytes([1]))

    def _on_button(self, data):
        """Private function for button notification

        Arguments:
            data {bytes} -- Data from device
        """
        val = data[0] == 1

        if self.debug:
            print("Button: {}".format(val))

        self.on_button(val)
        for callback in self._button_callbacks.values():
            callback(val)

    def on_button(self, value):
        """Function called on button notification

        Arguments:
            pressed {bool} -- If button is pressed
        """
        pass

    def _on_temperature(self, data):
        """Private function for temperature notification

        Arguments:
            data {bytes} -- Data from device
        """
        val = numpy.int16(numpy.uint16(int.from_bytes(data[0:2], byteorder='little')))

        if self.debug:
            print("Temperature: {}".format(val))

        self.on_temperature(val)
        for callback in self._temperature_callbacks.values():
            callback(val)

    def on_temperature(self, value):
        """Function called on temperature notification

        Arguments:
            value {int} -- Temperature of the wand
        """
        pass

    def _on_battery(self, data):
        """Private function for battery notification

        Arguments:
            data {bytes} -- Data from device
        """
        val = data[0]

        if self.debug:
            print("Battery: {}".format(val))

        self.on_battery(val)
        for callback in self._battery_callbacks.values():
            callback(val)

    def on_battery(self, value):
        """Function called on battery notification

        Arguments:
            value {int} -- Battery level of the wand
        """

    def handleNotification(self, cHandle, data):
        """Handle notifications subscribed to

        Arguments:
            cHandle {int} -- Handle of notification
            data {bytes} -- Data from device
        """
        if cHandle == self._position_notification_handle:
            self._on_position(data)
        elif cHandle == self._orientation_notification_handle:
            self._on_orientation(data)
        elif cHandle == self._button_notification_handle:
            self._on_button(data)
        elif cHandle == self._temp_notification_handle:
            self._on_temperature(data)
        elif cHandle == self._battery_notification_handle:
            self._on_battery(data)


class ScanDelegate(DefaultDelegate):
    def __init__(self):
        DefaultDelegate.__init__(self)

    def handleDiscovery(self, dev, isNewDev, isNewData):
        if isNewDev:
            print(f"Discovered device: {dev.addr}")
        elif isNewData:
            print(f"Received new data from: {dev.addr}") 

if __name__ == '__main__':
    # Create a new wand scanner
    scanner = Scanner().withDelegate(ScanDelegate())
    wand_device = None
    sphero_device = None
    try:
        # While we don't have any wands
        while wand_device is None or sphero_device is None:
            print("scanning for devices")
            devices = scanner.scan(10.0)
            for dev in devices:
                print(f"device: {dev}")
                name = dev.getValueText(9)
                # Check devices for wand and sphero
                if wand_device is None:
                    if name.startswith(KANO_PREFIX):
                        wand_device = dev
                if sphero_device is None:
                    if name.startswith(SPHERO_PREFIX):
                        sphero_device = dev
        # Initialize devices
    # Detect keyboard interrupt and disconnect wands
    except KeyboardInterrupt:
        wand.disconnect()
        sphero.disconnect()
    