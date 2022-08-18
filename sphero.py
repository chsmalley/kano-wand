import bluetooth


class sphero_mini:
    def __init__(self, MAC_ADDR):
        self.v_batt = None # will be updated with battery voltage when sphero.getBatteryVoltage() is called
        self.firmware_version = [] # will be updated with firware version when sphero.returnMainApplicationVersion() is called

        # Subscribe to notifications
        self.sphero_delegate = MyDelegate(self, user_delegate) # Pass a reference to this instance when initializing
        self.p.setDelegate(self.sphero_delegate)