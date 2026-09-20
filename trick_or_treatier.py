import sys
from gpiozero import Motor, Button, LED, DigitalOutputDevice
import time
try:
    import sphero_mini
except ImportError:
    sphero_mini = None
import random
import threading
import queue
from itertools import cycle
from flask import Flask, render_template, redirect, url_for
from flask.logging import default_handler
import plotly.graph_objs as go
import pandas as pd
import json
# import picamera
import time
import logging

HALLOWEEN_FILE = '~/trick-or-treat/trick_or_treat.log'


logging.basicConfig(
    filename="trick_or_treat.log",
    format='%(asctime)s,%(levelname)s,%(message)s',
    datefmt='%Y:%m:%d:%H:%M:%S',
    level=logging.INFO
)


# CONSTANTS
TRICK_TIME = 13
TREAT_TIME = 0.2
LIGHTS_TIME = 1.0
ROLL_TIME = 3
ROLL_STEP_TIME = 0.01
BUTTON_PRESS_DELAY = 0.1
EYE_TIME = 15
SPHERO_SPEED = 100  # int: (0 - 255)
CANDY_SPEED = 0.3  # float: (0 - 1)
BLOOD_SPEED = 0.5  # float: (0 - 1)
STIR_SPEED = 0.2  # float: (0 - 1)
TRICKS = [
    "BUBBLE",
    "PINGPONG",
    # "BAT",
    "TOY",
    "STIR"
]
TRICK_TIMES = {
    "BUBBLE": 10,
    "PINGPONG": 5,
    "BAT": 8,
    "TOY": 5,
    "STIR": 10
}
# GPIO PINS
TREAT_BUTTON_PIN = 14  
TRICK_BUTTON_PIN = 18
# GHOST_BUTTON_PIN = 0
TREAT_LED_PIN = 15
TRICK_LED_PIN = 22
TREAT_MOTOR_FORWARD_PIN = 17
TREAT_MOTOR_BACKWARD_PIN = 27
STIR_MOTOR_FORWARD_PIN = 7
STIR_MOTOR_BACKWARD_PIN = 8
PING_PONG_PIN = 13
BUBBLE_SWITCH = 19
BUBBLE_SWITCH_2 = 26
LIGHTS_PIN_ON = 2
LIGHTS_PIN_OFF = 3
BAT_PIN = 23
TOY_PIN = 24
DOORBELL_PIN = 6

# Trick or Treat object to handle devices
class TrickOrTreat():
    def __init__(self, sphero_mac: str):
        self.running = False
        self.current_trick = None
        self.trick_end_time = time.time()
        self.treat_queue = queue.Queue()
        # Setup motors
        self.treat_motor = Motor(forward=TREAT_MOTOR_FORWARD_PIN,
                                 backward=TREAT_MOTOR_BACKWARD_PIN)
        self.treat_motor.stop()
        self.stir_motor = Motor(forward=STIR_MOTOR_FORWARD_PIN,
                                backward=STIR_MOTOR_BACKWARD_PIN)
        self.stir_motor.stop()
        # Setup buttons
        # self.ghost_button = Button(GHOST_BUTTON_PIN)
        self.treat_button = Button(TREAT_BUTTON_PIN)
        self.trick_button = Button(TRICK_BUTTON_PIN)
        self.curr_trick_button = False
        self.curr_treat_button = False
        self.prev_trick_button = False
        self.prev_treat_button = False
        # Setup LEDs
        self.treat_led = LED(TREAT_LED_PIN)
        self.trick_led = LED(TRICK_LED_PIN)
        # Setup continuous tricks
        # Setup sphero ball
        if sphero_mac and sphero_mini is not None:
            self.sphero = sphero_mini.sphero_mini(sphero_mac)
            self.time_since_eye = time.time()
        else:
            self.sphero = None
            self.time_since_eye = None
        self.tricks = cycle(TRICKS)
        # Setup other tricks
        self.doorbell = DigitalOutputDevice(DOORBELL_PIN,
                                            active_high=False)
        self.bat = DigitalOutputDevice(BAT_PIN, active_high=False)
        self.toy = DigitalOutputDevice(TOY_PIN, active_high=False)
        self.ping_pong = DigitalOutputDevice(PING_PONG_PIN,
                                             active_high=False)
        self.lights_on = DigitalOutputDevice(LIGHTS_PIN_ON,
                                             active_high=False)
        self.lights_off = DigitalOutputDevice(LIGHTS_PIN_OFF,
                                              active_high=False)
        # Start by turning on the lights
        self.lights_on.on()
        time.sleep(BUTTON_PRESS_DELAY)
        self.lights_on.off()
        self.bubble_switch = DigitalOutputDevice(BUBBLE_SWITCH,
                                                 active_high=False)
        self.bubble_switch_2 = DigitalOutputDevice(BUBBLE_SWITCH_2,
                                                   active_high=False)
        # Setup tricks threads
        self.trick_thread = threading.Thread(
            target=self._handle_tricks,
            daemon=True,
        )
        self.treat_thread = threading.Thread(
            target=self._handle_treats,
            daemon=True,
        )
        self.continuous_trick_thread = \
            threading.Thread(
                target=self._handle_continuous_tricks,
                daemon=True,
            )

    def trigger_trick(self, trick):
        """Start a named trick from an external spell controller."""
        trick = trick.upper().strip()
        if trick not in TRICK_TIMES:
            raise ValueError(f"Unknown trick: {trick}")
        if not self.running:
            raise RuntimeError("Trick controller is not running")

        self.current_trick = trick
        self.trick_end_time = time.time() + TRICK_TIMES[trick]
        
    
    def _handle_continuous_tricks(self):
        while self.running:
            # Run continuous tricks
            if self.sphero:
                if (time.time() - self.time_since_eye) > EYE_TIME:
                    print("running sphero trick")
                    self._sphero_trick()
                    self.time_since_eye = time.time()

    def _handle_treats(self):
        while self.running:
            treat = self.treat_queue.get()
            if treat == "CANDY":
                print(f"treat button pressed")
                self._treat()
        
    def _handle_tricks(self):
        while self.running:
            # if time.time() < self.trick_end_time:
            #     self.trick_led.on()
            # else:
            #     self.trick_led.off()
            trick = self.current_trick
            if trick is not None:
                print(f"trick button pressed. Performing trick: {trick}")
            if trick == "BUBBLE":
                self._bubble_trick()
            elif trick == "PINGPONG":
                self._ping_pong_trick()
            elif trick == "TOY":
                self._toy_trick()
            elif trick == "BAT":
                self._bat_trick()
            elif trick == "STIR":
                self._stir_trick()
            elif trick is None:
                time.sleep(0.01)
            else:
                print(f"Unknown trick: {trick}")
            
    def _bubble_trick(self):
        self.trick_led.on()
        self.bubble_switch.on()
        self.bubble_switch_2.on()
        while (time.time() - self.trick_end_time) < 0:
            time.sleep(0.01)
        self.bubble_switch.off()
        self.bubble_switch_2.off()
        self.trick_led.off()

    def _ping_pong_trick(self):
        self.trick_led.on()
        self.ping_pong.on()
        while (time.time() - self.trick_end_time) < 0:
            time.sleep(0.01)
        self.ping_pong.off()
        self.trick_led.off()

    def _toy_trick(self):
        self.trick_led.on()
        self.toy.on()
        time.sleep(BUTTON_PRESS_DELAY)
        self.toy.off()
        while (time.time() - self.trick_end_time) < 0:
            time.sleep(0.01)
        self.toy.on()
        time.sleep(BUTTON_PRESS_DELAY)
        self.toy.off()
        self.trick_led.off()

    def _bat_trick(self):
        self.trick_led.on()
        self.bat.on()
        while (time.time() - self.trick_end_time) < 0:
            time.sleep(0.01)
        self.bat.off()
        self.trick_led.off()

    def _lights_trick(self):
        self.trick_led.on()
        while (time.time() - self.trick_end_time) < 0:
            self.lights_off.on()
            time.sleep(BUTTON_PRESS_DELAY)
            self.lights_off.off()
            time.sleep(LIGHTS_TIME)
            self.lights_on.on()
            time.sleep(BUTTON_PRESS_DELAY)
            self.lights_on.off()
            time.sleep(LIGHTS_TIME)
        self.trick_led.off()

    def _stir_trick(self):
        self.trick_led.on()
        self.doorbell.on()
        time.sleep(BUTTON_PRESS_DELAY)
        self.doorbell.off()
        self.stir_motor.forward(STIR_SPEED)
        while (time.time() - self.trick_end_time) < 0:
            time.sleep(0.01)
        self.stir_motor.stop()
        self.trick_led.off()

    def _sphero_trick2(self):
        self.sphero.setLEDColor(red=255, green=0, blue=0)
        for i in range(int(ROLL_TIME // ROLL_STEP_TIME)):
            self.sphero.roll(
                    SPHERO_SPEED,
                    int(360 * i * ROLL_STEP_TIME / ROLL_TIME)
            )
            self.sphero.wait(ROLL_STEP_TIME)
        self.sphero.roll(0, 0)
        self.sphero.wait(1)
        self.sphero.setLEDColor(red=0, green=0, blue=255)

    def _sphero_trick(self):
        self.sphero.setLEDColor(red=255, green=0, blue=0)
        start_time = time.time()
        while (time.time() - start_time) < ROLL_TIME:
            self.sphero.roll(SPHERO_SPEED, random.randint(0, 360))
            self.sphero.wait(ROLL_STEP_TIME)
        self.sphero.roll(0, 0)
        self.sphero.wait(1)
        self.sphero.setLEDColor(red=0, green=0, blue=255)
        
    def _treat(self):
        self.treat_led.on()
        self.treat_motor.forward(CANDY_SPEED)
        time.sleep(TREAT_TIME)
        self.treat_motor.backward()
        time.sleep(TREAT_TIME)
        self.treat_motor.forward()
        time.sleep(TREAT_TIME)
        time.sleep(TREAT_TIME)
        self.treat_motor.stop()
        self.treat_led.off()
    
    def run(self):
        self.running = True
        # Start threads
        self.continuous_trick_thread.start()
        self.trick_thread.start()
        self.treat_thread.start()
        while self.running:
            self.prev_treat_button = self.curr_treat_button
            self.prev_trick_button = self.curr_trick_button
            self.curr_trick_button = self.trick_button.is_pressed
            self.curr_treat_button = self.treat_button.is_pressed
            treat_pressed = self.prev_treat_button \
                    and not self.curr_treat_button
            trick_pressed = self.prev_trick_button \
                    and not self.curr_trick_button
            # print(f"curr: {self.curr_trick_button} prev: {self.prev_trick_button}")
            if time.time() > self.trick_end_time:
                self.current_trick = None
            if treat_pressed:
                logging.info("treat")
                self.treat_queue.put("CANDY")
            elif trick_pressed:
                if not self.current_trick:
                    logging.info("trick")
                    self.current_trick = next(self.tricks)
                    self.trick_end_time = time.time() + \
                            TRICK_TIMES[self.current_trick]
                else:
                    self.current_trick = None
                    self.trick_end_time = time.time()
            else:
                # Don't run too fast
                time.sleep(0.01)
        print("end of run")

    def stop(self):
        self.running = False
        self.current_trick = None
        self.treat_motor.stop()
        self.stir_motor.stop()
        self.trick_led.off()
        self.treat_led.off()
        self.bubble_switch.off()
        self.bubble_switch_2.off()
        self.bat.off()
        self.toy.off()
        self.ping_pong.off()
        self.doorbell.off()
        if self.sphero is not None:
            self.sphero.disconnect()
        print("end of stop")


if __name__ == '__main__':
    # EB:B6:31:82:7C:F0
    sphero_mac = sys.argv[1]
    # sphero_mac = "EB:B6:31:82:7C:F0"
    print("Welcome trick or treaters")
    trick_or_treat = TrickOrTreat(sphero_mac)
    # trick_or_treat = TrickOrTreat("")
    try:
        print("trick or treat start")
        trick_or_treat.run()
    except KeyboardInterrupt:
        trick_or_treat.stop()
    print("\ngoodbye")

