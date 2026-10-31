import logging
import queue
import random
import sys
import threading
import time
from itertools import cycle

from gpiozero import Motor, Button, LED, DigitalOutputDevice

try:
    import sphero_mini
except ImportError:
    sphero_mini = None


# CONSTANTS
TREAT_TIME = 0.2
LIGHTS_TIME = 10.0
ROLL_TIME = 3
ROLL_STEP_TIME = 0.01
BUTTON_PRESS_DELAY = 0.1
EYE_TIME = 15
SPHERO_SPEED = 100  # int: (0 - 255)
CANDY_SPEED = 0.3  # float: (0 - 1)
BLOOD_SPEED = 0.5  # float: (0 - 1)
STIR_SPEED = 0.2  # float: (0 - 1)
SPIDER_MOTOR_1_SPEED = 0.8  # float: (0 - 1)
SPIDER_MOTOR_2_SPEED = 0.9  # float: (0 - 1)
SPIDER_MOTOR_1_FORWARD_TIME = 2.0
SPIDER_MOTOR_1_BACKWARD_TIME = 1.5
SPIDER_MOTOR_2_DELAY = 5.0
SPIDER_MOTOR_2_FORWARD_TIME = 7.0
PING_PONG_SPEED = 0.6  # float: (0 - 1)
PING_PONG_TIME = 4.0
POTION_TIME = 10.0
POTION_SPEED = 0.4
TRICKS = [
    "BUBBLE",
    "PINGPONG",
    "BAT",
    "FLAME",
    "STIR",
    "SPIDER",
    "POTION",
]
TRICK_TIMES = {
    "BUBBLE": 10,
    "PINGPONG": PING_PONG_TIME,
    "BAT": 8,
    "FLAME": 8,
    "STIR": 10,
    "SPIDER": (SPIDER_MOTOR_1_FORWARD_TIME
               + SPIDER_MOTOR_1_BACKWARD_TIME
               + SPIDER_MOTOR_2_DELAY
               + SPIDER_MOTOR_2_FORWARD_TIME),
    "POTION": POTION_TIME,
    "LIGHTS": LIGHTS_TIME,
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
SPIDER_MOTOR_2_FORWARD_PIN = 11
SPIDER_MOTOR_2_BACKWARD_PIN = 5
SPIDER_MOTOR_1_FORWARD_PIN = 21
SPIDER_MOTOR_1_BACKWARD_PIN = 20
PING_PONG_MOTOR_FORWARD_PIN = 9
PING_PONG_MOTOR_BACKWARD_PIN = 10
POTION_MOTOR_FORWARD_PIN = 12
POTION_MOTOR_BACKWARD_PIN = 16
BUBBLE_SWITCH = 19
BUBBLE_SWITCH_2 = 26
LIGHTS_PIN_ON = 3
LIGHTS_PIN_OFF = 2
BAT_PIN = 23
FLAME_PIN = 13
DOORBELL_PIN = 6

# Trick or Treat object to handle devices
class TrickOrTreat():
    def __init__(self, sphero_mac: str):
        self.running = False
        self.current_trick = None
        self.active_tricks = {}
        self._active_tricks_lock = threading.Lock()
        self._trick_locks = {
            trick: threading.Lock() for trick in TRICK_TIMES
        }
        self._trick_threads = set()
        self.treat_queue = queue.Queue()
        self.trick_queue = queue.Queue()
        # Setup motors
        self.treat_motor = Motor(forward=TREAT_MOTOR_FORWARD_PIN,
                                 backward=TREAT_MOTOR_BACKWARD_PIN)
        self.treat_motor.stop()
        self.stir_motor = Motor(forward=STIR_MOTOR_FORWARD_PIN,
                                backward=STIR_MOTOR_BACKWARD_PIN)
        self.stir_motor.stop()
        self.spider_motor_1 = Motor(
            forward=SPIDER_MOTOR_1_FORWARD_PIN,
            backward=SPIDER_MOTOR_1_BACKWARD_PIN,
        )
        self.spider_motor_1.stop()
        self.spider_motor_2 = Motor(
            forward=SPIDER_MOTOR_2_FORWARD_PIN,
            backward=SPIDER_MOTOR_2_BACKWARD_PIN,
        )
        self.spider_motor_2.stop()
        self.ping_pong_motor = Motor(
            forward=PING_PONG_MOTOR_FORWARD_PIN,
            backward=PING_PONG_MOTOR_BACKWARD_PIN,
        )
        self.ping_pong_motor.stop()
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
        self.flame = DigitalOutputDevice(FLAME_PIN, active_high=False)
        self.potion_motor = Motor(
            forward=POTION_MOTOR_FORWARD_PIN,
            backward=POTION_MOTOR_BACKWARD_PIN,
        )
        self.potion_motor.stop()
        self.lights_on = DigitalOutputDevice(LIGHTS_PIN_ON,
                                             active_high=False)
        self.lights_off = DigitalOutputDevice(LIGHTS_PIN_OFF,
                                              active_high=False)
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
        """Queue a named trick for the controller's dispatcher."""
        trick = trick.upper().strip()
        if trick not in TRICK_TIMES:
            raise ValueError(f"Unknown trick: {trick}")
        if not self.running:
            raise RuntimeError("Trick controller is not running")

        self.trick_queue.put_nowait(trick)

    def trigger_treat(self):
        """Queue one treat for the controller's worker thread."""
        self.treat_queue.put_nowait("CANDY")

    
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
            try:
                trick = self.trick_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if trick is None:
                self.trick_queue.task_done()
                break

            with self._active_tricks_lock:
                if not self.running:
                    self.trick_queue.task_done()
                    break

                stop_event = threading.Event()
                worker = threading.Thread(
                    target=self._run_trick,
                    args=(trick, stop_event),
                    name=f"Trick-{trick}",
                    daemon=True,
                )
                was_idle = not self.active_tricks
                self.active_tricks[stop_event] = trick
                self.current_trick = trick
                self._trick_threads.add(worker)
                if was_idle:
                    self.trick_led.on()
                worker.start()

    def _run_trick(self, trick, stop_event):
        handlers = {
            "BUBBLE": self._bubble_trick,
            "PINGPONG": self._ping_pong_trick,
            "BAT": self._bat_trick,
            "FLAME": self._flame_trick,
            "LIGHTS": self._lights_trick,
            "STIR": self._stir_trick,
            "SPIDER": self._spider_trick,
            "POTION": self._potion_trick,
        }
        try:
            with self._trick_locks[trick]:
                if self.running and not stop_event.is_set():
                    deadline = time.monotonic() + TRICK_TIMES[trick]
                    logging.info("Performing trick: %s", trick)
                    handlers[trick](stop_event, deadline)
        except Exception:
            logging.exception("Trick failed: %s", trick)
        finally:
            with self._active_tricks_lock:
                self.active_tricks.pop(stop_event, None)
                self.current_trick = next(
                    reversed(self.active_tricks.values()),
                    None,
                )
                if not self.active_tricks:
                    self.trick_led.off()
                self._trick_threads.discard(threading.current_thread())
            self.trick_queue.task_done()

    def _wait_for_duration(self, stop_event, duration):
        return self.running and not stop_event.wait(max(0.0, duration))

    def _wait_until(self, stop_event, deadline):
        return self._wait_for_duration(
            stop_event,
            deadline - time.monotonic(),
        )

    def _stop_active_tricks(self):
        with self._active_tricks_lock:
            for stop_event in self.active_tricks:
                stop_event.set()

    def _bubble_trick(self, stop_event, deadline):
        self.bubble_switch.on()
        self.bubble_switch_2.on()
        try:
            self._wait_until(stop_event, deadline)
        finally:
            self.bubble_switch.off()
            self.bubble_switch_2.off()

    def _ping_pong_trick(self, stop_event, deadline):
        self.ping_pong_motor.forward(PING_PONG_SPEED)
        try:
            self._wait_until(stop_event, deadline)
        finally:
            self.ping_pong_motor.stop()

    def _flame_trick(self, stop_event, deadline):
        self.flame.on()
        try:
            self._wait_until(stop_event, deadline)
        finally:
            self.flame.off()

    def _bat_trick(self, stop_event, deadline):
        self.bat.on()
        try:
            self._wait_until(stop_event, deadline)
        finally:
            self.bat.off()

    def _lights_trick(self, stop_event, deadline):
        try:
            while self.running and time.monotonic() < deadline:
                self.lights_on.on()
                if not self._wait_for_duration(stop_event, BUTTON_PRESS_DELAY):
                    break
                self.lights_on.off()
                if not self._wait_for_duration(stop_event, LIGHTS_TIME):
                    break
                self.lights_off.on()
                if not self._wait_for_duration(stop_event, BUTTON_PRESS_DELAY):
                    break
                self.lights_off.off()
        finally:
            self.lights_on.off()
            self.lights_off.off()

    def _stir_trick(self, stop_event, deadline):
        self.doorbell.on()
        try:
            if not self._wait_for_duration(stop_event, BUTTON_PRESS_DELAY):
                return
            self.doorbell.off()
            self.stir_motor.forward(STIR_SPEED)
            self._wait_until(stop_event, deadline)
        finally:
            self.doorbell.off()
            self.stir_motor.stop()

    def _spider_trick(self, stop_event, deadline):
        try:
            self.spider_motor_1.forward(SPIDER_MOTOR_1_SPEED)
            if not self._wait_for_duration(
                stop_event, SPIDER_MOTOR_1_FORWARD_TIME
            ):
                return
            self.spider_motor_1.backward(SPIDER_MOTOR_1_SPEED)
            if not self._wait_for_duration(
                stop_event, SPIDER_MOTOR_1_BACKWARD_TIME
            ):
                return
            self.spider_motor_1.stop()

            if not self._wait_for_duration(stop_event, SPIDER_MOTOR_2_DELAY):
                return
            self.spider_motor_2.forward(SPIDER_MOTOR_2_SPEED)
            self._wait_until(stop_event, deadline)
        finally:
            self.spider_motor_1.stop()
            self.spider_motor_2.stop()

    def _potion_trick(self, stop_event, deadline):
        self.potion_motor.forward(POTION_SPEED)
        try:
            self._wait_until(stop_event, deadline)
        finally:
            self.potion_motor.stop()

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
            if treat_pressed:
                logging.info("treat")
                self.treat_queue.put("CANDY")
            elif trick_pressed:
                if not self.current_trick:
                    logging.info("trick")
                    self.trigger_trick(next(self.tricks))
                else:
                    self._stop_active_tricks()
            else:
                # Don't run too fast
                time.sleep(0.01)
        print("end of run")

    def stop(self):
        with self._active_tricks_lock:
            self.running = False
            for stop_event in self.active_tricks:
                stop_event.set()
        self.trick_queue.put_nowait(None)
        self.treat_queue.put_nowait(None)
        if self.trick_thread.is_alive():
            self.trick_thread.join(timeout=1)
        with self._active_tricks_lock:
            trick_threads = list(self._trick_threads)
        for thread in trick_threads:
            thread.join(timeout=1)
        self.current_trick = None
        self.treat_motor.stop()
        self.stir_motor.stop()
        self.ping_pong_motor.stop()
        self.spider_motor_1.stop()
        self.spider_motor_2.stop()
        self.trick_led.off()
        self.treat_led.off()
        self.bubble_switch.off()
        self.bubble_switch_2.off()
        self.bat.off()
        self.flame.off()
        self.potion_motor.stop()
        self.doorbell.off()
        if self.sphero is not None:
            self.sphero.disconnect()
        print("end of stop")


if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
    )
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

