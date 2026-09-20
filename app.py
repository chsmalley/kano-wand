import logging
import os
import queue
import threading
import time

import zmq

from flask import Flask, jsonify, render_template

from spell_receiver import SpellReceiver


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

ZMQ_HOST = os.getenv("ZMQ_HOST", "127.0.0.1")
ZMQ_PORT = int(os.getenv("ZMQ_PORT", "5555"))

DISPLAY_SPELL_SECONDS = 8

SPELL_TRICKS = {
    "FLIPENDO": "PINGPONG",
    "LOCOMOTOR": "TOY",
    "EXPELLIARMUS": "BUBBLE",
    "AVIS": "BAT",
    "REDUCTO": "STIR",
}

TRICK_CONTROLLER_ENABLED = os.getenv(
    "TRICK_CONTROLLER_ENABLED",
    "1",
) != "0"
SPHERO_MAC = os.getenv("SPHERO_MAC", "")
NERF_ENABLED = os.getenv("NERF_ENABLED", "1") != "0"
NERF_HOST = os.getenv("NERF_HOST", "192.168.0.26")
NERF_PORT = int(os.getenv("NERF_PORT", "5555"))


# ------------------------------------------------------------
# Flask application
# ------------------------------------------------------------

app = Flask(__name__)


# ------------------------------------------------------------
# Spell state
# ------------------------------------------------------------

spell_queue = queue.Queue()

current_spell = {
    "spell": None,
    "confidence": None,
    "timestamp": None,
}


state_lock = threading.Lock()


# ------------------------------------------------------------
# ZeroMQ receiver
# ------------------------------------------------------------

spell_receiver = SpellReceiver(
    host=ZMQ_HOST,
    port=ZMQ_PORT,
    spell_queue=spell_queue,
)

trick_controller = None
trick_controller_thread = None
nerf_context = None
nerf_publisher = None


def start_trick_controller():
    """Start the GPIO trick controller when hardware is available."""
    global trick_controller, trick_controller_thread

    if not TRICK_CONTROLLER_ENABLED:
        logging.info("Trick controller disabled by configuration.")
        return

    try:
        from trick_or_treatier import TrickOrTreat

        trick_controller = TrickOrTreat(SPHERO_MAC or None)
        trick_controller_thread = threading.Thread(
            target=trick_controller.run,
            name="TrickController",
            daemon=True,
        )
        trick_controller_thread.start()
        logging.info("Trick controller started.")
    except Exception:
        trick_controller = None
        logging.exception(
            "Unable to start trick controller; continuing without GPIO actions."
        )


def stop_trick_controller():
    if trick_controller is not None:
        trick_controller.stop()


def start_nerf_publisher():
    """Connect a ZeroMQ publisher to the remote Nerf Raspberry Pi."""
    global nerf_context, nerf_publisher

    if not NERF_ENABLED:
        logging.info("Nerf publisher disabled by configuration.")
        return

    nerf_context = zmq.Context()
    nerf_publisher = nerf_context.socket(zmq.PUB)
    nerf_publisher.setsockopt(zmq.LINGER, 0)
    nerf_publisher.connect(f"tcp://{NERF_HOST}:{NERF_PORT}")
    logging.info(
        "Nerf publisher connected to tcp://%s:%s",
        NERF_HOST,
        NERF_PORT,
    )


def stop_nerf_publisher():
    if nerf_publisher is not None:
        nerf_publisher.close(linger=0)
    if nerf_context is not None:
        nerf_context.term()


def send_nerf_command(spell, confidence):
    """Ask the remote Nerf Pi to perform the action for a spell."""
    if spell != "STUPEFY" or nerf_publisher is None:
        return

    nerf_publisher.send_json({
        "command": "shoot",
        "spell": spell,
        "confidence": confidence,
        "timestamp": time.time(),
    })
    logging.info("Sent Nerf shoot command to %s", NERF_HOST)


def run_spell_action(spell):
    """Trigger the GPIO trick associated with a recognized spell."""
    trick = SPELL_TRICKS.get(spell)
    if trick is None:
        return

    if trick_controller is None:
        logging.warning(
            "No trick controller available for %s (%s)",
            spell,
            trick,
        )
        return

    trick_controller.trigger_trick(trick)
    logging.info("Triggered trick %s for spell %s", trick, spell)


# ------------------------------------------------------------
# Spell processing
# ------------------------------------------------------------

def spell_processor():
    """
    Wait for spells from ZeroMQ and update the display state.

    Recognized spells also trigger their mapped Raspberry Pi trick.
    """

    global current_spell

    logging.info("Spell processor started.")

    while True:
        try:
            spell_data = spell_queue.get()

            spell = spell_data["spell"]
            confidence = spell_data["confidence"]

            logging.info(
                "Processing spell: %s (confidence=%s)",
                spell,
                confidence,
            )

            # Update the current display state.
            with state_lock:
                current_spell = {
                    "spell": spell,
                    "confidence": confidence,
                    "timestamp": time.time(),
                }

            run_spell_action(spell)
            send_nerf_command(spell, confidence)
            print(f"CAST: {spell}")

        except Exception:
            logging.exception(
                "Error processing spell."
            )


# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------

@app.route("/")
def index():
    """Main TV display."""

    return render_template("index.html")


@app.route("/api/spell")
def get_spell():
    """
    Return the currently displayed spell.

    Example response:

        {
            "spell": "LUMOS",
            "confidence": 0.94,
            "timestamp": 1726750000.123
        }
    """

    with state_lock:
        spell_data = current_spell.copy()

    # Automatically return to the idle state after the
    # configured display time.
    if spell_data["timestamp"] is not None:

        elapsed = time.time() - spell_data["timestamp"]

        if elapsed > DISPLAY_SPELL_SECONDS:
            spell_data = {
                "spell": None,
                "confidence": None,
                "timestamp": None,
            }

    return jsonify(spell_data)


# ------------------------------------------------------------
# Startup / shutdown
# ------------------------------------------------------------

def start_services():
    """Start background services."""

    start_trick_controller()
    start_nerf_publisher()
    spell_receiver.start()

    processor_thread = threading.Thread(
        target=spell_processor,
        name="SpellProcessor",
        daemon=True,
    )

    processor_thread.start()


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    start_services()

    try:
        app.run(
            host="0.0.0.0",
            port=5000,
            debug=False,
            threaded=True,
            use_reloader=False,
        )

    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received.")

    finally:
        spell_receiver.stop()
        stop_trick_controller()
        stop_nerf_publisher()
