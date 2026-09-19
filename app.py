import logging
import queue
import threading
import time

from flask import Flask, jsonify, render_template

from spell_receiver import SpellReceiver


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

ZMQ_HOST = "127.0.0.1"
ZMQ_PORT = 5555

DISPLAY_SPELL_SECONDS = 8


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


# ------------------------------------------------------------
# Spell processing
# ------------------------------------------------------------

def spell_processor():
    """
    Wait for spells from ZeroMQ and update the display state.

    Later this is where we can also launch the physical spell
    actions, such as turning GPIO pins on/off.
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

            # ------------------------------------------------
            # Physical spell action will eventually go here.
            # ------------------------------------------------
            #
            # Example:
            #
            # run_spell_action(spell)
            #
            # For now we just print it.
            #
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
