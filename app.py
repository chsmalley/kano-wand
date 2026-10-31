import csv
import logging
import os
import queue
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import zmq

from flask import Flask, jsonify, render_template

from spell_receiver import SpellReceiver


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

ZMQ_HOST = os.getenv("ZMQ_HOST", "127.0.0.1")
ZMQ_PORT = int(os.getenv("ZMQ_PORT", "5555"))
EVENT_LOG_PATH = Path(
    os.getenv(
        "EVENT_LOG_PATH",
        Path(__file__).resolve().parent / "event_log.csv",
    )
)

DISPLAY_SPELL_SECONDS = 8
ACTIVITY_INTERVAL_MINUTES = 15
ACTIVITY_WINDOW_HOURS = 24

SPELL_TRICKS = {
    "WINGARDIUM_LEVIOSA": "SPIDER",
    "FLIPENDO": "PINGPONG",
    "LOCOMOTOR": "PINGPONG",
    "EXPELLIARMUS": "BUBBLE",
    "AVIS": "BAT",
    "INCENDIO": "FLAME",
    "LUMOS": "LIGHTS",
    "REDUCTO": "STIR",
    "REDUCIO": "STIR",
    "ENGORGIO": "SPIDER",
    "AGUAMENTI": "POTION",
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
    "wand_id": None,
    "timestamp": None,
}


state_lock = threading.Lock()
event_log_lock = threading.Lock()


def record_event(event, *, spell="", confidence="", wand_id="", source=""):
    """Append one timestamped spell or treat event to the CSV event log."""
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "spell": spell,
        "confidence": confidence,
        "wand_id": wand_id,
        "source": source,
    }

    try:
        with event_log_lock:
            EVENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with EVENT_LOG_PATH.open("a", newline="", encoding="utf-8") as log_file:
                writer = csv.DictWriter(
                    log_file,
                    fieldnames=(
                        "timestamp",
                        "event",
                        "spell",
                        "confidence",
                        "wand_id",
                        "source",
                    ),
                )
                if log_file.tell() == 0:
                    writer.writeheader()
                writer.writerow(row)
    except OSError:
        logging.exception("Unable to append event to %s.", EVENT_LOG_PATH)


def get_activity_data():
    """Aggregate logged trick and treat events into rolling 15-minute bins."""
    now = datetime.now(timezone.utc)
    interval = timedelta(minutes=ACTIVITY_INTERVAL_MINUTES)
    window_start = now - timedelta(hours=ACTIVITY_WINDOW_HOURS)
    first_bucket = int(window_start.timestamp() // interval.total_seconds())
    current_bucket = int(now.timestamp() // interval.total_seconds())
    trick_names = sorted(set(SPELL_TRICKS.values()))
    series = {
        **{trick: [0] * (current_bucket - first_bucket + 1)
           for trick in trick_names},
        "Treats": [0] * (current_bucket - first_bucket + 1),
    }

    try:
        with event_log_lock:
            with EVENT_LOG_PATH.open(newline="", encoding="utf-8") as log_file:
                for row in csv.DictReader(log_file):
                    try:
                        timestamp = datetime.fromisoformat(
                            row["timestamp"].replace("Z", "+00:00")
                        )
                        if timestamp.tzinfo is None:
                            timestamp = timestamp.replace(tzinfo=timezone.utc)
                        timestamp = timestamp.astimezone(timezone.utc)
                    except (AttributeError, KeyError, ValueError):
                        logging.warning("Skipping event with invalid timestamp: %s", row)
                        continue

                    bucket = int(timestamp.timestamp() // interval.total_seconds())
                    if bucket < first_bucket or bucket > current_bucket:
                        continue

                    event = row.get("event")
                    if event == "treat":
                        series["Treats"][bucket - first_bucket] += 1
                    elif event == "spell":
                        trick = SPELL_TRICKS.get(row.get("spell", ""))
                        if trick is not None:
                            series[trick][bucket - first_bucket] += 1
    except FileNotFoundError:
        pass

    labels = [
        datetime.fromtimestamp(
            bucket * interval.total_seconds(),
            tz=timezone.utc,
        ).isoformat()
        for bucket in range(first_bucket, current_bucket + 1)
    ]
    return {
        "labels": labels,
        "series": series,
        "interval_minutes": ACTIVITY_INTERVAL_MINUTES,
        "window_hours": ACTIVITY_WINDOW_HOURS,
        "updated_at": now.isoformat(),
    }


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

        trick_controller = TrickOrTreat(
            SPHERO_MAC or None,
            on_treat=lambda source: record_event("treat", source=source),
        )
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
            wand_id = spell_data.get("wand_id")

            logging.info(
                "Processing spell: %s (confidence=%s, wand=%s)",
                spell,
                confidence,
                wand_id or "unknown",
            )

            record_event(
                "spell",
                spell=spell,
                confidence=confidence,
                wand_id=wand_id or "",
            )

            # Update the current display state.
            with state_lock:
                current_spell = {
                    "spell": spell,
                    "confidence": confidence,
                    "wand_id": wand_id,
                    "timestamp": time.time(),
                }

            if spell == "UNKNOWN":
                if trick_controller is not None:
                    trick_controller.trigger_treat()
                    logging.info("Added a treat for an unclassified gesture.")
            else:
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
                "wand_id": None,
                "timestamp": None,
            }

    return jsonify(spell_data)


@app.route("/api/activity")
def get_activity():
    """Return the rolling 24-hour trick and treat counts."""
    return jsonify(get_activity_data())


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
