import json
import logging
import threading
import queue

import zmq


class SpellReceiver:
    """
    Receives spell messages from a ZeroMQ PUB socket.

    Expected message format:
        {
            "spell": "LUMOS",
            "confidence": 0.94
        }

    The most recently received spell is placed into the queue so that
    the Flask application can process it without blocking the web server.
    """

    VALID_SPELLS = {
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
        "REDUCTO",
    }

    def __init__(
        self,
        host="127.0.0.1",
        port=5555,
        spell_queue=None,
    ):
        self.host = host
        self.port = port

        self.spell_queue = spell_queue or queue.Queue()

        self.running = False
        self.thread = None

        self.context = None
        self.socket = None

        self.logger = logging.getLogger(__name__)

    def start(self):
        """Start the ZeroMQ receiver thread."""

        if self.running:
            self.logger.warning("Spell receiver is already running.")
            return

        self.running = True

        self.thread = threading.Thread(
            target=self._receive_loop,
            name="SpellReceiver",
            daemon=True,
        )

        self.thread.start()

        self.logger.info(
            "Spell receiver started on tcp://%s:%s",
            self.host,
            self.port,
        )

    def stop(self):
        """Stop the receiver and clean up ZeroMQ."""

        if not self.running:
            return

        self.logger.info("Stopping spell receiver...")

        self.running = False

        # Closing the socket causes recv() to return/raise and allows
        # the receiver thread to exit.
        if self.socket is not None:
            try:
                self.socket.close(linger=0)
            except Exception:
                pass

        if self.thread is not None:
            self.thread.join(timeout=2)

        if self.context is not None:
            try:
                self.context.term()
            except Exception:
                pass

        self.socket = None
        self.context = None
        self.thread = None

        self.logger.info("Spell receiver stopped.")

    def _receive_loop(self):
        """Background thread that receives ZeroMQ messages."""

        self.context = zmq.Context()

        self.socket = self.context.socket(zmq.SUB)

        # Only receive messages matching this subscription.
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")

        connection_string = f"tcp://{self.host}:{self.port}"

        self.logger.info(
            "Connecting to ZeroMQ publisher at %s",
            connection_string,
        )

        self.socket.connect(connection_string)

        # Use a poller so that we can periodically check self.running
        # instead of blocking forever on recv().
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)

        try:
            while self.running:

                events = dict(poller.poll(timeout=500))

                if self.socket not in events:
                    continue

                try:
                    message = self.socket.recv_string(
                        flags=zmq.NOBLOCK
                    )
                except zmq.Again:
                    continue

                self._process_message(message)

        except zmq.ZMQError as e:
            if self.running:
                self.logger.error(
                    "ZeroMQ error in spell receiver: %s",
                    e,
                )

        except Exception:
            self.logger.exception(
                "Unexpected error in spell receiver."
            )

        finally:
            try:
                self.socket.close(linger=0)
            except Exception:
                pass

    def _process_message(self, message):
        """Parse and validate a received spell message."""

        self.logger.debug(
            "Received ZeroMQ message: %s",
            message,
        )

        try:
            data = json.loads(message)

        except json.JSONDecodeError:
            self.logger.error(
                "Invalid JSON received: %s",
                message,
            )
            return

        spell = data.get("spell")

        if not isinstance(spell, str):
            self.logger.error(
                "Message does not contain a valid spell: %s",
                data,
            )
            return

        spell = spell.upper().strip()

        if spell not in self.VALID_SPELLS:
            self.logger.warning(
                "Unknown spell received: %s",
                spell,
            )
            return

        confidence = data.get("confidence")

        try:
            if confidence is not None:
                confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = None

        spell_data = {
            "spell": spell,
            "confidence": confidence,
        }

        self.logger.info(
            "SPELL RECEIVED: %s (confidence=%s)",
            spell,
            confidence,
        )

        # Put the spell into the queue without blocking.
        try:
            self.spell_queue.put_nowait(spell_data)

        except queue.Full:
            self.logger.warning(
                "Spell queue is full; dropping spell: %s",
                spell,
            )

    def get_queue(self):
        """Return the queue used by the receiver."""

        return self.spell_queue


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    receiver = SpellReceiver()

    try:
        receiver.start()

        print("Spell receiver running.")
        print("Press Ctrl+C to stop.")

        while True:
            try:
                spell_data = receiver.get_queue().get(
                    timeout=1
                )

                print(
                    f"Received spell: "
                    f"{spell_data['spell']} "
                    f"(confidence={spell_data['confidence']})"
                )

            except queue.Empty:
                pass

    except KeyboardInterrupt:
        print("\nShutting down...")

    finally:
        receiver.stop()
