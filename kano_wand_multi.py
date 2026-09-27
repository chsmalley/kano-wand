"""Experimental multi-wand runner; core BLE behavior lives in kano_wand."""

import asyncio
import os
import queue
import random
import threading
import time

import zmq

from kano_wand import (
    KanoWand,
    RECONNECT_DELAY,
    RECONNECT_MAX_DELAY,
    WAND_ADDRESS,
    ZMQ_BIND_HOST,
    ZMQ_PORT,
    classify_worker,
    print_ble_diagnostics,
)

WAND_ADDRESS_2 = os.getenv("WAND_ADDRESS_2", "")
MAX_WANDS = 2
NOTIFICATION_TIMEOUT = float(os.getenv("WAND_NOTIFICATION_TIMEOUT", "5.0"))


class SpellPublisher:
    """Own the shared ZeroMQ PUB socket on a single thread."""

    def __init__(self):
        self.messages = queue.Queue()
        self.ready = threading.Event()
        self.thread = threading.Thread(
            target=self._publish_loop,
            name="SpellPublisher",
            daemon=True,
        )
        self.thread.start()
        if not self.ready.wait(timeout=5):
            raise RuntimeError("ZeroMQ spell publisher failed to start.")

    def publish(self, message):
        self.messages.put_nowait(message)

    def close(self):
        self.messages.put(None)
        self.thread.join(timeout=2)

    def _publish_loop(self):
        context = zmq.Context()
        publisher = context.socket(zmq.PUB)
        publisher.setsockopt(zmq.LINGER, 0)
        publisher.bind(f"tcp://{ZMQ_BIND_HOST}:{ZMQ_PORT}")
        self.ready.set()
        try:
            while True:
                message = self.messages.get()
                if message is None:
                    break
                publisher.send_json(message)
        finally:
            publisher.close(linger=0)
            context.term()


async def run_wand(wand):
    classifier_thread = threading.Thread(
        target=classify_worker,
        args=(wand,),
        name=f"Classifier-{wand.address}",
        daemon=True,
    )
    classifier_thread.start()
    reconnect_delay = RECONNECT_DELAY
    first_connection = True

    try:
        while not wand.shutdown_event.is_set():
            if not wand._connected_ready:
                connected = await wand.reconnect(
                    attempts=1,
                    delay=0,
                    reason=("initial_connection" if first_connection else "reconnect"),
                )
                first_connection = False
                if connected:
                    reconnect_delay = RECONNECT_DELAY
                    print(f"Wand {wand.address} connected.")
                    continue

                print(
                    f"Wand {wand.address} unavailable; retrying in "
                    f"{reconnect_delay:.1f}s."
                )
                await asyncio.sleep(
                    reconnect_delay + random.uniform(0.0, 0.5)
                )
                reconnect_delay = min(
                    reconnect_delay * 2,
                    RECONNECT_MAX_DELAY,
                )
                continue

            await asyncio.sleep(0.5)
            notification_age = time.monotonic() - wand.last_notification_time
            connection_lost = wand.bluetooth_disconnected.is_set()
            notifications_stalled = (
                wand._connected_ready
                and notification_age > NOTIFICATION_TIMEOUT
            )

            if not connection_lost and not notifications_stalled:
                reconnect_delay = RECONNECT_DELAY
                continue

            if notifications_stalled and not connection_lost:
                print(
                    f"WARNING: Wand {wand.address} has sent no BLE "
                    f"notifications for {notification_age:.1f}s; reconnecting."
                )
                wand.mark_notification_stall()
                reconnect_reason = "notification_stall"
            else:
                reconnect_reason = (
                    wand.last_disconnect_reason or "bluez_disconnect"
                )

            print_ble_diagnostics(wand)
            reconnected = await wand.reconnect(
                attempts=5,
                delay=0,
                reason=reconnect_reason,
            )
            if reconnected:
                reconnect_delay = RECONNECT_DELAY
                print(f"Wand {wand.address} connection restored.")
                continue

            print(
                f"Wand {wand.address} reconnect batch failed; retrying in "
                f"{reconnect_delay:.1f}s."
            )
            await asyncio.sleep(
                reconnect_delay + random.uniform(0.0, 0.5)
            )
            reconnect_delay = min(
                reconnect_delay * 2,
                RECONNECT_MAX_DELAY,
            )
    finally:
        wand.shutdown_event.set()
        classifier_thread.join(timeout=2)
        await wand.disconnect()
        print(f"Wand {wand.address} stopped.")


def read_wand_commands(command_queue):
    print("Runtime commands: add <BLE address> | drop <BLE address>")
    while True:
        try:
            command = input("wand> ").strip().split(maxsplit=1)
        except EOFError:
            return
        if len(command) == 2 and command[0].lower() in {"add", "drop"}:
            command_queue.put((command[0].lower(), command[1]))
        else:
            print("Use: add <BLE address> or drop <BLE address>")


async def main():
    publisher = SpellPublisher()
    command_queue = queue.Queue()
    active_wands = {}

    def add_wand(address):
        wand = KanoWand(address, publisher=publisher)
        task = asyncio.create_task(run_wand(wand), name=f"Wand-{address}")
        active_wands[address] = (wand, task)
        print(f"Added wand {address} ({len(active_wands)}/{MAX_WANDS}).")

    async def drop_wand(address):
        wand_entry = active_wands.pop(address, None)
        if wand_entry is None:
            print(f"Wand {address} is not active.")
            return
        _, task = wand_entry
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        print(f"Dropped wand {address} ({len(active_wands)}/{MAX_WANDS}).")

    try:
        initial_addresses = dict.fromkeys(
            address.strip()
            for address in (WAND_ADDRESS, WAND_ADDRESS_2)
            if address and address.strip()
        )
        for address in list(initial_addresses)[:MAX_WANDS]:
            add_wand(address)

        threading.Thread(
            target=read_wand_commands,
            args=(command_queue,),
            name="WandCommands",
            daemon=True,
        ).start()

        print("Running. Hold a wand button to cast; release to finish.")
        print("Press Ctrl+C to stop.")

        while True:
            await asyncio.sleep(0.1)

            for address, (_, task) in list(active_wands.items()):
                if task.done():
                    active_wands.pop(address, None)
                    try:
                        task.result()
                    except asyncio.CancelledError:
                        pass
                    except Exception as exc:
                        print(
                            f"Wand {address} task failed: "
                            f"{type(exc).__name__}: {exc}"
                        )

            while True:
                try:
                    action, address = command_queue.get_nowait()
                except queue.Empty:
                    break

                if action == "add":
                    if address in active_wands:
                        print(f"Wand {address} is already active.")
                    elif len(active_wands) >= MAX_WANDS:
                        print(f"Only {MAX_WANDS} wands can be active at once.")
                    else:
                        add_wand(address)
                else:
                    await drop_wand(address)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")
    finally:
        print("Shutting down...")
        tasks = [task for _, task in active_wands.values()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        publisher.close()
        print("Shutdown complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print()
        print("Ctrl+C received")
