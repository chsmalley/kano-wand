import asyncio
import json
import os
import re
import time
from dataclasses import asdict
from pathlib import Path

# Import your KanoWand class
from kano_wand import KanoWand

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

WAND_ADDRESS = "D0:1F:65:71:51:32"

TRAINING_FOLDER = Path("training_data")

# Number of times to record each spell
REPETITIONS = 5

spells = [
    "Stupefy",
    "Wingardium Leviosa",
    "Reducio",
    "Flipendo",
    "Expelliarmus",
    "Incendio",
    "Lumos",
    "Locomotor",
    "Engorgio",
    "Aguamenti",
    "Avis",
    "Reducto"
]


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------

def safe_folder_name(name):
    """
    Convert a spell name into a filesystem-safe folder name.
    """
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")


def gesture_to_dict(gesture):
    """
    Convert a Gesture dataclass into a dictionary suitable for JSON.

    The Gesture is expected to look like:

        Gesture(
            motion=[WandMotionState(...)],
            orientation=[WandOrientationState(...)]
        )
    """

    return {
        "motion": [
            asdict(sample)
            for sample in gesture.motion
        ],
        "orientation": [
            asdict(sample)
            for sample in gesture.orientation
        ]
    }


def save_gesture(gesture, spell, repetition, output_folder):
    """
    Save one completed gesture to a JSON file.
    """

    spell_folder = output_folder / safe_folder_name(spell)
    spell_folder.mkdir(parents=True, exist_ok=True)

    filename = f"recording_{repetition:03d}.json"
    filepath = spell_folder / filename

    data = {
        "spell": spell,
        "repetition": repetition,
        "recorded_at": time.time(),
        "motion": [
            asdict(sample)
            for sample in gesture.motion
        ],
        "orientation": [
            asdict(sample)
            for sample in gesture.orientation
        ]
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    return filepath


def create_manifest(output_folder, recordings):
    """
    Create a manifest describing all recordings in the dataset.
    """

    manifest = {
        "created_at": time.time(),
        "num_spells": len(spells),
        "repetitions_per_spell": REPETITIONS,
        "num_recordings": len(recordings),
        "recordings": recordings
    }

    filepath = output_folder / "manifest.json"

    with open(filepath, "w") as f:
        json.dump(manifest, f, indent=2)

    return filepath


# ----------------------------------------------------------------------
# Queue handling
# ----------------------------------------------------------------------

def clear_spell_queue(wand):
    """
    Remove any old gestures from the queue.

    This is important before starting a new recording so that an old
    gesture cannot accidentally be saved as the current spell.
    """

    while True:
        try:
            wand.spell_queue.get_nowait()
            wand.spell_queue.task_done()
        except Exception:
            break


async def ensure_connected(wand):
    """Recover the BLE connection if Bleak reports it has been lost."""
    if wand.bluetooth_disconnected.is_set() or not wand.client.is_connected:
        print()
        print("BLE connection is unavailable. Recovering...")
        return await wand.reconnect(attempts=6, delay=2.5)

    return True


def print_ble_diagnostics(wand):
    now = time.monotonic()
    print(
        "BLE diagnostics: "
        f"connected={wand.client.is_connected}, "
        f"motion={wand.notification_counts['motion']}, "
        f"orientation={wand.notification_counts['orientation']}, "
        f"button={wand.notification_counts['button']}, "
        f"button_down={wand.button_down_count}, "
        f"button_up={wand.button_up_count}, "
        f"last_any={now - wand.last_notification_time:.2f}s ago"
    )


async def wait_for_gesture(wand):
    """
    Wait for a button-held gesture while watching for BLE loss.

    If BLE drops before a gesture begins, reconnect and keep waiting.
    If BLE drops during a gesture, discard that partial gesture, reconnect,
    and require the user to repeat it rather than saving incomplete data.
    """

    print()
    print("Press and HOLD the wand button...")

    last_status = time.monotonic()

    while True:
        if wand.bluetooth_disconnected.is_set() or not wand.client.is_connected:
            if not await ensure_connected(wand):
                raise RuntimeError("Unable to recover the wand Bluetooth connection.")
            print("BLE recovered. Please press and HOLD the wand button...")
            last_status = time.monotonic()

        if wand.recording:
            print("Recording!")

            # Wait for release, but abort this recording if BLE disappears.
            while wand.recording:
                if wand.bluetooth_disconnected.is_set() or not wand.client.is_connected:
                    print()
                    print("BLE lost during recording. This attempt will NOT be saved.")
                    if not await ensure_connected(wand):
                        raise RuntimeError(
                            "Unable to recover the wand Bluetooth connection."
                        )
                    print("BLE recovered. Please repeat this spell.")
                    clear_spell_queue(wand)
                    wand.recording = False
                    wand.button_pressed = False
                    return await wait_for_gesture(wand)

                await asyncio.sleep(0.01)

            print("Button released.")

            for _ in range(100):
                try:
                    gesture = wand.spell_queue.get_nowait()
                    wand.spell_queue.task_done()
                    return gesture
                except Exception:
                    await asyncio.sleep(0.01)

            # A release without a queued gesture means the callback stream
            # was interrupted or the gesture was empty.  Do not save it.
            raise RuntimeError(
                "Button was released, but no completed gesture was received."
            )

        # While waiting for the user, periodically report diagnostics.
        if time.monotonic() - last_status >= 10:
            print_ble_diagnostics(wand)
            last_status = time.monotonic()

        await asyncio.sleep(0.01)


# ----------------------------------------------------------------------
# Recording
# ----------------------------------------------------------------------

async def record_spell(wand, spell, repetition, output_folder):
    """
    Record one repetition of a spell.
    """

    print()
    print("=" * 70)
    print(f"Spell: {spell}")
    print(f"Recording: {repetition} of {REPETITIONS}")
    print("=" * 70)

    print()
    print(f"Get ready to perform: {spell}")
    print()
    print("Hold the wand button down while performing the spell.")
    print("Release the button when the spell is complete.")

    # Allow the user a moment to get ready
    await asyncio.sleep(2)

    # Make sure there isn't an old gesture waiting in the queue
    clear_spell_queue(wand)

    # Wait for the actual gesture.  A BLE failure during capture returns
    # control here so the repetition can be retried rather than saving bad data.
    try:
        gesture = await wait_for_gesture(wand)
    except RuntimeError as e:
        print()
        print(f"Recording attempt failed: {e}")
        return None

    # Make sure we actually captured data
    motion_count = len(gesture.motion)
    orientation_count = len(gesture.orientation)

    if motion_count == 0 and orientation_count == 0:
        print("WARNING: No motion or orientation data was recorded.")
        return None

    # Save the gesture
    filepath = save_gesture(
        gesture,
        spell,
        repetition,
        output_folder
    )

    print()
    print(f"Saved: {filepath}")
    print(f"  Motion samples:      {motion_count}")
    print(f"  Orientation samples: {orientation_count}")

    if motion_count > 0:
        start_time = gesture.motion[0].timestamp
        end_time = gesture.motion[-1].timestamp
        print(f"  Motion duration:     {end_time - start_time:.3f} seconds")

    return {
        "spell": spell,
        "repetition": repetition,
        "file": str(filepath),
        "motion_samples": motion_count,
        "orientation_samples": orientation_count
    }


# ----------------------------------------------------------------------
# Main training session
# ----------------------------------------------------------------------

async def main():
    print()
    print("=" * 70)
    print("KANO WAND SPELL TRAINING DATA RECORDER")
    print("=" * 70)
    print()
    print(f"Spells:       {len(spells)}")
    print(f"Repetitions:  {REPETITIONS}")
    print(f"Total samples to record: {len(spells) * REPETITIONS}")
    print()
    print("Instructions:")
    print("  1. Get ready for the displayed spell.")
    print("  2. Press and HOLD the wand button.")
    print("  3. Perform the spell motion.")
    print("  4. Release the button when finished.")
    print()
    print("The raw motion and orientation data will be saved.")
    print()
    print("Press Ctrl+C at any time to stop safely.")
    print()

    # Create the training data folder
    TRAINING_FOLDER.mkdir(parents=True, exist_ok=True)

    wand = KanoWand(WAND_ADDRESS)

    recordings = []

    try:
        await wand.connect()

        # Give BLE notifications a moment to stabilize
        await asyncio.sleep(1)

        # --------------------------------------------------------------
        # Record every spell
        # --------------------------------------------------------------

        for spell_index, spell in enumerate(spells, start=1):

            print()
            print()
            print("#" * 70)
            print(f" SPELL {spell_index} OF {len(spells)}")
            print(f" {spell}")
            print("#" * 70)

            repetition = 1
            while repetition <= REPETITIONS:
                if not await ensure_connected(wand):
                    raise RuntimeError("Unable to reconnect to the wand.")

                result = await record_spell(
                    wand,
                    spell,
                    repetition,
                    TRAINING_FOLDER
                )

                if result is not None:
                    recordings.append(result)
                    repetition += 1
                else:
                    print()
                    print(
                        f"Re-attempting {spell}, repetition {repetition}. "
                        "No recording was saved."
                    )
                    await asyncio.sleep(1)
                    continue

                # Short pause between repetitions
                if repetition <= REPETITIONS:
                    print()
                    print("Get ready for the next repetition...")
                    await asyncio.sleep(2)

            # Longer pause between spells
            if spell_index < len(spells):
                print()
                print("-" * 70)
                print(f"Completed {spell}")
                print(f"Next spell: {spells[spell_index]}")
                print("-" * 70)
                print()
                print("Take a short break if needed.")
                await asyncio.sleep(3)

        # --------------------------------------------------------------
        # Create dataset manifest
        # --------------------------------------------------------------

        manifest_path = create_manifest(
            TRAINING_FOLDER,
            recordings
        )

        print()
        print()
        print("=" * 70)
        print("TRAINING DATA COLLECTION COMPLETE!")
        print("=" * 70)
        print()
        print(f"Recordings saved: {len(recordings)}")
        print(f"Data folder:      {TRAINING_FOLDER.resolve()}")
        print(f"Manifest:         {manifest_path.resolve()}")
        print()

        for spell in spells:
            spell_recordings = [
                r for r in recordings
                if r["spell"] == spell
            ]

            print(
                f"{spell:25s} "
                f"{len(spell_recordings)}/{REPETITIONS}"
            )

        print()
        print("The data is ready for preprocessing and training.")

    except KeyboardInterrupt:
        print()
        print()
        print("Ctrl+C detected.")
        print("Stopping recording...")

    except Exception as e:
        print()
        print(f"ERROR: {type(e).__name__}: {e}")

    finally:
        # --------------------------------------------------------------
        # Always disconnect the wand cleanly
        # --------------------------------------------------------------

        print()
        print("Disconnecting wand...")

        try:
            await wand.disconnect()
        except Exception as e:
            print(f"Error disconnecting wand: {e}")

        print("Wand disconnected.")


# ----------------------------------------------------------------------
# Program entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Handles Ctrl+C occurring while asyncio is shutting down.
        print()
        print("Training recorder stopped.")
