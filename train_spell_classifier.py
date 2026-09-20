import json
import hashlib
import numpy as np
from pathlib import Path
from collections import Counter

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from dtw_classifier import DTWNearestNeighborClassifier


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

TRAINING_FOLDERS = [
    Path("training_data_1"),
    Path("training_data_2"),
    Path("training_data_3"),
    Path("training_data_4"),
    Path("training_data_5"),
    Path("training_data_6"),
    ]
MODEL_FILE = "spell_classifier.joblib"
DTW_MODEL_FILE = "dtw_spell_classifier.joblib"

# Number of time points every gesture will be converted to.
# This makes gestures of different lengths comparable.
NUM_SAMPLES = 100
MIN_GESTURE_DURATION = 0.5
MAX_GESTURE_DURATION = 5.0
MAX_SAMPLE_GAP = 1.0


# ----------------------------------------------------------------------
# Load one JSON gesture
# ----------------------------------------------------------------------

def load_gesture(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def validate_gesture(gesture):
    """Return None for usable data, otherwise a reason to skip it."""
    motion = gesture.get("motion", [])
    orientation = gesture.get("orientation", [])
    all_samples = motion or orientation

    if len(all_samples) < 2:
        return "fewer than two samples"

    timestamps = sorted(
        float(sample["timestamp"])
        for sample in all_samples
    )
    duration = timestamps[-1] - timestamps[0]
    if duration < MIN_GESTURE_DURATION:
        return f"duration {duration:.3f}s is below {MIN_GESTURE_DURATION}s"
    if duration > MAX_GESTURE_DURATION:
        return f"duration {duration:.3f}s exceeds {MAX_GESTURE_DURATION}s"

    largest_gap = max(
        current - previous
        for previous, current in zip(timestamps, timestamps[1:])
    )
    if largest_gap > MAX_SAMPLE_GAP:
        return f"sample gap {largest_gap:.3f}s exceeds {MAX_SAMPLE_GAP}s"

    return None


# ----------------------------------------------------------------------
# Interpolate a time series
# ----------------------------------------------------------------------

def interpolate_stream(samples, timestamps, num_samples):
    """
    Resample a stream to exactly num_samples points.

    Each column is interpolated independently.
    """

    if len(samples) == 0:
        return np.zeros((num_samples, 1))

    samples = np.asarray(samples, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)

    order = np.argsort(timestamps)
    timestamps = timestamps[order]
    samples = samples[order]

    # Remove duplicate timestamps
    unique_indices = np.unique(timestamps, return_index=True)[1]
    unique_indices = np.sort(unique_indices)

    timestamps = timestamps[unique_indices]
    samples = samples[unique_indices]

    if len(timestamps) == 1:
        return np.repeat(
            samples,
            num_samples,
            axis=0
        )

    # Normalize time to 0-1
    old_time = np.linspace(0, 1, len(timestamps))
    new_time = np.linspace(0, 1, num_samples)

    result = np.zeros(
        (num_samples, samples.shape[1])
    )

    for column in range(samples.shape[1]):
        result[:, column] = np.interp(
            new_time,
            old_time,
            samples[:, column]
        )

    return result


# ----------------------------------------------------------------------
# Convert one gesture into a feature vector
# ----------------------------------------------------------------------

def gesture_to_features(gesture):
    """
    Convert motion + orientation streams into one fixed-length
    feature vector.

    Motion:
        mag_x
        mag_y
        mag_z
        acc_x
        acc_y
        acc_z
        pitch
        roll
        yaw

    Orientation:
        x
        y
        z
        w
    """

    # --------------------------------------------------------------
    # Motion
    # --------------------------------------------------------------

    motion = gesture["motion"]

    if len(motion) > 0:

        motion_timestamps = [
            sample["timestamp"]
            for sample in motion
        ]

        motion_values = [
            [
                sample["mag_x"],
                sample["mag_y"],
                sample["mag_z"],
                sample["acc_x"],
                sample["acc_y"],
                sample["acc_z"],
                sample["pitch"],
                sample["roll"],
                sample["yaw"]
            ]
            for sample in motion
        ]

        motion_resampled = interpolate_stream(
            motion_values,
            motion_timestamps,
            NUM_SAMPLES
        )

    else:
        motion_resampled = np.zeros(
            (NUM_SAMPLES, 9)
        )

    # --------------------------------------------------------------
    # Orientation
    # --------------------------------------------------------------

    orientation = gesture["orientation"]

    if len(orientation) > 0:

        orientation_timestamps = [
            sample["timestamp"]
            for sample in orientation
        ]

        orientation_values = [
            [
                sample["x"],
                sample["y"],
                sample["z"],
                sample["w"]
            ]
            for sample in orientation
        ]

        orientation_resampled = interpolate_stream(
            orientation_values,
            orientation_timestamps,
            NUM_SAMPLES
        )

    else:
        orientation_resampled = np.zeros(
            (NUM_SAMPLES, 4)
        )

    # --------------------------------------------------------------
    # Combine motion and orientation
    # --------------------------------------------------------------

    combined = np.hstack([
        motion_resampled,
        orientation_resampled
    ])

    # Flatten:
    #
    # 100 samples x 13 features
    #
    # becomes:
    #
    # 1300-element feature vector
    #
    return combined.flatten()


# ----------------------------------------------------------------------
# Load entire training dataset
# ----------------------------------------------------------------------

def load_training_data(
    training_folder,
    X,
    y,
    groups,
    seen_files,
    skipped,
):

    print()
    print("Loading training data...")
    print()

    for spell_folder in sorted(training_folder.iterdir()):

        if not spell_folder.is_dir():
            continue

        spell = spell_folder.name

        files = sorted(
            spell_folder.glob("recording_*.json")
        )

        print(
            f"{spell:25s}: {len(files)} recordings"
        )

        for filepath in files:

            try:
                file_hash = hashlib.sha256(filepath.read_bytes()).hexdigest()
                if file_hash in seen_files:
                    skipped["duplicate"] += 1
                    print(f"Skipping duplicate: {filepath}")
                    continue

                gesture = load_gesture(filepath)
                reason = validate_gesture(gesture)
                if reason is not None:
                    skipped[reason] += 1
                    print(f"Skipping {filepath}: {reason}")
                    continue

                features = gesture_to_features(
                    gesture
                )

                X.append(features)
                y.append(spell)
                groups.append(training_folder.name)
                seen_files.add(file_hash)

            except Exception as e:
                print(
                    f"ERROR reading {filepath}: {e}"
                )

    return X, y, groups


# ----------------------------------------------------------------------
# Train classifier
# ----------------------------------------------------------------------

def evaluate_classifier(model_factory, X, y, groups):
    validation_predictions = []
    validation_actual = []
    logo = LeaveOneGroupOut()

    for train_indices, test_indices in logo.split(X, y, groups):
        model = model_factory()
        model.fit(X[train_indices], y[train_indices])
        validation_predictions.extend(
            model.predict(X[test_indices])
        )
        validation_actual.extend(y[test_indices])

    labels = sorted(np.unique(y))
    print(classification_report(
        validation_actual,
        validation_predictions,
        labels=labels,
        zero_division=0,
    ))
    print("Confusion matrix:")
    print(confusion_matrix(
        validation_actual,
        validation_predictions,
        labels=labels,
    ))


def train_classifier(X, y, groups):

    print()
    print("=" * 70)
    print("TRAINING CLASSIFIER")
    print("=" * 70)
    print()

    print(f"Training samples: {len(X)}")
    print(f"Features/sample:  {X.shape[1]}")
    print(
        f"Spells:           {len(np.unique(y))}"
    )

    model_factories = {
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=500,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "dtw": lambda: DTWNearestNeighborClassifier(
            n_neighbors=3,
            sequence_length=25,
            window=4,
            temperature=1.0,
        ),
    }

    for name, model_factory in model_factories.items():
        print()
        print(f"Session-aware validation: {name}")
        evaluate_classifier(model_factory, X, y, groups)

    print()
    print("Training final model on all accepted recordings...")
    classifier = model_factories["extra_trees"]()
    classifier.fit(X, y)
    dtw_classifier = model_factories["dtw"]()
    dtw_classifier.fit(X, y)

    # --------------------------------------------------------------
    # Save model
    # --------------------------------------------------------------

    joblib.dump(
        classifier,
        MODEL_FILE
    )
    joblib.dump(
        dtw_classifier,
        DTW_MODEL_FILE,
    )

    print()
    print(f"Model saved to: {MODEL_FILE}")
    print(f"DTW model saved to: {DTW_MODEL_FILE}")

    return classifier


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    X = []
    y = []
    groups = []
    seen_files = set()
    skipped = Counter()
    for training_folder in TRAINING_FOLDERS:
        if not training_folder.exists():
            print(
                f"Training folder does not exist: "
                f"{training_folder}"
            )
            return

        X, y, groups = load_training_data(
            training_folder,
            X,
            y,
            groups,
            seen_files,
            skipped,
        )

        if len(X) == 0:
            print("No training data found.")
            return

    print()
    print(f"Skipped duplicate recordings: {skipped['duplicate']}")
    classifier = train_classifier(
        np.asarray(X),
        np.asarray(y),
        np.asarray(groups),
    )


if __name__ == "__main__":
    main()
