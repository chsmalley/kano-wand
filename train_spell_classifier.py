import json
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

TRAINING_FOLDER = Path("training_data")
MODEL_FILE = "spell_classifier.joblib"

# Number of time points every gesture will be converted to.
# This makes gestures of different lengths comparable.
NUM_SAMPLES = 100


# ----------------------------------------------------------------------
# Load one JSON gesture
# ----------------------------------------------------------------------

def load_gesture(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


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

def load_training_data():

    X = []
    y = []

    print()
    print("Loading training data...")
    print()

    for spell_folder in sorted(TRAINING_FOLDER.iterdir()):

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
                gesture = load_gesture(filepath)

                features = gesture_to_features(
                    gesture
                )

                X.append(features)
                y.append(spell)

            except Exception as e:
                print(
                    f"ERROR reading {filepath}: {e}"
                )

    return np.asarray(X), np.asarray(y)


# ----------------------------------------------------------------------
# Train classifier
# ----------------------------------------------------------------------

def train_classifier(X, y):

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

    # --------------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print()
    print(f"Training set: {len(X_train)}")
    print(f"Test set:     {len(X_test)}")

    # --------------------------------------------------------------
    # Classifier
    # --------------------------------------------------------------

    classifier = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1
            )
        )
    ])

    print()
    print("Training...")
    
    classifier.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------------

    predictions = classifier.predict(
        X_test
    )

    print()
    print("=" * 70)
    print("CLASSIFICATION RESULTS")
    print("=" * 70)
    print()

    print(
        classification_report(
            y_test,
            predictions
        )
    )

    print("Confusion matrix:")
    print()

    labels = sorted(np.unique(y))

    print(
        "Actual / Predicted"
    )

    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=labels
        )
    )

    # --------------------------------------------------------------
    # Save model
    # --------------------------------------------------------------

    joblib.dump(
        classifier,
        MODEL_FILE
    )

    print()
    print(f"Model saved to: {MODEL_FILE}")

    return classifier


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():

    if not TRAINING_FOLDER.exists():
        print(
            f"Training folder does not exist: "
            f"{TRAINING_FOLDER}"
        )
        return

    X, y = load_training_data()

    if len(X) == 0:
        print("No training data found.")
        return

    classifier = train_classifier(
        X,
        y
    )


if __name__ == "__main__":
    main()
