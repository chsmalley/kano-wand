import numpy as np


class DTWNearestNeighborClassifier:
    """Nearest-neighbor classifier for fixed-length multichannel gestures."""

    def __init__(
        self,
        n_neighbors=3,
        sequence_length=50,
        window=8,
        temperature=1.0,
    ):
        self.n_neighbors = n_neighbors
        self.sequence_length = sequence_length
        self.window = window
        self.temperature = temperature

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] % 13 != 0:
            raise ValueError("Expected flattened gesture sequences with 13 channels")

        self.n_channels_ = 13
        self.original_length_ = X.shape[1] // self.n_channels_
        self.classes_ = np.unique(y)
        self.y_ = np.asarray(y)

        # Normalize each sensor channel without mixing accelerometer,
        # magnetic, angular, and quaternion units.
        raw_sequences = X.reshape(-1, self.original_length_, self.n_channels_)
        self.channel_mean_ = raw_sequences.mean(axis=(0, 1))
        self.channel_scale_ = raw_sequences.std(axis=(0, 1))
        self.channel_scale_[self.channel_scale_ < 1e-6] = 1.0
        self.X_ = self._prepare_sequences(X)
        return self

    def _prepare_sequences(self, X):
        sequences = np.asarray(X, dtype=float).reshape(
            -1,
            self.original_length_,
            self.n_channels_,
        )
        sequences = (sequences - self.channel_mean_) / self.channel_scale_
        if self.original_length_ != self.sequence_length:
            old_time = np.linspace(0.0, 1.0, self.original_length_)
            new_time = np.linspace(0.0, 1.0, self.sequence_length)
            resampled = np.empty(
                (len(sequences), self.sequence_length, self.n_channels_)
            )
            for sample_index, sequence in enumerate(sequences):
                for channel in range(self.n_channels_):
                    resampled[sample_index, :, channel] = np.interp(
                        new_time,
                        old_time,
                        sequence[:, channel],
                    )
            sequences = resampled
        return sequences

    def _distance(self, left, right):
        rows = len(left)
        columns = len(right)
        window = max(self.window, abs(rows - columns))
        costs = np.full((rows + 1, columns + 1), np.inf)
        costs[0, 0] = 0.0
        local_costs = np.mean(
            (left[:, None, :] - right[None, :, :]) ** 2,
            axis=2,
        )

        for row in range(1, rows + 1):
            start = max(1, row - window)
            stop = min(columns, row + window) + 1
            for column in range(start, stop):
                costs[row, column] = local_costs[row - 1, column - 1] + min(
                    costs[row - 1, column],
                    costs[row, column - 1],
                    costs[row - 1, column - 1],
                )

        return float(np.sqrt(costs[rows, columns] / (rows + columns)))

    def _nearest(self, X):
        sequences = self._prepare_sequences(X)
        distances = np.empty((len(sequences), len(self.X_)))
        for sample_index, sequence in enumerate(sequences):
            distances[sample_index] = [
                self._distance(sequence, template)
                for template in self.X_
            ]
        return distances

    def predict_proba(self, X):
        distances = self._nearest(X)
        probabilities = np.zeros((len(distances), len(self.classes_)))
        neighbor_count = min(self.n_neighbors, len(self.X_))

        for row_index, row in enumerate(distances):
            nearest_indices = np.argsort(row)[:neighbor_count]
            nearest_distances = row[nearest_indices]
            weights = np.exp(
                -(nearest_distances - nearest_distances.min())
                / max(self.temperature, 1e-6)
            )
            for index, weight in zip(nearest_indices, weights):
                class_index = np.flatnonzero(
                    self.classes_ == self.y_[index]
                )[0]
                probabilities[row_index, class_index] += weight
            probabilities[row_index] /= probabilities[row_index].sum()

        return probabilities

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]
