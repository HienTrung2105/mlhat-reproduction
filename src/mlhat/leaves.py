"""Online multi-label prediction models used inside MLHAT leaves."""

from collections import Counter

import numpy as np
from numpy.typing import NDArray
from river import ensemble, linear_model, neighbors


class MultiLabelLeafPredictor:
    """Switch between Label-Powerset kNN and BR bagging logistic regression."""

    def __init__(
        self,
        n_labels: int,
        *,
        eta: int = 750,
        n_neighbors: int = 5,
        n_estimators: int = 10,
        seed: int = 1,
        prediction_threshold: float = 0.5,
    ) -> None:
        if n_labels <= 0:
            raise ValueError("n_labels must be positive")
        if eta <= 0:
            raise ValueError("eta must be positive")
        if n_neighbors <= 0 or n_estimators <= 0:
            raise ValueError("model sizes must be positive")

        self.n_labels = n_labels
        self.eta = eta
        self.n_neighbors = n_neighbors
        self.n_estimators = n_estimators
        self.seed = seed
        self.prediction_threshold = prediction_threshold

        # Low-complexity classifier: one class represents a complete labelset.
        self._low_model = neighbors.KNNClassifier(n_neighbors=n_neighbors)
        self._labelset_to_id: dict[tuple[int, ...], int] = {}
        self._id_to_labelset: dict[int, tuple[int, ...]] = {}

        # High-complexity classifier: one bagging ensemble per output label.
        # Use plain BaggingClassifier here. Drift detection belongs to the tree,
        # not to the leaf classifier.
        self._high_models = [
            ensemble.BaggingClassifier(
                model=linear_model.LogisticRegression(),
                n_models=n_estimators,
                seed=seed + label_index,
            )
            for label_index in range(n_labels)
        ]

        self.n_seen = 0
        self._labelset_counts: Counter[tuple[int, ...]] = Counter()

    @staticmethod
    def _as_river_features(
        features: NDArray[np.float64],
    ) -> dict[str, float]:
        return {
            f"x{index}": float(value)
            for index, value in enumerate(features)
        }

    def _low_complexity_proba(
        self,
        river_features: dict[str, float],
    ) -> NDArray[np.float64]:
        class_probabilities = self._low_model.predict_proba_one(river_features)
        result = np.zeros(self.n_labels, dtype=np.float64)

        for class_id, probability in class_probabilities.items():
            labelset = self._id_to_labelset.get(int(class_id))
            if labelset is None:
                continue
            result += float(probability) * np.asarray(labelset, dtype=np.float64)

        return np.clip(result, 0.0, 1.0)

    def _high_complexity_proba(
        self,
        river_features: dict[str, float],
    ) -> NDArray[np.float64]:
        result = np.zeros(self.n_labels, dtype=np.float64)

        for index, model in enumerate(self._high_models):
            probabilities = model.predict_proba_one(river_features)
            result[index] = float(probabilities.get(1, 0.0))

        return np.clip(result, 0.0, 1.0)

    def predict_proba_one(
        self,
        features: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
            raise ValueError("features must be a finite, nonempty 1-D vector")

        if self.n_seen == 0:
            return np.zeros(self.n_labels, dtype=np.float64)

        # If exactly one labelset has reached the leaf, entropy is zero.
        if len(self._labelset_counts) == 1:
            only_labelset = next(iter(self._labelset_counts))
            return np.asarray(only_labelset, dtype=np.float64)

        river_features = self._as_river_features(x)

        if self.n_seen < self.eta:
            return self._low_complexity_proba(river_features)

        return self._high_complexity_proba(river_features)

    def predict_one(
        self,
        features: NDArray[np.float64],
    ) -> NDArray[np.int64]:
        probabilities = self.predict_proba_one(features)
        return (probabilities >= self.prediction_threshold).astype(np.int64)

    def learn_one(
        self,
        features: NDArray[np.float64],
        labels: NDArray[np.int64],
        *,
        weight: int = 1,
    ) -> None:
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels, dtype=np.int64)

        if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
            raise ValueError("features must be a finite, nonempty 1-D vector")
        if y.shape != (self.n_labels,) or not np.isin(y, [0, 1]).all():
            raise ValueError("labels must be a binary vector of length n_labels")
        if weight < 0:
            raise ValueError("weight must be nonnegative")

        river_features = self._as_river_features(x)
        labelset = tuple(int(value) for value in y)

        # LP-kNN receives the outer MLHAT Poisson weight as repeated observations.
        # It stops learning when this leaf has crossed eta.
        if weight > 0 and self.n_seen < self.eta:
            class_id = self._labelset_to_id.get(labelset)
            if class_id is None:
                class_id = len(self._labelset_to_id)
                self._labelset_to_id[labelset] = class_id
                self._id_to_labelset[class_id] = labelset

            for _ in range(weight):
                self._low_model.learn_one(river_features, class_id)

        # The bagging ensemble applies its own internal online bootstrap.
        # Do not multiply this sample by the outer MLHAT weight a second time.
        for model, target in zip(self._high_models, y, strict=True):
            model.learn_one(river_features, int(target))

        if weight > 0:
            self.n_seen += weight
            self._labelset_counts[labelset] += weight