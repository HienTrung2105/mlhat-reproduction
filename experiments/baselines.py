"""BR+HAT baseline for the reproduction experiments."""

from collections.abc import Mapping

import numpy as np
import river
from numpy.typing import NDArray
from river import drift, tree


class BinaryRelevanceHAT:
    """One River Hoeffding Adaptive Tree per output label."""

    def __init__(
        self,
        n_labels: int,
        *,
        seed: int,
        config: Mapping[str, object],
    ) -> None:
        if n_labels <= 0:
            raise ValueError("n_labels must be positive")

        expected_version = str(config["river_version"])
        if river.__version__ != expected_version:
            raise RuntimeError(
                f"Config requires river=={expected_version}, "
                f"but installed version is {river.__version__}"
            )

        if int(config["n_models"]) != n_labels:
            raise ValueError(
                f"Config says n_models={config['n_models']}, "
                f"but dataset has {n_labels} labels"
            )

        self.n_labels = n_labels
        self.threshold = float(config["prediction_threshold"])

        splitter_name = str(config["splitter"])
        if splitter_name != "GaussianSplitter":
            raise ValueError(f"Unsupported splitter: {splitter_name}")

        self.models = [
            tree.HoeffdingAdaptiveTreeClassifier(
                grace_period=int(config["grace_period"]),
                delta=float(config["delta"]),
                tau=float(config["tau"]),
                split_criterion=str(config["split_criterion"]),
                leaf_prediction=str(config["leaf_prediction"]),
                nb_threshold=int(config["nb_threshold"]),
                splitter=tree.splitter.GaussianSplitter(),
                drift_detector=drift.ADWIN(),
                drift_window_threshold=int(config["drift_window_threshold"]),
                switch_significance=float(config["switch_significance"]),
                bootstrap_sampling=bool(config["bootstrap_sampling"]),
                seed=seed + label_index,
            )
            for label_index in range(n_labels)
        ]

    @staticmethod
    def _as_river_features(
        features: NDArray[np.float64],
    ) -> dict[str, float]:
        return {
            f"Att{index + 1}": float(value)
            for index, value in enumerate(features)
        }

    def predict_proba_one(
        self,
        features: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        river_features = self._as_river_features(features)
        probabilities = []

        for model in self.models:
            output = model.predict_proba_one(river_features)
            probabilities.append(float(output.get(1, 0.0)))

        return np.asarray(probabilities, dtype=np.float64)

    def predict_one(
        self,
        features: NDArray[np.float64],
    ) -> NDArray[np.int64]:
        return (self.predict_proba_one(features) >= self.threshold).astype(np.int64)

    def learn_one(
        self,
        features: NDArray[np.float64],
        labels: NDArray[np.int64],
    ) -> None:
        river_features = self._as_river_features(features)

        for model, label in zip(self.models, labels, strict=True):
            model.learn_one(river_features, int(label))