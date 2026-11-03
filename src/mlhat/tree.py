"""Incremental numeric MLHAT tree without the C-owned drift module."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .interfaces import validate_sample
from .leaves import MultiLabelLeafPredictor
from .splitting import NodeStatistics, choose_split


@dataclass
class _Node:
    stats: NodeStatistics
    leaf_predictor: MultiLabelLeafPredictor | None
    split_feature: int | None = None
    split_threshold: float | None = None
    left: "_Node | None" = None
    right: "_Node | None" = None
    last_split_attempt: int = 0

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class MLHATTree:
    """Incremental MLHAT main tree; drift alternates are integrated later."""

    def __init__(
        self,
        n_labels: int,
        n_features: int,
        *,
        config: dict[str, object],
        seed: int,
    ) -> None:
        if n_labels <= 0 or n_features <= 0:
            raise ValueError("n_labels and n_features must be positive")

        self.n_labels = n_labels
        self.n_features = n_features
        self.config = config
        self.seed = seed
        self.threshold = float(config["prediction_threshold"])
        self.delta_spl = float(config["delta_spl"])
        self.kappa_spl = int(config["kappa_spl"])
        self.poisson_lambda = float(config["lambda"])
        self.n_bins = int(config["numeric_splitter"]["n_bins"])  # type: ignore[index]
        self._rng = np.random.default_rng(seed)
        self._next_leaf_seed = seed
        self.last_prequential_error: float | None = None
        self.drift_enabled = False
        self.root = self._new_leaf()

    def _new_leaf(self) -> _Node:
        low_config = self.config["low_leaf"]
        high_config = self.config["high_leaf"]

        predictor = MultiLabelLeafPredictor(
            self.n_labels,
            eta=int(self.config["eta"]),
            n_neighbors=int(low_config["n_neighbors"]),  # type: ignore[index]
            n_estimators=int(high_config["n_estimators"]),  # type: ignore[index]
            seed=self._next_leaf_seed,
            prediction_threshold=self.threshold,
        )
        self._next_leaf_seed += 1

        return _Node(
            stats=NodeStatistics(self.n_features, self.n_labels),
            leaf_predictor=predictor,
        )

    def _route(
        self,
        features: NDArray[np.float64],
    ) -> tuple[_Node, list[_Node]]:
        node = self.root
        path = [node]

        while not node.is_leaf:
            assert node.split_feature is not None
            assert node.split_threshold is not None
            assert node.left is not None and node.right is not None

            if features[node.split_feature] <= node.split_threshold:
                node = node.left
            else:
                node = node.right
            path.append(node)

        return node, path

    def predict_proba_one(
        self,
        features: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        x = np.asarray(features, dtype=np.float64)
        if x.shape != (self.n_features,) or not np.isfinite(x).all():
            raise ValueError("features have the wrong shape or contain non-finite values")

        leaf, _ = self._route(x)
        assert leaf.leaf_predictor is not None
        return leaf.leaf_predictor.predict_proba_one(x)

    def predict_one(
        self,
        features: NDArray[np.float64],
    ) -> NDArray[np.int64]:
        probabilities = self.predict_proba_one(features)
        return (probabilities >= self.threshold).astype(np.int64)

    def learn_one(
        self,
        features: NDArray[np.float64],
        labels: NDArray[np.int64],
    ) -> None:
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels, dtype=np.int64)
        validate_sample(x, y, self.n_labels)

        if x.shape != (self.n_features,):
            raise ValueError("feature vector has the wrong length")

        # Save the pre-update Hamming error. C's drift manager will consume it.
        before = self.predict_one(x)
        self.last_prequential_error = float(np.mean(before != y))

        leaf, path = self._route(x)
        weight = int(self._rng.poisson(self.poisson_lambda))

        # Every node on the route receives the sample's weighted statistics.
        for node in path:
            node.stats.update(x, y, weight=weight)

        assert leaf.leaf_predictor is not None
        leaf.leaf_predictor.learn_one(x, y, weight=weight)

        if leaf.stats.n_seen - leaf.last_split_attempt < self.kappa_spl:
            return

        leaf.last_split_attempt = leaf.stats.n_seen
        candidate = choose_split(
            leaf.stats,
            delta_spl=self.delta_spl,
            n_bins=self.n_bins,
        )
        if candidate is None:
            return

        # Replace this leaf by an internal split and create two new leaves.
        leaf.split_feature = candidate.feature_index
        leaf.split_threshold = candidate.threshold
        leaf.leaf_predictor = None
        leaf.left = self._new_leaf()
        leaf.right = self._new_leaf()

        # Feed the splitting sample into the child it would follow.
        child = (
            leaf.left
            if x[candidate.feature_index] <= candidate.threshold
            else leaf.right
        )
        child.stats.update(x, y, weight=weight)
        assert child.leaf_predictor is not None
        child.leaf_predictor.learn_one(x, y, weight=weight)