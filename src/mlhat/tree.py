"""Incremental numeric MLHAT tree with adaptive background subtrees."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .drift import AdaptiveDriftManager, DriftAction
from .interfaces import validate_sample
from .leaves import MultiLabelLeafPredictor
from .splitting import NodeStatistics, choose_split


@dataclass
class _Node:
    node_id: int
    stats: NodeStatistics
    leaf_predictor: MultiLabelLeafPredictor | None
    drift_manager: AdaptiveDriftManager | None = None
    background: "_Node | None" = None
    split_feature: int | None = None
    split_threshold: float | None = None
    left: "_Node | None" = None
    right: "_Node | None" = None
    last_split_attempt: int = 0

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


@dataclass(frozen=True)
class DriftEvent:
    """One structural drift action, indexed by stream sample and tree node."""

    sample_index: int
    node_id: int
    depth: int
    action: DriftAction


class MLHATTree:
    """Online multi-label tree with optional node-level drift adaptation."""

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
        has_drift_config = "delta_alt" in config and "kappa_alt" in config
        self.drift_enabled = bool(config.get("drift_enabled", has_drift_config))
        if self.drift_enabled and not has_drift_config:
            raise ValueError("drift requires delta_alt and kappa_alt in config")
        self.delta_alt = float(config.get("delta_alt", 0.05))
        self.kappa_alt = int(config.get("kappa_alt", 200))
        self.adwin_delta = float(config.get("adwin_delta", 0.002))
        self._rng = np.random.default_rng(seed)
        self._next_leaf_seed = seed
        self._next_node_id = 0
        self.n_samples = 0
        self.drift_events: list[DriftEvent] = []
        self.last_prequential_error: float | None = None
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

        manager = (
            AdaptiveDriftManager(
                delta_alt=self.delta_alt,
                kappa_alt=self.kappa_alt,
                adwin_delta=self.adwin_delta,
            )
            if self.drift_enabled
            else None
        )
        node = _Node(
            node_id=self._next_node_id,
            stats=NodeStatistics(self.n_features, self.n_labels),
            leaf_predictor=predictor,
            drift_manager=manager,
        )
        self._next_node_id += 1
        return node

    @staticmethod
    def _route_from(
        root: _Node,
        features: NDArray[np.float64],
    ) -> tuple[_Node, list[_Node]]:
        node = root
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

    def _route(self, features: NDArray[np.float64]) -> tuple[_Node, list[_Node]]:
        return self._route_from(self.root, features)

    def _main_proba_from(
        self, root: _Node, features: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        leaf, _ = self._route_from(root, features)
        assert leaf.leaf_predictor is not None
        return leaf.leaf_predictor.predict_proba_one(features)

    def _prediction_components(
        self, root: _Node, features: NDArray[np.float64]
    ) -> list[tuple[NDArray[np.float64], float]]:
        """Collect Algorithm 3 predictions from one subtree.

        Args:
            root: Root of the main or background subtree to traverse.
            features: One validated feature vector.

        Returns:
            Pairs of label probabilities and error-derived weights.
        """
        leaf, path = self._route_from(root, features)
        assert leaf.leaf_predictor is not None
        error = leaf.drift_manager.main_error if leaf.drift_manager else 0.0
        components = [
            (leaf.leaf_predictor.predict_proba_one(features), max(0.0, 1.0 - error))
        ]

        # Algorithm 3 returns the majority labelset immediately for a pure leaf.
        if leaf.stats.is_pure:
            return components
        for node in path:
            if (
                node.background is not None
                and node.background.stats.n_seen > self.kappa_alt
            ):
                components.extend(self._prediction_components(node.background, features))
        return components

    def predict_proba_one(
        self,
        features: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        x = np.asarray(features, dtype=np.float64)
        if x.shape != (self.n_features,) or not np.isfinite(x).all():
            raise ValueError("features have the wrong shape or contain non-finite values")

        if not self.drift_enabled:
            return self._main_proba_from(self.root, x)

        components = self._prediction_components(self.root, x)
        total_weight = sum(weight for _, weight in components)
        if total_weight == 0.0:
            return np.mean([proba for proba, _ in components], axis=0)
        return sum((proba * weight for proba, weight in components)) / total_weight

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
        raw_x = np.asarray(features)
        raw_y = np.asarray(labels)
        validate_sample(raw_x, raw_y, self.n_labels)
        x = np.asarray(raw_x, dtype=np.float64)
        y = np.asarray(raw_y, dtype=np.int64)

        if x.shape != (self.n_features,):
            raise ValueError("feature vector has the wrong length")

        # Save the externally scored pre-update Hamming error.
        before = self.predict_one(x)
        self.last_prequential_error = float(np.mean(before != y))

        weight = int(self._rng.poisson(self.poisson_lambda))
        self.n_samples += 1
        if self.drift_enabled:
            self.root = self._process_drift(self.root, x, y)
        self._learn_subtree(self.root, x, y, weight)

    def _process_drift(
        self,
        root: _Node,
        features: NDArray[np.float64],
        labels: NDArray[np.int64],
        *,
        depth_offset: int = 0,
    ) -> _Node:
        """Apply node actions before training on the current labeled sample.

        Args:
            root: Root of the subtree being traversed.
            features: Current feature vector.
            labels: Current binary target vector.
            depth_offset: Depth of ``root`` relative to the main tree root.

        Returns:
            The subtree root, possibly replaced by its background branch.
        """
        _, path = self._route_from(root, features)
        for depth, node in enumerate(path, start=depth_offset):
            manager = node.drift_manager
            assert manager is not None
            main_prediction = self._main_proba_from(node, features) >= self.threshold
            main_error = float(np.mean(main_prediction != labels))

            background = node.background
            background_error = None
            background_weight = 0
            if background is not None:
                alternate_prediction = (
                    self._main_proba_from(background, features) >= self.threshold
                )
                background_error = float(np.mean(alternate_prediction != labels))
                background_weight = background.stats.n_seen

            action = manager.observe(
                main_error,
                background_error,
                main_weight=node.stats.n_seen,
                background_weight=background_weight,
            )
            if action is not DriftAction.NONE:
                self.drift_events.append(
                    DriftEvent(self.n_samples, node.node_id, depth, action)
                )

            if action is DriftAction.START_BACKGROUND:
                node.background = self._new_leaf()
            elif action is DriftAction.DISCARD_BACKGROUND:
                node.background = None
            elif action is DriftAction.PROMOTE_BACKGROUND:
                assert node.background is not None
                replacement = node.background
                assert replacement.drift_manager is not None
                replacement.drift_manager.reset_main()
                if node is root:
                    return replacement
                parent = path[depth - depth_offset - 1]
                if parent.left is node:
                    parent.left = replacement
                else:
                    assert parent.right is node
                    parent.right = replacement
                return root

            # Background branches receive their own per-node ADWIN updates.
            # A branch opened on this sample has no pre-update prediction yet.
            if node.background is not None and action is not DriftAction.START_BACKGROUND:
                previous_background = node.background
                node.background = self._process_drift(
                    node.background, features, labels, depth_offset=depth
                )
                if node.background is not previous_background:
                    manager.reset_background()
        return root

    def _learn_subtree(
        self,
        root: _Node,
        features: NDArray[np.float64],
        labels: NDArray[np.int64],
        weight: int,
    ) -> None:
        """Train a main or background subtree with one shared Poisson weight.

        Args:
            root: Root of the subtree to update.
            features: Current feature vector.
            labels: Current binary target vector.
            weight: Poisson draw shared by all branches for this sample.

        Returns:
            None. Node statistics and leaf predictors are updated in place.
        """
        leaf, path = self._route_from(root, features)

        # Every node on the route receives the sample's weighted statistics.
        for node in path:
            node.stats.update(features, labels, weight=weight)

        assert leaf.leaf_predictor is not None
        leaf.leaf_predictor.learn_one(features, labels, weight=weight)

        if leaf.stats.n_seen - leaf.last_split_attempt >= self.kappa_spl:
            leaf.last_split_attempt = leaf.stats.n_seen
            candidate = choose_split(
                leaf.stats,
                delta_spl=self.delta_spl,
                n_bins=self.n_bins,
            )
            if candidate is not None:
                leaf.split_feature = candidate.feature_index
                leaf.split_threshold = candidate.threshold
                leaf.leaf_predictor = None
                leaf.left = self._new_leaf()
                leaf.right = self._new_leaf()
                child = (
                    leaf.left
                    if features[candidate.feature_index] <= candidate.threshold
                    else leaf.right
                )
                child.stats.update(features, labels, weight=weight)
                assert child.leaf_predictor is not None
                child.leaf_predictor.learn_one(features, labels, weight=weight)

        for node in path:
            if node.background is not None:
                self._learn_subtree(node.background, features, labels, weight)
