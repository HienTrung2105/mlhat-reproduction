"""Behavioral checks for node drift decisions and tree integration."""

from collections.abc import Callable
from math import log, sqrt

import numpy as np
import pytest

from mlhat.drift import AdaptiveDriftManager, DriftAction, alternate_bound
from mlhat.tree import MLHATTree


class _ScriptedDetector:
    """Deterministic detector used to isolate the branch policy from ADWIN."""

    def __init__(self, alarm_on: int | None = None) -> None:
        self.alarm_on = alarm_on
        self.width = 0
        self.estimation = 0.0
        self.drift_detected = False

    def update(self, value: float) -> None:
        self.width += 1
        self.estimation = value
        self.drift_detected = self.width == self.alarm_on


def _factory(alarm_on: int | None = None) -> Callable[[], _ScriptedDetector]:
    return lambda: _ScriptedDetector(alarm_on)


def _config(*, drift_enabled: bool = True, kappa_alt: int = 2) -> dict[str, object]:
    return {
        "prediction_threshold": 0.5,
        "delta_spl": 1e-7,
        "kappa_spl": 1000,
        "lambda": 1,
        "numeric_splitter": {"n_bins": 10},
        "delta_alt": 0.05,
        "kappa_alt": kappa_alt,
        "adwin_delta": 0.002,
        "drift_enabled": drift_enabled,
        "eta": 750,
        "low_leaf": {"n_neighbors": 1},
        "high_leaf": {"n_estimators": 2},
    }


def test_alternate_bound_matches_eq16() -> None:
    expected = sqrt(2 * 0.6 * (1 - 0.2) * (100 + 80) * log(2 / 0.05) / (100 * 80))
    assert alternate_bound(0.6, 0.2, 100, 80, 0.05) == pytest.approx(expected)


def test_manager_warns_then_promotes_only_after_grace_period() -> None:
    manager = AdaptiveDriftManager(
        delta_alt=0.05, kappa_alt=2, detector_factory=_factory(alarm_on=2)
    )
    assert manager.observe(0.0, None, main_weight=10) is DriftAction.NONE
    assert manager.observe(1.0, None, main_weight=10) is DriftAction.START_BACKGROUND
    assert manager.observe(1.0, 0.0, main_weight=100, background_weight=2) is DriftAction.NONE
    assert (
        manager.observe(1.0, 0.0, main_weight=100, background_weight=100)
        is DriftAction.PROMOTE_BACKGROUND
    )


def test_manager_discards_worse_background_and_ignores_improvement_alarm() -> None:
    manager = AdaptiveDriftManager(
        delta_alt=0.05, kappa_alt=1, detector_factory=_factory(alarm_on=2)
    )
    manager.observe(0.0, None, main_weight=10)
    assert manager.observe(1.0, None, main_weight=10) is DriftAction.START_BACKGROUND
    assert (
        manager.observe(0.0, 1.0, main_weight=100, background_weight=100)
        is DriftAction.DISCARD_BACKGROUND
    )
    assert manager.background_detector is None

    improving = AdaptiveDriftManager(
        delta_alt=0.05, kappa_alt=1, detector_factory=_factory(alarm_on=2)
    )
    improving.observe(1.0, None, main_weight=10)
    assert improving.observe(0.0, None, main_weight=10) is DriftAction.NONE


def test_real_adwin_opens_background_after_sustained_error_increase() -> None:
    manager = AdaptiveDriftManager(delta_alt=0.05, kappa_alt=200)
    for _ in range(400):
        assert manager.observe(0.0, None, main_weight=400) is DriftAction.NONE

    actions = []
    for _ in range(400):
        action = manager.observe(1.0, None, main_weight=400)
        actions.append(action)
        if action is DriftAction.START_BACKGROUND:
            break
    assert DriftAction.START_BACKGROUND in actions


def test_tree_starts_and_trains_background_on_warning() -> None:
    tree = MLHATTree(1, 1, config=_config(), seed=1)
    tree._rng = type("FixedPoisson", (), {"poisson": lambda self, _: 1})()
    assert tree.root.drift_manager is not None
    tree.root.drift_manager = AdaptiveDriftManager(
        delta_alt=0.05, kappa_alt=2, detector_factory=_factory(alarm_on=2)
    )
    x = np.array([0.0])
    tree.learn_one(x, np.array([0]))
    tree.learn_one(x, np.array([1]))

    assert tree.root.background is not None
    assert tree.root.background.stats.n_seen == 1
    assert tree.root.stats.n_seen == 2
    assert tree.last_prequential_error == 1.0
    assert tree.drift_events[-1].action is DriftAction.START_BACKGROUND
    assert tree.drift_events[-1].sample_index == 2


def test_warning_at_child_only_opens_that_childs_background() -> None:
    tree = MLHATTree(1, 1, config=_config(), seed=1)
    tree._rng = type("FixedPoisson", (), {"poisson": lambda self, _: 1})()
    root = tree.root
    root.split_feature = 0
    root.split_threshold = 0.5
    root.leaf_predictor = None
    root.left = tree._new_leaf()
    root.right = tree._new_leaf()
    assert root.left.drift_manager is not None
    root.left.drift_manager = AdaptiveDriftManager(
        delta_alt=0.05, kappa_alt=2, detector_factory=_factory(alarm_on=2)
    )

    tree.learn_one(np.array([0.0]), np.array([0]))
    tree.learn_one(np.array([0.0]), np.array([1]))

    assert root.background is None
    assert root.left.background is not None
    assert root.right.background is None
    assert root.left.background.stats.n_seen == 1
    assert tree.drift_events[-1].depth == 1


def test_tree_promotes_background_without_training_old_root_again() -> None:
    tree = MLHATTree(1, 1, config=_config(), seed=1)
    tree._rng = type("FixedPoisson", (), {"poisson": lambda self, _: 1})()
    old_root = tree.root
    background = tree._new_leaf()
    old_root.background = background
    background.stats.update(np.array([0.0]), np.array([1]), weight=100)
    assert background.leaf_predictor is not None
    background.leaf_predictor.learn_one(np.array([0.0]), np.array([1]), weight=1)
    old_root.stats.update(np.array([0.0]), np.array([0]), weight=100)
    assert old_root.drift_manager is not None
    old_root.drift_manager.background_detector = _ScriptedDetector()
    # A large active error and a small alternate error make Eq. (16) pass.
    old_root.drift_manager.main_detector = _ScriptedDetector()
    old_root.drift_manager.main_detector.estimation = 1.0
    old_root.drift_manager.main_detector.width = 1
    old_root.drift_manager.kappa_alt = 2
    tree.learn_one(np.array([0.0]), np.array([1]))

    assert tree.root is background
    assert background.stats.n_seen == 101
    assert old_root.stats.n_seen == 100
    assert tree.drift_events[-1].action is DriftAction.PROMOTE_BACKGROUND


def test_tree_discards_worse_background_before_training_sample() -> None:
    tree = MLHATTree(1, 1, config=_config(), seed=1)
    tree._rng = type("FixedPoisson", (), {"poisson": lambda self, _: 1})()
    root = tree.root
    background = tree._new_leaf()
    root.background = background
    x = np.array([0.0])
    root.stats.update(x, np.array([0]), weight=100)
    assert root.leaf_predictor is not None
    root.leaf_predictor.learn_one(x, np.array([0]), weight=1)
    background.stats.update(x, np.array([1]), weight=100)
    assert background.leaf_predictor is not None
    background.leaf_predictor.learn_one(x, np.array([1]), weight=1)
    assert root.drift_manager is not None
    root.drift_manager.main_detector = _ScriptedDetector()
    root.drift_manager.main_detector.width = 1
    root.drift_manager.background_detector = _ScriptedDetector()

    tree.learn_one(x, np.array([0]))

    assert root.background is None
    assert root.stats.n_seen == 101
    assert background.stats.n_seen == 100
    assert tree.drift_events[-1].action is DriftAction.DISCARD_BACKGROUND


def test_background_prediction_requires_grace_period_and_is_error_weighted() -> None:
    tree = MLHATTree(1, 1, config=_config(kappa_alt=2), seed=1)
    root = tree.root
    x = np.array([0.0])
    root.stats.update(x, np.array([0]), weight=1)
    root.stats.update(x, np.array([1]), weight=1)
    background = tree._new_leaf()
    root.background = background

    class _FixedPredictor:
        def __init__(self, value: float) -> None:
            self.value = value

        def predict_proba_one(self, features: np.ndarray) -> np.ndarray:
            return np.array([self.value])

    root.leaf_predictor = _FixedPredictor(0.0)  # type: ignore[assignment]
    background.leaf_predictor = _FixedPredictor(1.0)  # type: ignore[assignment]
    assert root.drift_manager is not None
    assert background.drift_manager is not None
    root.drift_manager.main_detector = _ScriptedDetector()
    root.drift_manager.main_detector.estimation = 0.2
    background.drift_manager.main_detector = _ScriptedDetector()
    background.drift_manager.main_detector.estimation = 0.1

    background.stats.update(x, np.array([1]), weight=2)
    assert tree.predict_proba_one(x)[0] == pytest.approx(0.0)
    background.stats.update(x, np.array([1]), weight=1)
    assert tree.predict_proba_one(x)[0] == pytest.approx(0.9 / (0.8 + 0.9))


def test_drift_disabled_keeps_main_tree_only() -> None:
    tree = MLHATTree(1, 1, config=_config(drift_enabled=False), seed=1)
    assert tree.root.drift_manager is None
    tree.learn_one(np.array([0.0]), np.array([1]))
    assert tree.last_prequential_error == 1.0
    assert tree.drift_events == []
    assert tree.root.background is None


def test_tree_rejects_fractional_labels_before_casting() -> None:
    tree = MLHATTree(1, 1, config=_config(), seed=1)
    with pytest.raises(ValueError, match="binary"):
        tree.learn_one(np.array([0.0]), np.array([0.2]))
