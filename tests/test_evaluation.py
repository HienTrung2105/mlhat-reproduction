"""Verify actual prequential order and result calculations."""

import numpy as np
import pytest

from mlhat.evaluation import prequential_evaluate


class MemoryModel:
    """Simple fixture that repeats its last labels, for ordering tests only."""

    def __init__(self) -> None:
        self.last = np.array([0, 0], dtype=np.int64)
        self.actions: list[str] = []

    def predict_one(self, features: np.ndarray) -> np.ndarray:
        self.actions.append("predict")
        return self.last.copy()

    def learn_one(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.actions.append("learn")
        self.last = labels.copy()


def test_prequential_predicts_before_learning() -> None:
    model = MemoryModel()
    stream = [
        (np.array([0.0]), np.array([1, 0])),
        (np.array([1.0]), np.array([1, 0])),
        (np.array([2.0]), np.array([0, 1])),
    ]
    result = prequential_evaluate(model, stream, n_labels=2)
    assert result.n_samples == 3
    assert model.actions == ["predict", "learn"] * 3
    assert result.hamming_loss == pytest.approx(0.5)
    assert 0 <= result.micro_f1 <= 1


def test_empty_stream_is_rejected() -> None:
    with pytest.raises(ValueError, match="no samples"):
        prequential_evaluate(MemoryModel(), [], n_labels=2)


def test_evaluation_rejects_fractional_labels_instead_of_silently_rounding() -> None:
    with pytest.raises(ValueError, match="binary"):
        prequential_evaluate(MemoryModel(), [(np.array([1.0]), np.array([0.2, 1.0]))], n_labels=2)
