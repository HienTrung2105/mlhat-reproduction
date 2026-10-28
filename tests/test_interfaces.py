"""Tests for implemented glue code, not a claim MLHAT is implemented."""

import numpy as np
import pytest

from mlhat.interfaces import validate_sample
from mlhat.tree import MLHATTree


def test_validate_sample_accepts_binary_multi_label() -> None:
    validate_sample(np.array([0.5, 1.2]), np.array([0, 1, 0]), n_labels=3)


@pytest.mark.parametrize("labels", [np.array([0, 2]), np.array([1]), np.array([1, -1])])
def test_validate_sample_rejects_invalid_targets(labels: np.ndarray) -> None:
    with pytest.raises(ValueError):
        validate_sample(np.array([1.0, 2.0]), labels, n_labels=2)


def test_unimplemented_mlhat_fails_loudly() -> None:
    with pytest.raises(NotImplementedError, match="not implemented"):
        MLHATTree(n_labels=2).predict_one(np.array([1.0, 2.0]))
