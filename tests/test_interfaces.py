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


def test_mlhat_tree_predicts_binary_multi_label_vector() -> None:
    config = {
        "prediction_threshold": 0.5,
        "delta_spl": 1e-7,
        "kappa_spl": 200,
        "lambda": 1,
        "eta": 750,
        "low_leaf": {"n_neighbors": 5},
        "high_leaf": {"n_estimators": 10},
        "numeric_splitter": {"n_bins": 10},
    }

    model = MLHATTree(
        n_labels=2,
        n_features=2,
        config=config,
        seed=1,
    )

    prediction = model.predict_one(
        np.array([1.0, 2.0], dtype=np.float64)
    )

    assert prediction.shape == (2,)
    assert prediction.dtype == np.int64
    assert np.isin(prediction, [0, 1]).all()
