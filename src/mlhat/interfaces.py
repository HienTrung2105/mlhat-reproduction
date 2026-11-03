"""Shared interfaces for online multi-label classifiers."""

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class OnlineMultiLabelClassifier(Protocol):
    """Predict first, then learn one labeled stream sample at a time."""

    def predict_proba_one(
        self,
        features: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return one probability in [0, 1] for each label."""
        ...

    def predict_one(
        self,
        features: NDArray[np.float64],
    ) -> NDArray[np.int64]:
        """Return one binary prediction for each label."""
        ...

    def learn_one(
        self,
        features: NDArray[np.float64],
        labels: NDArray[np.int64],
    ) -> None:
        """Update the model using one labeled sample."""
        ...


def validate_sample(features: NDArray, labels: NDArray, n_labels: int) -> None:
    """Reject malformed features and non-binary multi-label targets."""
    x = np.asarray(features)
    y = np.asarray(labels)

    if x.ndim != 1 or x.size == 0 or not np.issubdtype(x.dtype, np.number):
        raise ValueError("features must be a nonempty numeric 1-D array")
    if not np.all(np.isfinite(x)):
        raise ValueError("features must be finite")
    if y.ndim != 1 or y.shape[0] != n_labels:
        raise ValueError("labels must be a 1-D vector of length n_labels")
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("labels must contain only binary values (0 or 1)")