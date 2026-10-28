"""Small contracts shared by A (tree), B (leaves/evaluation) and C (drift)."""

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class OnlineMultiLabelClassifier(Protocol):
    """Interface for predicting *before* learning each multi-label sample."""

    def predict_one(self, features: NDArray[np.float64]) -> NDArray[np.int64]:
        """Return binary predictions for one feature vector."""
        ...

    def learn_one(self, features: NDArray[np.float64], labels: NDArray[np.int64]) -> None:
        """Update the model using one feature vector and its known labels."""
        ...


def validate_sample(features: NDArray, labels: NDArray, n_labels: int) -> None:
    """Validate 1-D finite numeric features and binary multi-label targets.

    Args:
        features: One observation as a numeric 1-D NumPy vector.
        labels: Binary multi-label targets for the observation.
        n_labels: Number of expected output labels.

    Returns:
        None. Raises ValueError for incompatible inputs.
    """
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
