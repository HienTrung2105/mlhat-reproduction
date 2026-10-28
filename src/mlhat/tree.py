"""A — MLHAT tree. Skeleton ONLY; no research algorithm implemented."""

import numpy as np
from numpy.typing import NDArray


class MLHATTree:
    """Placeholder for the independently implemented MLHAT method.

    The team must first verify paper equations, node statistics, splits,
    update order and leaf/drift interfaces. Do not port author source code.
    """

    def __init__(self, n_labels: int) -> None:
        """Construct an unimplemented tree with a known label dimension."""
        if n_labels <= 0:
            raise ValueError("n_labels must be positive")
        self.n_labels = n_labels

    def predict_one(self, features: NDArray[np.float64]) -> NDArray[np.int64]:
        """Predict multi-label output for one sample (not implemented yet)."""
        raise NotImplementedError("MLHAT is not implemented; A/B/C must build it from the paper")

    def learn_one(self, features: NDArray[np.float64], labels: NDArray[np.int64]) -> None:
        """Learn from one labeled sample (not implemented yet)."""
        raise NotImplementedError("MLHAT is not implemented; implement and test the actual update")
