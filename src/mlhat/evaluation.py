"""B — Generic honest test-then-train evaluation (NOT the MLHAT method)."""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import f1_score, hamming_loss

from .interfaces import OnlineMultiLabelClassifier, validate_sample


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregate metrics computed from actual observed samples."""

    n_samples: int
    micro_f1: float
    hamming_loss: float


def prequential_evaluate(
    model: OnlineMultiLabelClassifier,
    samples: Iterable[tuple[NDArray[np.float64], NDArray[np.int64]]],
    *,
    n_labels: int,
) -> EvaluationResult:
    """Evaluate an online model by predicting *before* updating each sample.

    Args:
        model: Predictor that supports ``predict_one`` and ``learn_one``.
        samples: One-pass sequence of (feature vector, binary target vector).
        n_labels: Number of labels expected in each sample.

    Returns:
        Micro-F1 and Hamming loss computed over observed samples.

    Important:
        This is a generic helper and DOES NOT establish paper metric protocol.
        Confirm paper metrics and warmup/evaluation conventions before use
        in any reported experiment.
    """
    if n_labels <= 0:
        raise ValueError("n_labels must be positive")
    truths: list[NDArray[np.int64]] = []
    preds: list[NDArray[np.int64]] = []
    for features, labels in samples:
        raw_x = np.asarray(features)
        raw_y = np.asarray(labels)
        validate_sample(raw_x, raw_y, n_labels)
        x = np.asarray(raw_x, dtype=np.float64)
        y = np.asarray(raw_y, dtype=np.int64)
        prediction = np.asarray(model.predict_one(x), dtype=np.int64)
        if prediction.shape != (n_labels,) or not np.all(np.isin(prediction, [0, 1])):
            raise ValueError("predict_one must return a binary 1-D vector of length n_labels")
        truths.append(y.copy())
        preds.append(prediction.copy())
        model.learn_one(x, y)
    if not truths:
        raise ValueError("no samples received")
    return EvaluationResult(
        n_samples=len(truths),
        micro_f1=float(f1_score(np.stack(truths), np.stack(preds), average="micro", zero_division=0)),
        hamming_loss=float(hamming_loss(np.stack(truths), np.stack(preds))),
    )
