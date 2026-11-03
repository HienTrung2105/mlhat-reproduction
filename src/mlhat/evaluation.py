"""Prequential evaluation for online multi-label classifiers."""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .interfaces import OnlineMultiLabelClassifier, validate_sample


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregate metrics computed over an online stream."""

    n_samples: int
    example_f1: float
    hamming_loss: float
    subset_accuracy: float
    micro_f1: float


def prequential_evaluate(
    model: OnlineMultiLabelClassifier,
    samples: Iterable[tuple[NDArray[np.float64], NDArray[np.int64]]],
    *,
    n_labels: int,
) -> EvaluationResult:
    """Evaluate by predicting and scoring each sample before learning it.

    Args:
        model: Object implementing predict_one and learn_one.
        samples: One-pass stream of (feature vector, binary-label vector).
        n_labels: Expected number of labels per sample.

    Returns:
        Stream-average example F1, Hamming loss, subset accuracy,
        and micro F1.
    """
    if n_labels <= 0:
        raise ValueError("n_labels must be positive")

    n_samples = 0
    example_f1_sum = 0.0
    hamming_errors = 0
    exact_matches = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for features, labels in samples:
        raw_x = np.asarray(features)
        raw_y = np.asarray(labels)
        validate_sample(raw_x, raw_y, n_labels)

        x = np.asarray(raw_x, dtype=np.float64)
        y = np.asarray(raw_y, dtype=np.int64)

        # Predict using the model state from before learning this sample.
        raw_prediction = np.asarray(model.predict_one(x))

        if raw_prediction.shape != (n_labels,):
            raise ValueError(
                "predict_one must return a 1-D vector of length n_labels"
            )
        if not np.issubdtype(raw_prediction.dtype, np.number):
            raise ValueError("predictions must be numeric")
        if not np.isfinite(raw_prediction).all():
            raise ValueError("predictions must be finite")
        if not np.isin(raw_prediction, [0, 1]).all():
            raise ValueError("predictions must contain only 0 or 1")

        prediction = raw_prediction.astype(np.int64, copy=False)

        tp = int(np.count_nonzero((y == 1) & (prediction == 1)))
        fp = int(np.count_nonzero((y == 0) & (prediction == 1)))
        fn = int(np.count_nonzero((y == 1) & (prediction == 0)))

        # Example F1 is calculated separately for this sample.
        example_denominator = 2 * tp + fp + fn
        if example_denominator == 0:
            example_f1_sum += 0.0
        else:
            example_f1_sum += (2 * tp) / example_denominator

        hamming_errors += int(np.count_nonzero(y != prediction))
        exact_matches += int(np.array_equal(y, prediction))

        total_tp += tp
        total_fp += fp
        total_fn += fn
        n_samples += 1

        # Learn only after recording this sample's prediction and metrics.
        model.learn_one(x, y)

    if n_samples == 0:
        raise ValueError("no samples received")

    micro_denominator = 2 * total_tp + total_fp + total_fn
    if micro_denominator == 0:
        micro_f1 = 0.0
    else:
        micro_f1 = (2 * total_tp) / micro_denominator

    return EvaluationResult(
        n_samples=n_samples,
        example_f1=example_f1_sum / n_samples,
        hamming_loss=hamming_errors / (n_samples * n_labels),
        subset_accuracy=exact_matches / n_samples,
        micro_f1=micro_f1,
    )