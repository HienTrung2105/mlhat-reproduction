"""Online statistics and numeric split selection for MLHAT."""

from dataclasses import dataclass, field
from math import erf, log, log2, sqrt

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SplitCandidate:
    feature_index: int
    threshold: float
    gain: float


@dataclass
class NodeStatistics:
    n_features: int
    n_labels: int
    n_seen: int = 0
    positive_counts: NDArray[np.float64] = field(init=False)
    labelset_counts: dict[tuple[int, ...], int] = field(default_factory=dict)
    feature_min: NDArray[np.float64] = field(init=False)
    feature_max: NDArray[np.float64] = field(init=False)
    conditional_counts: NDArray[np.float64] = field(init=False)
    conditional_means: NDArray[np.float64] = field(init=False)
    conditional_m2: NDArray[np.float64] = field(init=False)

    def __post_init__(self) -> None:
        self.positive_counts = np.zeros(self.n_labels, dtype=np.float64)
        self.feature_min = np.full(self.n_features, np.inf, dtype=np.float64)
        self.feature_max = np.full(self.n_features, -np.inf, dtype=np.float64)

        # Last axis: 0 = label absent, 1 = label present.
        shape = (self.n_features, self.n_labels, 2)
        self.conditional_counts = np.zeros(shape, dtype=np.float64)
        self.conditional_means = np.zeros(shape, dtype=np.float64)
        self.conditional_m2 = np.zeros(shape, dtype=np.float64)

    @property
    def is_pure(self) -> bool:
        return self.n_seen > 0 and len(self.labelset_counts) == 1

    @property
    def majority_labelset(self) -> NDArray[np.int64]:
        if not self.labelset_counts:
            return np.zeros(self.n_labels, dtype=np.int64)

        winner = sorted(
            self.labelset_counts,
            key=lambda labelset: (-self.labelset_counts[labelset], labelset),
        )[0]
        return np.asarray(winner, dtype=np.int64)

    def entropy(self) -> float:
        if self.n_seen == 0:
            return 0.0

        total = 0.0
        for positive_count in self.positive_counts:
            p = float(positive_count / self.n_seen)
            total += _binary_entropy(p)
        return total

    def update(
        self,
        features: NDArray[np.float64],
        labels: NDArray[np.int64],
        *,
        weight: int = 1,
    ) -> None:
        if weight < 0:
            raise ValueError("weight must be nonnegative")
        if weight == 0:
            return

        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels, dtype=np.int64)

        if x.shape != (self.n_features,):
            raise ValueError("feature vector has the wrong length")
        if y.shape != (self.n_labels,):
            raise ValueError("label vector has the wrong length")

        self.n_seen += weight
        self.positive_counts += weight * y
        key = tuple(int(value) for value in y)
        self.labelset_counts[key] = self.labelset_counts.get(key, 0) + weight

        self.feature_min = np.minimum(self.feature_min, x)
        self.feature_max = np.maximum(self.feature_max, x)

        for label_index, label_value in enumerate(y):
            outcome = int(label_value)
            old_count = self.conditional_counts[:, label_index, outcome]
            new_count = old_count + weight
            delta = x - self.conditional_means[:, label_index, outcome]
            new_mean = (
                self.conditional_means[:, label_index, outcome]
                + delta * (weight / new_count)
            )
            self.conditional_m2[:, label_index, outcome] += (
                weight * delta * (x - new_mean)
            )
            self.conditional_means[:, label_index, outcome] = new_mean
            self.conditional_counts[:, label_index, outcome] = new_count


def _binary_entropy(probability: float) -> float:
    p = min(1.0, max(0.0, probability))
    if p == 0.0 or p == 1.0:
        return 0.0
    return -p * log2(p) - (1.0 - p) * log2(1.0 - p)


def _normal_cdf(
    value: float,
    *,
    count: float,
    mean: float,
    m2: float,
) -> float:
    if count <= 0:
        return 0.5

    variance = m2 / max(count - 1.0, 1.0)
    sigma = sqrt(max(variance, 0.0))
    if sigma < 1e-12:
        return 1.0 if value >= mean else 0.0

    z = (value - mean) / (sigma * sqrt(2.0))
    return 0.5 * (1.0 + erf(z))


def _conditional_entropy_after_split(
    stats: NodeStatistics,
    feature_index: int,
    threshold: float,
) -> float:
    result = 0.0

    for label_index in range(stats.n_labels):
        p_positive = float(
            stats.positive_counts[label_index] / stats.n_seen
        )
        p_negative = 1.0 - p_positive

        positive_cdf = _normal_cdf(
            threshold,
            count=stats.conditional_counts[feature_index, label_index, 1],
            mean=stats.conditional_means[feature_index, label_index, 1],
            m2=stats.conditional_m2[feature_index, label_index, 1],
        )
        negative_cdf = _normal_cdf(
            threshold,
            count=stats.conditional_counts[feature_index, label_index, 0],
            mean=stats.conditional_means[feature_index, label_index, 0],
            m2=stats.conditional_m2[feature_index, label_index, 0],
        )

        p_left = p_positive * positive_cdf + p_negative * negative_cdf
        p_right = 1.0 - p_left

        if p_left > 1e-12:
            p_positive_left = (
                p_positive * positive_cdf / p_left
            )
            result += p_left * _binary_entropy(p_positive_left)

        if p_right > 1e-12:
            p_positive_right = (
                p_positive * (1.0 - positive_cdf) / p_right
            )
            result += p_right * _binary_entropy(p_positive_right)

    return result


def choose_split(
    stats: NodeStatistics,
    *,
    delta_spl: float,
    n_bins: int = 10,
) -> SplitCandidate | None:
    if not 0.0 < delta_spl < 1.0:
        raise ValueError("delta_spl must be between 0 and 1")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    if stats.n_seen < 2 or stats.is_pure:
        return None

    current_entropy = stats.entropy()
    candidates: list[SplitCandidate] = []

    for feature_index in range(stats.n_features):
        low = float(stats.feature_min[feature_index])
        high = float(stats.feature_max[feature_index])
        if not np.isfinite(low) or not np.isfinite(high) or low >= high:
            continue

        thresholds = np.linspace(low, high, n_bins + 2)[1:-1]
        for threshold in thresholds:
            after = _conditional_entropy_after_split(
                stats, feature_index, float(threshold)
            )
            gain = current_entropy - after
            candidates.append(
                SplitCandidate(feature_index, float(threshold), float(gain))
            )

    if not candidates:
        return None

    best_by_feature: dict[int, SplitCandidate] = {}

    for candidate in candidates:
        previous = best_by_feature.get(candidate.feature_index)
        if previous is None or candidate.gain > previous.gain:
            best_by_feature[candidate.feature_index] = candidate

    ranked = sorted(
        best_by_feature.values(),
        key=lambda candidate: candidate.gain,
        reverse=True,
    )

    # Chưa có hai feature để so sánh thì chưa tách.
    if len(ranked) < 2:
        return None

    best, second = ranked[:2]

    n_labelsets = max(2, len(stats.labelset_counts))
    epsilon = sqrt(
        ((log2(n_labelsets) ** 2) * log(1.0 / delta_spl))
        / (2.0 * stats.n_seen)
    )

    if best.gain - second.gain > epsilon:
        return best
    return None
