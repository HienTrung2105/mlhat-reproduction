"""Concept-drift decisions for an MLHAT node.

ADWIN supplies a change signal and window error estimates. The branch policy
and the comparison bound follow Algorithm 1 and Eq. (16) of Esteban et al.
The River detector is infrastructure; no MLHAT implementation is imported.
"""

from collections.abc import Callable
from enum import Enum, auto
from math import isfinite, log, sqrt
from typing import Protocol

from river import drift


class ErrorDetector(Protocol):
    """Small interface required of an online error detector."""

    @property
    def estimation(self) -> float: ...

    @property
    def drift_detected(self) -> bool: ...

    @property
    def width(self) -> int: ...

    def update(self, value: float) -> object: ...


class DriftAction(Enum):
    """Structural change requested by one adaptive node."""

    NONE = auto()
    START_BACKGROUND = auto()
    PROMOTE_BACKGROUND = auto()
    DISCARD_BACKGROUND = auto()


def alternate_bound(
    main_error: float,
    background_error: float,
    main_weight: int,
    background_weight: int,
    delta_alt: float,
) -> float:
    """Return the alternate-tree comparison threshold from paper Eq. (16).

    Args:
        main_error: ADWIN estimate of the active branch Hamming loss.
        background_error: ADWIN estimate of the alternate branch loss.
        main_weight: Weighted instances seen by the active subtree root.
        background_weight: Weighted instances seen by the alternate root.
        delta_alt: Confidence parameter for alternate replacement.

    Returns:
        Nonnegative Hoeffding comparison threshold.
    """
    if not 0.0 < delta_alt < 1.0:
        raise ValueError("delta_alt must be between 0 and 1")
    if main_weight <= 0 or background_weight <= 0:
        raise ValueError("both subtree weights must be positive")
    if not (isfinite(main_error) and 0.0 <= main_error <= 1.0):
        raise ValueError("main_error must be in [0, 1]")
    if not (isfinite(background_error) and 0.0 <= background_error <= 1.0):
        raise ValueError("background_error must be in [0, 1]")

    variance_term = 2.0 * main_error * (1.0 - background_error)
    sample_term = (main_weight + background_weight) / (
        main_weight * background_weight
    )
    return sqrt(variance_term * sample_term * log(2.0 / delta_alt))


class AdaptiveDriftManager:
    """Monitor one active node and decide the lifecycle of its alternate.

    River ADWIN has one change flag and no separate warning flag. A change in
    the loss distribution toward *higher* error is treated as the paper's
    warning and starts an alternate. Each existing alternate receives its own
    pre-update error sequence. Counts used in Eq. (16) come from tree nodes,
    because the paper defines W using Poisson-weighted observations.
    """

    def __init__(
        self,
        *,
        delta_alt: float,
        kappa_alt: int,
        adwin_delta: float = 0.002,
        detector_factory: Callable[[], ErrorDetector] | None = None,
    ) -> None:
        if not 0.0 < delta_alt < 1.0:
            raise ValueError("delta_alt must be between 0 and 1")
        if kappa_alt < 0:
            raise ValueError("kappa_alt must be nonnegative")
        if not 0.0 < adwin_delta < 1.0:
            raise ValueError("adwin_delta must be between 0 and 1")

        self.delta_alt = delta_alt
        self.kappa_alt = kappa_alt
        self.adwin_delta = adwin_delta
        self._detector_factory = detector_factory or (
            lambda: drift.ADWIN(delta=adwin_delta)
        )
        self.main_detector = self._detector_factory()
        self.background_detector: ErrorDetector | None = None

    @property
    def main_error(self) -> float:
        """Current ADWIN estimate for active-branch Hamming loss."""
        return float(self.main_detector.estimation)

    @property
    def background_error(self) -> float | None:
        """Current alternate error estimate, when an alternate exists."""
        if self.background_detector is None:
            return None
        return float(self.background_detector.estimation)

    def reset_main(self) -> None:
        """Reset the promoted subtree's detector for future concept changes."""
        self.main_detector = self._detector_factory()

    def reset_background(self) -> None:
        """Start a fresh error window after a nested alternate is replaced."""
        if self.background_detector is None:
            raise ValueError("no alternate detector to reset")
        self.background_detector = self._detector_factory()

    def observe(
        self,
        main_error: float,
        background_error: float | None,
        *,
        main_weight: int,
        background_weight: int = 0,
    ) -> DriftAction:
        """Consume pre-update losses and select one structural action.

        Args:
            main_error: Hamming loss of the active subtree's prior prediction.
            background_error: Alternate prediction loss, or None if absent.
            main_weight: Active subtree root's current Poisson-weighted count.
            background_weight: Alternate root's current weighted count.

        Returns:
            An action for the tree to apply before training this sample.
        """
        for name, error in (("main_error", main_error), ("background_error", background_error)):
            if error is not None and (not isfinite(error) or not 0.0 <= error <= 1.0):
                raise ValueError(f"{name} must be in [0, 1]")
        if main_weight < 0 or background_weight < 0:
            raise ValueError("subtree weights must be nonnegative")
        if background_error is None and self.background_detector is not None:
            raise ValueError("background error is required while an alternate exists")
        if background_error is not None and self.background_detector is None:
            raise ValueError("an alternate must be started before observing its error")

        old_estimate = self.main_error if self.main_detector.width else None
        self.main_detector.update(main_error)

        if background_error is None:
            if (
                self.main_detector.drift_detected
                and old_estimate is not None
                and self.main_error > old_estimate
            ):
                self.background_detector = self._detector_factory()
                return DriftAction.START_BACKGROUND
            return DriftAction.NONE

        assert self.background_detector is not None
        self.background_detector.update(background_error)

        if main_weight <= 0 or background_weight <= self.kappa_alt:
            return DriftAction.NONE

        active = self.main_error
        alternate = self.background_error
        assert alternate is not None
        epsilon = alternate_bound(
            active, alternate, main_weight, background_weight, self.delta_alt
        )
        if active - alternate > epsilon:
            return DriftAction.PROMOTE_BACKGROUND
        if alternate - active > epsilon:
            self.background_detector = None
            return DriftAction.DISCARD_BACKGROUND
        return DriftAction.NONE
