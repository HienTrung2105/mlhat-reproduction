"""C — Concept-drift monitoring and adaptive branches (placeholder only)."""


class AdaptiveDriftManager:
    """Placeholder for the paper's detection/replacement policy.

    Do not replace paper-specific logic with generic ADWIN assumptions
    without checking the paper's exact method and update order.
    """

    def update(self, *, error: float) -> None:
        """Accept a prediction error signal once drift policy is implemented."""
        raise NotImplementedError("Implement paper-specific drift detector and branch management")
