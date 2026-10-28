"""A — Multi-label split criterion and Hoeffding decision, TODO from paper."""


def choose_split(*, sufficient_statistics: object) -> None:
    """Choose a split from node statistics according to the paper.

    Args:
        sufficient_statistics: Team-designed sufficient statistics object.

    Returns:
        Eventually a validated split decision; currently unimplemented.
    """
    raise NotImplementedError("Specify split statistic/criterion from paper, then implement")
