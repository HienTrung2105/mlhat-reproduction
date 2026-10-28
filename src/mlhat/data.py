"""B — Dataset ingestion helpers (paper dataset/schema decision still pending)."""

from collections.abc import Iterable, Mapping
from typing import Any


def load_huggingface_rows(
    dataset_id: str,
    *,
    split: str = "train",
    config_name: str | None = None,
    streaming: bool = True,
) -> Iterable[Mapping[str, Any]]:
    """Fetch rows programmatically from a Hugging Face dataset.

    Args:
        dataset_id: Verified, public Hugging Face dataset identifier.
        split: Published split name to load.
        config_name: Optional dataset configuration.
        streaming: Whether to use iterative HF streaming.

    Returns:
        Hugging Face row iterator/dataset. Caller must convert the exact
        paper-specific feature and multi-label schema explicitly.

    Note:
        No dataset ID is assumed. Kaggle support can be added after the
        group approves the experiment protocol.
    """
    if not dataset_id.strip():
        raise ValueError("Specify verified Hugging Face dataset_id")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError('Install data dependency: pip install -e ".[data]"') from exc
    return load_dataset(dataset_id, name=config_name, split=split, streaming=streaming)
