"""Programmatic loaders for the paper's datasets."""

from collections.abc import Iterable, Mapping
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


HF_YEAST_REPO_ID = "imodels/multitask-tabular-datasets"


def load_huggingface_rows(
    dataset_id: str,
    *,
    split: str = "train",
    config_name: str | None = None,
    streaming: bool = True,
) -> Iterable[Mapping[str, Any]]:
    """Fetch rows from a Hugging Face dataset with a standard dataset loader.

    Use load_yeast() for the Yeast CSV, whose file and schema are handled
    explicitly.
    """
    if not dataset_id.strip():
        raise ValueError("Specify verified Hugging Face dataset_id")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            'Install the data dependencies with: python -m pip install -e ".[data]"'
        ) from exc

    return load_dataset(
        dataset_id,
        name=config_name,
        split=split,
        streaming=streaming,
    )


def load_yeast(
    revision: str,
    *,
    cache_dir: str | Path = Path("data/raw/hf-cache"),
) -> tuple[
    NDArray[np.float64],
    NDArray[np.int64],
    tuple[str, ...],
    tuple[str, ...],
    str,
]:
    """Download one pinned Yeast CSV and validate its feature/label arrays.

    Args:
        revision: Full 40-character Hugging Face commit SHA.
        cache_dir: Local cache directory. Keep it under data/raw/ so Git ignores it.

    Returns:
        A tuple containing:
        - X: features with shape (2417, 103)
        - Y: binary labels with shape (2417, 14)
        - feature names in CSV order
        - label names in CSV order
        - SHA-256 digest of the downloaded CSV
    """
    if len(revision) != 40 or any(
        char not in "0123456789abcdef" for char in revision.lower()
    ):
        raise ValueError("revision must be the full 40-character Hugging Face commit SHA")

    try:
        from datasets import load_dataset
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            'Install the data dependencies with: python -m pip install -e ".[data]"'
        ) from exc

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # hf_hub_download accepts repo_type="dataset".
    csv_path = Path(
        hf_hub_download(
            repo_id=HF_YEAST_REPO_ID,
            filename="yeast.csv",
            repo_type="dataset",
            revision=revision,
            cache_dir=str(cache_path),
        )
    )

    hasher = hashlib.sha256()
    with csv_path.open("rb") as csv_file:
        for chunk in iter(lambda: csv_file.read(1024 * 1024), b""):
            hasher.update(chunk)
    csv_sha256 = hasher.hexdigest()

    # Parse the exact downloaded file. The source row order is preserved.
    dataset = load_dataset(
        "csv",
        data_files={"train": str(csv_path)},
        split="train",
        cache_dir=str(cache_path),
    )

    columns = dataset.column_names
    label_names = tuple(name for name in columns if name.endswith("__target"))
    feature_names = tuple(name for name in columns if not name.endswith("__target"))

    if not feature_names:
        raise ValueError("No feature columns found")
    if not label_names:
        raise ValueError("No label columns ending in '__target' found")

    rows = dataset.to_dict()

    try:
        X = np.column_stack(
            [np.asarray(rows[name], dtype=np.float64) for name in feature_names]
        )
        Y_float = np.column_stack(
            [np.asarray(rows[name], dtype=np.float64) for name in label_names]
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Could not convert Yeast CSV columns to numeric arrays") from exc

    if not np.isfinite(X).all():
        raise ValueError("Features contain NaN or infinity")
    if not np.isfinite(Y_float).all():
        raise ValueError("Labels contain NaN or infinity")
    if not np.isin(Y_float, [0.0, 1.0]).all():
        raise ValueError("Labels must contain only 0 and 1")

    Y = Y_float.astype(np.int64)

    expected_x_shape = (2417, 103)
    expected_y_shape = (2417, 14)

    if X.shape != expected_x_shape:
        raise ValueError(f"Expected X shape {expected_x_shape}; got {X.shape}")
    if Y.shape != expected_y_shape:
        raise ValueError(f"Expected Y shape {expected_y_shape}; got {Y.shape}")

    return X, Y, feature_names, label_names, csv_sha256