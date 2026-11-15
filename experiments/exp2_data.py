"""Load pinned Hyperplane ARFF streams from a Hugging Face dataset repo."""

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


DATASET_NAMES = ("SynHPSud", "SynHPGrad")
REFERENCE_PATH = Path(__file__).resolve().parents[1] / "data/manifests/hyperplane-reference.json"
_ATTRIBUTE = re.compile(r"^@attribute\s+(?:'([^']+)'|\"([^\"]+)\"|(\S+))\s+(.+)$", re.I)
_CLASS_COUNT = re.compile(r"(?:^|\s)-C\s+(-?\d+)(?=\s|['\"]|$)", re.I)


def _pinned_hash(value: Any, length: int, name: str) -> str:
    """Validate a full hexadecimal revision or checksum before downloading."""
    result = str(value).lower()
    if len(result) != length or any(char not in "0123456789abcdef" for char in result):
        raise ValueError(f"{name} must be a full {length}-character hex value")
    return result


def load_hyperplane(
    name: str,
    source: dict[str, Any],
    *,
    cache_dir: str | Path = Path("data/raw/hf-cache"),
) -> tuple[NDArray[np.float64], NDArray[np.int64], str]:
    """Download and validate one exact 50,000-row paper stream.

    Args:
        name: ``SynHPSud`` or ``SynHPGrad``.
        source: Pinned HF repo, revision, ARFF filename, and SHA-256.
        cache_dir: Ignored local directory for Hugging Face downloads.

    Returns:
        Features (50,000 by 30), labels (50,000 by 8), and file SHA-256.
    """
    if name not in DATASET_NAMES:
        raise ValueError(f"Unsupported EXP-02 dataset: {name}")
    required = {"repo_id", "file", "revision", "sha256", "label_location"}
    missing = required - source.keys()
    if missing:
        raise ValueError(
            f"{name} has no complete pinned Hugging Face source; missing {sorted(missing)}"
        )
    repo_id = str(source["repo_id"])
    filename = str(source["file"])
    if not repo_id.strip() or not filename.endswith(".arff"):
        raise ValueError("Specify a Hugging Face repo and an ARFF filename")
    revision = _pinned_hash(source["revision"], 40, "revision")
    expected_hash = _pinned_hash(source["sha256"], 64, "sha256")
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    author_hash = reference["datasets"][name]["author_sha256"]
    if expected_hash != author_hash:
        raise ValueError("HF file checksum must match the author's original ARFF")
    if source.get("label_location") != "start":
        raise ValueError("Author's Hyperplane ARFF files have labels at the start")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            'Install data dependencies: python -m pip install -e ".[data]"'
        ) from exc

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            revision=revision,
            cache_dir=str(cache_path),
        )
    )
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    actual_hash = digest.hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(f"{name} checksum mismatch: {actual_hash}")

    features: list[list[float]] = []
    labels: list[list[int]] = []
    attributes: list[tuple[str, str]] = []
    class_count: int | None = None
    in_data = False
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line or line.startswith("%"):
                continue
            if not in_data:
                lower = line.lower()
                if lower.startswith("@relation"):
                    match = _CLASS_COUNT.search(line)
                    if match:
                        class_count = int(match.group(1))
                elif lower.startswith("@attribute"):
                    match = _ATTRIBUTE.match(line)
                    if not match:
                        raise ValueError(f"Invalid ARFF attribute at line {line_number}")
                    attributes.append((next(part for part in match.groups()[:3] if part), match.group(4).strip().lower()))
                elif lower == "@data":
                    in_data = True
                    if len(attributes) != 38:
                        raise ValueError(f"Expected 38 ARFF attributes; got {len(attributes)}")
                    if class_count != 8:
                        raise ValueError(f"Expected ARFF relation -C 8; got {class_count}")
                    if any(kind.replace(" ", "") not in {"{0,1}", "{1,0}"} for _, kind in attributes[:8]):
                        raise ValueError("The first 8 ARFF attributes must be binary labels")
                    if any(kind not in {"numeric", "real", "integer"} for _, kind in attributes[8:]):
                        raise ValueError("The last 30 ARFF attributes must be numeric features")
                continue

            row = next(csv.reader([line], skipinitialspace=True))
            # The author's MOA export ends each dense data row with one comma.
            if len(row) == 39 and row[-1] == "":
                row.pop()
            if len(row) != 38:
                raise ValueError(f"Expected 38 values at ARFF line {line_number}; got {len(row)}")
            try:
                y_float = [float(value) for value in row[:8]]
                x = [float(value) for value in row[8:]]
            except ValueError as exc:
                raise ValueError(f"Nonnumeric value at ARFF line {line_number}") from exc
            if not np.isfinite(x).all() or not np.isfinite(y_float).all():
                raise ValueError(f"Nonfinite value at ARFF line {line_number}")
            if any(value not in (0.0, 1.0) for value in y_float):
                raise ValueError(f"Nonbinary label at ARFF line {line_number}")
            features.append(x)
            labels.append([int(value) for value in y_float])

    if not in_data or len(features) != 50_000:
        raise ValueError(f"Expected 50,000 ARFF data rows; got {len(features)}")
    return np.asarray(features, dtype=np.float64), np.asarray(labels, dtype=np.int64), actual_hash
