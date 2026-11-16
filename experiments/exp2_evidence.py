"""Validate saved EXP-02 evidence without loading data or constructing models."""

import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
from typing import Any
import zipfile

TEAM = "ml-group10-24khmt1-2026"
PROJECT = "mlhat-team10-exp2"
SNAPSHOT_SHA256 = "d229a956f7b7a25c07d0bac05f62121db27f911e9fc69e4ac4278562d438965b"
METRICS = ("example_f1", "hamming_loss", "subset_accuracy")
CURVES = ("rolling_f1", "faded_f1")
ACTIONS = ("START_BACKGROUND", "PROMOTE_BACKGROUND", "DISCARD_BACKGROUND")


def digest_file(path: Path) -> str:
    """Return the SHA-256 of a file's bytes without changing the file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _probability(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and in [0, 1]")


def validate_record(record: dict[str, Any], config: dict, manifest: dict) -> None:
    """Check one recorded run's identity, source, metrics, curve grids and events.

    Args:
        record: A parsed, completed local measurement; never modified here.
        config: Agreed EXP-02 configuration from the measured source snapshot.
        manifest: Verified HF manifest from the same snapshot.

    Returns:
        None. Raises ValueError if any consistency check fails.
    """
    dataset, model, seed = (record[k] for k in ("dataset", "model", "seed"))
    if (record.get("experiment") != "EXP-02" or dataset not in config["datasets"]
        or model not in config["models"] or type(seed) is not int or seed not in config["seeds"]):
        raise ValueError("Unexpected EXP-02 dataset/model/seed identity")
    if record["config"] != config:
        raise ValueError("Run config differs from measured snapshot")
    source = manifest["datasets"][dataset]
    if record["dataset_source"] != source or record["dataset_sha256"] != source["sha256"]:
        raise ValueError("Run dataset source differs from pinned HF manifest")
    if not record.get("git_commit") or type(record.get("git_dirty")) is not bool:
        raise ValueError("Missing measured Git provenance")
    duration = record["duration_seconds_excluding_download"]
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid measured duration")
    result = record["result"]
    if type(result["n_samples"]) is not int or result["n_samples"] != 50000:
        raise ValueError("Each completed run must have 50000 samples")
    for metric in METRICS:
        _probability(result[metric], metric)
    for curve in CURVES:
        start = config["rolling_f1_window"] if curve == "rolling_f1" else config["fading_step"]
        step = config["rolling_f1_step"] if curve == "rolling_f1" else config["fading_step"]
        points = result[curve]
        if [p["sample_index"] for p in points] != list(range(start, 50001, step)):
            raise ValueError(f"{curve} sample grid is incomplete, duplicated or unordered")
        for point in points:
            _probability(point["example_f1"], curve)
    # 50 non-overlapping 1000-example windows cover the entire 50000-example stream.
    windows = [p["example_f1"] for p in result["rolling_f1"] if p["sample_index"] % 1000 == 0]
    if len(windows) != 50 or not math.isclose(statistics.mean(windows), result["example_f1"], abs_tol=1e-10):
        raise ValueError("Whole-stream F1 disagrees with non-overlapping saved windows")
    last_index = 0
    active: set[int] = set()
    for event in result["drift_events"]:
        index, node, depth, action = (event[k] for k in ("sample_index", "node_id", "depth", "action"))
        if (type(index) is not int or not last_index <= index <= 50000 or index < 1
            or type(node) is not int or node < 0 or type(depth) is not int or depth < 0 or action not in ACTIONS):
            raise ValueError("Invalid or unordered drift event")
        if action == "START_BACKGROUND":
            if node in active:
                raise ValueError("Background started twice without closure")
            active.add(node)
        else:
            if node not in active:
                raise ValueError("Background closed without a recorded start")
            active.remove(node)
        last_index = index
    if model != "MLHAT" and result["drift_events"]:
        raise ValueError("Unexpected instrumented events for this model")


def load_evidence(directory: Path) -> tuple[list[tuple[Path, dict]], dict]:
    """Validate the exact 18-run Cartesian product and pinned measurement snapshot.

    Args:
        directory: Directory containing immutable measurement JSON and source zip.

    Returns:
        Sorted (path, record) pairs and a validation audit with source hashes/limits.
    """
    snapshot = directory / "source_snapshot.zip"
    if digest_file(snapshot) != SNAPSHOT_SHA256:
        raise ValueError("Source snapshot checksum differs from the measured batch")
    with zipfile.ZipFile(snapshot) as archive:
        config = json.loads(archive.read("configs/exp2_hyperplane.json"))
        manifest = json.loads(archive.read("data/manifests/hyperplane-hf.json"))
    expected = set(itertools.product(config["datasets"], config["models"], config["seeds"]))
    records = []
    seen = set()
    hashes = {}
    for path in sorted(directory.glob("*_seed*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        validate_record(record, config, manifest)
        identity = (record["dataset"], record["model"], record["seed"])
        if identity in seen:
            raise ValueError(f"Duplicate run identity: {identity}")
        seen.add(identity)
        records.append((path, record))
        hashes[path.name] = digest_file(path)
    if seen != expected:
        raise ValueError(f"Expected exactly 18 runs; missing={sorted(expected-seen)}, unexpected={sorted(seen-expected)}")
    return records, {
        "validated_runs": len(records), "samples_per_run": 50000,
        "expected_rolling_points_per_run": 197, "expected_faded_points_per_run": 1000,
        "source_snapshot_sha256": SNAPSHOT_SHA256, "measurement_sha256": hashes,
        "limitations": [
            "No per-sample predictions were saved: cannot independently reconstruct Hamming/subset metrics or fading recurrence without a new experiment.",
            "Whole-stream F1 is cross-checked against 50 disjoint saved windows; plot arrays are checked directly against stored curve points.",
            "BR+HAT internal structural events were not instrumented; absence is not a measured zero.",
            "Original-code comparison and equivalence of paper metric aggregation remain unverified.",
        ],
    }


def summarize(records: list[dict]) -> list[dict]:
    """Return means and sample SD (ddof=1) of three seeds per dataset/model/metric."""
    rows = []
    for dataset, model in sorted({(r["dataset"], r["model"]) for r in records}):
        group = [r for r in records if (r["dataset"], r["model"]) == (dataset, model)]
        for metric in METRICS:
            values = [r["result"][metric] for r in group]
            rows.append({"dataset": dataset, "model": model, "completed_seeds": ",".join(str(r["seed"]) for r in group),
                         "metric": metric, "mean": statistics.mean(values), "sample_sd": statistics.stdev(values)})
    return rows


def curve_means(records: list[dict], dataset: str, model: str, curve: str) -> list[dict]:
    """Average aligned saved curve points across seeds, for both plots and CSV export."""
    group = [r for r in records if (r["dataset"], r["model"]) == (dataset, model)]
    return [{"sample_index": p["sample_index"], "example_f1": statistics.mean(
        r["result"][curve][i]["example_f1"] for r in group)} for i, p in enumerate(group[0]["result"][curve])]


def run_id(record: dict, measurement_sha256: str) -> str:
    """Return a deterministic W&B ID, ensuring retrying this measurement reuses its run."""
    identity = f"{record['dataset']}/{record['model']}/{record['seed']}/{measurement_sha256}"
    return "exp2" + hashlib.sha256(identity.encode()).hexdigest()[:16]


def series(record: dict) -> list[dict]:
    """Merge recorded F1 and observed structural actions on explicit one-based sample steps."""
    result = record["result"]
    points: dict[int, dict] = {}
    for curve, metric in (("rolling_f1", "rolling_example_f1"), ("faded_f1", "faded_example_f1")):
        for point in result[curve]:
            points.setdefault(point["sample_index"], {})[metric] = point["example_f1"]
    counts = dict.fromkeys(ACTIONS, 0)
    for event in result["drift_events"]:
        counts[event["action"]] += 1
        points.setdefault(event["sample_index"], {}).update({f"drift_{k.lower()}": v for k, v in counts.items()})
    return [{"sample_index": step, **values} for step, values in sorted(points.items())]
