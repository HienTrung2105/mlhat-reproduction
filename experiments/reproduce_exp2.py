"""Run EXP-02 on verified Hugging Face mirrors of the author's ARFF streams."""

import argparse
from collections import deque
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
import subprocess
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray
import river
import scipy
import sklearn

from experiments.baselines import BinaryRelevanceHAT
from experiments.exp2_data import DATASET_NAMES, load_hyperplane
from mlhat.interfaces import OnlineMultiLabelClassifier, validate_sample
from mlhat.tree import MLHATTree


MODEL_NAMES = ("MLHAT", "BR+HAT", "MLHAT-no-drift")


@dataclass(frozen=True)
class StreamResult:
    """Whole-stream metrics plus time-series points for one model and seed."""

    n_samples: int
    example_f1: float
    hamming_loss: float
    subset_accuracy: float
    rolling_f1: list[dict[str, float | int]]
    faded_f1: list[dict[str, float | int]]
    drift_events: list[dict[str, Any]]


def evaluate_stream(
    model: OnlineMultiLabelClassifier,
    X: NDArray[np.float64],
    Y: NDArray[np.int64],
    *,
    rolling_window: int,
    rolling_step: int,
    fading_factor: float,
    fading_step: int,
) -> StreamResult:
    """Score each ordered sample before training and retain drift-time F1 curves.

    Args:
        model: Online multi-label classifier being evaluated.
        X: Ordered numeric features.
        Y: Ordered binary targets.
        rolling_window: Number of most recent examples in the group curve.
        rolling_step: Samples between rolling-curve points.
        fading_factor: Per-sample exponential discount for the paper-style curve.
        fading_step: Samples between faded-curve points.

    Returns:
        Whole-stream means, two F1 curves, and structural drift events.
    """
    if X.ndim != 2 or Y.ndim != 2 or len(X) != len(Y) or not len(X):
        raise ValueError("X and Y must be nonempty aligned two-dimensional arrays")
    if rolling_window <= 0 or rolling_step <= 0 or fading_step <= 0:
        raise ValueError("Window and reporting steps must be positive")
    if not 0.0 < fading_factor < 1.0:
        raise ValueError("fading_factor must be in (0, 1)")

    n_labels = Y.shape[1]
    recent: deque[float] = deque(maxlen=rolling_window)
    rolling_f1: list[dict[str, float | int]] = []
    faded_f1: list[dict[str, float | int]] = []
    f1_sum = 0.0
    hamming_sum = 0.0
    subset_sum = 0
    faded_sum = 0.0
    faded_weight = 0.0

    for index, (x, y) in enumerate(zip(X, Y, strict=True), start=1):
        validate_sample(x, y, n_labels)
        prediction = np.asarray(model.predict_one(x))
        if prediction.shape != (n_labels,) or not np.isin(prediction, (0, 1)).all():
            raise ValueError("model returned an invalid binary label vector")
        tp = int(np.count_nonzero((y == 1) & (prediction == 1)))
        fp = int(np.count_nonzero((y == 0) & (prediction == 1)))
        fn = int(np.count_nonzero((y == 1) & (prediction == 0)))
        denominator = 2 * tp + fp + fn
        f1 = 0.0 if denominator == 0 else (2 * tp) / denominator
        f1_sum += f1
        hamming_sum += float(np.mean(y != prediction))
        subset_sum += int(np.array_equal(y, prediction))
        recent.append(f1)
        faded_sum = fading_factor * faded_sum + f1
        faded_weight = fading_factor * faded_weight + 1.0

        if index >= rolling_window and index % rolling_step == 0:
            rolling_f1.append({"sample_index": index, "example_f1": float(np.mean(recent))})
        if index % fading_step == 0:
            faded_f1.append({"sample_index": index, "example_f1": faded_sum / faded_weight})

        model.learn_one(x, y)
        if index % 5000 == 0:
            print(f"  processed {index}/{len(X)} samples; F1={f1_sum / index:.4f}", flush=True)

    events = [
        {**asdict(event), "action": event.action.name}
        for event in getattr(model, "drift_events", ())
    ]
    return StreamResult(
        n_samples=len(X),
        example_f1=f1_sum / len(X),
        hamming_loss=hamming_sum / len(X),
        subset_accuracy=subset_sum / len(X),
        rolling_f1=rolling_f1,
        faded_f1=faded_f1,
        drift_events=events,
    )


def _model(name: str, config: dict[str, Any], seed: int) -> OnlineMultiLabelClassifier:
    """Construct one EXP-02 model with the shared seed and agreed config."""
    if name == "BR+HAT":
        return BinaryRelevanceHAT(n_labels=8, seed=seed, config=config["baseline"])
    if name in ("MLHAT", "MLHAT-no-drift"):
        tree_config = dict(config["mlhat"])
        tree_config["drift_enabled"] = name == "MLHAT"
        return MLHATTree(n_labels=8, n_features=30, config=tree_config, seed=seed)
    raise ValueError(f"Unknown EXP-02 model: {name}")


def _commit() -> str:
    """Return the exact Git commit, or fail rather than log an untraceable run."""
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
    ).strip()


def _git_dirty() -> bool:
    """Report whether tracked or untracked source changes lack a commit."""
    return bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())


def _wandb_series(result: StreamResult) -> list[tuple[int, dict[str, float | int]]]:
    """Merge F1 curves and cumulative drift counts into ordered W&B steps."""
    points: dict[int, dict[str, float | int]] = {}
    for point in result.rolling_f1:
        points[int(point["sample_index"])] = {
            "rolling_example_f1": float(point["example_f1"])
        }
    for point in result.faded_f1:
        points.setdefault(int(point["sample_index"]), {})["faded_example_f1"] = float(
            point["example_f1"]
        )
    counts = {"START_BACKGROUND": 0, "PROMOTE_BACKGROUND": 0, "DISCARD_BACKGROUND": 0}
    for event in result.drift_events:
        action = str(event["action"])
        if action in counts:
            counts[action] += 1
            points.setdefault(int(event["sample_index"]), {}).update(
                {f"drift_{key.lower()}": value for key, value in counts.items()}
            )
    return sorted(points.items())


def main() -> None:
    """Validate source manifests, run requested comparisons, and save evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="B-verified HF source manifest")
    parser.add_argument("--config", default="configs/exp2_hyperplane.json")
    parser.add_argument("--dataset", choices=DATASET_NAMES)
    parser.add_argument("--model", choices=MODEL_NAMES)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="mlhat-team10-exp2")
    parser.add_argument("--resume", action="store_true", help="Skip complete results matching commit, source and config")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if args.seed is not None and args.seed not in config["seeds"]:
        parser.error(f"seed must be one of {config['seeds']}")
    dirty = _git_dirty()
    if args.wandb:
        if dirty:
            parser.error("commit source changes before logging a reportable W&B run")
        try:
            import wandb  # noqa: F401
        except ImportError as exc:
            parser.error('install W&B first: python -m pip install -e ".[tracking]"')
    datasets = (args.dataset,) if args.dataset else DATASET_NAMES
    models = (args.model,) if args.model else MODEL_NAMES
    seeds = (args.seed,) if args.seed is not None else tuple(config["seeds"])
    commit = _commit()

    for dataset_name in datasets:
        if dataset_name not in manifest["datasets"]:
            raise ValueError(f"Manifest has no verified entry for {dataset_name}")
        source = manifest["datasets"][dataset_name]
        X, Y, digest = load_hyperplane(dataset_name, source)
        for seed in seeds:
            for model_name in models:
                output = Path("artifacts/exp2_hyperplane") / f"{dataset_name}_{model_name}_seed{seed}.json"
                if args.resume and output.exists():
                    previous = json.loads(output.read_text(encoding="utf-8"))
                    if (previous.get("git_commit") == commit
                        and previous.get("config") == config
                        and previous.get("dataset_sha256") == digest
                        and previous.get("result", {}).get("n_samples") == len(X)
                        and (not args.wandb or previous.get("wandb_url"))):
                        print(f"Already complete: {output}", flush=True)
                        continue
                print(f"Starting {dataset_name} {model_name} seed={seed}", flush=True)
                model = _model(model_name, config, seed)
                start = time.perf_counter()
                result = evaluate_stream(
                    model,
                    X,
                    Y,
                    rolling_window=int(config["rolling_f1_window"]),
                    rolling_step=int(config["rolling_f1_step"]),
                    fading_factor=float(config["fading_factor"]),
                    fading_step=int(config["fading_step"]),
                )
                duration = time.perf_counter() - start
                record: dict[str, Any] = {
                    "experiment": "EXP-02",
                    "dataset": dataset_name,
                    "model": model_name,
                    "seed": seed,
                    "git_commit": commit,
                    "git_dirty": dirty,
                    "dataset_source": source,
                    "dataset_sha256": digest,
                    "config": config,
                    "environment": {
                        "python": platform.python_version(),
                        "platform": platform.platform(),
                        "numpy": np.__version__,
                        "scipy": scipy.__version__,
                        "scikit_learn": sklearn.__version__,
                        "river": river.__version__,
                        "machine": platform.machine(),
                        "processor": platform.processor(),
                    },
                    "duration_seconds_excluding_download": duration,
                    "result": asdict(result),
                }
                output = Path("artifacts/exp2_hyperplane") / f"{dataset_name}_{model_name}_seed{seed}.json"
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(record, indent=2), encoding="utf-8")
                print(f"{dataset_name} {model_name} seed={seed}: F1={result.example_f1:.4f}; {output}")
                if args.wandb:
                    import wandb

                    with wandb.init(
                        project=args.wandb_project,
                        name=f"EXP-02-{dataset_name}-{model_name}-seed-{seed}",
                        job_type="reproduction",
                        config={key: value for key, value in record.items() if key != "result"},
                    ) as run:
                        for step, values in _wandb_series(result):
                            run.log(values, step=step)
                        run.summary.update({
                            "example_f1": result.example_f1,
                            "hamming_loss": result.hamming_loss,
                            "subset_accuracy": result.subset_accuracy,
                            "duration_seconds": duration,
                            "drift_event_count": len(result.drift_events),
                        })
                        record["wandb_url"] = run.url
                    output.write_text(json.dumps(record, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
