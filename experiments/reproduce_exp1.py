"""Run EXP-01 locally and optionally log runs to Weights & Biases."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from experiments.baselines import BinaryRelevanceHAT
from mlhat.data import load_yeast
from mlhat.evaluation import prequential_evaluate
from mlhat.tree import MLHATTree


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/exp1_yeast.json",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log each model/seed run to Weights & Biases",
    )
    parser.add_argument(
        "--wandb-project",
        default="mlhat-team10-exp1",
    )
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    dataset_config = config["dataset"]

    X, Y, feature_names, label_names, csv_sha256 = load_yeast(
        revision=dataset_config["revision"],
    )

    if csv_sha256 != dataset_config["csv_sha256"]:
        raise RuntimeError("Yeast CSV checksum differs from the experiment config")
    if X.shape != (
        dataset_config["expected_rows"],
        dataset_config["expected_features"],
    ):
        raise RuntimeError(f"Unexpected X shape: {X.shape}")
    if Y.shape != (
        dataset_config["expected_rows"],
        dataset_config["expected_labels"],
    ):
        raise RuntimeError(f"Unexpected Y shape: {Y.shape}")

    # Do not enable the official run until C has integrated the drift manager.
    if not args.wandb:
        print("W&B logging is OFF. Results will be saved locally.")

    results: list[dict[str, object]] = []
    output_dir = Path("artifacts/exp1_yeast")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "results.json"

    for seed in config["seeds"]:
        models = [
            (
                "BR+HAT",
                BinaryRelevanceHAT(
                    n_labels=Y.shape[1],
                    seed=int(seed),
                    config=config["baseline"],
                ),
            ),
            (
                "MLHAT",
                MLHATTree(
                    n_labels=Y.shape[1],
                    n_features=X.shape[1],
                    config=config["mlhat"],
                    seed=int(seed),
                ),
            ),
        ]

        for model_name, model in models:
            if model_name == "MLHAT" and not model.drift_enabled:
                print(
                    "WARNING: this tree has no C-owned drift alternates yet; "
                    "do not present this run as the final MLHAT reproduction."
                )

            start = time.perf_counter()
            evaluation = prequential_evaluate(
                model,
                (
                    (X[index].astype(np.float64), Y[index].astype(np.int64))
                    for index in range(len(X))
                ),
                n_labels=Y.shape[1],
            )
            duration_seconds = time.perf_counter() - start

            record: dict[str, object] = {
                "experiment": config["experiment"],
                "model": model_name,
                "seed": int(seed),
                "protocol": config["protocol"],
                "dataset_repo": dataset_config["repo_id"],
                "dataset_revision": dataset_config["revision"],
                "dataset_sha256": csv_sha256,
                "n_samples": evaluation.n_samples,
                "n_features": len(feature_names),
                "n_labels": len(label_names),
                "metrics": asdict(evaluation),
                "duration_seconds": duration_seconds,
                "config": config,
            }

            if args.wandb:
                import wandb

                run = wandb.init(
                    project=args.wandb_project,
                    name=f"EXP-01-{model_name}-seed-{seed}",
                    job_type="reproduction",
                    config={
                        "experiment": config["experiment"],
                        "model": model_name,
                        "seed": int(seed),
                        "dataset_revision": dataset_config["revision"],
                        "dataset_sha256": csv_sha256,
                        "protocol": config["protocol"],
                        "parameters": (
                            config["baseline"]
                            if model_name == "BR+HAT"
                            else config["mlhat"]
                        ),
                    },
                )
                run.log(asdict(evaluation))
                run.summary["duration_seconds"] = duration_seconds
                record["wandb_url"] = run.url
                run.finish()

            results.append(record)
            output_file.write_text(
                json.dumps(results, indent=2),
                encoding="utf-8",
            )
            print(
                f"{model_name}, seed={seed}: "
                f"example_f1={evaluation.example_f1:.4f}, "
                f"hamming_loss={evaluation.hamming_loss:.4f}, "
                f"subset_accuracy={evaluation.subset_accuracy:.4f}"
            )

    print(f"Saved results to: {output_file}")


if __name__ == "__main__":
    main()