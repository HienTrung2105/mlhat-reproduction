"""Small executable pipeline SMOKE TEST; NOT MLHAT or paper reproduction.

Run: python -m experiments.smoke_baseline
Optional real W&B demo run: python -m experiments.smoke_baseline --log-wandb --wandb-project NAME
"""

import argparse
from dataclasses import asdict

import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.linear_model import SGDClassifier

from mlhat.evaluation import prequential_evaluate


class BaselinePerLabelSGD:
    """Demo-only online baseline with one SGD binary classifier per label."""

    def __init__(self, n_labels: int, *, seed: int = 42) -> None:
        """Create untrained binary SGD models for a demo stream."""
        self.models = [SGDClassifier(loss="log_loss", random_state=seed + i) for i in range(n_labels)]
        self.has_seen_data = False

    def predict_one(self, features: np.ndarray) -> np.ndarray:
        """Predict each label; prior to training use the all-zero baseline."""
        if not self.has_seen_data:
            return np.zeros(len(self.models), dtype=np.int64)
        return np.array([int(m.predict(features.reshape(1, -1))[0]) for m in self.models], dtype=np.int64)

    def learn_one(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Incrementally update all binary classifiers on one observed sample."""
        for model, label in zip(self.models, labels, strict=True):
            model.partial_fit(features.reshape(1, -1), np.array([label]), classes=np.array([0, 1]))
        self.has_seen_data = True


def main() -> None:
    """Execute one deterministic synthetic toy-stream demonstration."""
    parser = argparse.ArgumentParser(description="Toy baseline smoke test — NOT paper reproduction")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-samples", type=int, default=80)
    parser.add_argument("--log-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="mlhat-team-demo")
    args = parser.parse_args()
    if args.n_samples < 2:
        parser.error("--n-samples must be >= 2")

    X, Y = make_multilabel_classification(
        n_samples=args.n_samples,
        n_features=10,
        n_classes=3,
        n_labels=2,
        random_state=args.seed,
        return_indicator="dense",
    )
    model = BaselinePerLabelSGD(n_labels=Y.shape[1], seed=args.seed)
    results = prequential_evaluate(
        model, ((X[i].astype(np.float64), Y[i].astype(np.int64)) for i in range(len(X))),
        n_labels=Y.shape[1],
    )
    print("STATUS: DEMO ONLY (synthetic data, simple baseline, NOT MLHAT, NOT reproduction)")
    print(f"Seed: {args.seed} | Samples: {results.n_samples}")
    print(
        f"Example-based F1: {results.example_f1:.4f} | "
        f"Hamming loss: {results.hamming_loss:.4f} | "
        f"Subset accuracy: {results.subset_accuracy:.4f} | "
        f"Micro-F1: {results.micro_f1:.4f}"
    )

    if args.log_wandb:
        try:
            import wandb
        except ImportError as exc:
            raise SystemExit('Install tracking dependency: pip install -e ".[tracking]"') from exc
        with wandb.init(
            project=args.wandb_project,
            job_type="smoke-test",
            tags=["demo-only", "not-mlhat", "synthetic-data"],
            config={"seed": args.seed, "n_samples": args.n_samples, "model": "per-label-sgd-demo"},
        ) as run:
            run.log(asdict(results))
            print(f"Demo run URL: {run.url}")
            print("Set project visibility to PUBLIC in W&B UI if course evidence requires it.")


if __name__ == "__main__":
    main()
