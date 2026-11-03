import json
from pathlib import Path

from mlhat.data import load_yeast


CONFIG_PATH = Path("configs/exp1_yeast.json")
MANIFEST_PATH = Path("data/manifests/yeast.json")


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    dataset_config = config["dataset"]

    X, Y, feature_names, label_names, csv_sha256 = load_yeast(
        revision=dataset_config["revision"]
    )

    mean_cardinality = float(Y.sum(axis=1).mean())
    density = mean_cardinality / Y.shape[1]

    print("X shape:", X.shape)
    print("Y shape:", Y.shape)
    print("First feature names:", feature_names[:5])
    print("Label names:", label_names)
    print("Average labels per example:", round(mean_cardinality, 3))
    print("Label density:", round(density, 3))
    print("CSV SHA-256:", csv_sha256)

    manifest = {
        "source_repo": dataset_config["repo_id"],
        "source_file": dataset_config["file"],
        "revision": dataset_config["revision"],
        "csv_sha256": csv_sha256,
        "rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_labels": int(Y.shape[1]),
        "feature_names": list(feature_names),
        "label_names": list(label_names),
        "target_suffix": dataset_config["target_suffix"],
        "row_order": "source CSV order; not shuffled",
        "average_cardinality": mean_cardinality,
        "label_density": density,
    }

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("Manifest saved:", MANIFEST_PATH)


if __name__ == "__main__":
    main()