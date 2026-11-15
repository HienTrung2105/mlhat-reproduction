"""Stage exact author ARFF files for a team-owned Hugging Face dataset upload.

This is a one-time data-preparation step. EXP-02 itself downloads from the
team's pinned Hugging Face dataset, never from the author's GitHub repository.
"""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

from experiments.exp2_data import REFERENCE_PATH


def stage_files(output_dir: Path) -> dict[str, str]:
    """Download both author files at a pinned commit and verify their bytes.

    Args:
        output_dir: Ignored directory whose files can be uploaded to HF.

    Returns:
        Mapping of dataset names to verified SHA-256 checksums.
    """
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    revision = reference["author_revision"]
    output_dir.mkdir(parents=True, exist_ok=True)
    digests: dict[str, str] = {}

    for name, info in reference["datasets"].items():
        destination = output_dir / f"{name}.arff"
        url = (
            "https://raw.githubusercontent.com/"
            f"{reference['author_repo']}/{revision}/{info['author_file']}"
        )
        if not destination.exists():
            partial = destination.with_suffix(".arff.part")
            digest = hashlib.sha256()
            try:
                with urllib.request.urlopen(url, timeout=120) as source, partial.open("wb") as target:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        target.write(chunk)
                        digest.update(chunk)
                if digest.hexdigest() != info["author_sha256"]:
                    raise ValueError(f"{name}: author file checksum differs from reference")
                partial.replace(destination)
            finally:
                partial.unlink(missing_ok=True)

        digest = hashlib.sha256()
        with destination.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != info["author_sha256"]:
            raise ValueError(f"{name}: staged file checksum mismatch")
        digests[name] = digest.hexdigest()

    card = (
        "---\nlicense: cc0-1.0\npretty_name: MLHAT Hyperplane streams\n"
        "---\n\n# MLHAT Hyperplane streams\n\n"
        "Exact ARFF files from [Esteban et al.'s MLHAT repository]"
        f"(https://github.com/{reference['author_repo']}/tree/{revision}/synthdata). "
        "The files are preserved byte-for-byte and in original stream order. "
        "Each row contains eight binary labels followed by 30 numeric features. "
        "Use test-then-train evaluation; do not shuffle.\n\n"
        "| File | SHA-256 |\n| --- | --- |\n"
        + "".join(f"| {name}.arff | `{digest}` |\n" for name, digest in digests.items())
        + "\nSource license: CC0-1.0 in the original repository. "
        "Paper: Esteban et al., *Hoeffding adaptive trees for multi-label "
        "classification on data streams*, Knowledge-Based Systems, 2024.\n"
    )
    (output_dir / "README.md").write_text(card, encoding="utf-8")
    return digests


def main() -> None:
    """Prepare uploadable data locally without publishing to an external service."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/raw/hyperplane-publish")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    for name, digest in stage_files(output_dir).items():
        print(f"{name}: {digest}")
    print(f"Upload-ready files staged in: {output_dir}")


if __name__ == "__main__":
    main()
