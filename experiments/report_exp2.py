"""Validate/report saved EXP-02 evidence; train nothing and upload only explicitly."""

import argparse
import csv
import json
import os
from pathlib import Path

from experiments.exp2_evidence import (
    ACTIONS, CURVES, METRICS, PROJECT, TEAM, curve_means, digest_file,
    load_evidence, summarize,
)
from experiments.exp2_upload import load_upload_ledger, upload_plan, upload_records


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write report rows with their declared field order; return None."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check_summary(records: list[dict], rows: list[dict]) -> None:
    """Cross-check means/sample SD against independent NumPy calculations."""
    import numpy as np

    for row in rows:
        values = [record["result"][row["metric"]] for record in records
                  if (record["dataset"], record["model"]) == (row["dataset"], row["model"])]
        if not np.allclose([row["mean"], row["sample_sd"]],
                           [np.mean(values), np.std(values, ddof=1)], rtol=1e-12, atol=1e-14):
            raise ValueError("Independent mean/sample SD check failed")


def export_plots(output: Path, records: list[dict]) -> None:
    """Render saved curves using one plotting path for seed means and individual runs.

    Args:
        output: Separate report directory, never the original evidence directory.
        records: Validated records, with aligned sample grids.

    Returns:
        None. Writes four mean PNG/CSV pairs, 36 per-run PNGs, and a hash manifest.
    """
    cache = output / "matplotlib-cache"
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache.resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positions = records[0]["config"]["drift_positions_one_based"]

    def render(path: Path, title: str, curve: str, lines: list[tuple[str, list[dict]]]) -> None:
        fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
        for label, points in lines:
            ax.plot([p["sample_index"] for p in points], [p["example_f1"] for p in points], label=label)
        for position in positions:
            ax.axvline(position, color="gray", linestyle="--", linewidth=0.8)
        definition = "window 1000; step 250" if curve == "rolling_f1" else "fading alpha=0.995; step 50"
        ax.set(title=f"{title}: {definition}", xlabel="Samples processed (one-based)",
               ylabel="Example F1", ylim=(0, 1))
        ax.legend()
        fig.savefig(path, dpi=180)
        plt.close(fig)

    for dataset in sorted({r["dataset"] for r in records}):
        for curve in CURVES:
            lines, rows = [], []
            for model in sorted({r["model"] for r in records}):
                points = curve_means(records, dataset, model, curve)
                lines.append((f"{model} (3 seeds)", points))
                rows.extend({"model": model, **point} for point in points)
            path = output / f"{dataset}_{curve}.png"
            render(path, dataset, curve, lines)
            write_csv(path.with_suffix(".csv"), rows)
    run_plots = output / "run_plots"
    run_plots.mkdir(exist_ok=True)
    for record in records:
        for curve in CURVES:
            path = run_plots / f"{record['dataset']}_{record['model']}_seed{record['seed']}_{curve}.png"
            render(path, f"{record['dataset']} / seed {record['seed']}", curve,
                   [(record["model"], record["result"][curve])])
    files = [*output.glob("*.png"), *output.glob("*_f1.csv"), *run_plots.glob("*.png")]
    manifest = {path.relative_to(output).as_posix(): digest_file(path) for path in sorted(files)}
    (output / "plot_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def export_registry(output: Path, pairs: list[tuple[Path, dict]], audit: dict, ledger: dict) -> None:
    """Write run metrics/events and only measurement-matched URLs; leave baseline counts unknown."""
    rows = []
    for path, record in pairs:
        result = record["result"]
        row = {key: record[key] for key in ("dataset", "model", "seed")}
        row.update({key: result[key] for key in ("n_samples", *METRICS)})
        row["seconds"] = record["duration_seconds_excluding_download"]
        row.update({action.lower(): sum(e["action"] == action for e in result["drift_events"])
                    if record["model"] != "BR+HAT" else "" for action in ACTIONS})
        entry = ledger["runs"].get(path.name, {})
        row["wandb_url"] = entry.get("url", "") if entry.get("measurement_sha256") == audit["measurement_sha256"][path.name] else ""
        row["git_dirty"] = record["git_dirty"]
        rows.append(row)
    write_csv(output / "run_registry.csv", rows)


def main() -> None:
    """Validate the completed batch and write derived reports outside its evidence directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("artifacts/exp2_hyperplane"), help="Read-only evidence input")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/exp2_reports"), help="New derived reports/upload state")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true", help="Validate without plotting or network access")
    mode.add_argument("--plan-upload", action="store_true", help="List stable IDs and recorded upload state; no network")
    mode.add_argument("--upload", action="store_true", help="Explicitly upload missing evidence to W&B; never train")
    parser.add_argument("--wandb-project", choices=[PROJECT], default=PROJECT)
    parser.add_argument("--wandb-entity", choices=[TEAM], default=TEAM)
    args = parser.parse_args()
    evidence = args.directory.resolve()
    output = args.output_dir.resolve()
    if output == evidence or evidence in output.parents or output in evidence.parents:
        parser.error("--output-dir must be separate from the original evidence directory")
    pairs, audit = load_evidence(evidence)
    records = [record for _, record in pairs]
    summaries = summarize(records)
    check_summary(records, summaries)
    audit["statistics_check"] = "all means and sample SD match NumPy ddof=1"
    audit["target"] = {"entity": TEAM, "project": PROJECT}
    output.mkdir(parents=True, exist_ok=True)
    (output / "validation.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    ledger = load_upload_ledger(evidence, output)
    write_csv(output / "summary.csv", summaries)
    if args.plan_upload:
        plan = upload_plan(pairs, audit, ledger)
        (output / "upload_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
        print(f"Local plan: {len(plan)} stable IDs; {sum(r['recorded_finished'] for r in plan)} already recorded Finished. No network/upload.")
    elif not args.validate_only:
        export_plots(output, records)
        if args.upload:
            import tempfile
            temporary = output / "wandb-temp"
            temporary.mkdir(exist_ok=True)
            tempfile.tempdir = str(temporary)
            import wandb
            if not wandb.login(relogin=False):
                parser.error("Run python -m wandb login in your terminal first")
            ledger = upload_records(evidence, pairs, audit, wandb, wandb.Api(timeout=60), reports_dir=output)
    export_registry(output, pairs, audit, ledger)
    if any(digest_file(path) != audit["measurement_sha256"][path.name] for path, _ in pairs):
        raise RuntimeError("Original measurement JSON changed during reporting")
    print(f"Validated 18/18 completed runs. Read-only evidence: {evidence}; reports: {output}")


if __name__ == "__main__":
    main()
