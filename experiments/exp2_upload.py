"""Publish saved EXP-02 evidence to the specified existing W&B Team only."""

import json
import math
import time
from pathlib import Path

from experiments.exp2_evidence import ACTIONS, CURVES, METRICS, PROJECT, TEAM, digest_file, run_id, series


def load_upload_ledger(directory: Path, reports_dir: Path) -> dict:
    """Read report-side upload state, falling back to the preserved original ledger.

    Args:
        directory: Read-only measurement directory.
        reports_dir: Directory for new report/upload state.

    Returns:
        Team-scoped ledger; never writes either input file.
    """
    path = reports_dir / "wandb_uploads.json"
    if not path.exists():
        path = directory / "wandb_uploads.json"
    ledger = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "entity": TEAM, "project": PROJECT, "runs": {}}
    if ledger.get("entity") != TEAM or ledger.get("project") != PROJECT:
        raise ValueError("Existing upload ledger belongs to a different Team/project")
    return ledger


def upload_plan(pairs: list[tuple[Path, dict]], audit: dict, ledger: dict) -> list[dict]:
    """Build a local-only plan with stable run IDs; does not call W&B or train."""
    rows = []
    for path, record in pairs:
        digest = audit["measurement_sha256"][path.name]
        identifier = run_id(record, digest)
        entry = ledger["runs"].get(path.name, {})
        matched = (entry.get("id") == identifier and entry.get("measurement_sha256") == digest
                   and entry.get("state") == "finished")
        rows.append({"measurement": path.name, "run_id": identifier,
                     "recorded_finished": matched, "url": entry.get("url", "") if matched else ""})
    return rows


def existing_run(api, path: str):
    """Return a remote run or None only for an explicit not-found response."""
    try:
        return api.run(path)
    except Exception as exc:
        # W&B 0.19 wraps its not-found ValueError as CommError.
        if not isinstance(exc, ValueError) and type(exc).__name__ != 'CommError':
            raise
        if any(text in str(exc).lower() for text in ("not find", "not found", "does not exist")):
            return None
        raise


def verify_remote(remote, record: dict, digest: str) -> None:
    """Verify finished remote config, metrics and all saved stream points before ledgering."""
    if remote.state != 'finished' or remote.summary.get('upload_schema_version') != 2:
        raise RuntimeError('Remote upload is not finished; retry the same upload command')
    if remote.config.get('measurement_sha256') != digest:
        raise ValueError('Remote measurement hash mismatch')
    for key in ('dataset', 'model', 'seed'):
        if remote.config.get(key) != record[key]:
            raise ValueError(f'Remote config mismatch: {key}')
    if remote.summary.get('n_samples') != 50000:
        raise ValueError('Remote n_samples mismatch')
    for metric in METRICS:
        if not math.isclose(remote.summary[metric], record['result'][metric], rel_tol=0, abs_tol=1e-12):
            raise ValueError(f'Remote summary mismatch: {metric}')
    expected = {row['sample_index']: row for row in series(record)}
    found = set()
    for row in remote.scan_history(page_size=1000):
        step = row.get('sample_index')
        if step is None:
            continue
        if step not in expected:
            raise ValueError(f'Unexpected remote stream step: {step}')
        # Media/table logs can inherit the last custom sample_index; only metric rows count.
        if not any(k in row for k in expected[step] if k != 'sample_index'):
            continue
        if any(k not in row or not math.isclose(row[k], value, rel_tol=0, abs_tol=1e-12)
               for k, value in expected[step].items()):
            raise ValueError(f'Remote curve/event values differ at step {step}')
        found.add(step)
    if found != expected.keys():
        raise ValueError(f'Remote stream points incomplete: {len(found)}/{len(expected)}')


def upload_records(directory: Path, pairs: list[tuple[Path, dict]], audit: dict, wandb, api, *, reports_dir: Path) -> dict:
    """Upload an already validated batch without editing measurement JSON or retraining.

    Args:
        directory: Read-only evidence directory with measurements and source snapshot.
        pairs: Exactly 18 immutable measurement path/record pairs.
        audit: Batch validation, including original measurement SHA-256 hashes.
        wandb: W&B SDK, injected for tests without network access.
        api: Authenticated API client, injected for tests.
        reports_dir: Separate directory holding plots and new upload state.

    Returns:
        Ledger of server-verified finished runs in the requested Team/project.
        Retrying reuses deterministic IDs and skips already uploaded sample steps.
    """
    if directory.resolve() == reports_dir.resolve() or directory.resolve() in reports_dir.resolve().parents:
        raise ValueError("Report output must be outside the original evidence directory")
    reports_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = reports_dir / "wandb_uploads.json"
    ledger = load_upload_ledger(directory, reports_dir)
    for path, record in pairs:
        digest = audit["measurement_sha256"][path.name]
        if digest_file(path) != digest:
            raise ValueError(f"Measurement changed after validation: {path}")
        identifier = run_id(record, digest)
        remote_path = f"{TEAM}/{PROJECT}/{identifier}"
        remote = existing_run(api, remote_path)
        if remote is not None and remote.config.get("measurement_sha256") != digest:
            raise ValueError(f"Remote run identity conflict: {remote_path}")
        complete = remote is not None and remote.state == "finished" and remote.summary.get("upload_schema_version") == 2
        if not complete:
            steps = set() if remote is None else {
                int(row["sample_index"]) for row in remote.scan_history(keys=["sample_index"])}
            parameters = dict(record["config"]["baseline" if record["model"] == "BR+HAT" else "mlhat"])
            if record["model"] != "BR+HAT":
                parameters["drift_enabled"] = record["model"] == "MLHAT"
            with wandb.init(
                entity=TEAM, project=PROJECT, id=identifier, resume="allow", mode="online",
                dir=str(reports_dir.resolve()),
                name=f"EXP-02-{record['dataset']}-{record['model']}-seed-{record['seed']}",
                group=record["dataset"], job_type="upload-completed-measurement",
                tags=["EXP-02", "measured-locally", "paper-comparison-pending"],
                config={**{k: v for k, v in record.items() if k not in ("result", "wandb_url")},
                        "effective_model_config": parameters, "measurement_sha256": digest,
                        "source_snapshot_sha256": audit["source_snapshot_sha256"],
                        "logging_mode": "upload_completed_local_measurement",
                        "metric_definitions": {"example_f1": "whole-stream sample mean; empty/empty F1=0",
                                               "rolling_f1": "window=1000, step=250",
                                               "faded_f1": "normalized decay=0.995 per sample, step=50"}},
            ) as run:
                run.define_metric("sample_index")
                for metric in ("rolling_example_f1", "faded_example_f1", *(f"drift_{a.lower()}" for a in ACTIONS)):
                    run.define_metric(metric, step_metric="sample_index")
                for row in series(record):
                    if row["sample_index"] not in steps:
                        run.log(row)
                result = record["result"]
                run.summary.update({key: result[key] for key in ("n_samples", *METRICS)})
                run.summary["f1_score"] = result["example_f1"]
                run.summary["duration_seconds"] = record["duration_seconds_excluding_download"]
                run.summary["structural_events_instrumented"] = record["model"] != "BR+HAT"
                if record["model"] != "BR+HAT":
                    for action in ACTIONS:
                        run.summary[action.lower()] = sum(e["action"] == action for e in result["drift_events"])
                rows = series(record)
                columns = ["sample_index", "rolling_example_f1", "faded_example_f1", *(f"drift_{a.lower()}" for a in ACTIONS)]
                run.log({"stream_metrics": wandb.Table(columns=columns, data=[[row.get(k) for k in columns] for row in rows]),
                         "drift_events": wandb.Table(columns=["sample_index", "node_id", "depth", "action"],
                                                      data=[[e[k] for k in ("sample_index", "node_id", "depth", "action")] for e in result["drift_events"]])})
                for curve in CURVES:
                    image = reports_dir / "run_plots" / f"{record['dataset']}_{record['model']}_seed{record['seed']}_{curve}.png"
                    run.log({f"plots/{curve}": wandb.Image(str(image)),
                             f"batch_plots/{curve}": wandb.Image(str(reports_dir / f"{record['dataset']}_{curve}.png"))})
                artifact = wandb.Artifact(f"exp2-{identifier}-evidence", type="experiment-evidence")
                artifact.add_file(str(path), name="measurement.json")
                artifact.add_file(str(directory / "source_snapshot.zip"), name="source_snapshot.zip")
                for filename in ("validation.json", "summary.csv", "plot_manifest.json"):
                    artifact.add_file(str(reports_dir / filename), name=filename)
                for curve in CURVES:
                    artifact.add_file(str(reports_dir / f"{record['dataset']}_{curve}.csv"))
                run.log_artifact(artifact).wait()
                run.summary["upload_schema_version"] = 2
            for attempt in range(30):
                api.flush()
                remote = api.run(remote_path)
                if remote.state == "finished" and remote.summary.get("upload_schema_version") == 2:
                    break
                time.sleep(2)
            else:
                raise RuntimeError(f"Upload not confirmed finished by server: {remote_path}; retry the same command")
        verify_remote(remote, record, digest)
        previous = ledger["runs"].get(path.name, {})
        public_verified = previous.get("public_visibility_verified", False) if previous.get("measurement_sha256") == digest else False
        ledger["runs"][path.name] = {"id": identifier, "url": remote.url, "state": remote.state,
                                    "measurement_sha256": digest, "public_visibility_verified": public_verified}
        temporary = ledger_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
        temporary.replace(ledger_path)
        print(f"Verified finished upload: {remote.url}", flush=True)
    return ledger
