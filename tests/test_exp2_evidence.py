"""Synthetic unit fixtures for evidence validation and an entirely mocked upload SDK.

These fixtures are not experimental measurements and never leave pytest's temp dir.
"""

import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from experiments import exp2_evidence as evidence
from experiments.exp2_upload import existing_run, load_upload_ledger, upload_plan, upload_records, verify_remote


@pytest.fixture
def batch(tmp_path, monkeypatch):
    tmp_path = tmp_path / 'evidence'
    tmp_path.mkdir()
    root = Path(__file__).parents[1]
    config = json.loads((root / 'configs/exp2_hyperplane.json').read_text())
    manifest = json.loads((root / 'data/manifests/hyperplane-hf.json').read_text())
    snapshot = tmp_path / 'source_snapshot.zip'
    with zipfile.ZipFile(snapshot, 'w') as archive:
        archive.writestr('configs/exp2_hyperplane.json', json.dumps(config))
        archive.writestr('data/manifests/hyperplane-hf.json', json.dumps(manifest))
    monkeypatch.setattr(evidence, 'SNAPSHOT_SHA256', evidence.digest_file(snapshot))
    for dataset in config['datasets']:
        for model in config['models']:
            for seed in config['seeds']:
                f1 = seed / 10
                record = {
                    'experiment': 'EXP-02', 'dataset': dataset, 'model': model, 'seed': seed,
                    'git_commit': 'a' * 40, 'git_dirty': True,
                    'config': config, 'dataset_source': manifest['datasets'][dataset],
                    'dataset_sha256': manifest['datasets'][dataset]['sha256'],
                    'duration_seconds_excluding_download': 1.0,
                    'result': {'n_samples': 50000, 'example_f1': f1, 'hamming_loss': 0.25, 'subset_accuracy': 0.1,
                               'rolling_f1': [{'sample_index': i, 'example_f1': f1} for i in range(1000, 50001, 250)],
                               'faded_f1': [{'sample_index': i, 'example_f1': f1} for i in range(50, 50001, 50)],
                               'drift_events': []},
                }
                (tmp_path / f'{dataset}_{model}_seed{seed}.json').write_text(json.dumps(record))
    return tmp_path, config, manifest


def test_exact_batch_and_sample_sd(batch):
    directory, _, _ = batch
    pairs, audit = evidence.load_evidence(directory)
    assert audit['validated_runs'] == 18
    rows = evidence.summarize([r for _, r in pairs])
    f1 = next(r for r in rows if r['metric'] == 'example_f1')
    assert f1['mean'] == pytest.approx(0.2)
    assert f1['sample_sd'] == pytest.approx(0.1)  # Sample SD, not population SD.


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'truncated', 'curve', 'mean', 'snapshot'])
def test_incomplete_or_inconsistent_evidence_is_rejected(batch, mutation):
    directory, _, _ = batch
    path = next(directory.glob('*_seed*.json'))
    record = json.loads(path.read_text())
    if mutation == 'missing':
        path.unlink()
    elif mutation == 'duplicate':
        (directory / 'duplicate_seed1.json').write_text(path.read_text())
    elif mutation == 'snapshot':
        (directory / 'source_snapshot.zip').write_bytes(b'altered')
    else:
        if mutation == 'truncated':
            record['result']['n_samples'] = 49999
        elif mutation == 'curve':
            record['result']['faded_f1'].pop()
        else:
            record['result']['example_f1'] = 0.99
        path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        evidence.load_evidence(directory)


class FakeSDK:
    """In-memory W&B interface: no network, authentication or real runs."""
    def __init__(self):
        self.remote = {}
        self.init_calls = []
        self.artifacts = []

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        remote_path = f"{kwargs['entity']}/{kwargs['project']}/{kwargs['id']}"
        if remote_path not in self.remote:
            self.remote[remote_path] = FakeRun(kwargs, remote_path, self)
        return self.remote[remote_path]

    def run(self, path):
        if path not in self.remote:
            raise ValueError('Could not find run')
        return self.remote[path]

    def flush(self):
        pass

    def Artifact(self, name, **kwargs):
        artifact = SimpleNamespace(files=[], add_file=lambda path, **kwargs: artifact.files.append(Path(path).name))
        self.artifacts.append(artifact)
        return artifact

    def Table(self, **kwargs):
        return kwargs

    def Image(self, path):
        return path


class FakeRun:
    def __init__(self, kwargs, path, sdk):
        self.config, self.summary, self.history = kwargs['config'], {}, []
        self.state, self.url = 'running', f'https://wandb.ai/{path}'
        self.sdk = sdk

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *args):
        self.state = 'finished' if exc_type is None else 'failed'

    def define_metric(self, *args, **kwargs):
        pass

    def log(self, row):
        self.history.append(row)

    def log_artifact(self, artifact):
        return SimpleNamespace(wait=lambda: None)

    def scan_history(self, keys=None, **kwargs):
        return [row for row in self.history if keys is None or all(key in row for key in keys)]


def test_upload_is_team_scoped_idempotent_and_measurements_are_immutable(batch):
    directory, _, _ = batch
    pairs, audit = evidence.load_evidence(directory)
    sdk = FakeSDK()
    before = {p.name: evidence.digest_file(p) for p, _ in pairs}
    reports = directory.parent / 'reports'
    first = upload_records(directory, pairs, audit, sdk, sdk, reports_dir=reports)
    assert len(first['runs']) == len(sdk.remote) == len(sdk.init_calls) == 18
    assert all(call['entity'] == evidence.TEAM and call['project'] == evidence.PROJECT for call in sdk.init_calls)
    assert all(call['mode'] == 'online' and call['resume'] == 'allow' for call in sdk.init_calls)
    assert all('measurement.json' not in a.files and 'source_snapshot.zip' in a.files for a in sdk.artifacts)
    # SDK was passed the actual original JSON path; filenames carry dataset/model/seed.
    assert all(any('_seed' in name for name in a.files) for a in sdk.artifacts)
    saved = next(iter(sdk.remote.values()))
    metric_row = next(row for row in saved.history if 'faded_example_f1' in row)
    original_value = metric_row['faded_example_f1']
    metric_row['faded_example_f1'] = 0.99
    with pytest.raises(ValueError, match='curve/event values'):
        verify_remote(saved, pairs[0][1], before[pairs[0][0].name])
    metric_row['faded_example_f1'] = original_value
    upload_records(directory, pairs, audit, sdk, sdk, reports_dir=reports)
    assert len(sdk.init_calls) == 18
    assert {p.name: evidence.digest_file(p) for p, _ in pairs} == before
    record = pairs[0][1]
    remote = next(iter(sdk.remote.values()))
    remote.summary['example_f1'] = float('nan')
    with pytest.raises(ValueError, match='summary mismatch'):
        verify_remote(remote, record, before[pairs[0][0].name])


def test_upload_retries_a_partial_run_using_the_same_id(batch):
    directory, _, _ = batch
    pairs, audit = evidence.load_evidence(directory)
    path, record = pairs[0]
    digest = audit['measurement_sha256'][path.name]
    sdk = FakeSDK()
    identity = evidence.run_id(record, digest)
    remote = sdk.init(entity=evidence.TEAM, project=evidence.PROJECT, id=identity,
                      config={**record, 'measurement_sha256': digest})
    remote.history = evidence.series(record)[:10]
    upload_records(directory, pairs, audit, sdk, sdk, reports_dir=directory.parent / 'reports')
    assert len(sdk.remote) == 18
    steps = [row['sample_index'] for row in remote.history if 'sample_index' in row]
    assert len(steps) == len(set(steps)) == 1000


def test_missing_run_handles_sdk_wrapped_error_but_does_not_swallow_auth_failure():
    class CommError(Exception):
        pass

    def missing(_):
        raise CommError('Could not find run team/project/run')

    assert existing_run(SimpleNamespace(run=missing), 'team/project/run') is None

    def unauthorized(_):
        raise CommError('HTTP 403: permission denied')

    with pytest.raises(CommError, match='403'):
        existing_run(SimpleNamespace(run=unauthorized), 'team/project/run')


def test_report_and_upload_plan_preserve_every_input_file(batch, monkeypatch):
    import sys
    from experiments import report_exp2

    directory, _, _ = batch
    reports = directory.parent / 'reports'
    before = {p.name: evidence.digest_file(p) for p in directory.iterdir()}
    monkeypatch.setitem(sys.modules, 'wandb', None)  # Any accidental SDK import fails.
    for mode in ('--validate-only', '--plan-upload'):
        monkeypatch.setattr(sys, 'argv', ['report_exp2', '--directory', str(directory), '--output-dir', str(reports), mode])
        report_exp2.main()
    assert before == {p.name: evidence.digest_file(p) for p in directory.iterdir()}
    plan = json.loads((reports / 'upload_plan.json').read_text())
    assert len(plan) == 18 and len({row['run_id'] for row in plan}) == 18
    assert not any(row['recorded_finished'] for row in plan)


@pytest.mark.parametrize('nested', ['', 'derived'])
def test_report_rejects_output_inside_evidence(batch, monkeypatch, nested):
    import sys
    from experiments import report_exp2

    directory, _, _ = batch
    monkeypatch.setattr(sys, 'argv', ['report_exp2', '--directory', str(directory),
                                    '--output-dir', str(directory / nested), '--validate-only'])
    with pytest.raises(SystemExit) as error:
        report_exp2.main()
    assert error.value.code == 2


def test_plan_reuses_preserved_public_ledger_and_upload_writes_only_reports(batch):
    directory, _, _ = batch
    reports = directory.parent / 'reports'
    pairs, audit = evidence.load_evidence(directory)
    sdk = FakeSDK()
    ledger = upload_records(directory, pairs, audit, sdk, sdk, reports_dir=reports)
    for entry in ledger['runs'].values():
        entry['public_visibility_verified'] = True
    # Simulate an existing public ledger preserved with the measurement batch.
    (directory / 'wandb_uploads.json').write_text(json.dumps(ledger))
    (reports / 'wandb_uploads.json').unlink()
    before = {p.name: evidence.digest_file(p) for p in directory.iterdir()}
    inherited = load_upload_ledger(directory, reports)
    assert all(row['recorded_finished'] for row in upload_plan(pairs, audit, inherited))
    updated = upload_records(directory, pairs, audit, sdk, sdk, reports_dir=reports)
    assert len(sdk.init_calls) == 18  # Already finished: no new run initialized.
    assert all(row['public_visibility_verified'] for row in updated['runs'].values())
    assert before == {p.name: evidence.digest_file(p) for p in directory.iterdir()}
