"""Check EXP-02 stream ordering and source-format edge cases."""

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import exp2_data
from experiments.reproduce_exp2 import _model, _wandb_series, completed_output, evaluate_stream
from mlhat.drift import DriftAction
from mlhat.tree import DriftEvent


class _RememberLastLabel:
    """Tiny online model whose next prediction proves test-then-train order."""

    def __init__(self) -> None:
        self.last = np.zeros(2, dtype=np.int64)
        self.drift_events: list[DriftEvent] = []
        self.seen = 0

    def predict_one(self, features: np.ndarray) -> np.ndarray:
        return self.last.copy()

    def learn_one(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.last = labels.copy()
        self.seen += 1
        if self.seen == 2:
            self.drift_events.append(DriftEvent(2, 0, 0, DriftAction.START_BACKGROUND))


def test_exp2_curves_use_predictions_before_learning() -> None:
    X = np.arange(4, dtype=np.float64).reshape(-1, 1)
    Y = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.int64)
    result = evaluate_stream(
        _RememberLastLabel(), X, Y,
        rolling_window=2, rolling_step=2, fading_factor=0.5, fading_step=2,
    )
    assert result.n_samples == 4
    assert result.example_f1 == 0.5
    assert result.hamming_loss == 0.375
    assert result.subset_accuracy == 0.5
    assert result.rolling_f1 == [
        {"sample_index": 2, "example_f1": 0.5},
        {"sample_index": 4, "example_f1": 0.5},
    ]
    assert [point['example_f1'] for point in result.faded_f1] == pytest.approx([2 / 3, 2 / 3])
    assert result.drift_events == [
        {"sample_index": 2, "node_id": 0, "depth": 0, "action": "START_BACKGROUND"}
    ]
    series = _wandb_series(result)
    assert [step for step, _ in series] == [2, 4]
    assert series[0][1]['drift_start_background'] == 1
    assert series[0][1]['faded_example_f1'] == pytest.approx(2 / 3)


def test_exp2_arff_reads_labels_first_and_ignores_final_comma(tmp_path, monkeypatch) -> None:
    path = tmp_path / "SynHPSud.arff"
    header = "@relation 'stream: -C 8'\n" + "".join(
        f"@attribute class{i} {{0,1}}\n" for i in range(8)
    ) + "".join(f"@attribute att{i} numeric\n" for i in range(30)) + "@data\n"
    first = "1,0,0,0,0,0,0,1," + ",".join(["0.25"] * 30) + ",\n"
    rest = "0," * 8 + "0.5," * 30 + "\n"
    path.write_text(header + first + rest * 49_999, encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(exp2_data, "REFERENCE_PATH", tmp_path / "reference.json")
    exp2_data.REFERENCE_PATH.write_text(
        '{"datasets":{"SynHPSud":{"author_sha256":"' + digest + '"}}}',
        encoding="utf-8",
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=lambda **_: path))
    source = {
        "repo_id": "team/hyperplane", "file": "SynHPSud.arff",
        "revision": "a" * 40, "sha256": digest, "label_location": "start",
    }
    X, Y, actual = exp2_data.load_hyperplane("SynHPSud", source, cache_dir=tmp_path)
    assert X.shape == (50_000, 30)
    assert Y.shape == (50_000, 8)
    assert Y[0].tolist() == [1, 0, 0, 0, 0, 0, 0, 1]
    assert X[0, 0] == 0.25
    assert actual == digest


def test_exp2_rejects_unverified_hf_hash() -> None:
    source = {
        "repo_id": "team/hyperplane", "file": "SynHPSud.arff",
        "revision": "a" * 40, "sha256": "0" * 64, "label_location": "start",
    }
    with pytest.raises(ValueError, match="author's original ARFF"):
        exp2_data.load_hyperplane("SynHPSud", source)


def test_exp2_all_three_model_variants_accept_the_same_ordered_stream() -> None:
    config = json.loads(
        (Path(__file__).parents[1] / "configs/exp2_hyperplane.json").read_text(encoding="utf-8")
    )
    X = np.tile(np.linspace(0.0, 1.0, 30), (6, 1)).astype(np.float64)
    Y = np.tile(np.array([1, 0, 1, 0, 0, 0, 0, 0], dtype=np.int64), (6, 1))
    for name in ("MLHAT", "BR+HAT", "MLHAT-no-drift"):
        model = _model(name, config, seed=1)
        result = evaluate_stream(
            model, X, Y,
            rolling_window=4, rolling_step=2, fading_factor=0.995, fading_step=2,
        )
        assert result.n_samples == 6
        assert len(result.rolling_f1) == 2
        assert len(result.faded_f1) == 3


def test_existing_measurement_is_never_overwritten(tmp_path) -> None:
    path = tmp_path / 'run.json'
    config = {'experiment': 'EXP-02'}
    assert not completed_output(path, config, 'digest', 'revision', resume=False, wandb=False)
    path.write_text(json.dumps({'git_commit': 'revision', 'config': config,
                               'dataset_sha256': 'digest', 'result': {'n_samples': 50000}}))
    before = path.read_bytes()
    assert completed_output(path, config, 'digest', 'revision', resume=True, wandb=False)
    for resume, commit in ((False, 'revision'), (True, 'different-revision')):
        with pytest.raises(FileExistsError, match='Refusing to overwrite'):
            completed_output(path, config, 'digest', commit, resume=resume, wandb=False)
    assert path.read_bytes() == before


def test_runner_checks_collisions_before_download_or_training(tmp_path, monkeypatch) -> None:
    from experiments import reproduce_exp2

    config = Path(__file__).parents[1] / 'configs/exp2_hyperplane.json'
    manifest = Path(__file__).parents[1] / 'data/manifests/hyperplane-hf.json'
    (tmp_path / 'SynHPSud_MLHAT_seed1.json').write_text('{}')
    monkeypatch.setattr(sys, 'argv', ['reproduce_exp2', '--config', str(config), '--manifest', str(manifest),
                                    '--output-dir', str(tmp_path)])
    def forbidden(*args, **kwargs):
        raise AssertionError('Must not download or create a model for existing output')
    monkeypatch.setattr(reproduce_exp2, 'load_hyperplane', forbidden)
    monkeypatch.setattr(reproduce_exp2, '_model', forbidden)
    with pytest.raises(SystemExit) as error:
        reproduce_exp2.main()
    assert error.value.code == 2
