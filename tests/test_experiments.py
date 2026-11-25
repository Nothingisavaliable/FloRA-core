import json

import pytest

from flora_core.experiments import load_experiment_config, run_experiment


def test_config_paths_are_relative_to_config(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "name": "smoke", "seed": 2, "output_dir": "runs/a",
        "model": {}, "data": {}, "environment": {},
    }))
    config = load_experiment_config(path)
    assert config.output_dir == tmp_path / "runs/a"


def test_runner_records_config_and_metrics(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "name": "smoke", "seed": 2, "output_dir": "runs/a",
        "model": {}, "data": {}, "environment": {},
    }))
    config = load_experiment_config(path)
    record = run_experiment(config, lambda cfg: {"score": 0.75})
    saved = json.loads((config.output_dir / "run.json").read_text())
    assert saved["status"] == record.status == "completed"
    assert saved["metrics"] == {"score": 0.75}
    assert len(saved["config_sha256"]) == 64


def test_runner_persists_failure_before_raising(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "name": "smoke", "seed": 2, "output_dir": "runs/a",
        "model": {}, "data": {}, "environment": {},
    }))
    config = load_experiment_config(path)
    with pytest.raises(RuntimeError, match="failed"):
        run_experiment(config, lambda cfg: (_ for _ in ()).throw(ValueError("bad run")))
    record = json.loads((config.output_dir / "run.json").read_text())
    assert record["status"] == "failed"
    assert "ValueError: bad run" in record["error"]
