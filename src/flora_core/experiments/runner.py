"""Reproducible run lifecycle around a user-supplied experiment function."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import platform
import sys
import traceback
from typing import Any, Callable

import numpy as np

from .config import ExperimentConfig


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    experiment: str
    seed: int
    status: str
    started_at: str
    finished_at: str
    config_sha256: str
    metrics: dict[str, Any]
    error: str | None = None


def run_experiment(
    config: ExperimentConfig,
    experiment: Callable[[ExperimentConfig], dict[str, Any]],
) -> RunRecord:
    """Run callback, save config/metadata/metrics; failures are recorded then raised."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = config.output_dir / "config.json"
    payload = asdict(config)
    payload["output_dir"] = str(config.output_dir)
    config_text = json.dumps(payload, sort_keys=True, indent=2, default=str) + "\n"
    config_path.write_text(config_text, encoding="utf-8")
    digest = hashlib.sha256(config_text.encode()).hexdigest()
    run_id = f"{config.name}-{config.seed}-{digest[:10]}"
    started = datetime.now(timezone.utc).isoformat()
    status, metrics, error, failure = "completed", {}, None, None
    try:
        result = experiment(config)
        if not isinstance(result, dict):
            raise TypeError("experiment callback must return a metrics dictionary")
        metrics = result
        json.dumps(metrics, allow_nan=False)
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        failure = exc
    record = RunRecord(
        run_id=run_id,
        experiment=config.name,
        seed=config.seed,
        status=status,
        started_at=started,
        finished_at=datetime.now(timezone.utc).isoformat(),
        config_sha256=digest,
        metrics=metrics,
        error=error,
    )
    (config.output_dir / "run.json").write_text(
        json.dumps(asdict(record), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if error is not None:
        raise RuntimeError(f"experiment {run_id} failed: {error}") from failure
    return record
