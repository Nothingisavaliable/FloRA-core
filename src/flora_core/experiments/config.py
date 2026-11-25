"""Small, simulator-agnostic experiment contract with explicit paths/seeds."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int
    output_dir: Path
    model: dict[str, Any]
    data: dict[str, Any]
    environment: dict[str, Any]

    def __post_init__(self):
        if not self.name.strip():
            raise ValueError("experiment name must be non-empty")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        for key, value in (("model", self.model), ("data", self.data), ("environment", self.environment)):
            if not isinstance(value, dict):
                raise ValueError(f"{key} must be an object")


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"name", "seed", "output_dir", "model", "data", "environment"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"missing experiment config keys: {sorted(missing)}")
    output_dir = Path(payload["output_dir"]).expanduser()
    if not output_dir.is_absolute():
        output_dir = path.parent / output_dir
    return ExperimentConfig(
        name=str(payload["name"]),
        seed=int(payload["seed"]),
        output_dir=output_dir.resolve(),
        model=dict(payload["model"]),
        data=dict(payload["data"]),
        environment=dict(payload["environment"]),
    )
