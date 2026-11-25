"""Train the Global Motion Predictor on replayed SpaceMouse demonstrations.

Use flora-replay-demo --collect-point-cloud to prepare each episode.
The model predicts a fixed-length remaining trajectory from a stable initial
scene cloud, the current cup state, and the requested realization.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import torch
from torch.utils.data import DataLoader

from flora_core.demonstrations.dataset import discover_local_dp3_episodes
from flora_core.models import GlobalLocalDiffusionConfig, GlobalTrajectoryPredictor
from flora_core.realization.schema import realization_network_condition


TARGET_CONVENTION = "uniform_remaining_recorded_pose_v1"


def _load_examples(paths: Sequence[Path], horizon: int, queries_per_phase: int) -> list[dict]:
    examples = []
    point_count = None
    for path in paths:
        with np.load(path, allow_pickle=False) as episode:
            cloud = np.array(episode["point_cloud_task"][0], dtype=np.float32)
            state = np.asarray(episode["state_task"], dtype=np.float32)
            phase = np.asarray(episode["phase_id"])
            times = np.asarray(episode["time"], dtype=np.float64)
            metadata = json.loads(str(episode["metadata"].item()))
        if cloud.ndim != 2 or cloud.shape[1] != 6 or not len(cloud) or not np.isfinite(cloud).all():
            raise ValueError(f"{path}: expected a finite initial point cloud with shape [N,6]")
        if point_count is None:
            point_count = len(cloud)
        if len(cloud) != point_count:
            raise ValueError(f"{path}: replay all episodes in a split with the same point count")
        if (state.ndim != 2 or state.shape[1] != 11 or len(state) < 2
                or phase.shape != (len(state),) or times.shape != (len(state),)
                or not np.isin(phase, (0, 1, 2)).all()
                or not np.isfinite(state).all() or not np.isfinite(times).all()
                or np.any(np.diff(times) <= 0)):
            raise ValueError(f"{path}: invalid per-frame state, phase, or timestamps")
        condition = realization_network_condition(metadata)
        rotations = Slerp(times, Rotation.from_quat(state[:, [4, 5, 6, 3]]))
        for label in np.unique(phase[:-1]):
            indices = np.flatnonzero(phase[:-1] == label)
            selection = np.rint(
                np.linspace(0, len(indices) - 1, min(queries_per_phase, len(indices)))
            ).astype(int)
            for index in indices[selection]:
                suffix_times = np.linspace(times[index], times[-1], horizon)
                positions = np.column_stack(
                    [np.interp(suffix_times, times, state[:, axis]) for axis in range(3)]
                )
                rot6d = rotations(suffix_times).as_matrix()[:, :2, :].reshape(horizon, 6)
                examples.append({
                    "fused_point_cloud": cloud,
                    "fused_state": state[index],
                    "progress": state[index, 10:11],
                    "target": np.column_stack((positions, rot6d)).astype(np.float32),
                    **condition,
                })
    return examples


def _run_epoch(model, loader, device, optimizer=None) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = dict.fromkeys(("loss", "position_loss", "rotation_loss"), 0.0)
    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            target = batch.pop("target")
            if training:
                optimizer.zero_grad(set_to_none=True)
            losses = model.training_loss(model(**batch), target)
            loss = losses["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite Global trajectory loss")
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
            for key in totals:
                totals[key] += float(losses[key].detach()) * len(target)
    return {key: value / len(loader.dataset) for key, value in totals.items()}


def _save_checkpoint(path: Path, checkpoint: dict) -> None:
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            torch.save(checkpoint, stream)
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def train_global(
    train_roots: str | Path | Sequence[str | Path],
    out_dir: str | Path,
    *,
    val_roots: str | Path | Sequence[str | Path] = (),
    epochs: int = 10,
    batch_size: int = 16,
    queries_per_phase: int = 16,
    learning_rate: float = 1e-4,
    device: str = "auto",
    seed: int = 3407,
) -> Path:
    """Fit Global on prepared episodes and return the path to last.pt."""
    if min(epochs, batch_size, queries_per_phase) < 1:
        raise ValueError("epochs, batch size, and queries per phase must be positive")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning rate must be finite and positive")
    train_paths = discover_local_dp3_episodes(train_roots)
    val_paths = discover_local_dp3_episodes(val_roots)
    if not train_paths or (val_roots and not val_paths):
        raise ValueError("each requested split must contain at least one dp3_episode.npz")
    if set(train_paths) & set(val_paths):
        raise ValueError("training and validation contain the same episode")
    torch.manual_seed(seed)
    target_device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    )
    config = GlobalLocalDiffusionConfig(
        global_scene_source="scene_state",
        global_scene_representation="pointcloud",
    )
    train = _load_examples(train_paths, config.global_horizon, queries_per_phase)
    val = _load_examples(val_paths, config.global_horizon, queries_per_phase)
    loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size) if val else None
    model = GlobalTrajectoryPredictor(config).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    last_path = out_dir / "last.pt"
    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        metrics = {"epoch": epoch, "train": _run_epoch(model, loader, target_device, optimizer)}
        if val_loader is not None:
            metrics["validation"] = _run_epoch(model, val_loader, target_device)
        checkpoint = {
            "config": asdict(config),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "target_convention": TARGET_CONVENTION,
            "metrics": metrics,
            "data_manifest": {
                "train": list(map(str, train_paths)),
                "validation": list(map(str, val_paths)),
            },
        }
        _save_checkpoint(last_path, checkpoint)
        if val_loader is not None and metrics["validation"]["loss"] < best_loss:
            best_loss = metrics["validation"]["loss"]
            _save_checkpoint(out_dir / "best.pt", checkpoint)
        print(json.dumps(metrics), flush=True)
    return last_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, required=True, action="append",
                        help="Prepared episode or directory. Repeat for multiple roots.")
    parser.add_argument("--val-root", type=Path, action="append", default=[],
                        help="Optional held-out episode or directory.")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--queries-per-phase", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args(argv)
    try:
        path = train_global(
            args.train_root, args.out_dir, val_roots=args.val_root, epochs=args.epochs,
            batch_size=args.batch_size, queries_per_phase=args.queries_per_phase,
            learning_rate=args.learning_rate, device=args.device, seed=args.seed,
        )
    except (ValueError, OSError, KeyError, FloatingPointError) as exc:
        parser.exit(1, f"{exc}\n")
    print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
