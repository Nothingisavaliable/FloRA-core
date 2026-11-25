"""Windowed Local-DP3 dataset with trajectory/phase-balanced sampling.

Each training item is aligned at one current timestep t:

    observation: point_cloud/state at [t-To+1, ..., t]
    action:      raw task-frame actions [t, ..., t+H-1]
    realization: m=(u_o,u_g,c)
    Global:      full teacher trajectory + reference index at t

Only complete action horizons are indexed in this first implementation. The
sampler gives every episode equal probability mass, then redistributes that
episode's mass across transport / active-pour / hold strata. This prevents
long trajectories and the terminal hold from dominating training.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import argparse
import json
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, Sampler
except ImportError as exc:
    raise RuntimeError(
        "LocalDP3WindowDataset requires PyTorch; install flora-core[local]"
    ) from exc

from flora_core.realization.schema import realization_network_condition


STRATUM_TRANSPORT = 0
STRATUM_POUR = 1
STRATUM_HOLD = 2
STRATUM_NAMES = {
    STRATUM_TRANSPORT: "transport",
    STRATUM_POUR: "pour",
    STRATUM_HOLD: "hold",
}
DEFAULT_STRATUM_MASS = {
    "transport": 0.45,
    "pour": 0.40,
    "hold": 0.15,
}

_REQUIRED_ARRAYS = {
    "point_cloud_task",
    "state_task",
    "actions_task",
    "global_trajectory",
    "phase_id",
    "phase_progress",
    "source_tilt_angle_deg",
    "metadata",
}


@dataclass(frozen=True)
class LocalDP3EpisodeInfo:
    path: Path
    frame_count: int
    point_count: int
    global_horizon: int
    source_parameter: np.ndarray
    target_parameter: np.ndarray
    channel_id: int
    collection_row_id: str | None
    scene_id: str | None
    realization_id: str | None


@dataclass(frozen=True)
class LocalDP3WindowConfig:
    obs_steps: int = 2
    action_horizon: int = 16
    hold_tilt_tolerance_deg: float = 1e-3
    cache_episodes: int = 4

    def __post_init__(self) -> None:
        if self.obs_steps < 1:
            raise ValueError("obs_steps must be positive")
        if self.action_horizon < 1:
            raise ValueError("action_horizon must be positive")
        if self.hold_tilt_tolerance_deg < 0.0:
            raise ValueError("hold_tilt_tolerance_deg must be non-negative")
        if self.cache_episodes < 0:
            raise ValueError("cache_episodes must be non-negative")


def discover_local_dp3_episodes(
    roots: str | Path | Sequence[str | Path],
) -> list[Path]:
    """Return sorted unique dp3_episode.npz files below one or more roots."""
    if isinstance(roots, (str, Path)):
        roots = [roots]
    found: set[Path] = set()
    for value in roots:
        root = Path(value).expanduser()
        if root.is_file():
            if root.name != "dp3_episode.npz":
                raise ValueError(
                    f"Local-DP3 episode file must be named dp3_episode.npz: {root}"
                )
            found.add(root.resolve())
            continue
        if not root.exists():
            raise FileNotFoundError(root)
        for path in root.rglob("dp3_episode.npz"):
            found.add(path.resolve())
    return sorted(found)


def _load_metadata(archive: np.lib.npyio.NpzFile, path: Path) -> dict:
    try:
        metadata = json.loads(str(archive["metadata"].item()))
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: metadata must be a scalar JSON string") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: metadata must decode to an object")
    return metadata


def _small_episode_index(
    path: Path,
    config: LocalDP3WindowConfig,
) -> tuple[LocalDP3EpisodeInfo, np.ndarray, np.ndarray]:
    """Read only small arrays needed to build the global window index."""
    with np.load(path, allow_pickle=False) as archive:
        archive_files = set(archive.files)
        missing = sorted(_REQUIRED_ARRAYS - archive_files)
        if missing:
            raise ValueError(f"{path}: missing Local-DP3 arrays: {missing}")

        state = np.asarray(archive["state_task"])
        actions = np.asarray(archive["actions_task"])
        phase = np.asarray(archive["phase_id"])
        progress = np.asarray(archive["phase_progress"])
        tilt = np.asarray(archive["source_tilt_angle_deg"])
        global_trajectory = np.asarray(archive["global_trajectory"])
        metadata = _load_metadata(archive, path)
        legacy_v2 = metadata.get("teacher_representation", {}).get("schema_version") == 2
        explicit_phase = legacy_v2 or metadata.get("format") == "flora_spacemouse_demo_v1"
        if explicit_phase:
            missing_v2 = {"global_reference_index", "global_source_time"} - archive_files
            if missing_v2:
                raise ValueError(f"{path}: missing Global mapping arrays: {sorted(missing_v2)}")
            source_time = np.asarray(archive["global_source_time"], dtype=np.float64)
            if source_time.shape != (len(global_trajectory),) or not np.isfinite(source_time).all() or np.any(np.diff(source_time) <= 0):
                raise ValueError(f"{path}: global_source_time must be finite and strictly increasing")
        if "global_reference_index" in archive.files:
            reference = np.asarray(archive["global_reference_index"])
        else:
            # Legacy geometric-teacher episodes used the same uniform
            # time-parameterization to construct global_trajectory, so this is
            # the exact alignment implied by that archived trajectory.
            reference = np.rint(
                np.linspace(
                    0.0,
                    float(global_trajectory.shape[0] - 1),
                    int(state.shape[0]),
                )
            ).astype(np.int16)

        observation = metadata.get("local_dp3_observation", {})
        shape_per_frame = observation.get("shape_per_frame")
        if (
            not isinstance(shape_per_frame, list)
            or len(shape_per_frame) != 2
            or int(shape_per_frame[1]) != 6
        ):
            raise ValueError(
                f"{path}: local_dp3_observation.shape_per_frame must be [N,6]"
            )
        point_count = int(shape_per_frame[0])

    if state.ndim != 2 or state.shape[1] != 11:
        raise ValueError(f"{path}: state_task must be [T,11], got {state.shape}")
    frame_count = int(state.shape[0])
    if actions.shape != (frame_count, 6):
        raise ValueError(
            f"{path}: actions_task must be [T,6], got {actions.shape}"
        )
    for name, array in (
        ("phase_id", phase),
        ("phase_progress", progress),
        ("source_tilt_angle_deg", tilt),
        ("global_reference_index", reference),
    ):
        if array.shape != (frame_count,):
            raise ValueError(f"{path}: {name} must be [T], got {array.shape}")
    if global_trajectory.ndim != 2 or global_trajectory.shape[1] != 9:
        raise ValueError(
            f"{path}: global_trajectory must be [L,9], got {global_trajectory.shape}"
        )
    if frame_count < config.obs_steps + config.action_horizon - 1:
        raise ValueError(
            f"{path}: episode is too short for obs_steps={config.obs_steps} "
            f"and action_horizon={config.action_horizon}"
        )
    if not np.isfinite(state).all() or not np.isfinite(actions).all():
        raise ValueError(f"{path}: state/action contains non-finite values")
    if not np.isfinite(global_trajectory).all():
        raise ValueError(f"{path}: global_trajectory contains non-finite values")
    if np.any(reference < 0) or np.any(reference >= len(global_trajectory)):
        raise ValueError(f"{path}: global_reference_index is outside Global horizon")
    if not np.isfinite(progress).all() or np.any((progress < 0.0) | (progress > 1.0)):
        raise ValueError(f"{path}: phase_progress must lie in [0,1]")

    if explicit_phase:
        if not np.isin(phase, (0, 1, 2)).all() or np.any(np.diff(phase.astype(np.int64)) < 0):
            raise ValueError(f"{path}: phases must follow transport/tilt/hold")
        if np.any(np.diff(reference.astype(np.int64)) < 0):
            raise ValueError(f"{path}: Global reference must be monotone")
        if not np.allclose(state[:, 7:10], np.eye(3)[phase]) or not np.allclose(state[:, 10], progress):
            raise ValueError(f"{path}: state phase/progress disagrees with labels")
    realization = realization_network_condition(metadata)
    info = LocalDP3EpisodeInfo(
        path=path,
        frame_count=frame_count,
        point_count=point_count,
        global_horizon=int(global_trajectory.shape[0]),
        source_parameter=realization["source_parameter"].astype(np.float32),
        target_parameter=realization["target_parameter"].astype(np.float32),
        channel_id=int(np.asarray(realization["channel_id"]).item()),
        collection_row_id=metadata.get("collection_row_id"),
        scene_id=(
            str(metadata["scene"].get("scene_id"))
            if isinstance(metadata.get("scene"), dict)
            and metadata["scene"].get("scene_id") is not None
            else None
        ),
        realization_id=(
            str(metadata["realization"].get("realization_id"))
            if isinstance(metadata.get("realization"), dict)
            and metadata["realization"].get("realization_id") is not None
            else None
        ),
    )

    first = config.obs_steps - 1
    last_exclusive = frame_count - config.action_horizon + 1
    timesteps = np.arange(first, last_exclusive, dtype=np.int32)

    max_tilt = float(np.max(tilt, initial=0.0))
    current_phase = phase[timesteps]
    current_tilt = tilt[timesteps]
    strata = np.full(len(timesteps), STRATUM_POUR, dtype=np.int8)
    strata[current_phase == 0] = STRATUM_TRANSPORT
    hold = (current_phase == 1) & (
        current_tilt >= max_tilt - float(config.hold_tilt_tolerance_deg)
    )
    if explicit_phase:
        hold = current_phase == 2
        if legacy_v2 and np.any(reference[phase == 2] != len(global_trajectory) - 1):
            raise ValueError(f"{path}: v2 hold must reference terminal Global waypoint")
    strata[hold] = STRATUM_HOLD
    unknown = ~np.isin(current_phase, (0, 1, 2) if explicit_phase else (0, 1))
    if np.any(unknown):
        values = sorted(set(int(v) for v in current_phase[unknown]))
        raise ValueError(f"{path}: unsupported phase_id values: {values}")
    return info, timesteps, strata


class LocalDP3WindowDataset(Dataset):
    """Lazy multi-episode dataset for Local DP3 training windows."""

    def __init__(
        self,
        episodes: Sequence[str | Path],
        *,
        config: LocalDP3WindowConfig = LocalDP3WindowConfig(),
    ) -> None:
        self.config = config
        self.episode_paths = [Path(value).expanduser().resolve() for value in episodes]
        if not self.episode_paths:
            raise ValueError("LocalDP3WindowDataset requires at least one episode")
        missing = [path for path in self.episode_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(missing[0])

        self.episodes: list[LocalDP3EpisodeInfo] = []
        window_episode: list[np.ndarray] = []
        window_timestep: list[np.ndarray] = []
        window_stratum: list[np.ndarray] = []
        for episode_index, path in enumerate(self.episode_paths):
            info, timesteps, strata = _small_episode_index(path, config)
            self.episodes.append(info)
            window_episode.append(
                np.full(len(timesteps), episode_index, dtype=np.int32)
            )
            window_timestep.append(timesteps)
            window_stratum.append(strata)

        self.window_episode = np.concatenate(window_episode)
        self.window_timestep = np.concatenate(window_timestep)
        self.window_stratum = np.concatenate(window_stratum)
        if len(self.window_episode) == 0:
            raise ValueError("no complete Local-DP3 windows were found")
        self._cache: OrderedDict[int, dict[str, np.ndarray | dict]] = OrderedDict()

    @classmethod
    def from_roots(
        cls,
        roots: str | Path | Sequence[str | Path],
        *,
        config: LocalDP3WindowConfig = LocalDP3WindowConfig(),
    ) -> "LocalDP3WindowDataset":
        paths = discover_local_dp3_episodes(roots)
        if not paths:
            raise ValueError(f"no dp3_episode.npz files found under {roots}")
        return cls(paths, config=config)

    def __len__(self) -> int:
        return int(len(self.window_episode))

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = OrderedDict()
        return state

    def _load_episode(self, episode_index: int) -> dict[str, np.ndarray | dict]:
        cached = self._cache.get(int(episode_index))
        if cached is not None:
            self._cache.move_to_end(int(episode_index))
            return cached

        info = self.episodes[int(episode_index)]
        with np.load(info.path, allow_pickle=False) as archive:
            arrays: dict[str, np.ndarray | dict] = {
                key: np.asarray(archive[key]).copy()
                for key in _REQUIRED_ARRAYS
                if key != "metadata"
            }
            arrays["metadata"] = _load_metadata(archive, info.path)
            if "point_cloud_sampling_group" in archive.files:
                arrays["point_cloud_sampling_group"] = np.asarray(archive["point_cloud_sampling_group"]).copy()
            if "global_reference_index" in archive.files:
                arrays["global_reference_index"] = np.asarray(
                    archive["global_reference_index"]
                ).copy()
            else:
                global_trajectory = np.asarray(arrays["global_trajectory"])
                arrays["global_reference_index"] = np.rint(
                    np.linspace(
                        0.0,
                        float(global_trajectory.shape[0] - 1),
                        info.frame_count,
                    )
                ).astype(np.int16)

        cloud = np.asarray(arrays["point_cloud_task"])
        if cloud.shape != (info.frame_count, info.point_count, 6):
            raise ValueError(
                f"{info.path}: point_cloud_task must be "
                f"[{info.frame_count},{info.point_count},6], got {cloud.shape}"
            )
        if cloud.dtype != np.float32:
            cloud = cloud.astype(np.float32)
            arrays["point_cloud_task"] = cloud
        if not np.isfinite(cloud).all():
            raise ValueError(f"{info.path}: point_cloud_task contains non-finite values")
        semantic = cloud[..., 3:6]
        if not np.allclose(semantic.sum(axis=-1), 1.0, atol=1e-6):
            raise ValueError(
                f"{info.path}: point semantic channels must be one-hot"
            )
        if np.any((semantic < -1e-6) | (semantic > 1.0 + 1e-6)):
            raise ValueError(
                f"{info.path}: point semantic channels must lie in [0,1]"
            )

        cache_limit = int(self.config.cache_episodes)
        if cache_limit > 0:
            self._cache[int(episode_index)] = arrays
            self._cache.move_to_end(int(episode_index))
            while len(self._cache) > cache_limit:
                self._cache.popitem(last=False)
        return arrays

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_index = int(self.window_episode[index])
        timestep = int(self.window_timestep[index])
        stratum = int(self.window_stratum[index])
        info = self.episodes[episode_index]
        arrays = self._load_episode(episode_index)

        obs_start = timestep - self.config.obs_steps + 1
        action_end = timestep + self.config.action_horizon

        point_cloud = np.asarray(arrays["point_cloud_task"])[
            obs_start : timestep + 1
        ]
        state = np.asarray(arrays["state_task"])[obs_start : timestep + 1]
        action = np.asarray(arrays["actions_task"])[timestep:action_end]
        global_trajectory = np.asarray(arrays["global_trajectory"])
        reference = int(np.asarray(arrays["global_reference_index"])[timestep])
        phase_id = int(np.asarray(arrays["phase_id"])[timestep])
        phase_progress = float(np.asarray(arrays["phase_progress"])[timestep])

        return {
            "current_point_cloud": torch.from_numpy(
                np.ascontiguousarray(point_cloud, dtype=np.float32)
            ),
            "current_state": torch.from_numpy(
                np.ascontiguousarray(state, dtype=np.float32)
            ),
            "action": torch.from_numpy(
                np.ascontiguousarray(action, dtype=np.float32)
            ),
            "source_parameter": torch.from_numpy(info.source_parameter.copy()),
            "target_parameter": torch.from_numpy(info.target_parameter.copy()),
            "channel_id": torch.tensor(info.channel_id, dtype=torch.long),
            "global_trajectory": torch.from_numpy(
                np.ascontiguousarray(global_trajectory, dtype=np.float32)
            ),
            "global_reference_index": torch.tensor(reference, dtype=torch.long),
            "phase_id": torch.tensor(phase_id, dtype=torch.long),
            "phase_progress": torch.tensor(phase_progress, dtype=torch.float32),
            "sampling_stratum": torch.tensor(stratum, dtype=torch.long),
            "episode_index": torch.tensor(episode_index, dtype=torch.long),
            "timestep": torch.tensor(timestep, dtype=torch.long),
        }

    def window_counts(self) -> dict[str, int]:
        return {
            STRATUM_NAMES[stratum]: int(np.count_nonzero(self.window_stratum == stratum))
            for stratum in STRATUM_NAMES
        }

    def episode_window_counts(self) -> list[dict[str, int]]:
        output = []
        for episode_index in range(len(self.episodes)):
            mask = self.window_episode == episode_index
            output.append(
                {
                    STRATUM_NAMES[stratum]: int(
                        np.count_nonzero(mask & (self.window_stratum == stratum))
                    )
                    for stratum in STRATUM_NAMES
                }
            )
        return output

    def balanced_sampling_weights(
        self,
        stratum_mass: Mapping[str, float] = DEFAULT_STRATUM_MASS,
    ) -> torch.Tensor:
        """Return weights with equal episode mass and configured phase mass."""
        masses = {
            name: float(stratum_mass.get(name, 0.0))
            for name in STRATUM_NAMES.values()
        }
        if any(value < 0.0 or not np.isfinite(value) for value in masses.values()):
            raise ValueError("stratum masses must be finite and non-negative")
        if sum(masses.values()) <= 0.0:
            raise ValueError("at least one stratum mass must be positive")

        weights = np.zeros(len(self), dtype=np.float64)
        episode_mass = 1.0 / float(len(self.episodes))
        for episode_index in range(len(self.episodes)):
            episode_mask = self.window_episode == episode_index
            available: dict[int, np.ndarray] = {}
            for stratum, name in STRATUM_NAMES.items():
                indices = np.flatnonzero(
                    episode_mask & (self.window_stratum == stratum)
                )
                if len(indices) and masses[name] > 0.0:
                    available[stratum] = indices
            if not available:
                raise ValueError(
                    f"episode {episode_index} has no windows in enabled strata"
                )
            normalizer = sum(
                masses[STRATUM_NAMES[stratum]] for stratum in available
            )
            for stratum, indices in available.items():
                stratum_total = (
                    episode_mass
                    * masses[STRATUM_NAMES[stratum]]
                    / normalizer
                )
                weights[indices] = stratum_total / float(len(indices))
        if not np.isclose(weights.sum(), 1.0, atol=1e-10):
            raise RuntimeError(
                f"internal sampling weights do not sum to one: {weights.sum()}"
            )
        return torch.from_numpy(weights)


class BalancedWindowSampler(Sampler[int]):
    """Replacement sampler for trajectory-balanced, phase-balanced windows."""

    def __init__(
        self,
        dataset: LocalDP3WindowDataset,
        *,
        num_samples: int | None = None,
        stratum_mass: Mapping[str, float] = DEFAULT_STRATUM_MASS,
        seed: int = 3407,
    ) -> None:
        self.dataset = dataset
        self.num_samples = int(num_samples if num_samples is not None else len(dataset))
        if self.num_samples < 1:
            raise ValueError("num_samples must be positive")
        self.weights = dataset.balanced_sampling_weights(stratum_mass).double()
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        sampled = torch.multinomial(
            self.weights,
            self.num_samples,
            replacement=True,
            generator=generator,
        )
        return iter(sampled.tolist())


def dataset_summary(dataset: LocalDP3WindowDataset) -> dict:
    weights = dataset.balanced_sampling_weights()
    episode_mass = []
    for episode_index in range(len(dataset.episodes)):
        mask = torch.from_numpy(dataset.window_episode == episode_index)
        episode_mass.append(float(weights[mask].sum()))
    stratum_mass = {}
    for stratum, name in STRATUM_NAMES.items():
        mask = torch.from_numpy(dataset.window_stratum == stratum)
        stratum_mass[name] = float(weights[mask].sum())
    first = dataset[0]
    return {
        "episodes": len(dataset.episodes),
        "windows": len(dataset),
        "raw_window_counts": dataset.window_counts(),
        "balanced_probability_mass": stratum_mass,
        "episode_probability_mass_min": min(episode_mass),
        "episode_probability_mass_max": max(episode_mass),
        "first_window_shapes": {
            key: list(value.shape)
            for key, value in first.items()
            if isinstance(value, torch.Tensor) and value.ndim > 0
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="+",
        type=Path,
        help="dp3_episode.npz file(s) or directories searched recursively",
    )
    parser.add_argument("--obs-steps", type=int, default=2)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--cache-episodes", type=int, default=2)
    args = parser.parse_args(argv)

    dataset = LocalDP3WindowDataset.from_roots(
        args.roots,
        config=LocalDP3WindowConfig(
            obs_steps=args.obs_steps,
            action_horizon=args.action_horizon,
            cache_episodes=args.cache_episodes,
        ),
    )
    print(json.dumps(dataset_summary(dataset), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
