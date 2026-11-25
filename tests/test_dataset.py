import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from flora_core.demonstrations.dataset import (
    BalancedWindowSampler,
    DEFAULT_STRATUM_MASS,
    LocalDP3WindowConfig,
    LocalDP3WindowDataset,
    STRATUM_HOLD,
    STRATUM_POUR,
    STRATUM_TRANSPORT,
    dataset_summary,
    discover_local_dp3_episodes,
)


def _write_episode(
    path: Path,
    *,
    frames: int,
    source: float,
    target: float,
    channel: str,
    phase_switch: int,
    hold_start: int,
    include_reference: bool = True,
) -> None:
    point_count = 32
    cloud = np.zeros((frames, point_count, 6), dtype=np.float32)
    for t in range(frames):
        cloud[t, :, 0] = np.linspace(-0.4, 0.1, point_count, dtype=np.float32)
        cloud[t, :, 1] = 0.001 * t
        cloud[t, :8, 3] = 1.0
        cloud[t, 8:16, 4] = 1.0
        cloud[t, 16:, 5] = 1.0

    state = np.zeros((frames, 11), dtype=np.float32)
    state[:, 0] = np.linspace(-0.3, 0.0, frames, dtype=np.float32)
    state[:, 3] = 1.0
    action = np.zeros((frames, 6), dtype=np.float32)
    action[:hold_start, 0] = 0.01
    action[phase_switch:hold_start, 4] = 0.02

    phase = np.zeros(frames, dtype=np.int8)
    phase[phase_switch:] = 1
    progress = np.zeros(frames, dtype=np.float32)
    for phase_id in (0, 1):
        idx = np.flatnonzero(phase == phase_id)
        progress[idx] = np.linspace(0.0, 1.0, len(idx), dtype=np.float32)

    tilt = np.zeros(frames, dtype=np.float32)
    active_count = max(1, hold_start - phase_switch)
    tilt[phase_switch:hold_start] = np.linspace(
        5.0, 110.0, active_count, dtype=np.float32
    )
    tilt[hold_start:] = 115.0

    global_trajectory = np.zeros((32, 9), dtype=np.float32)
    global_trajectory[:, 0] = np.linspace(-0.3, 0.0, 32, dtype=np.float32)
    global_trajectory[:, 3] = 1.0
    global_trajectory[:, 7] = 1.0
    reference = np.rint(
        np.linspace(0.0, 31.0, frames)
    ).astype(np.int16)

    metadata = {
        "collection_row_id": path.parent.name,
        "scene": {"scene_id": "SYNTH"},
        "realization": {
            "realization_id": f"R_{channel}",
            "normalized_specification": {
                "schema": "normalized_realization_specification_v2_1d_target",
                "u_o": source,
                "u_g": target,
                "c": channel,
            },
        },
        "local_dp3_observation": {
            "schema": "flora_local_dp3_point_cloud_v1",
            "point_cloud_key": "point_cloud_task",
            "shape_per_frame": [point_count, 6],
            "frame": "task",
        },
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "point_cloud_task": cloud,
        "state_task": state,
        "actions_task": action,
        "global_trajectory": global_trajectory,
        "phase_id": phase,
        "phase_progress": progress,
        "source_tilt_angle_deg": tilt,
        "metadata": np.asarray(json.dumps(metadata)),
    }
    if include_reference:
        payload["global_reference_index"] = reference
    np.savez_compressed(path, **payload)


def test_local_dp3_window_shapes_and_alignment(tmp_path: Path):
    episode = tmp_path / "one" / "dp3_episode.npz"
    _write_episode(
        episode,
        frames=40,
        source=0.25,
        target=1.0 / 3.0,
        channel="c_left",
        phase_switch=14,
        hold_start=24,
    )
    dataset = LocalDP3WindowDataset(
        [episode],
        config=LocalDP3WindowConfig(
            obs_steps=2,
            action_horizon=8,
            cache_episodes=1,
        ),
    )

    assert len(dataset) == 32
    first = dataset[0]
    assert first["current_point_cloud"].shape == (2, 32, 6)
    assert first["current_state"].shape == (2, 11)
    assert first["action"].shape == (8, 6)
    assert first["global_trajectory"].shape == (32, 9)
    assert first["source_parameter"].shape == (1,)
    assert first["target_parameter"].shape == (1,)
    assert first["channel_id"].item() == 0
    assert first["timestep"].item() == 1
    assert first["sampling_stratum"].item() == STRATUM_TRANSPORT

    counts = dataset.window_counts()
    assert counts == {
        "transport": 13,
        "pour": 10,
        "hold": 9,
    }

    pour_index = int(np.flatnonzero(dataset.window_stratum == STRATUM_POUR)[0])
    hold_index = int(np.flatnonzero(dataset.window_stratum == STRATUM_HOLD)[0])
    assert dataset[pour_index]["phase_id"].item() == 1
    assert dataset[hold_index]["phase_id"].item() == 1


def test_balanced_weights_equalize_episode_mass_and_control_hold(tmp_path: Path):
    first = tmp_path / "a" / "dp3_episode.npz"
    second = tmp_path / "b" / "dp3_episode.npz"
    _write_episode(
        first,
        frames=40,
        source=0.25,
        target=0.0,
        channel="c_left",
        phase_switch=14,
        hold_start=24,
    )
    _write_episode(
        second,
        frames=55,
        source=0.75,
        target=2.0 / 3.0,
        channel="c_over",
        phase_switch=20,
        hold_start=31,
    )
    dataset = LocalDP3WindowDataset(
        [first, second],
        config=LocalDP3WindowConfig(obs_steps=2, action_horizon=8),
    )
    weights = dataset.balanced_sampling_weights()

    for episode_index in (0, 1):
        mask = torch.from_numpy(dataset.window_episode == episode_index)
        assert torch.isclose(
            weights[mask].sum(),
            torch.tensor(0.5, dtype=weights.dtype),
            atol=1e-10,
        )

    for stratum, name in (
        (STRATUM_TRANSPORT, "transport"),
        (STRATUM_POUR, "pour"),
        (STRATUM_HOLD, "hold"),
    ):
        mask = torch.from_numpy(dataset.window_stratum == stratum)
        assert torch.isclose(
            weights[mask].sum(),
            torch.tensor(DEFAULT_STRATUM_MASS[name], dtype=weights.dtype),
            atol=1e-10,
        )

    summary = dataset_summary(dataset)
    assert summary["episodes"] == 2
    assert summary["episode_probability_mass_min"] == pytest.approx(0.5)
    assert summary["episode_probability_mass_max"] == pytest.approx(0.5)
    assert summary["balanced_probability_mass"]["hold"] == pytest.approx(0.15)


def test_balanced_sampler_is_reproducible_per_epoch(tmp_path: Path):
    episode = tmp_path / "one" / "dp3_episode.npz"
    _write_episode(
        episode,
        frames=40,
        source=0.50,
        target=0.0,
        channel="c_right",
        phase_switch=14,
        hold_start=24,
    )
    dataset = LocalDP3WindowDataset(
        [episode],
        config=LocalDP3WindowConfig(obs_steps=2, action_horizon=8),
    )
    sampler = BalancedWindowSampler(dataset, num_samples=100, seed=73)
    epoch0_a = list(iter(sampler))
    epoch0_b = list(iter(sampler))
    assert epoch0_a == epoch0_b

    sampler.set_epoch(1)
    epoch1 = list(iter(sampler))
    assert epoch1 != epoch0_a


def test_discover_local_dp3_episodes(tmp_path: Path):
    a = tmp_path / "a" / "dp3_episode.npz"
    b = tmp_path / "b" / "nested" / "dp3_episode.npz"
    _write_episode(
        a,
        frames=40,
        source=0.25,
        target=0.0,
        channel="c_left",
        phase_switch=14,
        hold_start=24,
    )
    _write_episode(
        b,
        frames=40,
        source=0.50,
        target=1.0 / 3.0,
        channel="c_right",
        phase_switch=14,
        hold_start=24,
    )
    assert discover_local_dp3_episodes(tmp_path) == [a.resolve(), b.resolve()]



def test_legacy_episode_without_global_reference_index_is_adapted(tmp_path: Path):
    episode = tmp_path / "legacy" / "dp3_episode.npz"
    _write_episode(
        episode,
        frames=40,
        source=0.25,
        target=0.0,
        channel="c_left",
        phase_switch=14,
        hold_start=24,
        include_reference=False,
    )
    dataset = LocalDP3WindowDataset(
        [episode],
        config=LocalDP3WindowConfig(obs_steps=2, action_horizon=8),
    )
    first = dataset[0]
    last = dataset[len(dataset) - 1]
    assert first["global_reference_index"].item() == 1
    expected_last = round(
        last["timestep"].item() * 31 / (40 - 1)
    )
    assert last["global_reference_index"].item() == expected_last
