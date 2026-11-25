"""OSC shadow-rollout evaluation of task-frame Global Motion."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
from scipy.spatial.transform import Rotation


@runtime_checkable
class GlobalMotionBackend(Protocol):
    """Robot/simulator operations needed to replay Global Motion through OSC."""

    def snapshot_state(self) -> Any: ...
    def restore_state(self, state: Any) -> None: ...
    def robot_environment_collision_pairs(self) -> list[tuple[str, str]]: ...
    def tcp_pose_world(self) -> np.ndarray: ...
    def pose_task_to_world(self, pose7: np.ndarray) -> np.ndarray: ...
    def object_pose_to_tcp_pose(self, object_pose_world: np.ndarray) -> np.ndarray: ...
    def osc_action_for_pose_error(self, target_tcp_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...
    def step(self, action: np.ndarray) -> None: ...
    def joint_limit_margin(self) -> float: ...
    def joint_safe(self) -> bool: ...
    def collision(self) -> bool: ...


@dataclass(frozen=True)
class GlobalMotionConfig:
    position_tolerance_m: float = 0.015
    orientation_tolerance_rad: float = 0.12
    control_substeps_per_waypoint: int = 4

    def __post_init__(self) -> None:
        if self.position_tolerance_m <= 0 or self.orientation_tolerance_rad <= 0:
            raise ValueError("tracking tolerances must be positive")
        if self.control_substeps_per_waypoint < 1:
            raise ValueError("control_substeps_per_waypoint must be positive")


def _validated_trajectory(value: np.ndarray) -> np.ndarray:
    trajectory = np.asarray(value, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != 9 or len(trajectory) == 0:
        raise ValueError("global trajectory must be nonempty [L,9]")
    if not np.isfinite(trajectory).all():
        raise ValueError("global trajectory must be finite")
    return trajectory


def _rot6d_to_pose7(row: np.ndarray) -> np.ndarray:
    first, second = row[3:6].copy(), row[6:9].copy()
    n1 = np.linalg.norm(first)
    if n1 < 1e-8:
        raise ValueError("degenerate first Rot6D vector")
    first /= n1
    second -= np.dot(first, second) * first
    n2 = np.linalg.norm(second)
    if n2 < 1e-8:
        raise ValueError("degenerate second Rot6D vector")
    second /= n2
    matrix = np.stack((first, second, np.cross(first, second)), axis=0)
    quat_xyzw = Rotation.from_matrix(matrix).as_quat()
    return np.r_[row[:3], quat_xyzw[[3, 0, 1, 2]]]


def evaluate_stage2(
    backend: GlobalMotionBackend,
    trajectory: np.ndarray,
    *,
    config: GlobalMotionConfig = GlobalMotionConfig(),
) -> dict[str, Any]:
    """OSC shadow-follow a complete Global path and measure tracking cost."""
    plan = _validated_trajectory(trajectory)
    saved = backend.snapshot_state()
    positions: list[float] = []
    orientations: list[float] = []
    min_margin = float("inf")
    joint_failure = collision_failure = None
    collision_pairs: list[tuple[str, str]] = []
    saturation_count = saturation_total = checked = completed_waypoints = 0
    try:
        # The event state is part of the remaining-motion rollout too.
        if backend.collision():
            collision_failure = 0
            collision_pairs = list(getattr(backend, "last_collision_pairs", ()))
        for index, row in enumerate(plan):
            if collision_failure is not None:
                break
            object_task = _rot6d_to_pose7(row)
            object_world = backend.pose_task_to_world(object_task)
            target_tcp = backend.object_pose_to_tcp_pose(object_world)
            for _ in range(config.control_substeps_per_waypoint):
                # Hold each reference for the configured control interval and
                # recompute OSC feedback at every simulator control step.
                action, saturated = backend.osc_action_for_pose_error(target_tcp)
                saturated = np.asarray(saturated, dtype=bool)
                saturation_count += int(saturated.sum())
                saturation_total += saturated.size
                backend.step(np.asarray(action))
                checked += 1
                if backend.collision():
                    collision_failure = index
                    collision_pairs = list(getattr(backend, "last_collision_pairs", ()))
                    break
            if collision_failure is not None:
                break
            reached = np.asarray(backend.tcp_pose_world(), dtype=np.float64)
            positions.append(float(np.linalg.norm(target_tcp[:3] - reached[:3])))
            q_target = target_tcp[[4, 5, 6, 3]]
            q_reached = reached[[4, 5, 6, 3]]
            orientations.append(float((
                Rotation.from_quat(q_target).inv() * Rotation.from_quat(q_reached)
            ).magnitude()))
            completed_waypoints += 1
            min_margin = min(min_margin, float(backend.joint_limit_margin()))
            if backend.joint_safe() is False and joint_failure is None:
                joint_failure = index
    finally:
        backend.restore_state(saved)

    max_pos = max(positions, default=float("inf"))
    max_rot = max(orientations, default=float("inf"))
    score = max(max_pos / config.position_tolerance_m,
                max_rot / config.orientation_tolerance_rad)
    safe = joint_failure is None and collision_failure is None
    within_tolerance = max_pos <= config.position_tolerance_m and max_rot <= config.orientation_tolerance_rad
    tracking_ok = safe and within_tolerance
    return {
        "tracking_ok": bool(tracking_ok),
        "failure_reason": None if tracking_ok else (
            "collision" if collision_failure is not None else
            "joint_safety" if joint_failure is not None else "tracking_error"
        ),
        "steps": checked,
        "completed_waypoints": completed_waypoints,
        "control_substeps_per_waypoint": config.control_substeps_per_waypoint,
        "planned_steps": len(plan),
        "max_position_error_m": max_pos,
        "max_orientation_error_rad": max_rot,
        "mean_position_error_m": float(np.mean(positions)) if positions else float("inf"),
        "mean_orientation_error_rad": float(np.mean(orientations)) if orientations else float("inf"),
        "tracking_score": float(score),
        "minimum_joint_margin_rad": None if not np.isfinite(min_margin) else min_margin,
        "first_joint_unsafe_step": joint_failure,
        "first_collision_step": collision_failure,
        "collision_pairs": [list(pair) for pair in collision_pairs],
        "controller_saturation_fraction": 0.0 if saturation_total == 0 else saturation_count / saturation_total,
    }


def _stage1_from_rollout(rollout: dict[str, Any]) -> dict[str, Any]:
    collision = rollout["first_collision_step"] is not None
    return {
        "stage1_feasible": not collision,
        "stage1_failure": None if not collision else {
            "reason": "robot_environment_collision",
            "waypoint": rollout["first_collision_step"],
            "collision_pairs": rollout["collision_pairs"],
        },
        "stage1_gate_mode": "collision_only",
        "checked_waypoints": rollout["completed_waypoints"],
        "collision_pairs": rollout["collision_pairs"],
    }


def evaluate_stage1(
    backend: GlobalMotionBackend,
    trajectory: np.ndarray,
    *,
    config: GlobalMotionConfig = GlobalMotionConfig(),
) -> dict[str, Any]:
    """Run an OSC shadow rollout and return its Stage-1 collision decision."""
    return _stage1_from_rollout(evaluate_stage2(backend, trajectory, config=config))


def evaluate_global_motion(
    backend: GlobalMotionBackend,
    trajectory: np.ndarray,
    *,
    config: GlobalMotionConfig = GlobalMotionConfig(),
) -> dict[str, Any]:
    """Filter by OSC-rollout collision, then rank survivors by Tracking Cost."""
    # Stage-1 and Stage-2 use the same rollout from the same initial state.
    rollout = evaluate_stage2(backend, trajectory, config=config)
    stage1 = _stage1_from_rollout(rollout)
    stage2 = None if not stage1["stage1_feasible"] else rollout
    return {
        "stage1": stage1,
        "stage2": stage2,
        "feasible": bool(stage1["stage1_feasible"]),
        "score": None if stage2 is None else stage2["tracking_score"],
    }
