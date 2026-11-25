#!/usr/bin/env python3
"""Cup-to-bowl task-frame construction and coordinate transforms.

The frame is fixed for an episode:
  * origin: initial bowl/target body origin
  * +z: world up
  * +x: horizontal direction from the initial cup center to the bowl center
  * +y: z cross x

With this convention the initial cup normally has negative task-frame x and
motion from the cup toward the bowl has positive task-frame x.

All poses use MuJoCo's ``[x, y, z, qw, qx, qy, qz]`` convention.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


FRAME_CONVENTION = "bowl_origin_z_up_x_cup_to_bowl_initial_v1"


def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-10:
        raise ValueError("quaternion norm is zero")
    return quat / norm


def quat_wxyz_to_rotation(quat: np.ndarray) -> Rotation:
    w, x, y, z = _normalize_quat_wxyz(quat)
    return Rotation.from_quat([x, y, z, w])


def rotation_to_quat_wxyz(rotation: Rotation) -> np.ndarray:
    x, y, z, w = rotation.as_quat()
    return np.asarray([w, x, y, z], dtype=np.float64)


def make_cup_to_bowl_task_frame(
    cup_pos_world: np.ndarray,
    bowl_pos_world: np.ndarray,
    *,
    min_horizontal_distance: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(task_frame_pose7_world, R_world_task)`` for one episode."""
    cup = np.asarray(cup_pos_world, dtype=np.float64).reshape(3)
    bowl = np.asarray(bowl_pos_world, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(cup)) or not np.all(np.isfinite(bowl)):
        raise ValueError("cup and bowl positions must be finite")

    x_axis = bowl - cup
    x_axis[2] = 0.0
    distance = float(np.linalg.norm(x_axis))
    if distance < float(min_horizontal_distance):
        raise ValueError(
            "initial cup-bowl horizontal distance is too small to define task +x "
            f"({distance:.3e} m)"
        )
    x_axis /= distance
    z_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    # Recompute x to remove accumulated numerical non-orthogonality.
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    rotation = np.column_stack([x_axis, y_axis, z_axis])
    quat = rotation_to_quat_wxyz(Rotation.from_matrix(rotation))
    pose = np.concatenate([bowl, quat])
    return pose.astype(np.float32), rotation


def world_positions_to_task(
    positions_world: np.ndarray,
    task_frame_world: np.ndarray,
) -> np.ndarray:
    pose = np.asarray(task_frame_world, dtype=np.float64).reshape(7)
    rotation = quat_wxyz_to_rotation(pose[3:7]).as_matrix()
    positions = np.asarray(positions_world, dtype=np.float64)
    return ((positions - pose[:3]) @ rotation).astype(np.float32)


def world_pose_series_to_task(
    poses_world: np.ndarray,
    task_frame_world: np.ndarray,
) -> np.ndarray:
    poses = np.asarray(poses_world, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError(f"poses_world must have shape [T, 7], got {poses.shape}")
    task = np.asarray(task_frame_world, dtype=np.float64).reshape(7)
    task_rotation = quat_wxyz_to_rotation(task[3:7])
    positions = world_positions_to_task(poses[:, :3], task)
    quaternions = np.asarray(
        [
            rotation_to_quat_wxyz(task_rotation.inv() * quat_wxyz_to_rotation(quat))
            for quat in poses[:, 3:7]
        ],
        dtype=np.float32,
    )
    return np.concatenate([positions, quaternions], axis=1).astype(np.float32)


def world_vectors_to_task(vectors_world: np.ndarray, task_frame_world: np.ndarray) -> np.ndarray:
    task = np.asarray(task_frame_world, dtype=np.float64).reshape(7)
    rotation = quat_wxyz_to_rotation(task[3:7]).as_matrix()
    vectors = np.asarray(vectors_world, dtype=np.float64)
    return (vectors @ rotation).astype(np.float32)


def pose_velocity_actions(poses: np.ndarray, dt: float) -> np.ndarray:
    """Finite-difference a pose series into linear/angular velocity actions."""
    pose = np.asarray(poses, dtype=np.float64)
    if pose.ndim != 2 or pose.shape[1] != 7:
        raise ValueError(f"poses must have shape [T, 7], got {pose.shape}")
    actions = np.zeros((len(pose), 6), dtype=np.float64)
    if len(pose) < 2:
        return actions.astype(np.float32)
    step = max(float(dt), 1e-8)
    actions[:-1, :3] = np.diff(pose[:, :3], axis=0) / step
    for index in range(len(pose) - 1):
        r0 = quat_wxyz_to_rotation(pose[index, 3:7])
        r1 = quat_wxyz_to_rotation(pose[index + 1, 3:7])
        actions[index, 3:6] = (r1 * r0.inv()).as_rotvec() / step
    actions[-1] = actions[-2]
    return actions.astype(np.float32)
