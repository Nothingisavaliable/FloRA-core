"""Runtime contract between evaluated global motion and local closed-loop control.

The upper layer evaluates one nominal global task-space motion. The local
policy may adapt continuously around that motion, but the evaluated reference
remains committed until the structural envelope is violated. A violation
invalidates the current evaluation and requests a new global prediction and
embodiment evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def _as_float_array(value: Sequence[float] | np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Convert row-major Rot6D (first two matrix rows) to a rotation matrix."""
    rot6d = np.asarray(rot6d, dtype=np.float64)
    if rot6d.shape[-1] != 6:
        raise ValueError("rot6d must end in dimension 6")
    first = rot6d[..., 0:3]
    second_raw = rot6d[..., 3:6]
    first = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-12)
    second = second_raw - np.sum(first * second_raw, axis=-1, keepdims=True) * first
    second = second / np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-12)
    third = np.cross(first, second)
    return np.stack((first, second, third), axis=-2)


def _quat_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    return Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()


def _matrix_geodesic_rad(a: np.ndarray, b: np.ndarray) -> float:
    relative = a @ b.T
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(cosine))


@dataclass(frozen=True)
class EvaluatedGlobalPlan:
    """One global motion that has already passed embodiment evaluation."""

    trajectory: np.ndarray
    realization_id: str
    evaluation_id: str = ""

    def __post_init__(self) -> None:
        trajectory = np.asarray(self.trajectory, dtype=np.float64)
        if trajectory.ndim != 2 or trajectory.shape[1] != 9:
            raise ValueError(
                f"trajectory must be [L,9] position+Rot6D, got {trajectory.shape}"
            )
        if trajectory.shape[0] < 2:
            raise ValueError("trajectory requires at least two waypoints")
        if not np.isfinite(trajectory).all():
            raise ValueError("trajectory must be finite")
        object.__setattr__(self, "trajectory", trajectory.copy())
        if not self.realization_id:
            raise ValueError("realization_id must be non-empty")


@dataclass(frozen=True)
class StructuralEnvelope:
    """Allowed local deformation around the evaluated global motion."""

    max_position_error_m: float = 0.05
    max_orientation_error_rad: float = np.deg2rad(20.0)
    progress_search_radius: int = 2

    def __post_init__(self) -> None:
        if self.max_position_error_m <= 0:
            raise ValueError("max_position_error_m must be positive")
        if self.max_orientation_error_rad <= 0:
            raise ValueError("max_orientation_error_rad must be positive")
        if self.progress_search_radius < 0:
            raise ValueError("progress_search_radius must be non-negative")


@dataclass(frozen=True)
class ConsistencyStatus:
    consistent: bool
    needs_reevaluation: bool
    realization_match: bool
    reference_index: int
    position_error_m: float
    orientation_error_rad: float
    reasons: tuple[str, ...]


class GlobalLocalContract:
    """Stateful execution contract for one committed evaluated global motion."""

    def __init__(self, envelope: StructuralEnvelope = StructuralEnvelope()) -> None:
        self.envelope = envelope
        self._plan: EvaluatedGlobalPlan | None = None
        self._invalidated_reason: str | None = None

    @property
    def plan(self) -> EvaluatedGlobalPlan | None:
        return self._plan

    @property
    def has_valid_plan(self) -> bool:
        return self._plan is not None and self._invalidated_reason is None

    def commit(self, plan: EvaluatedGlobalPlan) -> None:
        """Commit a plan only after the upper-layer evaluator accepts it."""
        self._plan = plan
        self._invalidated_reason = None

    def invalidate(self, reason: str) -> None:
        if not reason:
            raise ValueError("invalidation reason must be non-empty")
        if self._plan is None:
            raise RuntimeError("cannot invalidate before a plan is committed")
        self._invalidated_reason = reason

    def clear(self) -> None:
        self._plan = None
        self._invalidated_reason = None

    def check(
        self,
        current_pose7: Sequence[float] | np.ndarray,
        progress: float,
        realization_id: str,
    ) -> ConsistencyStatus:
        """Check whether local execution remains inside the evaluated structure."""
        if self._plan is None:
            raise RuntimeError("no evaluated global plan is committed")

        if self._invalidated_reason is not None:
            return ConsistencyStatus(
                consistent=False,
                needs_reevaluation=True,
                realization_match=False,
                reference_index=-1,
                position_error_m=float("inf"),
                orientation_error_rad=float("inf"),
                reasons=(self._invalidated_reason,),
            )

        pose = _as_float_array(current_pose7, (7,), "current_pose7")
        progress = float(np.clip(progress, 0.0, 1.0))
        realization_match = realization_id == self._plan.realization_id
        reasons: list[str] = []
        if not realization_match:
            reasons.append("realization_changed")

        trajectory = self._plan.trajectory
        length = trajectory.shape[0]
        nominal = int(round(progress * (length - 1)))
        low = max(0, nominal - self.envelope.progress_search_radius)
        high = min(length - 1, nominal + self.envelope.progress_search_radius)
        candidate_indices = range(low, high + 1)

        current_position = pose[:3]
        current_rotation = _quat_wxyz_to_matrix(pose[3:7])
        reference_rotations = _rot6d_to_matrix(trajectory[:, 3:9])

        best_index = nominal
        best_cost = float("inf")
        best_position_error = float("inf")
        best_orientation_error = float("inf")
        for index in candidate_indices:
            position_error = float(
                np.linalg.norm(current_position - trajectory[index, :3])
            )
            orientation_error = _matrix_geodesic_rad(
                current_rotation, reference_rotations[index]
            )
            cost = (
                position_error / self.envelope.max_position_error_m
                + orientation_error / self.envelope.max_orientation_error_rad
            )
            if cost < best_cost:
                best_cost = cost
                best_index = index
                best_position_error = position_error
                best_orientation_error = orientation_error

        if best_position_error > self.envelope.max_position_error_m:
            reasons.append("position_envelope_exceeded")
        if best_orientation_error > self.envelope.max_orientation_error_rad:
            reasons.append("orientation_envelope_exceeded")

        consistent = len(reasons) == 0
        return ConsistencyStatus(
            consistent=consistent,
            needs_reevaluation=not consistent,
            realization_match=realization_match,
            reference_index=best_index,
            position_error_m=best_position_error,
            orientation_error_rad=best_orientation_error,
            reasons=tuple(reasons),
        )


def interpolate_global_reference(
    trajectory: np.ndarray,
    progress: float,
) -> np.ndarray:
    """Interpolate one task-frame pose7 reference from an [L,9] global motion."""
    trajectory = np.asarray(trajectory, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != 9:
        raise ValueError("trajectory must be [L,9]")
    progress = float(np.clip(progress, 0.0, 1.0))
    x = progress * (trajectory.shape[0] - 1)
    low = int(np.floor(x))
    high = min(low + 1, trajectory.shape[0] - 1)
    alpha = x - low

    position = (1.0 - alpha) * trajectory[low, :3] + alpha * trajectory[high, :3]
    matrices = _rot6d_to_matrix(trajectory[:, 3:9])
    if high == low:
        matrix = matrices[low]
    else:
        rotations = Rotation.from_matrix(np.stack((matrices[low], matrices[high])))
        matrix = Slerp([0.0, 1.0], rotations)([alpha]).as_matrix()[0]
    quaternion_xyzw = Rotation.from_matrix(matrix).as_quat()
    quaternion_wxyz = quaternion_xyzw[[3, 0, 1, 2]]
    return np.concatenate((position, quaternion_wxyz))
