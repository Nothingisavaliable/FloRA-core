"""core task-space realization fidelity, independent of the teacher label.

The requested specification and scene affordance define reference geometry;
observed factor values are extracted from the *actual* SE(3) trajectory.
This evaluates kinematic realization, not true liquid outlet or task success.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from flora_core.realization.collection import classify_topology
from flora_core.realization.schema import (
    canonicalize_realization_metadata,
    source_outlet_local,
    _load_affordance_cache,
    scene_grounded_geometry,
)


@dataclass(frozen=True)
class EvaluatorConfig:
    outlet_tolerance_deg: float = 12.0
    axis_tolerance_deg: float = 15.0
    # Require the requested local outlet direction and global tilt axis to be
    # sustained over the active pour, rather than accepting a brief crossing.
    angular_min_time_fraction: float = 0.8
    min_tilt_deg: float = 15.0
    min_pour_frames: int = 3
    # Avoid using the 115-degree terminal hold to infer the rotation axis.
    # Estimate the direction from the pre-pour pose to the first sufficiently
    # tilted pour pose, and check the last orientation separately.
    max_pour_tilt_deg: float = 75.0
    # Explicit opt-in: do not silently change the frozen angular-only metric.
    # Calibrate on held-out data / physical target geometry before enabling.
    target_xy_tolerance_m: float | None = None
    # Whole-pour spatial consistency. Explicitly opt in with both parameters.
    # Spatial radius is relative to target safe-opening radius.
    target_region_radius_fraction: float | None = None
    target_region_min_time_fraction: float | None = None


def _rotations(poses: np.ndarray) -> Rotation:
    quat = np.asarray(poses[:, 3:7], dtype=np.float64)
    norms = np.linalg.norm(quat, axis=1)
    if np.any(norms < 1e-8):
        raise ValueError("zero quaternion in trajectory")
    return Rotation.from_quat(quat[:, [1, 2, 3, 0]] / norms[:, None])


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) /
        (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))))


def _wrap_deg(value: float) -> float:
    return float((value + 180.0) % 360.0 - 180.0)


def evaluate_realization(
    poses_task: np.ndarray,
    phase_id: np.ndarray,
    metadata: dict[str, Any],
    config: EvaluatorConfig = EvaluatorConfig(),
) -> dict[str, Any]:
    """Compare observed geometric factors to the requested realization.

    'u_o' refers to the cup-local low-side rim direction implied by pour
    kinematics. Without liquid observations this is *not* a measured spout.
    The signed global tilt axis determines u_g. The pre-pour local axis
    determines u_o. Topology uses the existing first-gate-crossing definition.
    """
    poses = np.asarray(poses_task, dtype=np.float64)
    phase = np.asarray(phase_id)
    if poses.ndim != 2 or poses.shape[1] != 7 or phase.shape != (len(poses),):
        raise ValueError("expected poses [T,7] and phase_id [T]")
    if not np.isfinite(poses).all():
        raise ValueError("non-finite poses")
    move = np.flatnonzero(phase == 0)
    pour = np.flatnonzero(phase == 1)
    # Channel fidelity remains observable even if the pour is missing.
    # An absent obstacle or an unusable move segment makes c unobservable.
    scene = metadata["scene"]
    realization = canonicalize_realization_metadata(metadata)
    if realization is None:
        raise ValueError("missing realization metadata")
    specification = realization["normalized_specification"]
    obstacles = scene.get("obstacles", [])
    requested_c = str(specification["c"])
    topology = (
        classify_topology(poses[move, :3], obstacles)["observed_topology_class"]
        if len(move) >= 2 and obstacles else None
    )
    c_pass = (topology == requested_c) if topology is not None else None
    if len(move) < 2 or len(pour) < config.min_pour_frames:
        return {
            "valid": False, "realization_success": False,
            "reason": "insufficient move/pour phase",
            "factor_success": {"u_o": None, "u_g": None, "c": c_pass},
            "observed_topology": topology, "requested_topology": requested_c,
        }
    rotation = _rotations(poses)
    r_pre = rotation[int(move[-1])]
    r_end = rotation[int(pour[-1])]
    # Evaluate every well-defined active-pour rotation.  Restricting the window
    # to 15-75 degrees avoids noisy near-zero axes and the terminal hold while
    # preserving an unambiguous signed axis.
    active_axes = []
    for idx in pour:
        rv = (rotation[int(idx)] * r_pre.inv()).as_rotvec()
        angle = np.degrees(np.linalg.norm(rv))
        if config.min_tilt_deg <= angle <= config.max_pour_tilt_deg:
            active_axes.append(rv / np.linalg.norm(rv))
    final_tilt = float(np.degrees(
        (r_end * r_pre.inv()).magnitude()))
    if not active_axes:
        return {
            "valid": False, "realization_success": False,
            "reason": "no valid rotation segment", "tilt_deg": final_tilt,
            "factor_success": {"u_o": None, "u_g": None, "c": c_pass},
            "observed_topology": topology, "requested_topology": requested_c,
        }
    min_time_fraction = float(config.angular_min_time_fraction)
    if not 0.0 <= min_time_fraction <= 1.0:
        raise ValueError("angular_min_time_fraction must be in [0,1]")
    axes_task = np.asarray(active_axes, dtype=np.float64)
    axes_local = r_pre.inv().apply(axes_task)
    # Keep the earliest active frame as the representative observation for
    # backward-compatible scalar diagnostics. Factor success is determined by
    # the temporal pass fractions below.
    axis_task = axes_task[0]
    axis_local = axes_local[0]
    # The teacher rotates around local [-sin(theta), cos(theta), 0].
    theta_observed = math.degrees(math.atan2(-axis_local[0], axis_local[1]))
    source = _load_affordance_cache(scene, "source")["source"]
    center = np.asarray(source.get("rim_center_local_m", [0, 0, 0]), dtype=float)
    outlet = source_outlet_local(float(specification["u_o"]), scene)
    relative = outlet[:2] - center[:2]
    theta_expected = math.degrees(math.atan2(relative[1], relative[0]))
    outlet_error = abs(_wrap_deg(theta_observed - theta_expected))
    theta_observed_all = np.degrees(np.arctan2(-axes_local[:, 0], axes_local[:, 1]))
    outlet_errors = np.abs(
        (theta_observed_all - theta_expected + 180.0) % 360.0 - 180.0
    )
    target = _load_affordance_cache(scene, "target")["target"]
    beta = 2.0 * np.pi * float(specification["u_g"]) + math.radians(
        float(target.get("angle_offset_deg", 0.0)))
    # Signed axis verified on teacher trajectories (positive tilt convention).
    expected_axis = np.array([math.sin(beta), -math.cos(beta), 0.0])
    axis_error = _angle_deg(axis_task, expected_axis)
    axis_errors = np.asarray(
        [_angle_deg(axis, expected_axis) for axis in axes_task], dtype=np.float64
    )
    outlet_pass_fraction = float(np.mean(outlet_errors <= config.outlet_tolerance_deg))
    axis_pass_fraction = float(np.mean(axis_errors <= config.axis_tolerance_deg))
    # Spatial proxy: actual designated cup-rim point relative to the
    # scene-grounded target interaction position, both in the task frame.
    # This is not a particle impact point. Use continuous errors for threshold
    # calibration rather than silently imposing an arbitrary distance cutoff.
    target_goal = scene_grounded_geometry(metadata)[3:6]
    rim_world = poses[pour, :3] + rotation[pour].apply(
        np.ascontiguousarray(np.broadcast_to(outlet, (len(pour), 3)))
    )
    xy_distances = np.linalg.norm(rim_world[:, :2] - target_goal[None, :2], axis=1)
    target_xy_min = float(np.min(xy_distances))
    target_xy_median = float(np.median(xy_distances))
    # Whole-pour metric: a short visit cannot satisfy the time requirement.
    # The specified target point is the center of a candidate local interaction
    # region, not a measured impact location.
    if (config.target_region_radius_fraction is None) != (
        config.target_region_min_time_fraction is None
    ):
        raise ValueError("Specify both spatial region fraction and time fraction")
    region_radius = None
    region_time_fraction = None
    region_pass = None
    if config.target_region_radius_fraction is not None:
        radius_fraction = float(config.target_region_radius_fraction)
        min_fraction = float(config.target_region_min_time_fraction)
        if not (0 < radius_fraction <= 1) or not (0 <= min_fraction <= 1):
            raise ValueError("region radius fraction must be in (0,1] and time fraction in [0,1]")
        safe_radius = float(target["safe_opening_radius_m"])
        if not np.isfinite(safe_radius) or safe_radius <= 0:
            raise ValueError("target safe opening radius must be positive")
        region_radius = safe_radius * radius_fraction
        region_time_fraction = float(np.mean(xy_distances <= region_radius))
        region_pass = bool(region_time_fraction >= min_fraction)
    spatial_pass = (
        None if config.target_xy_tolerance_m is None
        else bool(target_xy_min <= config.target_xy_tolerance_m)
    )
    if region_pass is not None:
        spatial_pass = region_pass if spatial_pass is None else (spatial_pass and region_pass)
    passes = {
        "u_o": outlet_pass_fraction >= min_time_fraction,
        "u_g": (axis_pass_fraction >= min_time_fraction) and (spatial_pass is not False),
        "c": c_pass,
    }
    return {
        "valid": True,
        "realization_success": bool(all(passes.values())),
        "factor_success": {k: (bool(v) if v is not None else None) for k, v in passes.items()},
        "outlet_error_deg": float(outlet_error),
        "angular_valid_frame_count": int(len(axes_task)),
        "angular_min_time_fraction": min_time_fraction,
        "outlet_angle_pass_fraction": outlet_pass_fraction,
        "global_axis_pass_fraction": axis_pass_fraction,
        "outlet_error_median_deg": float(np.median(outlet_errors)),
        "rotation_axis_error_median_deg": float(np.median(axis_errors)),
        "target_xy_distance_min_m": target_xy_min,
        "target_xy_distance_median_m": target_xy_median,
        "target_xy_distance_mean_m": float(np.mean(xy_distances)),
        "target_xy_distance_p90_m": float(np.quantile(xy_distances, .90)),
        "target_region_radius_m": region_radius,
        "target_region_time_fraction": region_time_fraction,
        "target_region_pass": region_pass,
        "target_xy_tolerance_m": config.target_xy_tolerance_m,
        "target_xy_pass": spatial_pass,
        "rotation_axis_error_deg": float(axis_error),
        "observed_outlet_deg": float(theta_observed),
        "expected_outlet_deg": float(theta_expected),
        "observed_axis_task": axis_task.tolist(),
        "expected_axis_task": expected_axis.tolist(),
        "tilt_deg": final_tilt,
        "observed_topology": topology,
        "requested_topology": requested_c,
    }
