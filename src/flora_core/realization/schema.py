"""Versioned realization metadata view for old and new demonstrations.

This module never rewrites an episode.  It exposes the normalized v2 contract
for both legacy v1 metadata and newly collected v2 metadata.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


CHANNEL_TO_ID = {"c_left": 0, "c_right": 1, "c_over": 2}
# Scene-grounded realization layout: [p_o(3), p_g(3)].
SCENE_GROUNDED_DIM = 6
LEGACY_SOURCE_LEVELS = {"uo25": 0.25, "uo50": 0.50, "uo75": 0.75}
LEGACY_TARGET_LEVELS = {
    "sector_0": 0.0,
    "sector_120": 1.0 / 3.0,
    "sector_240": 2.0 / 3.0,
}
# Fixed waypoint count for the scene-instantiated channel polyline γ_c, matching
# ``channel_tokens`` so the realization plan tensor stays [B, 18, dim].
N_CORRIDOR_WAYPOINTS = 16


def realization_condition_from_id(realization_id: str) -> dict[str, np.ndarray | int]:
    parts = realization_id.split("_")
    if len(parts) != 5 or parts[0] != "M" or parts[2] != "sector":
        raise ValueError(f"invalid realization ID: {realization_id!r}")
    source_level = parts[1]
    target_level = f"sector_{parts[3]}"
    channel = f"c_{parts[4]}"
    if (source_level not in LEGACY_SOURCE_LEVELS
            or target_level not in LEGACY_TARGET_LEVELS
            or channel not in CHANNEL_TO_ID):
        raise ValueError(f"unknown realization ID: {realization_id!r}")
    return {
        "source_parameter": np.asarray([LEGACY_SOURCE_LEVELS[source_level]], dtype=np.float32),
        "target_parameter": np.asarray([LEGACY_TARGET_LEVELS[target_level]], dtype=np.float32),
        "channel_id": CHANNEL_TO_ID[channel],
    }


def realization_ids_for_topology(topology_class: str) -> list[str]:
    if topology_class not in CHANNEL_TO_ID:
        raise ValueError(f"unknown topology class: {topology_class!r}")
    channel = topology_class.removeprefix("c_")
    return [
        f"M_{source}_sector_{sector}_{channel}"
        for source in LEGACY_SOURCE_LEVELS
        for sector in ("0", "120", "240")
    ]


def canonicalize_realization_metadata(metadata: dict[str, Any]) -> dict[str, Any] | None:
    realization = metadata.get("realization")
    if realization is None:
        return None
    if "normalized_specification" in realization:
        unified = deepcopy(realization)
        raw_u_g = unified["normalized_specification"]["u_g"]
        if isinstance(raw_u_g, (list, tuple)):
            unified["normalized_specification"]["u_g"] = float(raw_u_g[1])
            unified["normalized_specification"]["legacy_2d_u_g_adapter"] = True
        return unified

    # Read-only legacy v1 adapter.  In v1 the normalized values and their
    # instantiated patches/path were nested together under affordance/corridor.
    affordance = realization.get("affordance", {})
    corridor = realization.get("corridor", {})
    scene = metadata.get("scene", {})
    topology = str(realization.get("topology_class", ""))
    source_level = str(affordance.get("source_patch_level", "legacy_unknown"))
    target_level = str(affordance.get("target_patch_level", "legacy_unknown"))
    u_o = affordance.get("u_o", LEGACY_SOURCE_LEVELS.get(source_level))
    raw_u_g = affordance.get("u_g", LEGACY_TARGET_LEVELS.get(target_level))
    u_g = raw_u_g[1] if isinstance(raw_u_g, (list, tuple)) else raw_u_g
    if u_o is None or u_g is None:
        raise ValueError(
            "legacy realization lacks recoverable normalized u_o/u_g values; "
            "keep it for trajectory-only use or provide an explicit mapping"
        )
    centerline = np.asarray(corridor.get("centerline_task", []), dtype=np.float64)
    task_distance = float(
        scene.get("source_target_initial_xy_distance", metadata.get("task_distance_m", 1.0))
    )
    normalized_centerline = (
        centerline / max(task_distance, 1e-9) if centerline.size else centerline
    )
    return {
        "realization_id": realization.get("realization_id"),
        "normalized_specification": {
            "schema": "normalized_realization_specification_v2_1d_target",
            "u_o": float(u_o),
            "u_o_level": source_level,
            "u_g": float(u_g),
            "u_g_level": target_level,
            "c": topology,
            "legacy_adapter": True,
        },
        "instantiated_geometry": {
            "schema": "scene_specific_realization_instantiation_v1",
            "source_geometry_id": scene.get("source_geometry_id"),
            "target_geometry_id": scene.get("target_geometry_id"),
            "p_o_source_local": affordance.get("source_anchor_local"),
            "p_o_task_initial": affordance.get("source_anchor_task_initial"),
            "p_g_task": affordance.get("target_guidance_center_task"),
            "interaction_target_task": affordance.get("target_anchor_task"),
            "gamma_c_task": centerline.tolist(),
            "gamma_c_normalized_by_task_distance": normalized_centerline.tolist(),
            "task_distance_m": task_distance,
            "visual_source_patch_local": affordance.get("source_guidance_patch_local", []),
            "visual_target_patch_task": affordance.get("target_guidance_patch_task", []),
            "corridor_radius_m": corridor.get("radius_m"),
            "legacy_adapter": True,
        },
        "validation": deepcopy(realization.get("validation")),
    }


def realization_network_condition(metadata: dict[str, Any]) -> dict[str, np.ndarray]:
    """Return only the normalized realization tensors consumed by the model."""
    realization = canonicalize_realization_metadata(metadata)
    if realization is None:
        raise ValueError("episode has no structured realization metadata")
    specification = realization["normalized_specification"]
    topology = str(specification["c"])
    if topology not in CHANNEL_TO_ID:
        raise ValueError(f"unknown normalized spatial channel: {topology}")
    return {
        "source_parameter": np.asarray([specification["u_o"]], dtype=np.float32),
        "target_parameter": np.asarray([specification["u_g"]], dtype=np.float32),
        "channel_id": np.asarray(CHANNEL_TO_ID[topology], dtype=np.int64),
    }


def resample_polyline(seq, target_len: int, feature_dim: int = 3) -> np.ndarray:
    """Resample an ordered polyline to ``target_len`` via per-channel linear interp.

    Corridors are already arc-length-uniform (0.0125 m spacing), so uniform-parameter
    resampling is valid.  An empty polyline yields zeros rather than erroring, so
    archive/test-input writers can degrade to a neutral channel.
    """
    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim == 1 and seq.size == 0:
        return np.zeros((target_len, feature_dim), dtype=np.float32)
    if seq.ndim == 1:
        seq = seq.reshape(-1, feature_dim)
    if len(seq) == target_len:
        return seq.copy()
    old_idx = np.linspace(0.0, 1.0, len(seq), dtype=np.float32)
    new_idx = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    return np.stack(
        [np.interp(new_idx, old_idx, seq[:, d]) for d in range(seq.shape[1])],
        axis=1,
    ).astype(np.float32)


def realization_instantiated_condition(
    metadata: dict[str, Any],
    n_corridor: int = N_CORRIDOR_WAYPOINTS,
) -> dict[str, np.ndarray]:
    """Return the scene-instantiated realization tensors Γ(s,m) = (p_o, p_g, γ_c).

    ``p_o``/``p_g`` are task-frame interaction positions; ``γ_c`` is the ordered
    corridor centerline resampled to ``n_corridor`` waypoints.  All three live in the
    same task frame as the archived point cloud / state.
    """
    realization = canonicalize_realization_metadata(metadata)
    if realization is None:
        raise ValueError("episode has no structured realization metadata")
    geometry = realization.get("instantiated_geometry")
    if geometry is None:
        raise ValueError("realization has no instantiated_geometry block")
    p_o = geometry.get("p_o_task_initial")
    p_g = geometry.get("p_g_task")
    gamma_c = geometry.get("gamma_c_task")
    if p_o is None or p_g is None or gamma_c is None:
        raise ValueError(
            "instantiated_geometry lacks p_o_task_initial/p_g_task/gamma_c_task"
        )
    return {
        "source_anchor_task": np.asarray(p_o, dtype=np.float32).reshape(3),
        "target_goal_task": np.asarray(p_g, dtype=np.float32).reshape(3),
        "corridor_waypoints": resample_polyline(gamma_c, int(n_corridor)),
    }


def _quat_wxyz_to_matrix(quaternion) -> np.ndarray:
    """Convert a [w,x,y,z] quaternion to a 3x3 rotation matrix."""
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape[-1] != 4:
        raise ValueError(f"expected a wxyz quaternion, got shape {q.shape}")
    return Rotation.from_quat(q[..., [1, 2, 3, 0]]).as_matrix()


def _realization_scalars(metadata: dict[str, Any]) -> tuple[float, float]:
    """Return ``(u_o, u_g)`` from any supported realization metadata layout."""
    realization = metadata.get("realization", {})
    if isinstance(realization, dict) and "normalized_specification" in realization:
        spec = realization["normalized_specification"]
        return float(spec["u_o"]), float(spec["u_g"])
    if isinstance(realization, dict) and "u_o" in realization and "u_g" in realization:
        return float(realization["u_o"]), float(realization["u_g"])
    affordance = realization.get("affordance", {}) if isinstance(realization, dict) else {}
    u_o = affordance.get("u_o")
    u_g = affordance.get("u_g")
    if u_o is None or u_g is None:
        raise ValueError("realization lacks recoverable u_o/u_g values")
    u_g = u_g[1] if isinstance(u_g, (list, tuple)) else u_g
    return float(u_o), float(u_g)


def _load_affordance_cache(scene: dict[str, Any], role: str) -> dict[str, Any]:
    cache = scene.get(f"{role}_affordance_cache")
    if cache is None:
        raise ValueError(f"scene lacks {role}_affordance_cache")
    if isinstance(cache, dict):
        return cache
    with open(str(cache), encoding="utf-8") as stream:
        return json.load(stream)


def _cup_task_pose(scene: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return the source cup's task-frame (position, rotation matrix)."""
    task_frame_world = np.asarray(scene["task_frame_world"], dtype=np.float64)
    task_rot = _quat_wxyz_to_matrix(task_frame_world[3:7])
    cup_pos_world = np.asarray(
        [scene["cup_xy"][0], scene["cup_xy"][1], scene["cup_z"]], dtype=np.float64
    )
    cup_pos_task = task_rot.T @ (cup_pos_world - task_frame_world[:3])
    cup_rot_world = _quat_wxyz_to_matrix(np.asarray(scene["cup_quat"], dtype=np.float64))
    return cup_pos_task, task_rot.T @ cup_rot_world


def source_outlet_local(u_o: float, scene: dict[str, Any]) -> np.ndarray:
    """Return the requested ``u_o`` rim point in the source cup's local frame.

    This is the source outlet expressed in cup-local coordinates (the rim point
    that becomes the pour outlet at ``u_o``).  It deliberately excludes the cup
    task-frame pose so callers can transform it by any pose, e.g. a predicted
    trajectory waypoint.
    """
    cache = _load_affordance_cache(scene, "source")
    source = cache["source"]
    parameterization = str(source.get("parameterization", "normalized_rim_arc_length"))
    if parameterization == "normalized_preferred_pour_arc":
        center = np.deg2rad(float(source["pour_arc_center_deg_local"]))
        span = np.deg2rad(float(source["pour_arc_span_deg"]))
        theta = center + (u_o - 0.5) * span
    else:
        theta = 2.0 * np.pi * u_o + np.deg2rad(float(source.get("angle_offset_deg", 0.0)))
    rim_center = np.asarray(
        source.get("rim_center_local_m", [0.0, 0.0, 0.0]), dtype=np.float64
    )
    source_local = np.asarray(
        [
            float(source["rim_radius_m"]) * np.cos(theta),
            float(source["rim_radius_m"]) * np.sin(theta),
            float(source["lip_z_m"]),
        ],
        dtype=np.float64,
    )
    source_local[:2] += rim_center[:2]
    return source_local


def source_interaction_patch_local(
    u_o: float, scene: dict[str, Any]
) -> np.ndarray:
    """Instantiate ``u_o`` as the frozen 64-point source-local region.

    This is the pure ``Gamma_o(G_o, u_o)`` front-end used by the core grounded
    source bottleneck.  It depends only on the verified source affordance cache
    and the requested scalar.  It intentionally does not read a trajectory,
    future state, liquid result, or evaluator output.
    """
    cache = _load_affordance_cache(scene, "source")
    source = cache["source"]
    allowed = source.get("allowed_intervals_u", [[0.0, 1.0]])
    if not any(float(low) <= float(u_o) <= float(high) for low, high in allowed):
        raise ValueError(f"u_o={u_o:.4f} is outside the verified source affordance intervals")
    parameterization = str(source.get("parameterization", "normalized_rim_arc_length"))
    if parameterization == "normalized_preferred_pour_arc":
        center = np.deg2rad(float(source["pour_arc_center_deg_local"]))
        span = np.deg2rad(float(source["pour_arc_span_deg"]))
        theta = center + (float(u_o) - 0.5) * span
    else:
        theta = 2.0 * np.pi * float(u_o) + np.deg2rad(
            float(source.get("angle_offset_deg", 0.0))
        )
    rim_center = np.asarray(
        source.get("rim_center_local_m", [0.0, 0.0, 0.0]), dtype=np.float64
    )
    arc_axis = np.linspace(-0.25, 0.25, 8)
    height_axis = np.linspace(-0.012, 0.002, 8)
    arc_offset, height_offset = np.meshgrid(arc_axis, height_axis)
    arc_offset = arc_offset.reshape(-1)
    height_offset = height_offset.reshape(-1)
    radial_offset = 0.0015 * (height_offset - height_offset.max()) / max(
        float(np.ptp(height_offset)), 1e-9
    )
    patch = np.column_stack(
        (
            rim_center[0]
            + (float(source["rim_radius_m"]) + radial_offset)
            * np.cos(theta + arc_offset),
            rim_center[1]
            + (float(source["rim_radius_m"]) + radial_offset)
            * np.sin(theta + arc_offset),
            float(source["lip_z_m"]) + height_offset,
        )
    ).astype(np.float32)
    if patch.shape != (64, 3) or not np.isfinite(patch).all():
        raise ValueError(f"invalid source interaction patch: {patch.shape}")
    return patch


def _source_anchor_task(u_o: float, scene: dict[str, Any]) -> np.ndarray:
    """Instantiate the source outlet coordinate ``u_o`` as a task-frame position."""
    cup_pos_task, cup_rot_task = _cup_task_pose(scene)
    return cup_pos_task + cup_rot_task @ source_outlet_local(u_o, scene)


def _target_guidance_task(u_g: float, scene: dict[str, Any]) -> np.ndarray:
    """Instantiate the target cyclic coordinate ``u_g`` as a task-frame position."""
    cache = _load_affordance_cache(scene, "target")
    target = cache["target"]
    beta = 2.0 * np.pi * u_g + np.deg2rad(float(target.get("angle_offset_deg", 0.0)))
    radius = float(target["safe_opening_radius_m"]) * float(
        np.clip(float(target.get("default_radial_u", 0.60)), 0.0, 1.0)
    )
    z = float(target.get("receive_floor_z_task_m", target["opening_z_task_m"]))
    return np.asarray(
        [radius * np.cos(beta), radius * np.sin(beta), z], dtype=np.float64
    )


def scene_grounded_geometry(metadata: dict[str, Any]) -> np.ndarray:
    """Return the scene-instantiated B2 realization geometry.

    Layout ``[p_o(3), p_g(3)]``.  ``p_o``/``p_g`` are *recomputed from the
    current* ``u_o``/``u_g`` using the scene's verified affordance cache, so an
    intervention on either factor moves its concrete task-frame interaction
    position.  The geometry therefore encodes ``Γ(s, m)`` for the requested
    ``m``, not the archived natural realization.  Object orientation is
    deliberately excluded in this first version per the core spec.
    """
    scene = metadata["scene"]
    u_o, u_g = _realization_scalars(metadata)
    p_o = _source_anchor_task(u_o, scene)
    p_g = _target_guidance_task(u_g, scene)
    geometry = np.concatenate((p_o, p_g)).astype(np.float32)
    if geometry.shape != (SCENE_GROUNDED_DIM,):
        raise ValueError(
            f"scene-grounded geometry is {geometry.shape}, expected {SCENE_GROUNDED_DIM}"
        )
    return geometry


def source_outlet_local_from_metadata(metadata: dict[str, Any]) -> np.ndarray:
    """Return the requested u_o's cup-local rim point from full metadata."""
    u_o, _ = _realization_scalars(metadata)
    return source_outlet_local(u_o, metadata["scene"]).astype(np.float32)


def interaction_orientation_anchor(
    u_o: float, u_g: float, scene: dict[str, Any]
) -> dict[str, Any]:
    """Return the analytic orientation anchor for a requested realization (u_o, u_g).

    The anchor fixes the two *directional* DoFs of the interaction orientation:

    - ``u_o`` selects which cup-local rim region is the active outlet; together
      with ``u_g`` it pins the yaw/alignment ``phi*`` so the outlet aligns with
      the task-frame pour direction.
    - ``u_g`` pins the tilt axis ``a_tilt* = perp(d_g*)`` (the same axis the
      evaluator's ``expected_axis`` is derived from).

    Tilt *magnitude* and translation are left to the network.  The returned dict
    carries everything the model needs to structurally rebuild ``R*[k*]`` from a
    predicted tilt magnitude:

      rim_local [3]        cup-local outlet rim point r_o = r_rim(u_o)
      tilt_axis [3]        task-frame tilt axis a_tilt* = [-sin beta, cos beta, 0]
      yaw float            phi* = atan2(d_g_y, d_g_x) - atan2(dir_init_y, dir_init_x)
      initial_rotation     R_initial = _cup_task_pose(scene)[1] (cup asset -> task)

    Geometry matches the gate test ``construct_anchor_orientation`` exactly.
    """
    r_o = source_outlet_local(u_o, scene)
    source = _load_affordance_cache(scene, "source")["source"]
    rim_center = np.asarray(
        source.get("rim_center_local_m", [0.0, 0.0, 0.0]), dtype=np.float64
    )
    dir_o = r_o[:2] - rim_center[:2]
    dir_o = dir_o / max(float(np.linalg.norm(dir_o)), 1e-12)

    target = _load_affordance_cache(scene, "target")["target"]
    beta = 2.0 * np.pi * u_g + np.deg2rad(float(target.get("angle_offset_deg", 0.0)))
    d_g = np.asarray([np.cos(beta), np.sin(beta), 0.0], dtype=np.float64)
    a_tilt = np.asarray([-np.sin(beta), np.cos(beta), 0.0], dtype=np.float64)

    R_initial = _cup_task_pose(scene)[1]
    dir_init = (R_initial @ np.asarray([dir_o[0], dir_o[1], 0.0], dtype=np.float64))[:2]
    dir_init = dir_init / max(float(np.linalg.norm(dir_init)), 1e-12)
    phi = math.atan2(d_g[1], d_g[0]) - math.atan2(dir_init[1], dir_init[0])

    return {
        "rim_local": np.asarray(r_o, dtype=np.float32),
        "tilt_axis": a_tilt.astype(np.float32),
        "yaw": float(phi),
        "initial_rotation": np.asarray(R_initial, dtype=np.float32),
    }


def interaction_orientation_anchor_from_metadata(
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Return the orientation anchor for the realization in full metadata."""
    u_o, u_g = _realization_scalars(metadata)
    return interaction_orientation_anchor(u_o, u_g, metadata["scene"])
