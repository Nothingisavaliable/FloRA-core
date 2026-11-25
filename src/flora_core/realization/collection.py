"""Scene--Realization support for task-space pouring data collection.

This module deliberately contains no robot model.  It resolves a sparse
Scene--Realization protocol in the fixed cup-to-bowl task frame, instantiates
static task obstacles, maps normalized affordance coordinates to object-local
anchors, and plans a topology-constrained positional corridor with a small PRM.
"""


from __future__ import annotations


import heapq
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from flora_core.geometry.frames import (
    FRAME_CONVENTION,
    make_cup_to_bowl_task_frame,
    quat_wxyz_to_rotation,
    rotation_to_quat_wxyz,
)
from flora_core.libero_utils import find_libero_root


TOPOLOGIES = ("c_left", "c_right", "c_over")


GUIDANCE_GROUP = 4


def _read_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _resolve_config_path(base: Path, value: str) -> str:
    if value.startswith("libero://"):
        relative = value.removeprefix("libero://")
        return str((find_libero_root() / relative).resolve())
    candidate = Path(value)
    return str(candidate if candidate.is_absolute() else (base / candidate).resolve())


def _geometry_asset_id(raw: dict[str, Any], scene: dict[str, Any], role: str) -> tuple[str, str]:
    factor_key = "source_geometry_id" if role == "source" else "target_geometry_id"
    registry_key = "source_geometries" if role == "source" else "target_geometries"
    geometry_level = scene.get(factor_key)
    if geometry_level is None:
        legacy = str(scene["geometry_id"])
        return legacy, legacy
    geometry_level = str(geometry_level)
    registry = raw.get(registry_key, {})
    if geometry_level not in registry:
        raise KeyError(f"unknown {factor_key} {geometry_level!r}")
    return geometry_level, str(registry[geometry_level]["asset_id"])


def _compose_geometry_bundle(raw: dict[str, Any], base: Path, scene: dict[str, Any]):
    source_level, source_asset_id = _geometry_asset_id(raw, scene, "source")
    target_level, target_asset_id = _geometry_asset_id(raw, scene, "target")
    assets = raw.get("assets", {})
    if source_asset_id not in assets or target_asset_id not in assets:
        raise KeyError(f"unknown source/target asset: {source_asset_id!r}, {target_asset_id!r}")
    source_asset = dict(assets[source_asset_id])
    target_asset = dict(assets[target_asset_id])
    source_cache_path = _resolve_config_path(base, str(source_asset["affordance_cache"]))
    target_cache_path = _resolve_config_path(base, str(target_asset["affordance_cache"]))
    source_cache = _read_json(Path(source_cache_path))
    target_cache = _read_json(Path(target_cache_path))
    asset = {
        "source_geometry_id": source_level,
        "target_geometry_id": target_level,
        "source_asset_id": source_asset_id,
        "target_asset_id": target_asset_id,
        "source_xml": _resolve_config_path(base, str(source_asset["source_xml"])),
        "target_xml": _resolve_config_path(base, str(target_asset["target_xml"])),
        "source_scale_xyz": list(source_asset.get("source_scale_xyz", [1.0] * 3)),
        "target_scale_xyz": list(target_asset.get("target_scale_xyz", [1.0] * 3)),
        "source_nominal_quat_wxyz": list(source_asset.get("source_nominal_quat_wxyz", [1.0, 0.0, 0.0, 0.0])),
        "source_affordance_cache": source_cache_path,
        "target_affordance_cache": target_cache_path,
        "source_liquid_cavity": dict(source_asset.get("source_liquid_cavity", {})),
        "target_liquid_cavity": dict(target_asset.get("target_liquid_cavity", {})),
    }
    affordance = {
        "source": dict(source_cache["source"]),
        "target": dict(target_cache["target"]),
        "source_method": source_cache.get("source_method", "geometry_verified"),
        "target_method": target_cache.get("target_method", "geometry_verified"),
        "review": {"source": source_cache.get("review", {}), "target": target_cache.get("review", {})},
    }
    return asset, affordance


def load_collection_row(
    protocol_path: Path,
    design_path: Path,
    collection_row_id: str,
) -> dict[str, Any]:
    """Instantiate exactly one selected covering-array row."""
    raw = _read_json(protocol_path)
    design = _read_json(design_path)
    rows = {
        str(item["collection_row_id"]): item
        for item in design.get("collection_rows", [])
        if str(item.get("status", "collect")) in {"collect", "evaluate"}
    }
    if collection_row_id not in rows:
        raise KeyError(
            f"unknown collection_row_id {collection_row_id!r}; choices={sorted(rows)}"
        )
    selected = rows[collection_row_id]
    scene = {"scene_id": str(selected["scene_id"]), **dict(selected["scene"])}
    realization = {
        "realization_id": str(selected["realization_id"]),
        **dict(selected["realization"]),
    }
    layout_id = str(scene["obstacle_layout_id"])
    layouts = raw.get("obstacle_layouts", {})
    base = protocol_path.resolve().parent
    if layout_id not in layouts:
        raise KeyError(f"collection row references unknown obstacle layout: {layout_id}")
    asset, cache = _compose_geometry_bundle(raw, base, scene)
    topology = str(realization["topology_class"])
    valid_topologies = list(layouts[layout_id].get("valid_topologies", TOPOLOGIES))
    combination_valid = topology in valid_topologies
    return {
        "schema": str(raw.get("schema", "scene_realization_collection_v1")),
        "protocol_path": str(protocol_path.resolve()),
        "collection_design_path": str(design_path.resolve()),
        "collection_row_id": collection_row_id,
        "scene": scene,
        "realization": realization,
        "asset": asset,
        "affordance": cache,
        "obstacle_layout": dict(layouts[layout_id]),
        "combination_valid": combination_valid,
        "invalid_reason": "" if combination_valid else f"{topology} unavailable in {layout_id}",
    }



def apply_scene_factors(
    sampled_scene: dict[str, Any], protocol: dict[str, Any], distance_axis_world_xy: np.ndarray
) -> dict[str, Any]:
    """Apply configured d and psi while preserving the collector's world placement."""
    out = dict(sampled_scene)
    factor = protocol["scene"]
    bowl_xy = np.asarray(out["bowl_xy"], dtype=np.float64)
    axis = np.asarray(distance_axis_world_xy, dtype=np.float64).reshape(2)
    axis /= max(float(np.linalg.norm(axis)), 1e-9)
    distance = float(factor["distance_m"])
    out["cup_xy"] = (bowl_xy - distance * axis).tolist()
    out["source_target_initial_xy_distance"] = distance

    # psi is a yaw offset relative to the registered asset's nominal pose.
    nominal = np.asarray(protocol["asset"].get("source_nominal_quat_wxyz", out["cup_quat"]), dtype=np.float64)
    psi = math.radians(float(factor.get("orientation_deg", 0.0)))
    q = Rotation.from_rotvec(np.array([0.0, 0.0, psi])) * quat_wxyz_to_rotation(nominal)
    out["cup_quat"] = rotation_to_quat_wxyz(q).tolist()
    out.update(
        {
            "scene_id": factor["scene_id"],
            "distance_level": factor.get("distance_level", "custom"),
            "orientation_level": factor.get("orientation_level", "custom"),
            "orientation_task_deg": float(factor.get("orientation_deg", 0.0)),
            "obstacle_layout_id": factor["obstacle_layout_id"],
            "source_geometry_id": protocol["asset"]["source_geometry_id"],
            "target_geometry_id": protocol["asset"]["target_geometry_id"],
            "source_asset_id": protocol["asset"]["source_asset_id"],
            "target_asset_id": protocol["asset"]["target_asset_id"],
            "source_asset_xml": protocol["asset"]["source_xml"],
            "target_asset_xml": protocol["asset"]["target_xml"],
            "source_scale_xyz": list(protocol["asset"].get("source_scale_xyz", [1.0, 1.0, 1.0])),
            "target_scale_xyz": list(protocol["asset"].get("target_scale_xyz", [1.0, 1.0, 1.0])),
            "source_affordance_cache": protocol["asset"]["source_affordance_cache"],
            "target_affordance_cache": protocol["asset"]["target_affordance_cache"],
            "realization_id": protocol["realization"]["realization_id"],
            "topology_class": protocol["realization"]["topology_class"],
            "task_frame_convention": FRAME_CONVENTION,
            "combination_valid": bool(protocol["combination_valid"]),
            "invalid_reason": protocol["invalid_reason"],
        }
    )
    if "geometry_id" in factor:
        out["geometry_id"] = factor["geometry_id"]
    if protocol.get("collection_row_id"):
        out["collection_row_id"] = str(protocol["collection_row_id"])
        out["collection_design_path"] = str(protocol["collection_design_path"])
    return out


def _table_top_world_z(args: Any) -> float:
    if bool(getattr(args, "libero_table", True)):
        return 0.0 if bool(getattr(args, "shift_libero_table_to_zero", False)) else 0.9
    return 0.0


def resolve_obstacles(
    protocol: dict[str, Any], cup_pos_world: np.ndarray, bowl_pos_world: np.ndarray, args: Any
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    task_frame, rotation = make_cup_to_bowl_task_frame(cup_pos_world, bowl_pos_world)
    table_z_task = float(_table_top_world_z(args) - task_frame[2])
    distance = float(np.linalg.norm(np.asarray(cup_pos_world[:2]) - np.asarray(bowl_pos_world[:2])))
    resolved: list[dict[str, Any]] = []
    for index, raw in enumerate(protocol["obstacle_layout"].get("obstacles", [])):
        size_world = np.asarray(raw["size_m"], dtype=np.float64)
        # The MuJoCo box remains aligned with the tabletop/world axes.  Convert
        # its full extents to the task-frame AABB used by planning and collision
        # checks so a displayed world-x width has the same physical meaning in
        # both representations.
        size_task = np.abs(rotation.T) @ size_world
        center_task = np.array(
            [
                float(raw["x_task_m"]) if "x_task_m" in raw
                else float(raw.get("x_fraction_of_distance", -0.5)) * distance,
                float(raw.get("y_task_m", 0.0)),
                table_z_task + 0.5 * float(size_world[2]),
            ],
            dtype=np.float64,
        )
        center_world = np.asarray(task_frame[:3], dtype=np.float64) + rotation @ center_task
        resolved.append(
            {
                "obstacle_id": str(raw.get("obstacle_id", f"obstacle_{index}")),
                "type": "box",
                "center_task": center_task.tolist(),
                "center_world": center_world.tolist(),
                "size_m": size_task.tolist(),
                "size_world_m": size_world.tolist(),
                "rgba": list(raw.get("rgba", [0.55, 0.18, 0.12, 1.0])),
                "point_cloud_semantic": "static_environment",
            }
        )
    return task_frame.astype(np.float32), resolved



def instantiate_affordance(
    protocol: dict[str, Any], task_frame_world: np.ndarray, cup_quat_world: np.ndarray
) -> dict[str, Any]:
    cache = protocol["affordance"]
    realization = protocol["realization"]
    u_o = float(realization["u_o"])
    raw_ug = realization["u_g"]
    if isinstance(raw_ug, (list, tuple)):
        # Backward-compatible read of the former [radial, angular] schema.
        u_gr, u_ga = float(raw_ug[0]), float(raw_ug[1])
    else:
        u_gr, u_ga = float(cache["target"].get("default_radial_u", 0.60)), float(raw_ug)

    source = cache["source"]
    allowed = source.get("allowed_intervals_u", [[0.0, 1.0]])
    if not any(float(low) <= u_o <= float(high) for low, high in allowed):
        raise ValueError(f"u_o={u_o:.4f} is outside the verified source affordance intervals")
    parameterization = str(source.get("parameterization", "normalized_rim_arc_length"))
    if parameterization == "normalized_preferred_pour_arc":
        center = math.radians(float(source["pour_arc_center_deg_local"]))
        span = math.radians(float(source["pour_arc_span_deg"]))
        theta = center + (u_o - 0.5) * span
    else:
        theta = 2.0 * math.pi * u_o + math.radians(float(source.get("angle_offset_deg", 0.0)))
    rim_center = np.asarray(source.get("rim_center_local_m", [0.0, 0.0, 0.0]), dtype=np.float64)
    source_local = np.array(
        [
            float(source["rim_radius_m"]) * math.cos(theta),
            float(source["rim_radius_m"]) * math.sin(theta),
            float(source["lip_z_m"]),
        ],
        dtype=np.float64,
    )
    source_local[:2] += rim_center[:2]
    target = cache["target"]
    beta = 2.0 * math.pi * u_ga + math.radians(float(target.get("angle_offset_deg", 0.0)))
    radius = float(target["safe_opening_radius_m"]) * float(np.clip(u_gr, 0.0, 1.0))
    target_surface_task = np.array(
        [radius * math.cos(beta), radius * math.sin(beta), float(target["opening_z_task_m"])],
        dtype=np.float64,
    )
    target_approach_task = target_surface_task.copy()
    target_approach_task[2] += float(realization.get("anchor_height_m", 0.07))

    task_rot = quat_wxyz_to_rotation(np.asarray(task_frame_world[3:7])).as_matrix()
    cup_rot_world = quat_wxyz_to_rotation(cup_quat_world).as_matrix()
    cup_rot_task = task_rot.T @ cup_rot_world
    goal_cup_task = target_approach_task - cup_rot_task @ source_local

    # Operator-facing schematic patches.  OpenAD defines the candidate
    # affordance range, while these deterministic samples visualize the
    # normalized interaction variables without claiming pointwise inference
    # accuracy.  They intentionally retain the same 64-point visual language
    # used by the method figure.
    patch_count = 64
    # Use an 8x8 arc-by-height band so the outlet reads as a region on the
    # upper cup wall, rather than a thin line or a single precise spout point.
    arc_axis = np.linspace(-0.25, 0.25, 8)
    height_axis = np.linspace(-0.012, 0.002, 8)
    arc_offset, height_offset = np.meshgrid(arc_axis, height_axis)
    arc_offset = arc_offset.reshape(-1)
    height_offset = height_offset.reshape(-1)
    radial_offset = 0.0015 * (height_offset - height_offset.max()) / max(
        float(np.ptp(height_offset)), 1e-9
    )
    source_patch_local = np.column_stack(
        (
            rim_center[0]
            + (float(source["rim_radius_m"]) + radial_offset)
            * np.cos(theta + arc_offset),
            rim_center[1]
            + (float(source["rim_radius_m"]) + radial_offset)
            * np.sin(theta + arc_offset),
            float(source["lip_z_m"]) + height_offset,
        )
    )
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    disk_index = np.arange(patch_count, dtype=np.float64)
    disk_radius = 0.010 * np.sqrt((disk_index + 0.5) / patch_count)
    disk_angle = disk_index * golden_angle
    target_guidance_center_task = target_surface_task.copy()
    target_guidance_center_task[2] = float(
        target.get("receive_floor_z_task_m", target["opening_z_task_m"])
    )
    target_patch_task = target_guidance_center_task[None, :] + np.column_stack(
        (
            disk_radius * np.cos(disk_angle),
            disk_radius * np.sin(disk_angle),
            0.0012 * np.sin(2.0 * disk_angle),
        )
    )
    instantiated = {
        "source_anchor_local": source_local.tolist(),
        "source_pour_direction_local": [math.cos(theta), math.sin(theta), 0.0],
        "target_anchor_task": target_surface_task.tolist(),
        "target_approach_task": target_approach_task.tolist(),
        "goal_cup_origin_task": goal_cup_task.tolist(),
        "source_guidance_patch_local": source_patch_local.astype(np.float32).tolist(),
        "target_guidance_patch_task": target_patch_task.astype(np.float32).tolist(),
        "target_guidance_center_task": target_guidance_center_task.tolist(),
        "target_guidance_surface": "container_interior_floor",
        "guidance_patch_size": patch_count,
        "guidance_patch_semantics": "normalized_affordance_schematic_not_raw_openad_response",
        "source_patch_level": realization.get("u_o_level", "uo50"),
        "target_patch_level": realization.get("u_g_level", "sector_0"),
        "u_o": u_o,
        "u_g": u_ga,
        "target_fixed_radial_u": u_gr,
        "source_region_id": source.get("region_id", "pour_outlet"),
        "target_region_id": target.get("region_id", "receive_liquid"),
        "source": str(cache.get("source_method", "openad_plus_geometry_verified")),
    }
    if realization.get("virtual_handle_rule") == "opposite_selected_source_outlet":
        # The rotationally symmetric slim cup has no physical handle.  For the
        # matched psi x u_o diagnostic we give it the same source-side frame
        # semantics as a handled mug: the side opposite the selected outlet is
        # treated as a virtual handle / non-pouring side.  Save both the
        # object-local definition and its scene-specific task-frame instance.
        virtual_handle_local = np.array(
            [-math.cos(theta), -math.sin(theta), 0.0], dtype=np.float64
        )
        virtual_handle_task = cup_rot_task @ virtual_handle_local
        instantiated.update(
            {
                "virtual_handle_rule": "opposite_selected_source_outlet",
                "virtual_handle_direction_local": virtual_handle_local.tolist(),
                "virtual_handle_angle_deg_local": float(
                    math.degrees(math.atan2(virtual_handle_local[1], virtual_handle_local[0]))
                    % 360.0
                ),
                "virtual_handle_direction_task_initial": virtual_handle_task.tolist(),
                "virtual_handle_angle_deg_task_initial": float(
                    math.degrees(math.atan2(virtual_handle_task[1], virtual_handle_task[0]))
                    % 360.0
                ),
            }
        )
    return instantiated


def _inflated_boxes(obstacles: list[dict[str, Any]], radius: float, z_margin: float) -> list[np.ndarray]:
    boxes = []
    for item in obstacles:
        center = np.asarray(item["center_task"], dtype=np.float64)
        half = 0.5 * np.asarray(item["size_m"], dtype=np.float64)
        expand = np.array([radius, radius, z_margin], dtype=np.float64)
        boxes.append(np.stack([center - half - expand, center + half + expand]))
    return boxes


def _point_free(point: np.ndarray, boxes: list[np.ndarray]) -> bool:
    return not any(np.all(point >= box[0]) and np.all(point <= box[1]) for box in boxes)


def _edge_free(a: np.ndarray, b: np.ndarray, boxes: list[np.ndarray], resolution: float = 0.01) -> bool:
    steps = max(2, int(math.ceil(float(np.linalg.norm(b - a)) / resolution)) + 1)
    return all(_point_free(p, boxes) for p in np.linspace(a, b, steps))


def _dijkstra(nodes: np.ndarray, neighbors: list[list[tuple[int, float]]], start: int, goal: int) -> list[int]:
    costs = [math.inf] * len(nodes)
    parent = [-1] * len(nodes)
    costs[start] = 0.0
    queue = [(0.0, start)]
    while queue:
        cost, node = heapq.heappop(queue)
        if cost != costs[node]:
            continue
        if node == goal:
            break
        for nxt, weight in neighbors[node]:
            candidate = cost + weight
            if candidate < costs[nxt]:
                costs[nxt], parent[nxt] = candidate, node
                heapq.heappush(queue, (candidate, nxt))
    if not math.isfinite(costs[goal]):
        return []
    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    return list(reversed(path))


def _prm_segment(
    start: np.ndarray,
    goal: np.ndarray,
    boxes: list[np.ndarray],
    bounds: np.ndarray,
    rng: np.random.Generator,
    samples: int,
    neighbors_k: int,
) -> np.ndarray:
    if _edge_free(start, goal, boxes):
        return np.stack([start, goal])
    candidates = [start, goal]
    attempts = 0
    while len(candidates) < samples + 2 and attempts < samples * 30:
        attempts += 1
        point = rng.uniform(bounds[0], bounds[1])
        if _point_free(point, boxes):
            candidates.append(point)
    nodes = np.asarray(candidates, dtype=np.float64)
    graph: list[list[tuple[int, float]]] = [[] for _ in nodes]
    distances = np.linalg.norm(nodes[:, None, :] - nodes[None, :, :], axis=2)
    for i in range(len(nodes)):
        for j in np.argsort(distances[i])[1 : neighbors_k + 1]:
            j = int(j)
            if _edge_free(nodes[i], nodes[j], boxes):
                weight = float(distances[i, j])
                graph[i].append((j, weight))
                graph[j].append((i, weight))
    indices = _dijkstra(nodes, graph, 0, 1)
    if not indices:
        raise RuntimeError("PRM could not connect a topology-constrained segment")
    return nodes[np.asarray(indices, dtype=np.int64)]


def _resample_polyline(points: np.ndarray, spacing: float) -> np.ndarray:
    if len(points) < 2:
        return points.copy()
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    total = float(cumulative[-1])
    count = max(2, int(math.ceil(total / max(spacing, 1e-4))) + 1)
    query = np.linspace(0.0, total, count)
    return np.stack([np.interp(query, cumulative, points[:, axis]) for axis in range(3)], axis=1)


def _smooth_polyline_collision_free(
    points: np.ndarray,
    boxes: list[np.ndarray],
    iterations: int = 3,
) -> tuple[np.ndarray, bool]:
    """Round corners only when every resulting segment remains collision-free."""
    original = np.asarray(points, dtype=np.float64)
    smoothed = original.copy()
    for _ in range(max(0, int(iterations))):
        if len(smoothed) < 3:
            break
        refined = [smoothed[0]]
        for first, second in zip(smoothed[:-1], smoothed[1:]):
            refined.extend((0.75 * first + 0.25 * second, 0.25 * first + 0.75 * second))
        refined.append(smoothed[-1])
        candidate = np.asarray(refined, dtype=np.float64)
        if not all(_edge_free(a, b, boxes, resolution=0.005) for a, b in zip(candidate[:-1], candidate[1:])):
            return original, False
        smoothed = candidate
    return smoothed, not np.array_equal(smoothed, original)


def plan_corridor(
    start_task: np.ndarray,
    goal_task: np.ndarray,
    obstacles: list[dict[str, Any]],
    topology: str,
    *,
    seed: int,
    cup_clearance_radius_m: float,
    corridor_radius_m: float,
    workspace_z_task: tuple[float, float],
    tabletop_z_task: float | None = None,
    samples: int = 240,
) -> dict[str, Any]:
    if topology not in TOPOLOGIES:
        raise ValueError(topology)
    # The centerline is collision-free for the cup swept volume.  The visual
    # corridor is guidance tolerance and may taper near obstacles and the goal;
    # uniformly adding its full radius can incorrectly make the goal infeasible.
    start = np.asarray(start_task, dtype=np.float64)
    goal = np.asarray(goal_task, dtype=np.float64)
    bounds = np.array(
        [
            [min(start[0], goal[0]) - 0.08, -0.36, workspace_z_task[0]],
            [max(start[0], goal[0]) + 0.08, 0.36, workspace_z_task[1]],
        ],
        dtype=np.float64,
    )
    boxes = _inflated_boxes(obstacles, cup_clearance_radius_m, cup_clearance_radius_m)
    if tabletop_z_task is not None:
        # Model the table as a half-space beneath its top surface.  Since paths
        # represent the cup origin, raise that surface by the spherical swept-
        # volume clearance.  The wide AABB is equivalent to a plane within the
        # bounded PRM workspace and does not become a topology-class obstacle.
        floor_top = float(tabletop_z_task) + float(cup_clearance_radius_m)
        table_box = np.array(
            [
                [bounds[0, 0] - 1.0, bounds[0, 1] - 1.0, bounds[0, 2] - 1.0],
                [bounds[1, 0] + 1.0, bounds[1, 1] + 1.0, floor_top],
            ],
            dtype=np.float64,
        )
        boxes.append(table_box)
    if not _point_free(start, boxes):
        raise RuntimeError("corridor start is in collision with the inflated static environment")
    if not _point_free(goal, boxes):
        raise RuntimeError("corridor goal is in collision with the inflated static environment")
    if not obstacles:
        path = np.stack([start, goal])
        portal = 0.5 * (start + goal)
    else:
        primary = obstacles[0]
        center = np.asarray(primary["center_task"], dtype=np.float64)
        half = 0.5 * np.asarray(primary["size_m"], dtype=np.float64)
        portal = center.copy()
        lateral_clearance = cup_clearance_radius_m + corridor_radius_m + 0.015
        if topology == "c_left":
            portal[1] = center[1] + half[1] + lateral_clearance
            portal[2] = max(start[2], goal[2])
        elif topology == "c_right":
            portal[1] = center[1] - half[1] - lateral_clearance
            portal[2] = max(start[2], goal[2])
        else:
            portal[1] = center[1]
            portal[2] = center[2] + half[2] + cup_clearance_radius_m + corridor_radius_m + 0.015
        if not np.all(portal >= bounds[0]) or not np.all(portal <= bounds[1]) or not _point_free(portal, boxes):
            raise RuntimeError(f"topology portal is not feasible: {topology}")
        rng = np.random.default_rng(seed)
        first = _prm_segment(start, portal, boxes, bounds, rng, samples // 2, 12)
        second = _prm_segment(portal, goal, boxes, bounds, rng, samples // 2, 12)
        path = np.concatenate([first, second[1:]], axis=0)

    # Keep the topology portal as a hard semantic constraint.  Chaikin
    # smoothing is collision checked, but collision freedom alone does not
    # guarantee that the rounded path remains in the requested left/right/over
    # homotopy class.  In rare cases it can round a side path upward enough
    # that the final gate crossing is classified as c_over.  Preserve the
    # current behavior for ordinary rows and fall back to the unsmoothed PRM
    # path only when smoothing/resampling changes the requested topology.
    raw_path = np.asarray(path, dtype=np.float64).copy()
    smoothed_path, smoothing_applied = _smooth_polyline_collision_free(
        raw_path, boxes, iterations=3
    )
    path = _resample_polyline(smoothed_path, spacing=0.0125)
    smoothing_mode = (
        "chaikin_collision_checked_v1" if smoothing_applied else "none_collision_fallback"
    )
    if obstacles:
        observed = classify_topology(path, obstacles)["observed_topology_class"]
        if observed != topology:
            fallback = _resample_polyline(raw_path, spacing=0.0125)
            fallback_observed = classify_topology(
                fallback, obstacles
            )["observed_topology_class"]
            if fallback_observed == topology:
                path = fallback
                smoothing_applied = False
                smoothing_mode = "none_topology_preserving_fallback_v1"
            else:
                # Passing through a side/over portal is not sufficient to fix
                # the homotopy class: an unconstrained PRM segment can cross
                # the obstacle gate earlier and only visit the requested portal
                # afterwards.  Replan only this rare failure case with the
                # obstacle x-plane as a hard gate.  The pre-portal segment may
                # not enter x > gate_x and the post-portal segment may not enter
                # x < gate_x.  Smoothing each segment separately preserves the
                # portal endpoint and therefore the requested first crossing.
                gate_x = float(center[0])
                if not (start[0] <= gate_x + 1e-9 and goal[0] >= gate_x - 1e-9):
                    raise RuntimeError(
                        "topology gate is not ordered between start and goal: "
                        f"start_x={start[0]}, gate_x={gate_x}, goal_x={goal[0]}"
                    )
                before_bounds = bounds.copy()
                after_bounds = bounds.copy()
                before_bounds[1, 0] = min(before_bounds[1, 0], gate_x)
                after_bounds[0, 0] = max(after_bounds[0, 0], gate_x)
                strict_rng = np.random.default_rng(seed)
                strict_first = _prm_segment(
                    start, portal, boxes, before_bounds, strict_rng, samples // 2, 12
                )
                strict_second = _prm_segment(
                    portal, goal, boxes, after_bounds, strict_rng, samples // 2, 12
                )
                strict_first, first_smoothed = _smooth_polyline_collision_free(
                    strict_first, boxes, iterations=3
                )
                strict_second, second_smoothed = _smooth_polyline_collision_free(
                    strict_second, boxes, iterations=3
                )
                strict_first = _resample_polyline(strict_first, spacing=0.0125)
                strict_second = _resample_polyline(strict_second, spacing=0.0125)
                path = np.concatenate((strict_first, strict_second[1:]), axis=0)
                strict_observed = classify_topology(
                    path, obstacles
                )["observed_topology_class"]
                if strict_observed != topology:
                    raise RuntimeError(
                        "topology-gated PRM failed requested topology: "
                        f"requested={topology}, smoothed={observed}, "
                        f"unsmoothed={fallback_observed}, strict={strict_observed}"
                    )
                smoothing_applied = bool(first_smoothed or second_smoothed)
                smoothing_mode = "topology_gate_preserving_fallback_v1"
    return {
        "planner": "topology_portal_prm_v1",
        "collision_model": "static_environment_inflated_by_spherical_cup_clearance",
        "tabletop_collision": tabletop_z_task is not None,
        "tabletop_z_task_m": None if tabletop_z_task is None else float(tabletop_z_task),
        "smoothing": smoothing_mode,
        "topology_class": topology,
        "centerline_task": path.astype(np.float32).tolist(),
        "portal_task": np.asarray(portal, dtype=np.float32).tolist(),
        "radius_m": float(corridor_radius_m),
        "cup_clearance_radius_m": float(cup_clearance_radius_m),
        "waypoint_count": int(len(path)),
        "valid": True,
    }


def point_to_polyline_distance(point: np.ndarray, polyline: np.ndarray) -> tuple[float, int]:
    point = np.asarray(point, dtype=np.float64)
    line = np.asarray(polyline, dtype=np.float64)
    if len(line) == 1:
        return float(np.linalg.norm(point - line[0])), 0
    best = math.inf
    best_index = 0
    for i, (a, b) in enumerate(zip(line[:-1], line[1:])):
        delta = b - a
        t = float(np.dot(point - a, delta) / max(float(np.dot(delta, delta)), 1e-12))
        closest = a + np.clip(t, 0.0, 1.0) * delta
        distance = float(np.linalg.norm(point - closest))
        if distance < best:
            best, best_index = distance, i
    return best, best_index


def classify_topology(path_task: np.ndarray, obstacles: list[dict[str, Any]]) -> dict[str, Any]:
    path = np.asarray(path_task, dtype=np.float64)
    if len(path) == 0 or not obstacles:
        return {"observed_topology_class": "direct", "gate_crossing_task": None}
    item = obstacles[0]
    center = np.asarray(item["center_task"], dtype=np.float64)
    half = 0.5 * np.asarray(item["size_m"], dtype=np.float64)
    signed_x = path[:, 0] - center[0]
    # Use the first forward crossing of the obstacle gate and interpolate it.
    # Selecting the nearest sampled frame makes the label depend on frame rate
    # and can choose opposite sides of the gate before/after trajectory cleanup.
    forward = np.flatnonzero((signed_x[:-1] <= 0.0) & (signed_x[1:] >= 0.0))
    if len(forward):
        index = int(forward[0])
        denom = float(path[index + 1, 0] - path[index, 0])
        alpha = float((center[0] - path[index, 0]) / denom) if abs(denom) > 1e-12 else 0.0
        crossing = path[index] + np.clip(alpha, 0.0, 1.0) * (path[index + 1] - path[index])
    else:
        any_crossing = np.flatnonzero(signed_x[:-1] * signed_x[1:] <= 0.0)
        if len(any_crossing):
            index = int(any_crossing[0])
            denom = float(path[index + 1, 0] - path[index, 0])
            alpha = float((center[0] - path[index, 0]) / denom) if abs(denom) > 1e-12 else 0.0
            crossing = path[index] + np.clip(alpha, 0.0, 1.0) * (path[index + 1] - path[index])
        else:
            index = int(np.argmin(np.abs(signed_x)))
            crossing = path[index]
    if crossing[1] > center[1] + half[1]:
        label = "c_left"
    elif crossing[1] < center[1] - half[1]:
        label = "c_right"
    elif crossing[2] > center[2] + half[2]:
        label = "c_over"
    else:
        label = "ambiguous"
    return {"observed_topology_class": label, "gate_crossing_task": crossing.tolist()}
