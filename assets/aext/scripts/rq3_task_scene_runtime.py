#!/usr/bin/env python3
"""Inject the frozen RQ1 task-side obstacle into the robot RQ3 runtime."""

from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from robosuite.models.objects import MujocoXMLObject
from scipy.spatial.transform import Rotation

from libero.libero.envs.base_object import OBJECTS_DICT


ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "assets/rq3_task_obstacle.xml"
COLLECTION_PROTOCOL = ROOT / "configs/cross_scene_collection_v1.json"
# RQ1 冻结几何：支撑桌面 world z=0.9，task frame 原点在目标碗中心 world z=0.97，
# 故 task 帧里桌面在 -0.07；scene_height 同时平移两者，相对偏移不变。
LIBERO_TABLE_TOP_WORLD_Z = 0.9
RQ1_TASK_FRAME_WORLD_Z = 0.97
TABLE_TOP_TASK_Z = LIBERO_TABLE_TOP_WORLD_Z - RQ1_TASK_FRAME_WORLD_Z  # -0.07
# RQ1 冻结几何：nominal task frame 恒为绕 z 轴 90°（已对全部 72 个 teacher npz 验证）。
# 用于把 WORLD 轴的 size_m 一次性转到 canonical task-frame AABB，与 deployment yaw 无关。
NOMINAL_TASK_ROTATION = Rotation.from_euler("z", 90.0, degrees=True)
INSTANCE_NAME = "rq3_task_obstacle_1"
CATEGORY = "rq3_task_obstacle"
SOURCE_INSTANCE = "libero_mug_1"
TARGET_INSTANCE = "akita_black_bowl_1"
SOURCE_CATEGORIES = {
    "G_o1": "rq3_go1_source",
    "G_o2": "rq3_go2_source",
    "G_o3": "rq3_go3_source",
}
TARGET_CATEGORIES = {
    "G_g1": "rq3_gg1_target",
    "G_g2": "rq3_gg2_target",
    "G_g3": "rq3_gg3_target",
}


if CATEGORY not in OBJECTS_DICT:
    class Rq3TaskObstacle(MujocoXMLObject):
        def __init__(self, name=INSTANCE_NAME, joints=None):
            super().__init__(
                str(ASSET), name=name, joints=joints, obj_type="all",
                duplicate_collision_geoms=False,
            )
            self.category_name = CATEGORY
            self.rotation = (0.0, 0.0)
            self.rotation_axis = "z"
            self.object_properties = {"vis_site_names": {}}

    OBJECTS_DICT[CATEGORY] = Rq3TaskObstacle


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scaled_numbers(text: str, scales: tuple[float, ...]) -> str:
    values = tuple(float(value) for value in text.split())
    if len(values) != len(scales):
        raise ValueError(f"cannot apply {len(scales)} scales to {text!r}")
    return " ".join(f"{value * scale:.12g}" for value, scale in zip(values, scales))


def _scale_libero_object(asset: ET.Element, object_body: ET.Element,
                         scale_xyz: tuple[float, float, float]) -> None:
    """Apply the exact RQ1 collector scaling convention to a LIBERO object."""
    sx, sy, sz = map(float, scale_xyz)
    if min(sx, sy, sz) <= 0.0:
        raise ValueError(f"object scale must be positive, got {scale_xyz}")
    for mesh in asset.findall(".//mesh"):
        current = mesh.get("scale", "1 1 1")
        mesh.set("scale", _scaled_numbers(current, (sx, sy, sz)))
    for elem in object_body.iter():
        if "pos" in elem.attrib:
            elem.set("pos", _scaled_numbers(elem.get("pos", ""), (sx, sy, sz)))
        if "fromto" in elem.attrib:
            elem.set("fromto", _scaled_numbers(
                elem.get("fromto", ""), (sx, sy, sz, sx, sy, sz)
            ))
        if elem.tag not in {"geom", "site"} or "size" not in elem.attrib:
            continue
        size = elem.get("size", "")
        count = len(size.split())
        if elem.tag == "site":
            elem.set("size", _scaled_numbers(size, (min(sx, sy, sz),) * count))
            continue
        geom_type = elem.get("type", "sphere")
        if geom_type in {"cylinder", "capsule"}:
            if not math.isclose(sx, sy, rel_tol=1e-7, abs_tol=1e-9):
                raise ValueError(f"{geom_type} requires equal x/y scaling: {scale_xyz}")
            elem.set("size", _scaled_numbers(size, (sx, sz)))
        elif geom_type in {"box", "ellipsoid"}:
            elem.set("size", _scaled_numbers(size, (sx, sy, sz)))
        elif geom_type == "sphere":
            if not (math.isclose(sx, sy) and math.isclose(sx, sz)):
                raise ValueError(f"sphere requires uniform scaling: {scale_xyz}")
            elem.set("size", _scaled_numbers(size, (sx,)))
        elif count == 3:
            elem.set("size", _scaled_numbers(size, (sx, sy, sz)))


def _geometry_specs() -> dict[str, dict[str, Any]]:
    protocol = json.loads(COLLECTION_PROTOCOL.read_text(encoding="utf-8"))
    specs: dict[str, dict[str, Any]] = {}
    for geometry_id, category, role in [
        *((gid, category, "source") for gid, category in SOURCE_CATEGORIES.items()),
        *((gid, category, "target") for gid, category in TARGET_CATEGORIES.items()),
    ]:
        mapping = protocol[f"{role}_geometries"][geometry_id]
        asset_id = mapping["asset_id"]
        asset = protocol["assets"][asset_id]
        path = Path(asset[f"{role}_xml"])
        scale = tuple(map(float, asset.get(f"{role}_scale_xyz", [1.0, 1.0, 1.0])))
        if not path.exists():
            raise FileNotFoundError(f"RQ1 {geometry_id} {role} asset does not exist: {path}")
        specs[geometry_id] = {
            "geometry_id": geometry_id,
            "role": role,
            "category": category,
            "asset_id": asset_id,
            "xml": path,
            "xml_sha256": _file_sha256(path),
            "scale_xyz": scale,
        }
    return specs


GEOMETRY_SPECS = _geometry_specs()


def _register_geometry_category(spec: dict[str, Any]) -> None:
    category = str(spec["category"])
    if category in OBJECTS_DICT:
        return
    rotation = (-math.pi / 2, -math.pi / 2) if spec["geometry_id"] == "G_o1" else (
        (math.pi / 2, math.pi / 2) if spec["role"] == "target" else (0.0, 0.0)
    )
    rotation_axis = "z" if spec["role"] == "source" else "x"

    class Rq3GeometryObject(MujocoXMLObject):
        def __init__(self, name=category, joints=None):
            if joints is None:
                joints = [dict(type="free", damping="0.0005")]
            super().__init__(
                str(spec["xml"]), name=name, joints=joints, obj_type="all",
                duplicate_collision_geoms=False,
            )
            _scale_libero_object(self.asset, self._obj, spec["scale_xyz"])
            self.category_name = category
            self.rotation = rotation
            self.rotation_axis = rotation_axis
            self.object_properties = {"vis_site_names": {}}

        @property
        def bottom_offset(self):
            return super().bottom_offset * np.asarray(spec["scale_xyz"], dtype=np.float64)

        @property
        def top_offset(self):
            return super().top_offset * np.asarray(spec["scale_xyz"], dtype=np.float64)

        @property
        def horizontal_radius(self):
            return float(super().horizontal_radius) * float(spec["scale_xyz"][0])

    Rq3GeometryObject.__name__ = "".join(part.title() for part in category.split("_"))
    OBJECTS_DICT[category] = Rq3GeometryObject


for _spec in GEOMETRY_SPECS.values():
    _register_geometry_category(_spec)


def load_scene(bank_path: Path, scene_id: str) -> dict[str, Any]:
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    if bank.get("schema") != "rq3_embodiment_scene_bank_v2_rq1_realizations":
        raise ValueError("task-scene injection requires an RQ3 v2 RQ1-realization bank")
    matches = [row for row in bank["scenes"] if row["scene_id"] == scene_id]
    if len(matches) != 1:
        raise ValueError(f"expected one scene {scene_id}, found {len(matches)}")
    scene = dict(matches[0])
    source_id = scene["scene"]["source_geometry_id"]
    target_id = scene["scene"]["target_geometry_id"]
    if source_id not in SOURCE_CATEGORIES or target_id not in TARGET_CATEGORIES:
        raise ValueError(f"unsupported RQ1 geometry pair {(source_id, target_id)}")
    source_spec = dict(GEOMETRY_SPECS[source_id])
    source_override = scene.get("runtime_geometry_override", {}).get("source")
    if source_override:
        override_xml = Path(source_override["xml"]).resolve()
        if not override_xml.is_file():
            raise FileNotFoundError(f"source geometry override does not exist: {override_xml}")
        source_spec.update({
            "asset_id": source_override.get("asset_id", f"{source_spec['asset_id']}_variant"),
            "xml": override_xml,
            "xml_sha256": _file_sha256(override_xml),
            "category": source_override.get(
                "category", f"{source_spec['category']}_{_file_sha256(override_xml)[:10]}"
            ),
        })
        _register_geometry_category(source_spec)
    scene["runtime_geometry"] = {
        "source": {**source_spec, "xml": str(source_spec["xml"]),
                   "scale_xyz": list(source_spec["scale_xyz"])},
        "target": {**GEOMETRY_SPECS[target_id], "xml": str(GEOMETRY_SPECS[target_id]["xml"]),
                   "scale_xyz": list(GEOMETRY_SPECS[target_id]["scale_xyz"])},
        "instance_name_contract": {"source": SOURCE_INSTANCE, "target": TARGET_INSTANCE},
    }
    return scene


def _replace_object_category(text: str, instance: str, category: str) -> str:
    pattern = re.compile(rf"(^\s*{re.escape(instance)}\s*-\s*)[A-Za-z0-9_]+(\s*$)", re.MULTILINE)
    text, count = pattern.subn(rf"\g<1>{category}\g<2>", text)
    if count != 1:
        raise ValueError(f"expected exactly one BDDL declaration for {instance}, found {count}")
    return text


def inject_fixture_bddl(base_bddl: Path, scene_row: dict[str, Any]) -> tuple[Path, str]:
    text = base_bddl.read_text(encoding="utf-8")
    runtime = scene_row["runtime_geometry"]
    text = _replace_object_category(text, SOURCE_INSTANCE, runtime["source"]["category"])
    text = _replace_object_category(text, TARGET_INSTANCE, runtime["target"]["category"])
    if CATEGORY not in text:
        start = text.find("(:fixtures")
        if start < 0:
            raise ValueError(f"BDDL has no fixtures section: {base_bddl}")
        depth = 0
        end = None
        for index in range(start, len(text)):
            if text[index] == "(":
                depth += 1
            elif text[index] == ")":
                depth -= 1
                if depth == 0:
                    end = index
                    break
        if end is None:
            raise ValueError("unbalanced fixtures section")
        text = text[:end] + f"    {INSTANCE_NAME} - {CATEGORY}\n  " + text[end:]
    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".bddl", prefix="rq3_task_scene_", delete=False, encoding="utf-8"
    )
    with handle:
        handle.write(text)
    return Path(handle.name), hashlib.sha256(text.encode()).hexdigest()


def _deployed_obstacle_geometry(
    size_world: np.ndarray, task_rotation: Rotation
) -> tuple[np.ndarray, Rotation]:
    """Return a task-axis box and the rotation mapping it into world.

    The frozen RQ1 box is world-aligned in the nominal scene while the nominal
    task frame has yaw=90 degrees.  Converting its extents to task axes and
    then orienting those task axes with ``task_rotation`` is equivalent to
    retaining the world-axis extents and applying the layout delta.  Mixing
    task-axis extents with the layout-delta rotation turns the long axis by an
    additional 90 degrees.

    Returns R_deployed = R_layout @ R_nominal where R_nominal = R_z(90°).
    The task-axis size [0.18,0.04,0.10] becomes [0.04,0.18,0.10] after the
    90° rotation, placing the long axis (0.18m) along obstacle local y.
    R_deployed then rotates this oriented box into the deployment yaw angle.
    """
    # ``size_world`` belongs to the frozen nominal RQ1 scene.  Convert it to
    # canonical task axes exactly once with R_nominal; deployment yaw belongs
    # only in the body rotation below.  Using task_rotation here as well would
    # apply the layout yaw twice and change the local box dimensions.
    size_task = np.abs(NOMINAL_TASK_ROTATION.as_matrix().T) @ np.asarray(
        size_world, dtype=np.float64
    )
    return size_task, task_rotation


def configure(raw: Any, scene_row: dict[str, Any], task_frame_world: np.ndarray) -> dict[str, Any]:
    protocol = json.loads(COLLECTION_PROTOCOL.read_text(encoding="utf-8"))
    scene = scene_row["scene"]
    obstacles = protocol["obstacle_layouts"][scene["obstacle_layout_id"]]["obstacles"]
    if len(obstacles) != 1:
        raise ValueError("RQ3 v2 task-scene runtime currently requires exactly one RQ1 task obstacle")
    obstacle = obstacles[0]
    size_world = np.asarray(obstacle["size_m"], dtype=np.float64)
    task_frame_world = np.asarray(task_frame_world, dtype=np.float64)
    task_rotation = Rotation.from_quat([
        task_frame_world[4], task_frame_world[5], task_frame_world[6], task_frame_world[3]
    ])
    # ``size_m`` is stored in WORLD axes (the RQ1 MuJoCo box is world-aligned).
    # Convert extents with the fixed nominal task rotation, then orient that
    # canonical task-axis box with the deployed task rotation.  This is
    # equivalent to applying only the layout delta to the original world-axis
    # box, so its long axis remains perpendicular to cup->bowl task x.
    size_task, obstacle_rotation = _deployed_obstacle_geometry(size_world, task_rotation)
    center_task = np.asarray([
        obstacle.get("x_task_m", obstacle.get("x_fraction_of_distance", -0.5) * scene["distance_m"]),
        obstacle.get("y_task_m", 0.0),
        TABLE_TOP_TASK_Z + 0.5 * float(size_world[2]),
    ], dtype=np.float64)
    center_world = task_frame_world[:3] + task_rotation.apply(center_task)
    sim = raw.sim
    fixture = raw.fixtures_dict.get(INSTANCE_NAME)
    if fixture is None:
        raise ValueError("compiled environment lacks the RQ1 task-side obstacle fixture")
    body_id = sim.model.body_name2id(fixture.root_body)
    sim.model.body_pos[body_id] = center_world
    qx, qy, qz, qw = obstacle_rotation.as_quat()
    sim.model.body_quat[body_id] = [qw, qx, qy, qz]
    geom_ids = np.flatnonzero(np.asarray(sim.model.geom_bodyid) == body_id)
    for geom_id in geom_ids:
        sim.model.geom_size[geom_id, :3] = 0.5 * size_task
        name = sim.model.geom_id2name(int(geom_id)) or ""
        if "visual" in name:
            sim.model.geom_contype[geom_id] = 0
            sim.model.geom_conaffinity[geom_id] = 0
        else:
            sim.model.geom_contype[geom_id] = 1
            sim.model.geom_conaffinity[geom_id] = 1
        sim.model.geom_rgba[geom_id] = obstacle.get("rgba", [0.45, 0.16, 0.10, 1.0])
    sim.forward()
    expected_world_aabb_extent = (
        np.abs(obstacle_rotation.as_matrix()) @ size_task
    )
    lower = np.full(3, np.inf, dtype=np.float64)
    upper = np.full(3, -np.inf, dtype=np.float64)
    for geom_id in geom_ids:
        if int(sim.model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            raise ValueError("RQ3 task obstacle runtime AABB requires box geoms")
        # ``model.geom_aabb`` is a compile-time cache and is not rebuilt when
        # ``geom_size`` is changed at runtime.  Read the live box half extents
        # instead; ``geom_xpos`` already contains the transformed geom center.
        half_local = np.asarray(sim.model.geom_size[geom_id, :3], dtype=np.float64)
        rotation_world = np.asarray(sim.data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        center_geom_world = np.asarray(sim.data.geom_xpos[geom_id], dtype=np.float64)
        half_world = np.abs(rotation_world) @ half_local
        lower = np.minimum(lower, center_geom_world - half_world)
        upper = np.maximum(upper, center_geom_world + half_world)
    world_aabb_extent = upper - lower
    runtime_geometry = scene_row["runtime_geometry"]
    for instance in (SOURCE_INSTANCE, TARGET_INSTANCE):
        if instance not in raw.objects_dict:
            raise ValueError(f"compiled environment lacks RQ3 geometry instance {instance}")
    return {
        "scene_id": scene_row["scene_id"],
        "obstacle_layout_id": scene["obstacle_layout_id"],
        "center_task_m": center_task.tolist(),
        "center_world_m": center_world.tolist(),
        "size_m": size_task.tolist(),
        "body_quat_wxyz": [qw, qx, qy, qz],
        "orientation_rule": "task_axis_size_with_R_deployed",
        "expected_world_aabb_extent_m": expected_world_aabb_extent.tolist(),
        "world_aabb_extent_m": world_aabb_extent.tolist(),
        "whole_scene_transform": "T_world_task @ T_task_obstacle",
        "geometry_adapter": runtime_geometry,
        "geometry_adapter_status": "compiled_from_frozen_rq1_asset_xml",
    }
