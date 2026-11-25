"""Replay a recorded trajectory in a lightweight MuJoCo scene."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
from flora_core.libero_utils import find_libero_root

from flora_core.geometry.frames import (
    make_cup_to_bowl_task_frame,
    world_positions_to_task,
)


SOURCE_BODY = "source_cup"
TARGET_BODY = "target_container"


@dataclass(frozen=True)
class ReplayConfig:
    """Runtime settings for direct, object-level trajectory replay."""

    width: int = 640
    height: int = 480
    camera: str = "replay"
    render_stride: int = 1

    collect_point_cloud: bool = False
    point_cloud_points: int = 1024
    point_width: int = 320
    point_height: int = 240
    point_cameras: tuple[str, ...] = ("replay", "topdown")
    point_cloud_seed: int = 3407
    point_obstacle_reserved_fraction: float = 0.25
    point_crop_min_task: tuple[float, float, float] = (-0.65, -0.45, -0.05)
    point_crop_max_task: tuple[float, float, float] = (0.25, 0.45, 0.90)

    def __post_init__(self) -> None:
        if self.width < 16 or self.height < 16:
            raise ValueError("render dimensions must be at least 16 pixels")
        if self.render_stride < 1:
            raise ValueError("render_stride must be at least 1")
        if self.point_cloud_points < 1:
            raise ValueError("point_cloud_points must be positive")
        if self.point_width < 16 or self.point_height < 16:
            raise ValueError("point-cloud render dimensions must be at least 16 pixels")
        if not self.point_cameras:
            raise ValueError("point_cameras must contain at least one camera")
        if not 0.0 <= self.point_obstacle_reserved_fraction <= 0.5:
            raise ValueError(
                "point_obstacle_reserved_fraction must lie in [0,0.5]"
            )
        crop_min = np.asarray(self.point_crop_min_task, dtype=np.float64)
        crop_max = np.asarray(self.point_crop_max_task, dtype=np.float64)
        if crop_min.shape != (3,) or crop_max.shape != (3,):
            raise ValueError("point-cloud crop bounds must be 3D")
        if not np.all(crop_max > crop_min):
            raise ValueError("point-cloud crop max must exceed min on every axis")


def _prefix_references(element: ET.Element, prefix: str) -> None:
    for attribute in ("name", "mesh", "material", "texture"):
        if attribute in element.attrib:
            element.set(attribute, f"{prefix}_{element.get(attribute)}")
    for child in element:
        _prefix_references(child, prefix)


def _absolutize_files(element: ET.Element, directory: Path) -> None:
    if "file" in element.attrib:
        path = Path(element.get("file", ""))
        if not path.is_absolute():
            element.set("file", str((directory / path).resolve()))
    for child in element:
        _absolutize_files(child, directory)


def _scaled_numbers(value: str, scale: tuple[float, ...]) -> str:
    numbers = [float(item) for item in value.split()]
    if len(numbers) != len(scale):
        return value
    return " ".join(f"{number * factor:.9g}" for number, factor in zip(numbers, scale))


def _scale_object(
    assets: ET.Element,
    body: ET.Element,
    scale_xyz: tuple[float, float, float],
) -> None:
    sx, sy, sz = (float(value) for value in scale_xyz)
    if min(sx, sy, sz) <= 0.0:
        raise ValueError(f"object scale must be positive: {scale_xyz}")
    for mesh in assets.findall(".//mesh"):
        mesh.set("scale", _scaled_numbers(mesh.get("scale", "1 1 1"), (sx, sy, sz)))
    for element in body.iter():
        if "pos" in element.attrib:
            element.set("pos", _scaled_numbers(element.get("pos", ""), (sx, sy, sz)))
        if "fromto" in element.attrib:
            element.set(
                "fromto",
                _scaled_numbers(element.get("fromto", ""), (sx, sy, sz, sx, sy, sz)),
            )
        if element.tag not in {"geom", "site"} or "size" not in element.attrib:
            continue
        size = element.get("size", "")
        count = len(size.split())
        if element.tag == "site":
            element.set("size", _scaled_numbers(size, (min(sx, sy, sz),) * count))
            continue
        kind = element.get("type", "sphere")
        if kind in {"cylinder", "capsule"}:
            if not math.isclose(sx, sy, rel_tol=1e-7, abs_tol=1e-9):
                raise ValueError(f"{kind} requires equal x/y scale: {scale_xyz}")
            element.set("size", _scaled_numbers(size, (sx, sz)))
        elif kind in {"box", "ellipsoid"}:
            element.set("size", _scaled_numbers(size, (sx, sy, sz)))
        elif kind == "sphere":
            if not (math.isclose(sx, sy) and math.isclose(sx, sz)):
                raise ValueError(f"sphere requires uniform scale: {scale_xyz}")
            element.set("size", _scaled_numbers(size, (sx,)))
        elif count == 3:
            element.set("size", _scaled_numbers(size, (sx, sy, sz)))


def _load_object_fragment(
    path: Path,
    *,
    prefix: str,
    body_name: str,
    pose: np.ndarray,
    scale_xyz: tuple[float, float, float],
    mocap: bool,
) -> tuple[str, str]:
    root = ET.parse(path).getroot()
    source_assets = root.find("asset")
    source_body = root.find(".//body[@name='object']")
    if source_assets is None or source_body is None:
        raise ValueError(f"asset XML lacks asset/object body: {path}")
    assets = ET.fromstring(ET.tostring(source_assets, encoding="unicode"))
    body = ET.fromstring(ET.tostring(source_body, encoding="unicode"))
    _absolutize_files(assets, path.parent)
    _scale_object(assets, body, scale_xyz)
    _prefix_references(assets, prefix)
    _prefix_references(body, prefix)
    body.set("name", body_name)
    body.set("pos", " ".join(map(str, np.asarray(pose[:3], dtype=float))))
    body.set("quat", " ".join(map(str, np.asarray(pose[3:7], dtype=float))))
    if mocap:
        body.set("mocap", "true")
    # Collision shells stay visible enough for contact debugging; visual mesh
    # geoms retain their registered material.
    for geom in body.findall(".//geom"):
        if geom.get("group") == "0" and "rgba" not in geom.attrib:
            geom.set("rgba", "0.8 0.8 0.8 0.15")
    return (
        "\n".join(ET.tostring(child, encoding="unicode") for child in assets),
        ET.tostring(body, encoding="unicode"),
    )


def _resolve_asset_path(value: str, libero_root: Path) -> Path:
    if value.startswith("libero://"):
        value = value.removeprefix("libero://")
        path = libero_root / value
    else:
        path = Path(value).expanduser()
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"container asset does not exist: {path}")
    return path


def _camera_xyaxes(position: np.ndarray, look_at: np.ndarray) -> str:
    forward = np.asarray(look_at, dtype=np.float64) - np.asarray(position, dtype=np.float64)
    forward /= max(float(np.linalg.norm(forward)), 1e-9)
    camera_z = -forward
    camera_x = np.cross(np.asarray([0.0, 0.0, 1.0]), camera_z)
    if np.linalg.norm(camera_x) < 1e-8:
        camera_x = np.asarray([1.0, 0.0, 0.0])
    camera_x /= np.linalg.norm(camera_x)
    camera_y = np.cross(camera_z, camera_x)
    camera_y /= np.linalg.norm(camera_y)
    return " ".join(f"{value:.9g}" for value in np.r_[camera_x, camera_y])


def _obstacle_xml(obstacles: list[dict[str, Any]]) -> str:
    output = []
    for index, obstacle in enumerate(obstacles):
        center = obstacle.get("center_world")
        size = obstacle.get("size_world_m", obstacle.get("size_m"))
        if center is None or size is None:
            raise ValueError("obstacle lacks center_world/size_world_m")
        half = 0.5 * np.asarray(size, dtype=np.float64)
        rgba = obstacle.get("rgba", [0.55, 0.18, 0.12, 1.0])
        name = str(obstacle.get("obstacle_id", f"obstacle_{index}"))
        output.append(
            f'<body name="{name}" pos="{" ".join(map(str, center))}">'
            f'<geom name="{name}_geom" type="box" size="{" ".join(map(str, half))}" '
            f'rgba="{" ".join(map(str, rgba))}" friction="1 0.02 0.001"/>'
            "</body>"
        )
    return "\n".join(output)


def _scene_xml(
    metadata: dict[str, Any],
    source_pose: np.ndarray,
    target_pose: np.ndarray,
    libero_root: Path,
    config: ReplayConfig,
) -> str:
    scene = metadata["scene"]
    source_path = _resolve_asset_path(str(scene["source_asset_xml"]), libero_root)
    target_path = _resolve_asset_path(str(scene["target_asset_xml"]), libero_root)
    source_assets, source_body = _load_object_fragment(
        source_path,
        prefix="source",
        body_name=SOURCE_BODY,
        pose=source_pose,
        scale_xyz=tuple(scene.get("source_scale_xyz", [1.0, 1.0, 1.0])),
        mocap=True,
    )
    target_assets, target_body = _load_object_fragment(
        target_path,
        prefix="target",
        body_name=TARGET_BODY,
        pose=target_pose,
        scale_xyz=tuple(scene.get("target_scale_xyz", [1.0, 1.0, 1.0])),
        mocap=False,
    )
    midpoint = 0.5 * (source_pose[:3] + target_pose[:3])
    task_x = target_pose[:3] - source_pose[:3]
    task_x[2] = 0.0
    task_x /= max(float(np.linalg.norm(task_x)), 1e-9)
    task_y = np.asarray([-task_x[1], task_x[0], 0.0])
    camera_position = midpoint - 0.18 * task_x - 0.62 * task_y + np.asarray([0.0, 0.0, 0.5])
    look_at = midpoint + np.asarray([0.0, 0.0, 0.10])
    topdown_position = midpoint + np.asarray([0.0, 0.0, 1.0])
    table_z = float(target_pose[2]) - 0.025
    return f"""
<mujoco model="flora_replay">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <visual><global offwidth="{config.width}" offheight="{config.height}"/></visual>
  <default><geom contype="1" conaffinity="1" margin="0.001"/></default>
  <asset>
    <material name="table_material" rgba="0.52 0.42 0.30 1"/>
    {source_assets}
    {target_assets}
  </asset>
  <worldbody>
    <light name="key" pos="0 -0.5 1.8" diffuse="0.9 0.9 0.9"/>
    <light name="fill" pos="0.5 0.5 1.4" diffuse="0.45 0.45 0.45"/>
    <camera name="replay" pos="{' '.join(map(str, camera_position))}"
            xyaxes="{_camera_xyaxes(camera_position, look_at)}"/>
    <camera name="topdown" pos="{' '.join(map(str, topdown_position))}"
            xyaxes="1 0 0 0 1 0"/>
    <geom name="table" type="box" pos="0 0 {table_z}" size="0.5 0.6 0.025"
          material="table_material" friction="1 0.005 0.0001"/>
    {source_body}
    {target_body}
    {_obstacle_xml(list(scene.get('obstacles', [])))}
  </worldbody>
</mujoco>
"""


def _descendants(model: Any, root: int) -> set[int]:
    result = {int(root)}
    changed = True
    while changed:
        changed = False
        for body_id in range(model.nbody):
            if int(model.body_parentid[body_id]) in result and body_id not in result:
                result.add(body_id)
                changed = True
    return result


def _contact_groups(
    model: Any,
    data: Any,
    source_bodies: set[int],
    target_bodies: set[int],
) -> tuple[set[str], float]:
    groups: set[str] = set()
    minimum_distance = math.inf
    for index in range(data.ncon):
        contact = data.contact[index]
        body1 = int(model.geom_bodyid[contact.geom1])
        body2 = int(model.geom_bodyid[contact.geom2])
        if (body1 in source_bodies) == (body2 in source_bodies):
            continue
        source_is_first = body1 in source_bodies
        other = body2 if source_is_first else body1
        other_geom = int(contact.geom2 if source_is_first else contact.geom1)
        geom_name = model.geom(other_geom).name or ""
        if other in target_bodies:
            name = TARGET_BODY
        elif geom_name == "table":
            name = "table"
        else:
            name = model.body(other).name or geom_name or "world"
        groups.add(str(name))
        minimum_distance = min(minimum_distance, float(contact.dist))
    return groups, minimum_distance



def _camera_depth_segmentation_points(
    model: Any,
    data: Any,
    *,
    camera: str,
    depth_renderer: Any,
    segmentation_renderer: Any,
    source_bodies: set[int],
    target_bodies: set[int],
    obstacle_bodies: set[int],
    task_frame_world: np.ndarray,
    crop_min_task: np.ndarray,
    crop_max_task: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project one camera into task-frame xyz + semantic one-hot.

    MuJoCo camera coordinates use +x right, +y up, and -z forward. The
    network-visible semantics remain [source, target, environment].  A
    separate boolean obstacle mask is returned only for sampling/auditing so
    obstacles can be preserved without changing the Local-DP3 6D input.
    """
    import mujoco

    camera_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, str(camera)
    )
    if camera_id < 0:
        raise ValueError(f"unknown point-cloud camera: {camera!r}")

    depth_renderer.update_scene(data, camera=str(camera))
    depth = np.asarray(depth_renderer.render(), dtype=np.float64)

    segmentation_renderer.update_scene(data, camera=str(camera))
    segmentation = np.asarray(segmentation_renderer.render())
    if segmentation.ndim != 3 or segmentation.shape[-1] != 2:
        raise RuntimeError(
            f"MuJoCo segmentation render must be [H,W,2], got {segmentation.shape}"
        )
    if depth.shape != segmentation.shape[:2]:
        raise RuntimeError(
            "depth and segmentation renders must have identical image shapes"
        )

    height, width = depth.shape
    fovy = math.radians(float(model.cam_fovy[camera_id]))
    focal = 0.5 * float(height) / math.tan(0.5 * fovy)
    cx = 0.5 * float(width - 1)
    cy = 0.5 * float(height - 1)

    v, u = np.indices((height, width), dtype=np.float64)
    valid = np.isfinite(depth) & (depth > 1e-6)
    object_id = segmentation[..., 0].astype(np.int64)
    object_type = segmentation[..., 1].astype(np.int64)
    valid &= object_id >= 0
    valid &= object_type == int(mujoco.mjtObj.mjOBJ_GEOM)
    if not np.any(valid):
        return (
            np.zeros((0, 6), dtype=np.float32),
            np.zeros((0,), dtype=np.bool_),
        )

    z = depth[valid]
    x = (u[valid] - cx) * z / focal
    y = -(v[valid] - cy) * z / focal
    points_camera = np.column_stack((x, y, -z))

    rotation_world_camera = np.asarray(
        data.cam_xmat[camera_id], dtype=np.float64
    ).reshape(3, 3)
    camera_position_world = np.asarray(
        data.cam_xpos[camera_id], dtype=np.float64
    )
    points_world = (
        points_camera @ rotation_world_camera.T
        + camera_position_world[None, :]
    )
    points_task = world_positions_to_task(points_world, task_frame_world)

    geom_ids = object_id[valid]
    body_ids = np.asarray(model.geom_bodyid[geom_ids], dtype=np.int64)
    semantic = np.zeros((len(points_task), 3), dtype=np.float32)
    source_mask = np.isin(body_ids, np.fromiter(source_bodies, dtype=np.int64))
    target_mask = np.isin(body_ids, np.fromiter(target_bodies, dtype=np.int64))
    obstacle_mask = np.isin(
        body_ids,
        np.fromiter(obstacle_bodies, dtype=np.int64),
    )
    semantic[source_mask, 0] = 1.0
    semantic[target_mask, 1] = 1.0
    semantic[~(source_mask | target_mask), 2] = 1.0

    crop = np.all(points_task >= crop_min_task[None, :], axis=1)
    crop &= np.all(points_task <= crop_max_task[None, :], axis=1)
    if not np.any(crop):
        return (
            np.zeros((0, 6), dtype=np.float32),
            np.zeros((0,), dtype=np.bool_),
        )
    return (
        np.concatenate(
            (points_task[crop].astype(np.float32), semantic[crop]),
            axis=1,
        ),
        obstacle_mask[crop].astype(np.bool_, copy=False),
    )

def _sample_semantic_point_cloud(
    points: np.ndarray,
    *,
    count: int,
    seed: int,
    obstacle_mask: np.ndarray | None = None,
    obstacle_reserved_fraction: float = 0.25,
    return_sampling_group: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Deterministically sample a fixed-size task-relevant point cloud.

    Source and target each reserve one quarter of the budget.  Visible
    obstacles reserve an additional configurable fraction (default one
    quarter).  Obstacles keep the network-visible environment one-hot label;
    the reservation is only a sampling prior so the table cannot wash out
    collision-relevant geometry.

    Sampling-group ids used only for audit visualization are:
      0 source, 1 target, 2 obstacle, 3 other environment.
    """
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 6:
        raise ValueError(f"semantic point cloud must be [N,6], got {points.shape}")
    if len(points) == 0:
        raise RuntimeError("no visible task-frame points survived point-cloud crop")
    if not 0.0 <= float(obstacle_reserved_fraction) <= 0.5:
        raise ValueError("obstacle_reserved_fraction must lie in [0,0.5]")

    if obstacle_mask is None:
        obstacle_mask = np.zeros(len(points), dtype=np.bool_)
    else:
        obstacle_mask = np.asarray(obstacle_mask, dtype=np.bool_)
        if obstacle_mask.shape != (len(points),):
            raise ValueError(
                f"obstacle_mask must be [N], got {obstacle_mask.shape}"
            )

    rng = np.random.default_rng(int(seed))
    selected: list[int] = []
    chosen = np.zeros(len(points), dtype=bool)

    semantic_id = np.argmax(points[:, 3:6], axis=1)

    def reserve_candidates(candidates: np.ndarray, quota: int) -> None:
        if quota <= 0 or len(candidates) == 0:
            return
        candidates = candidates[~chosen[candidates]]
        if len(candidates) == 0:
            return
        take = min(int(quota), len(candidates))
        picked = rng.choice(candidates, size=take, replace=False)
        selected.extend(int(value) for value in picked)
        chosen[picked] = True

    object_reserve = int(count // 4)
    reserve_candidates(np.flatnonzero(semantic_id == 0), object_reserve)
    reserve_candidates(np.flatnonzero(semantic_id == 1), object_reserve)
    obstacle_reserve = int(round(count * float(obstacle_reserved_fraction)))
    reserve_candidates(np.flatnonzero(obstacle_mask), obstacle_reserve)

    remaining = int(count - len(selected))
    unused = np.flatnonzero(~chosen)
    if remaining > 0 and len(unused) > 0:
        take = min(remaining, len(unused))
        picked = rng.choice(unused, size=take, replace=False)
        selected.extend(int(value) for value in picked)
        chosen[picked] = True
        remaining -= take

    if remaining > 0:
        picked = rng.choice(np.arange(len(points)), size=remaining, replace=True)
        selected.extend(int(value) for value in picked)

    selected_array = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected_array)
    selected_array = selected_array[:count]
    sampled = points[selected_array].astype(np.float32)

    if not return_sampling_group:
        return sampled

    sampled_semantic = semantic_id[selected_array]
    sampled_obstacle = obstacle_mask[selected_array]
    group = np.full(len(sampled), 3, dtype=np.int8)
    group[sampled_semantic == 0] = 0
    group[sampled_semantic == 1] = 1
    group[sampled_obstacle] = 2
    return sampled, group

def _collect_point_cloud_frame(
    model: Any,
    data: Any,
    *,
    depth_renderer: Any,
    segmentation_renderer: Any,
    source_bodies: set[int],
    target_bodies: set[int],
    obstacle_bodies: set[int],
    task_frame_world: np.ndarray,
    config: ReplayConfig,
    frame_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    crop_min = np.asarray(config.point_crop_min_task, dtype=np.float32)
    crop_max = np.asarray(config.point_crop_max_task, dtype=np.float32)
    camera_results = [
        _camera_depth_segmentation_points(
            model,
            data,
            camera=camera,
            depth_renderer=depth_renderer,
            segmentation_renderer=segmentation_renderer,
            source_bodies=source_bodies,
            target_bodies=target_bodies,
            obstacle_bodies=obstacle_bodies,
            task_frame_world=task_frame_world,
            crop_min_task=crop_min,
            crop_max_task=crop_max,
        )
        for camera in config.point_cameras
    ]
    nonempty = [
        (cloud, obstacle)
        for cloud, obstacle in camera_results
        if len(cloud)
    ]
    if not nonempty:
        raise RuntimeError("all point-cloud cameras returned empty cropped clouds")
    fused = np.concatenate([cloud for cloud, _ in nonempty], axis=0)
    fused_obstacle = np.concatenate(
        [obstacle for _, obstacle in nonempty],
        axis=0,
    )
    return _sample_semantic_point_cloud(
        fused,
        count=config.point_cloud_points,
        seed=int(config.point_cloud_seed) + 104729 * int(frame_index),
        obstacle_mask=fused_obstacle,
        obstacle_reserved_fraction=config.point_obstacle_reserved_fraction,
        return_sampling_group=True,
    )

def _load_episode(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"source_pose_world", "target_pose_world", "metadata"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"episode lacks required arrays: {missing}")
        metadata = json.loads(str(archive["metadata"].item()))
        arrays = {key: np.asarray(archive[key]).copy() for key in archive.files if key != "metadata"}
    poses = arrays["source_pose_world"]
    if poses.ndim != 2 or poses.shape[1] != 7 or len(poses) < 2:
        raise ValueError(f"source_pose_world must be [T,7], got {poses.shape}")
    return metadata, arrays


def replay_episode(
    episode_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    video_path: str | Path | None = None,
    libero_root: str | Path | None = None,
    config: ReplayConfig = ReplayConfig(),
) -> dict[str, Any]:
    """Kinematically replay one saved trajectory and return an audit report."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    episode_path = Path(episode_path).resolve()
    metadata, arrays = _load_episode(episode_path)
    poses = np.asarray(arrays["source_pose_world"], dtype=np.float64)
    target = np.asarray(arrays["target_pose_world"], dtype=np.float64).reshape(7)
    root = (
        Path(libero_root).expanduser().resolve()
        if libero_root is not None
        else find_libero_root()
    )
    xml = _scene_xml(metadata, poses[0], target, root, config)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    source_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, SOURCE_BODY)
    mocap_id = int(model.body_mocapid[source_id])
    if mocap_id < 0:
        raise RuntimeError("source body is not mocap-controlled")
    source_bodies = _descendants(model, source_id)
    target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TARGET_BODY)
    target_bodies = _descendants(model, target_id)
    obstacle_bodies: set[int] = set()
    for obstacle in metadata["scene"].get("obstacles", []):
        obstacle_name = str(obstacle.get("obstacle_id", ""))
        if not obstacle_name:
            continue
        obstacle_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, obstacle_name
        )
        if obstacle_id >= 0:
            obstacle_bodies.update(_descendants(model, obstacle_id))
    actual = np.empty_like(poses)
    contact_count = np.zeros(len(poses), dtype=np.int32)
    penetration = np.full(len(poses), np.nan, dtype=np.float32)
    contact_names: list[list[str]] = []

    point_cloud_task = None
    point_cloud_sampling_group = None
    point_semantic_counts = None
    point_sampling_group_counts = None
    depth_renderer = None
    segmentation_renderer = None
    task_frame_world = None
    if config.collect_point_cloud:
        if "task_frame_world" in arrays:
            task_frame_world = np.asarray(
                arrays["task_frame_world"], dtype=np.float32
            ).reshape(7)
        else:
            task_frame_world, _ = make_cup_to_bowl_task_frame(
                poses[0, :3], target[:3]
            )
        point_cloud_task = np.empty(
            (len(poses), config.point_cloud_points, 6),
            dtype=np.float32,
        )
        point_cloud_sampling_group = np.empty(
            (len(poses), config.point_cloud_points),
            dtype=np.int8,
        )
        point_semantic_counts = np.zeros((len(poses), 3), dtype=np.int32)
        point_sampling_group_counts = np.zeros((len(poses), 4), dtype=np.int32)
        depth_renderer = mujoco.Renderer(
            model,
            height=config.point_height,
            width=config.point_width,
        )
        depth_renderer.enable_depth_rendering()
        segmentation_renderer = mujoco.Renderer(
            model,
            height=config.point_height,
            width=config.point_width,
        )
        segmentation_renderer.enable_segmentation_rendering()

    writer = None
    renderer = None
    if video_path is not None:
        import cv2

        video_path = Path(video_path)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        fps = float(np.asarray(arrays.get("fps", 20.0)).item()) / config.render_stride
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (config.width, config.height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"could not open video writer: {video_path}")
        renderer = mujoco.Renderer(model, height=config.height, width=config.width)
    try:
        for index, pose in enumerate(poses):
            data.mocap_pos[mocap_id] = pose[:3]
            data.mocap_quat[mocap_id] = pose[3:7]
            mujoco.mj_forward(model, data)
            groups, minimum = _contact_groups(
                model, data, source_bodies, target_bodies
            )
            contact_names.append(sorted(groups))
            contact_count[index] = len(groups)
            if math.isfinite(minimum):
                penetration[index] = minimum
            actual[index, :3] = data.xpos[source_id]
            actual[index, 3:7] = data.xquat[source_id]
            if point_cloud_task is not None:
                cloud, sampling_group = _collect_point_cloud_frame(
                    model,
                    data,
                    depth_renderer=depth_renderer,
                    segmentation_renderer=segmentation_renderer,
                    source_bodies=source_bodies,
                    target_bodies=target_bodies,
                    obstacle_bodies=obstacle_bodies,
                    task_frame_world=task_frame_world,
                    config=config,
                    frame_index=index,
                )
                point_cloud_task[index] = cloud
                point_cloud_sampling_group[index] = sampling_group
                point_semantic_counts[index] = np.bincount(
                    np.argmax(cloud[:, 3:6], axis=1),
                    minlength=3,
                )
                point_sampling_group_counts[index] = np.bincount(
                    sampling_group,
                    minlength=4,
                )
            if renderer is not None and index % config.render_stride == 0:
                renderer.update_scene(data, camera=config.camera)
                frame_bgr = np.ascontiguousarray(renderer.render()[:, :, ::-1])
                writer.write(frame_bgr)
            mujoco.mj_step(model, data)
    finally:
        if renderer is not None:
            renderer.close()
        if depth_renderer is not None:
            depth_renderer.close()
        if segmentation_renderer is not None:
            segmentation_renderer.close()
        if writer is not None:
            writer.release()

    position_error = np.linalg.norm(actual[:, :3] - poses[:, :3], axis=1)
    quaternion_alignment = np.abs(np.sum(actual[:, 3:7] * poses[:, 3:7], axis=1))
    quaternion_alignment = np.clip(quaternion_alignment, -1.0, 1.0)
    orientation_error = 2.0 * np.arccos(quaternion_alignment)
    unique_contacts = sorted({name for names in contact_names for name in names})
    report = {
        "schema": "flora_geometric_mujoco_replay_v1",
        "episode": str(episode_path),
        "frames": int(len(poses)),
        "kinematic_mocap_replay": True,
        "robot_executed": False,
        "liquid_validated": False,
        "max_position_tracking_error_m": float(np.max(position_error)),
        "max_orientation_tracking_error_rad": float(np.max(orientation_error)),
        "contact_frames": int(np.count_nonzero(contact_count)),
        "contact_bodies": unique_contacts,
        "minimum_contact_distance_m": (
            float(np.nanmin(penetration)) if np.isfinite(penetration).any() else None
        ),
        "video": None if video_path is None else str(Path(video_path).resolve()),
        "point_cloud_collected": bool(point_cloud_task is not None),
        "point_cloud_shape": (
            None if point_cloud_task is None else list(point_cloud_task.shape)
        ),
        "point_cloud_cameras": (
            None if point_cloud_task is None else list(config.point_cameras)
        ),
        "point_cloud_render_size": (
            None
            if point_cloud_task is None
            else [int(config.point_height), int(config.point_width)]
        ),
        "point_cloud_semantic_count_min": (
            None
            if point_semantic_counts is None
            else point_semantic_counts.min(axis=0).tolist()
        ),
        "point_cloud_semantic_count_mean": (
            None
            if point_semantic_counts is None
            else point_semantic_counts.mean(axis=0).tolist()
        ),
        "point_cloud_sampling_group_names": (
            None
            if point_sampling_group_counts is None
            else ["source", "target", "obstacle", "other_environment"]
        ),
        "point_cloud_sampling_group_count_min": (
            None
            if point_sampling_group_counts is None
            else point_sampling_group_counts.min(axis=0).tolist()
        ),
        "point_cloud_sampling_group_count_mean": (
            None
            if point_sampling_group_counts is None
            else point_sampling_group_counts.mean(axis=0).tolist()
        ),
    }
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "replay_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        np.savez_compressed(
            output_dir / "replay_trace.npz",
            commanded_pose_world=poses.astype(np.float32),
            actual_pose_world=actual.astype(np.float32),
            position_error_m=position_error.astype(np.float32),
            orientation_error_rad=orientation_error.astype(np.float32),
            contact_group_count=contact_count,
            minimum_contact_distance_m=penetration,
            **(
                {
                    "point_cloud_semantic_count": point_semantic_counts,
                    "point_cloud_sampling_group_count": point_sampling_group_counts,
                }
                if point_semantic_counts is not None
                else {}
            ),
        )
        if point_cloud_task is not None:
            augmented_metadata = dict(metadata)
            augmented_metadata["local_dp3_observation"] = {
                "schema": "flora_local_dp3_point_cloud_v1",
                "point_cloud_key": "point_cloud_task",
                "shape_per_frame": [config.point_cloud_points, 6],
                "frame": "task",
                "features": [
                    "x",
                    "y",
                    "z",
                    "source_one_hot",
                    "target_one_hot",
                    "environment_one_hot",
                ],
                "cameras": list(config.point_cameras),
                "render_height": int(config.point_height),
                "render_width": int(config.point_width),
                "sampling": "task_relevant_reserved_then_uniform_v2",
                "source_reserved_fraction": 0.25,
                "target_reserved_fraction": 0.25,
                "obstacle_reserved_fraction": float(
                    config.point_obstacle_reserved_fraction
                ),
                "sampling_group_key": "point_cloud_sampling_group",
                "sampling_group_names": [
                    "source",
                    "target",
                    "obstacle",
                    "other_environment",
                ],
                "crop_min_task": list(config.point_crop_min_task),
                "crop_max_task": list(config.point_crop_max_task),
            }
            augmented = dict(arrays)
            augmented["point_cloud_task"] = point_cloud_task.astype(np.float32)
            augmented["point_cloud_sampling_group"] = (
                point_cloud_sampling_group.astype(np.int8)
            )
            if augmented_metadata.get("teacher_representation", {}).get("schema_version") == 2:
                missing_v2 = {"global_reference_index", "global_source_time"} - set(augmented)
                if missing_v2:
                    raise ValueError(f"v2 teacher missing mapping arrays: {sorted(missing_v2)}")
            if "global_reference_index" not in augmented:
                global_trajectory = np.asarray(
                    augmented["global_trajectory"], dtype=np.float32
                )
                augmented["global_progress"] = np.linspace(
                    0.0, 1.0, len(poses), dtype=np.float32
                )
                augmented["global_reference_index"] = np.rint(
                    augmented["global_progress"]
                    * float(global_trajectory.shape[0] - 1)
                ).astype(np.int16)
                augmented_metadata["local_dp3_observation"][
                    "global_reference_index_source"
                ] = "legacy_uniform_teacher_adapter_v1"
            augmented["metadata"] = np.asarray(
                json.dumps(augmented_metadata, sort_keys=True)
            )
            np.savez_compressed(
                output_dir / "dp3_episode.npz",
                **augmented,
            )
    return report
