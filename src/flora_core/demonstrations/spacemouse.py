"""Record guided, human-operated cup trajectories in MuJoCo."""

from __future__ import annotations

import argparse
from collections import deque
from contextlib import closing
import json
from pathlib import Path
from struct import unpack_from
import tempfile
from time import monotonic, sleep
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from flora_core.geometry.frames import (
    FRAME_CONVENTION,
    make_cup_to_bowl_task_frame,
    pose_velocity_actions,
    quat_wxyz_to_rotation,
    rotation_to_quat_wxyz,
    world_pose_series_to_task,
)
from flora_core.libero_utils import find_libero_root
from flora_core.realization import collection
from flora_core.simulation.replay import (
    SOURCE_BODY,
    TARGET_BODY,
    ReplayConfig,
    _contact_groups,
    _descendants,
    _scene_xml,
)

DT = 0.05


def _build_scene(
    protocol: dict[str, Any]
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any], dict[str, Any]]:
    distance = float(protocol["scene"]["distance_m"])
    table_height = 0.9
    nominal = np.asarray(protocol["asset"]["source_nominal_quat_wxyz"], dtype=np.float64)
    base = {
        # Match the collection protocol's canonical LIBERO placement: motion
        # is world +y, so world-x obstacle width becomes task-frame lateral
        # width rather than incorrectly blocking the source at reset.
        "cup_xy": [0.0, -distance],
        "cup_z": table_height,
        "cup_quat": nominal.tolist(),
        "bowl_xy": [0.0, 0.0],
        "bowl_z": table_height,
        "bowl_quat": [1.0, 0.0, 0.0, 0.0],
    }
    scene = collection.apply_scene_factors(base, protocol, np.asarray([0.0, 1.0]))
    cup_world = np.asarray([*scene["cup_xy"], scene["cup_z"]], dtype=np.float64)
    bowl_world = np.asarray([*scene["bowl_xy"], scene["bowl_z"]], dtype=np.float64)
    task_frame, task_rotation = make_cup_to_bowl_task_frame(cup_world, bowl_world)
    _, obstacles = collection.resolve_obstacles(
        protocol,
        cup_world,
        bowl_world,
        SimpleNamespace(libero_table=True, shift_libero_table_to_zero=False),
    )
    affordance = collection.instantiate_affordance(
        protocol, task_frame, np.asarray(scene["cup_quat"], dtype=np.float64)
    )
    start = task_rotation.T @ (cup_world - bowl_world)
    cup_rotation_task = task_rotation.T @ quat_wxyz_to_rotation(scene["cup_quat"]).as_matrix()
    affordance["source_anchor_task_initial"] = (
        start + cup_rotation_task @ np.asarray(affordance["source_anchor_local"])
    ).tolist()
    clearance = 0.070
    minimum_z = clearance + 0.005
    planning_start = start.copy()
    planning_start[2] = max(float(planning_start[2]), minimum_z)
    planning_goal = np.asarray(affordance["goal_cup_origin_task"], dtype=np.float64)
    planning_goal[2] = max(float(planning_goal[2]), minimum_z)
    corridor = collection.plan_corridor(
        planning_start,
        planning_goal,
        obstacles,
        str(protocol["realization"]["topology_class"]),
        seed=0,
        cup_clearance_radius_m=clearance,
        corridor_radius_m=0.065,
        workspace_z_task=(0.0, 0.6),
        tabletop_z_task=0.0,
    )
    corridor.update(
        {
            "initial_supported_start_task": start.tolist(),
            "free_space_planning_start_task": planning_start.tolist(),
            "interaction_goal_cup_origin_task": np.asarray(
                affordance["goal_cup_origin_task"]
            ).tolist(),
            "free_space_planning_goal_task": planning_goal.tolist(),
        }
    )
    scene.update(
        {
            "task_frame_world": task_frame.tolist(),
            "cup_quat_task": rotation_to_quat_wxyz(
                Rotation.from_matrix(cup_rotation_task)
            ).tolist(),
            "obstacles": obstacles,
            "task_frame_convention": FRAME_CONVENTION,
        }
    )
    return scene, task_frame, affordance, corridor


def _resample_global_trajectory(poses: np.ndarray, horizon: int) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    old = np.linspace(0.0, 1.0, len(poses))
    new = np.linspace(0.0, 1.0, int(horizon))
    position = np.stack([np.interp(new, old, poses[:, axis]) for axis in range(3)], axis=1)
    rotations = Rotation.from_quat(poses[:, [4, 5, 6, 3]])
    sampled = Slerp(old, rotations)(new).as_matrix()
    rot6d = sampled[:, :2, :].reshape(-1, 6)
    return np.concatenate((position, rot6d), axis=1).astype(np.float32)


def _phase_progress(phase_id: np.ndarray) -> np.ndarray:
    output = np.zeros(len(phase_id), dtype=np.float32)
    for phase in np.unique(phase_id):
        indices = np.flatnonzero(phase_id == phase)
        output[indices] = np.linspace(0.0, 1.0, len(indices), dtype=np.float32)
    return output

def _episode_payload(protocol, scene, frame, affordance, corridor, world, phases, target):
    """Convert recorded poses to the replay and Local DP3 episode schema."""
    world = np.asarray(world, dtype=np.float32)
    phase = np.asarray(phases, dtype=np.int8)
    if len(world) < 2 or not ((phase == 0).any() and (phase == 1).any()):
        raise ValueError("Record both transport and pour before saving.")
    poses = world_pose_series_to_task(world, frame).astype(np.float32)
    progress = _phase_progress(phase)
    rotations = Rotation.from_quat(world[:, [4, 5, 6, 3]])
    upright = rotations.apply(rotations[0].inv().apply([0, 0, 1]))
    tilt = np.degrees(np.arccos(np.clip(upright[:, 2], -1, 1)))
    actions = pose_velocity_actions(poses, DT).astype(np.float32)
    centerline = np.asarray(corridor["centerline_task"])
    transport = poses[phase == 0, :3]
    deviation = [
        collection.point_to_polyline_distance(position, centerline)
        for position in poses[:, :3]
    ]
    spec = protocol["realization"]
    distance = float(scene["source_target_initial_xy_distance"])
    metadata = {
        "format": "flora_spacemouse_demo_v1",
        "human_demonstration": True,
        "synthetic": False,
        "robot_executed": False,
        "liquid_validated": False,
        "collection_row_id": protocol["collection_row_id"],
        "protocol_path": protocol["protocol_path"],
        "collection_design_path": protocol["collection_design_path"],
        "scene": scene,
        "realization": {
            "realization_id": spec["realization_id"],
            "normalized_specification": {
                "schema": "normalized_realization_specification_v2_1d_target",
                "u_o": spec["u_o"],
                "u_o_level": spec["u_o_level"],
                "u_g": spec["u_g"],
                "u_g_level": spec["u_g_level"],
                "c": spec["topology_class"],
            },
            "instantiated_geometry": {
                "schema": "scene_specific_realization_instantiation_v1",
                "source_geometry_id": scene["source_geometry_id"],
                "target_geometry_id": scene["target_geometry_id"],
                "p_o_source_local": affordance["source_anchor_local"],
                "p_o_task_initial": affordance["source_anchor_task_initial"],
                "p_g_task": affordance["target_guidance_center_task"],
                "interaction_target_task": affordance["target_anchor_task"],
                "gamma_c_task": centerline.tolist(),
                "gamma_c_normalized_by_task_distance": (centerline / distance).tolist(),
                "task_distance_m": distance,
                "corridor_radius_m": corridor["radius_m"],
            },
            "validation": collection.classify_topology(transport, scene["obstacles"]),
        },
        "phase_names": {"0": "transport", "1": "pour", "2": "hold"},
    }
    horizon = 32
    global_progress = np.linspace(0, 1, len(world), dtype=np.float32)
    return {
        "metadata": np.asarray(json.dumps(metadata)),
        "source_pose_world": world,
        "object_pose_world": world,
        "source_pose_task": poses,
        "object_pose_task": poses,
        "target_pose_world": target,
        "target_pose_world_series": np.repeat(target[None], len(world), axis=0),
        "task_frame_world": frame,
        "actions_task": actions,
        "actions_task_velocity": actions,
        "state_task": np.column_stack((poses, np.eye(3)[phase], progress)).astype(np.float32),
        "phase_id": phase,
        "phase_progress": progress,
        "source_tilt_angle_deg": tilt.astype(np.float32),
        "global_trajectory": _resample_global_trajectory(poses, horizon),
        "global_source_time": np.linspace(0, (len(world) - 1) * DT, horizon),
        "global_progress": global_progress,
        "global_reference_index": np.rint(global_progress * (horizon - 1)).astype(np.int16),
        "corridor_deviation": np.asarray([item[0] for item in deviation], dtype=np.float32),
        "corridor_waypoint": np.asarray([item[1] for item in deviation], dtype=np.int32),
        "time": np.arange(len(world), dtype=np.float64) * DT,
        "dt": np.asarray(DT),
        "fps": np.asarray(1 / DT),
    }


def _save_episode(out_dir, payload):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"spacemouse_{uuid4().hex}.npz"
    with tempfile.NamedTemporaryFile(dir=out_dir, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            np.savez_compressed(stream, **payload)
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return path


def _draw_guidance(mujoco, visual, centerline, anchors, waypoint):
    """Put guidance in the viewer overlay so replay observations stay clean."""
    visual.ngeom = 0
    for index, (start, end) in enumerate(zip(centerline[:-1], centerline[1:])):
        if visual.ngeom >= visual.maxgeom - len(anchors):
            break
        geom = visual.geoms[visual.ngeom]
        mujoco.mjv_initGeom(
            geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
            np.eye(3).ravel(), np.array([0.1, 0.75, 1, 1 if index == waypoint else 0.45]),
        )
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.005, start, end)
        visual.ngeom += 1
    for point in anchors:
        geom = visual.geoms[visual.ngeom]
        mujoco.mjv_initGeom(
            geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, 0.014), point,
            np.eye(3).ravel(), np.array([1, 0.65, 0.1, 0.85]),
        )
        visual.ngeom += 1


def _read_motion(device, axes):
    """Consume pending 3Dconnexion HID reports using AEXT's axis mapping."""
    for _ in range(64):
        report = bytes(device.read(64))
        if not report:
            break
        if report[0] == 1 and len(report) >= 7:
            y, x, z = unpack_from("<3h", report, 1)
            axes[:3] = [x, y, -z]
            if len(report) >= 13:
                roll, pitch, yaw = unpack_from("<3h", report, 7)
                axes[3:] = [pitch, roll, -yaw]
        elif report[0] == 2 and len(report) >= 7:
            roll, pitch, yaw = unpack_from("<3h", report, 1)
            axes[3:] = [pitch, roll, -yaw]
    motion = axes / 350.0
    motion[np.abs(motion) < 0.05] = 0
    for part in (motion[:3], motion[3:]):
        part /= max(1.0, np.linalg.norm(part))
    return motion[:3], motion[3:]


def _collect(args, protocol, scene, frame, affordance, corridor):
    import mujoco
    import mujoco.viewer
    import hid

    source = np.array([*scene["cup_xy"], scene["cup_z"], *scene["cup_quat"]])
    target = np.array([*scene["bowl_xy"], scene["bowl_z"], *scene["bowl_quat"]])
    root = args.libero_root or find_libero_root()
    model = mujoco.MjModel.from_xml_string(
        _scene_xml({"scene": scene}, source, target, root, ReplayConfig())
    )
    data = mujoco.MjData(model)
    source_id = model.body(SOURCE_BODY).id
    mocap = model.body_mocapid[source_id]
    source_bodies = _descendants(model, source_id)
    target_bodies = _descendants(model, model.body(TARGET_BODY).id)
    frame_rotation = quat_wxyz_to_rotation(frame[3:])
    centerline = np.asarray(corridor["centerline_task"])
    world_line = frame[:3] + frame_rotation.apply(centerline)
    anchor_local = np.asarray(affordance["source_anchor_local"])
    target_anchor = frame[:3] + frame_rotation.apply(affordance["target_guidance_center_task"])

    def place(pose):
        data.mocap_pos[mocap] = pose[:3]
        data.mocap_quat[mocap] = pose[3:]
        mujoco.mj_forward(model, data)

    place(source)
    initial_groups, initial_depth = _contact_groups(model, data, source_bodies, target_bodies)
    table_limit = min(-0.001, initial_depth) if initial_groups == {"table"} else -0.001
    keys = deque()
    devices = hid.enumerate(args.vendor_id, args.product_id)
    if not devices:
        raise RuntimeError("No matching SpaceMouse found. Check the USB connection and device IDs.")
    # Prefer the multi-axis HID interface on composite USB devices.
    entry = next((item for item in devices if item["usage"] == 8 and item["usage_page"] == 1), devices[0])
    device = hid.device()
    axes = np.zeros(6)
    pose = source.copy()
    world, phases = [], []
    phase = 0
    recording = False
    print("Enter starts | P marks pour | H marks hold | S saves | R resets | Q quits", flush=True)
    with closing(device):
        device.open_path(entry["path"])
        device.set_nonblocking(True)
        with mujoco.viewer.launch_passive(model, data, key_callback=keys.append) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = model.camera("replay").id
            while viewer.is_running():
                deadline = monotonic() + DT
                while keys:
                    key = keys.popleft()
                    if key == ord("Q"):
                        return
                    if key == 257 and not recording:
                        pose = source.copy()
                        recording = True
                        print("Recording transport", flush=True)
                    if key == ord("P") and recording and phase == 0 and world:
                        phase = 1
                        print("Recording pour", flush=True)
                    if key == ord("H") and recording and phase == 1 and phases[-1] == 1:
                        phase = 2
                        print("Recording hold", flush=True)
                    if key == ord("S") and recording:
                        try:
                            payload = _episode_payload(
                                protocol, scene, frame, affordance, corridor, world, phases, target
                            )
                            path = _save_episode(args.out_dir, payload)
                        except (ValueError, OSError) as exc:
                            print(exc, flush=True)
                            continue
                        print(f"Saved {path}", flush=True)
                        key = ord("R")
                    if key == ord("R"):
                        world.clear()
                        phases.clear()
                        phase = 0
                        pose = source.copy()
                        recording = False
                        print("Ready. Press Enter to record.", flush=True)

                translation, rotation = _read_motion(device, axes)
                with viewer.lock():
                    if recording and world:
                        candidate = pose.copy()
                        candidate[:3] += frame_rotation.apply(translation * args.move_speed * DT)
                        delta = Rotation.from_rotvec(frame_rotation.apply(rotation * args.turn_speed * DT))
                        candidate[3:] = rotation_to_quat_wxyz(delta * quat_wxyz_to_rotation(pose[3:]))
                        place(candidate)
                        groups, depth = _contact_groups(model, data, source_bodies, target_bodies)
                        limit = table_limit if groups == {"table"} else -0.001
                        if depth >= limit - 1e-8:
                            pose = candidate
                    place(pose)
                    task_position = frame_rotation.inv().apply(pose[:3] - frame[:3])
                    _, waypoint = collection.point_to_polyline_distance(task_position, centerline)
                    source_anchor = pose[:3] + quat_wxyz_to_rotation(pose[3:]).apply(anchor_local)
                    _draw_guidance(
                        mujoco, viewer.user_scn, world_line, (source_anchor, target_anchor), waypoint
                    )
                    if recording:
                        world.append(pose.copy())
                        phases.append(phase)
                viewer.sync()
                sleep(max(0.0, deadline - monotonic()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-config", required=True, type=Path)
    parser.add_argument("--collection-design", required=True, type=Path)
    parser.add_argument("--collection-row-id", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--libero-root", type=Path)
    parser.add_argument("--vendor-id", type=lambda value: int(value, 0), default=0x256F)
    parser.add_argument("--product-id", type=lambda value: int(value, 0), default=0xC635)
    parser.add_argument("--move-speed", type=float, default=0.12, help="Maximum translation in m/s")
    parser.add_argument("--turn-speed", type=float, default=1.0, help="Maximum rotation in rad/s")
    args = parser.parse_args(argv)
    if any(not np.isfinite(speed) or speed <= 0 for speed in (args.move_speed, args.turn_speed)):
        parser.error("movement speeds must be finite and positive")
    try:
        protocol = collection.load_collection_row(
            args.collection_config, args.collection_design, args.collection_row_id
        )
        if not protocol["combination_valid"]:
            raise ValueError(protocol["invalid_reason"])
        scene, frame, affordance, corridor = _build_scene(protocol)
        _collect(args, protocol, scene, frame, affordance, corridor)
    except ImportError as exc:
        parser.exit(1, f"{exc}. Install flora-core[spacemouse].\n")
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
