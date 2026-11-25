"""Robosuite OSC adapter for Panda, Sawyer, and UR5e shadow evaluation.

This adapter operates on a caller-created, reset, single-arm robosuite/LIBERO
environment. Scene creation, object attachment, target/obstacle placement and
asset selection remain explicit inputs supplied by the experiment.
"""
from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
from scipy.spatial.transform import Rotation

from flora_core.robotics.global_motion import GlobalMotionBackend
from flora_core.libero_utils import find_libero_root


SUPPORTED_ROBOTS = ("Panda", "Sawyer", "UR5e")
_STATE_FIELDS = ("qpos", "qvel", "ctrl", "act", "mocap_pos", "mocap_quat", "userdata")
_CONTROLLER_FIELDS = (
    "goal_pos", "goal_ori", "initial_joint", "initial_ee_pos",
    "initial_ee_ori_mat", "ori_ref", "relative_ori", "input_min",
    "input_max", "output_min", "output_max",
)


def _compose_pose7(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    first, second = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    ra = Rotation.from_quat(first[[4, 5, 6, 3]])
    rb = Rotation.from_quat(second[[4, 5, 6, 3]])
    q = (ra * rb).as_quat()
    return np.r_[first[:3] + ra.apply(second[:3]), q[[3, 0, 1, 2]]]


def _inverse_pose7(pose: np.ndarray) -> np.ndarray:
    value = np.asarray(pose, dtype=np.float64)
    rotation = Rotation.from_quat(value[[4, 5, 6, 3]])
    inverse = rotation.inv()
    q = inverse.as_quat()
    return np.r_[inverse.apply(-value[:3]), q[[3, 0, 1, 2]]]


def register_libero_robots(libero_root: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Register the three mounted robot models from the vendored AEXT helper."""
    path = Path(__file__).resolve().parents[3] / "assets" / "aext" / "scripts" / "three_arm_experiment" / "robot_registry.py"
    spec = importlib.util.spec_from_file_location("flora_core_aext_robot_registry", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load robot registry script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    root = find_libero_root() if libero_root is None else Path(libero_root).expanduser().resolve()
    return module.register_three_arm_models(root)


class RobosuiteOSCBackend:
    """Implement the GlobalMotionBackend protocol over a robosuite env.

    ``task_frame_world`` and ``object_T_tcp`` are pose7 values in wxyz
    convention. ``robot_collision_geoms`` and ``environment_collision_geoms``
    define the paper's forbidden robot/environment pair policy. Legal grasp
    contacts are excluded by construction because held-object geoms should not
    be listed as robot/environment pairs.
    """

    def __init__(
        self,
        env: Any,
        *,
        robot_name: str,
        task_frame_world: np.ndarray,
        object_T_tcp: np.ndarray,
        robot_collision_geoms: set[str] | None = None,
        environment_collision_geoms: set[str] | None = None,
        osc_action_from_error: Callable[[np.ndarray], np.ndarray] | None = None,
        control_output_bounds: tuple[np.ndarray, np.ndarray] | None = None,
        gripper_hold_command: float = 1.0,
    ) -> None:
        if robot_name not in SUPPORTED_ROBOTS:
            raise ValueError(f"robot_name must be one of {SUPPORTED_ROBOTS}")
        if len(env.robots) != 1:
            raise ValueError("RobosuiteOSCBackend requires exactly one robot")
        self.env = env
        self.robot_name = robot_name
        self.robot = env.robots[0]
        self.sim = env.sim
        self.task_frame_world = np.asarray(task_frame_world, dtype=np.float64).reshape(7)
        self.object_T_tcp = np.asarray(object_T_tcp, dtype=np.float64).reshape(7)
        self.robot_collision_geoms = set(robot_collision_geoms or ())
        self.environment_collision_geoms = set(environment_collision_geoms or ())
        self._osc_action_from_error = osc_action_from_error
        controller = self.robot.controller
        self.control_output_bounds = (
            (np.asarray(controller.output_min, dtype=np.float64).copy(),
             np.asarray(controller.output_max, dtype=np.float64).copy())
            if control_output_bounds is None else control_output_bounds
        )
        self.gripper_hold_command = float(gripper_hold_command)
        self.last_collision_pairs: list[tuple[str, str]] = []
        self._site_name = self.sim.model.site_id2name(int(self.robot.eef_site_id))
        self._qpos_ids = list(self.robot._ref_joint_pos_indexes)
        self._qvel_ids = list(self.robot._ref_joint_vel_indexes)
        if not self.robot_collision_geoms:
            self.robot_collision_geoms.update(getattr(self.robot.robot_model, "contact_geoms", ()))
            self.robot_collision_geoms.update(getattr(self.robot.gripper, "contact_geoms", ()))
        self._validate_collision_geometries()
        self._validate_robot_identity()

    def _validate_robot_identity(self) -> None:
        model = self.robot.robot_model
        names = " ".join((type(model).__name__, str(getattr(model, "name", "")))).lower()
        token = {"Panda": "panda", "Sawyer": "sawyer", "UR5e": "ur5"}[self.robot_name]
        if token not in names:
            raise ValueError(
                f"robot_name={self.robot_name!r} does not match environment model {type(model).__name__!r}"
            )

    def _validate_collision_geometries(self) -> None:
        if not self.robot_collision_geoms:
            raise ValueError("robot collision geometry list is empty")
        if not self.environment_collision_geoms:
            raise ValueError(
                "environment_collision_geoms must include forbidden target, obstacle, and table geoms"
            )
        known = {
            self.sim.model.geom_id2name(index)
            for index in range(int(self.sim.model.ngeom))
        }
        unknown = (self.robot_collision_geoms | self.environment_collision_geoms) - known
        if unknown:
            raise ValueError(f"collision geometry names not found in MuJoCo model: {sorted(unknown)}")
        overlap = self.robot_collision_geoms & self.environment_collision_geoms
        if overlap:
            raise ValueError(f"robot and environment collision geometry lists overlap: {sorted(overlap)}")

    def _controller(self) -> Any:
        return self.robot.controller

    def snapshot_state(self) -> dict[str, Any]:
        data = {key: np.asarray(getattr(self.sim.data, key)).copy()
                for key in _STATE_FIELDS if hasattr(self.sim.data, key)}
        if hasattr(self.sim.data, "time"):
            data["time"] = float(self.sim.data.time)
        controller = self._controller()
        controls = {}
        for key in _CONTROLLER_FIELDS:
            if hasattr(controller, key):
                value = getattr(controller, key)
                if isinstance(value, np.ndarray):
                    controls[key] = value.copy()
                elif isinstance(value, (int, float, bool, type(None), list, tuple)):
                    controls[key] = deepcopy(value)
        interpolators = {}
        for name in ("interpolator_pos", "interpolator_ori"):
            interpolator = getattr(controller, name, None)
            if interpolator is not None:
                interpolators[name] = {
                    key: (np.asarray(getattr(interpolator, key)).copy()
                          if isinstance(getattr(interpolator, key), np.ndarray)
                          else deepcopy(getattr(interpolator, key)))
                    for key in ("start", "goal", "step")
                    if hasattr(interpolator, key)
                }
        return {"data": data, "controller": controls, "interpolators": interpolators}

    def restore_state(self, state: dict[str, Any]) -> None:
        for key, value in state["data"].items():
            if key == "time":
                self.sim.data.time = float(value)
            elif hasattr(self.sim.data, key):
                getattr(self.sim.data, key)[...] = value
        self.sim.forward()
        controller = self._controller()
        for key, value in state["controller"].items():
            setattr(controller, key, value.copy() if isinstance(value, np.ndarray) else deepcopy(value))
        for name, values in state.get("interpolators", {}).items():
            interpolator = getattr(controller, name, None)
            if interpolator is not None:
                for key, value in values.items():
                    setattr(interpolator, key, value.copy() if isinstance(value, np.ndarray) else deepcopy(value))
        refresh = getattr(self.env, "_refresh_observations", None)
        if callable(refresh):
            refresh()

    def pose_task_to_world(self, pose7: np.ndarray) -> np.ndarray:
        return _compose_pose7(self.task_frame_world, pose7)

    def object_pose_to_tcp_pose(self, object_pose_world: np.ndarray) -> np.ndarray:
        return _compose_pose7(object_pose_world, self.object_T_tcp)

    def current_task_pose(self) -> np.ndarray:
        return _compose_pose7(_inverse_pose7(self.task_frame_world), self.tcp_pose_world())

    def set_held_object_world_pose(self, pose7: np.ndarray) -> None:
        # The physical grasp/attachment is part of the provided scene backend.
        # No teleport is performed during OSC shadow rollout.
        del pose7

    def _site_pose(self) -> np.ndarray:
        site_id = int(self.robot.eef_site_id)
        position = np.asarray(self.sim.data.site_xpos[site_id], dtype=np.float64)
        matrix = np.asarray(self.sim.data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
        q = Rotation.from_matrix(matrix).as_quat()[[3, 0, 1, 2]]
        return np.r_[position, q]

    def tcp_pose_world(self) -> np.ndarray:
        return self._site_pose()

    def joint_state(self) -> np.ndarray:
        return np.asarray(self.sim.data.qpos[self._qpos_ids], dtype=np.float64).copy()

    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        lower, upper = [], []
        for name in self.robot.robot_joints:
            jid = self.sim.model.joint_name2id(name)
            if self.sim.model.jnt_limited[jid]:
                lo, hi = self.sim.model.jnt_range[jid]
            else:
                lo, hi = -np.pi, np.pi
            lower.append(float(lo)); upper.append(float(hi))
        return np.asarray(lower), np.asarray(upper)

    def set_joint_state(self, q: np.ndarray) -> None:
        value = np.asarray(q, dtype=np.float64)
        if value.shape != (len(self._qpos_ids),):
            raise ValueError("joint vector shape does not match robot")
        self.sim.data.qpos[self._qpos_ids] = value
        self.sim.data.qvel[self._qvel_ids] = 0.0
        self.sim.forward()

    def solve_ik(self, tcp_pose_world: np.ndarray, seed: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        from flora_core.robotics.ik import solve_site_ik

        lower, upper = self.joint_limits()
        pose = np.asarray(tcp_pose_world, dtype=np.float64)
        quat = pose[[4, 5, 6, 3]]
        target_rot = Rotation.from_quat(quat).as_matrix()
        q, info = solve_site_ik(
            self.sim, int(self.robot.eef_site_id), self._site_name,
            self._qpos_ids, self._qvel_ids, lower, upper,
            pose[:3], target_rot, seed_q=seed,
        )
        return q, {"converged": info["converged"],
                   "position_error_m": info["pos_error_m"],
                   "orientation_error_rad": info["rot_error_rad"], **info}

    def robot_environment_collision_pairs(self) -> list[tuple[str, str]]:
        pairs = []
        for index in range(int(self.sim.data.ncon)):
            contact = self.sim.data.contact[index]
            first = self.sim.model.geom_id2name(int(contact.geom1)) or ""
            second = self.sim.model.geom_id2name(int(contact.geom2)) or ""
            if ((first in self.robot_collision_geoms and second in self.environment_collision_geoms)
                    or (second in self.robot_collision_geoms and first in self.environment_collision_geoms)):
                pairs.append((first, second))
        self.last_collision_pairs = pairs
        return pairs

    def joint_limit_margin(self) -> float:
        q = self.joint_state(); lower, upper = self.joint_limits()
        return float(np.min(np.minimum(q - lower, upper - q)))

    def joint_safe(self) -> bool:
        return self.joint_limit_margin() >= 0.0

    def collision(self) -> bool:
        return bool(self.robot_environment_collision_pairs())

    def osc_action_for_pose_error(self, target_tcp_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        target = np.asarray(target_tcp_world, dtype=np.float64)
        current = self.tcp_pose_world()
        dp = target[:3] - current[:3]
        target_q = target[[4, 5, 6, 3]]
        current_q = current[[4, 5, 6, 3]]
        dr = (Rotation.from_quat(target_q) * Rotation.from_quat(current_q).inv()).as_rotvec()
        physical = np.r_[dp, dr]
        if self._osc_action_from_error is not None:
            action = np.asarray(self._osc_action_from_error(physical), dtype=np.float32)
        else:
            # robosuite OSC_POSE accepts normalized Cartesian position and
            # axis-angle deltas; clip to its declared physical output range.
            controller = self._controller()
            lo = np.asarray(controller.output_min, dtype=np.float64)
            hi = np.asarray(controller.output_max, dtype=np.float64)
            clipped = np.clip(physical, lo, hi)
            alpha = (clipped - lo) / (hi - lo)
            input_lo = np.asarray(controller.input_min, dtype=np.float64)
            input_hi = np.asarray(controller.input_max, dtype=np.float64)
            action = (input_lo + alpha * (input_hi - input_lo)).astype(np.float32)
        if self.control_output_bounds is None:
            saturation = np.zeros(6, dtype=bool)
        else:
            lo, hi = (np.asarray(x, dtype=np.float64) for x in self.control_output_bounds)
            saturation = (physical < lo) | (physical > hi)
        return action, saturation

    def step(self, action: np.ndarray) -> None:
        command = np.asarray(action, dtype=np.float32).reshape(-1)
        action_dim = int(getattr(self.robot, "action_dim", command.size))
        if command.size == action_dim - 1:
            command = np.r_[command, self.gripper_hold_command].astype(np.float32)
        if command.size != action_dim:
            raise ValueError(f"OSC action has {command.size} values; expected {action_dim}")
        self.env.step(command)


def create_libero_osc_backend(
    env_factory: Callable[[], Any],
    *,
    robot_name: str,
    libero_root: str | Path | None = None,
    task_frame_world: np.ndarray,
    object_T_tcp: np.ndarray,
    **backend_options: Any,
) -> RobosuiteOSCBackend:
    """Register the requested AEXT robot, build the scene, and wrap its env.

    ``env_factory`` must create a resettable one-arm OSC_POSE environment with
    the source held according to the supplied grasp profile and the paper's
    task scene already instantiated.
    """
    if robot_name not in SUPPORTED_ROBOTS:
        raise ValueError(f"robot_name must be one of {SUPPORTED_ROBOTS}")
    register_libero_robots(libero_root)
    env = env_factory()
    if len(getattr(env, "robots", ())) != 1:
        raise ValueError("env_factory must create a single-arm environment")
    controller = env.robots[0].controller
    controller_name = getattr(controller, "name", type(controller).__name__)
    if controller_name != "OSC_POSE":
        raise ValueError(f"the robot backend requires OSC_POSE, got {controller_name}")
    if getattr(controller, "impedance_mode", "fixed") != "fixed":
        raise ValueError("the robot backend requires fixed-impedance OSC_POSE")
    return RobosuiteOSCBackend(env, robot_name=robot_name,
                               task_frame_world=task_frame_world,
                               object_T_tcp=object_T_tcp, **backend_options)
