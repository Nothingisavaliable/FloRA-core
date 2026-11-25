"""Particle-fluid backends for pouring simulation.

The first backend still uses MuJoCo free-body sphere particles. Future backends
can replace only this layer while the robot/container simulation stays in
MuJoCo or LIBERO.
"""

from abc import ABC, abstractmethod
import contextlib
from dataclasses import dataclass
import io
import os

import mujoco
import numpy as np

try:
    os.environ.setdefault("TI_LOG_LEVEL", "error")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import taichi as ti
except Exception:  # pragma: no cover - optional dependency
    ti = None

_TAICHI_AVAILABLE = ti is not None


if ti is None:  # pragma: no cover - import-only fallback for optional backend
    class _MissingTaichiTypes:
        @staticmethod
        def vector(*args, **kwargs):
            del args, kwargs
            return object

    class _MissingTaichi:
        f32 = object
        i32 = object
        types = _MissingTaichiTypes()

        @staticmethod
        def data_oriented(cls):
            return cls

        @staticmethod
        def kernel(fn):
            return fn

        @staticmethod
        def template():
            return object

    ti = _MissingTaichi()


PARTICLE_PREFIX = "water_particle_"
TARGET_BOX_CAPACITY = 96


@dataclass(frozen=True)
class TargetBounds:
    xy: np.ndarray
    radius: float
    bottom_z: float
    top_z: float
    spill_z: float = 0.08


@dataclass(frozen=True)
class FluidCounts:
    in_target: int
    spilled: int


class FluidBackend(ABC):
    """Minimal one-way coupled fluid interface.

    MuJoCo updates rigid bodies first. The fluid backend then updates or reads
    water state using rigid-body poses/bounds. Water does not push forces back
    to MuJoCo in this simplified pouring setup.
    """

    @abstractmethod
    def step(self, dt: float, rigid_body_poses: dict[str, np.ndarray]) -> None:
        """Advance fluid state by dt."""

    @abstractmethod
    def get_particle_positions(self) -> np.ndarray:
        """Return particle positions as an (N, 3) array."""

    @abstractmethod
    def count(self, target: TargetBounds) -> FluidCounts:
        """Count particles inside the target and particles considered spilled."""


class MujocoParticleBackend(FluidBackend):
    """Fluid backend backed by MuJoCo free-body sphere particles."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        particle_count: int,
        particle_prefix: str = PARTICLE_PREFIX,
    ) -> None:
        self.model = model
        self.data = data
        self.particle_count = particle_count
        self.body_ids = np.array(
            [
                mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    f"{particle_prefix}{i}",
                )
                for i in range(particle_count)
            ],
            dtype=int,
        )

    def step(self, dt: float, rigid_body_poses: dict[str, np.ndarray]) -> None:
        del dt, rigid_body_poses
        # MuJoCo already steps these particles with the rigid-body scene.

    def get_particle_positions(self) -> np.ndarray:
        return self.data.xpos[self.body_ids].copy()

    def count(self, target: TargetBounds) -> FluidCounts:
        positions = self.get_particle_positions()
        in_target = 0
        spilled = 0
        # MuJoCo particles are physical free bodies, so every modeled particle
        # is active. Particles still in the source cannot satisfy target bounds.
        for pos in positions:
            inside_xy = np.linalg.norm(pos[:2] - target.xy) < target.radius
            inside_z = target.bottom_z < pos[2] < target.top_z
            if inside_xy and inside_z:
                in_target += 1
            elif pos[2] < target.spill_z:
                spilled += 1
        return FluidCounts(in_target=in_target, spilled=spilled)


@ti.data_oriented  # type: ignore[union-attr]
class TaichiParticleBackend(FluidBackend):
    """One-way coupled Taichi particle liquid.

    This is intentionally a small first backend, not a full SPH solver yet. It
    owns particle positions/velocities in Taichi, uses scripted MuJoCo rigid
    body poses as moving collision boundaries, and writes positions back to
    MuJoCo mocap bodies for visualization.
    """

    _taichi_initialized = False

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        particle_count: int,
        initial_positions: np.ndarray,
        particle_radius: float = 0.008,
        particle_prefix: str = PARTICLE_PREFIX,
    ) -> None:
        if not _TAICHI_AVAILABLE:
            raise RuntimeError("TaichiParticleBackend requires `pip install taichi`.")

        if not TaichiParticleBackend._taichi_initialized:
            init_kwargs = {"arch": ti.cpu, "default_fp": ti.f32}
            cpu_threads = int(os.environ.get("FLORA_TAICHI_CPU_THREADS", "0"))
            if cpu_threads > 0:
                init_kwargs["cpu_max_num_threads"] = cpu_threads
            ti.init(**init_kwargs)
            TaichiParticleBackend._taichi_initialized = True

        self.ti = ti
        self.model = model
        self.data = data
        self.particle_count = particle_count
        self.particle_radius = particle_radius
        self.gravity = -9.81
        self.damping = 0.45
        self.table_z = particle_radius
        self.smoothing_radius = max(3.2 * particle_radius, 0.020)
        self.rest_density = 1.35
        self.pressure_stiffness = 2.2
        self.near_pressure_stiffness = 4.0
        self.viscosity = 0.35
        self.surface_tension = 0.08
        self.max_speed = 1.25
        self.fluid_substeps = 16
        self.source_inner_radius = 0.036
        self.source_bottom_z = 0.008
        self.source_lip_z = 0.071
        self.prev_source_pos: np.ndarray | None = None
        self.source_initial_pos = np.array([-0.12, 0.0, 0.17], dtype=np.float32)

        self.x = ti.Vector.field(3, dtype=ti.f32, shape=particle_count)
        self.v = ti.Vector.field(3, dtype=ti.f32, shape=particle_count)
        self.local_x = ti.Vector.field(3, dtype=ti.f32, shape=particle_count)
        self.released = ti.field(dtype=ti.i32, shape=particle_count)
        self.neighbor_count = ti.field(dtype=ti.f32, shape=particle_count)
        self.target_box_pos = ti.Vector.field(
            3, dtype=ti.f32, shape=TARGET_BOX_CAPACITY
        )
        self.target_box_rot = ti.Matrix.field(
            3, 3, dtype=ti.f32, shape=TARGET_BOX_CAPACITY
        )
        self.target_box_inv_rot = ti.Matrix.field(
            3, 3, dtype=ti.f32, shape=TARGET_BOX_CAPACITY
        )
        self.target_box_half = ti.Vector.field(
            3, dtype=ti.f32, shape=TARGET_BOX_CAPACITY
        )
        self.x.from_numpy(initial_positions.astype(np.float32))
        self.v.from_numpy(np.zeros((particle_count, 3), dtype=np.float32))
        self.local_x.from_numpy(
            (initial_positions.astype(np.float32) - self.source_initial_pos).astype(
                np.float32
            )
        )
        self.released.from_numpy(np.zeros(particle_count, dtype=np.int32))
        (
            self.target_box_count,
            target_box_pos,
            target_box_rot,
            target_box_inv_rot,
            target_box_half,
        ) = self._extract_target_box_colliders()
        self.target_box_pos.from_numpy(target_box_pos)
        self.target_box_rot.from_numpy(target_box_rot)
        self.target_box_inv_rot.from_numpy(target_box_inv_rot)
        self.target_box_half.from_numpy(target_box_half)

        self.body_ids = np.array(
            [
                mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    f"{particle_prefix}{i}",
                )
                for i in range(particle_count)
            ],
            dtype=int,
        )
        self.mocap_ids = model.body_mocapid[self.body_ids].copy()
        if np.any(self.mocap_ids < 0):
            raise ValueError("Taichi visual particles must be mocap bodies.")
        self.sync_to_mujoco()

    @classmethod
    def is_available(cls) -> bool:
        return _TAICHI_AVAILABLE

    def step(self, dt: float, rigid_body_poses: dict[str, np.ndarray]) -> None:
        source_pos = rigid_body_poses["source_pos"].astype(np.float32)
        source_angle = float(rigid_body_poses["source_angle"])
        if self.prev_source_pos is None:
            source_delta = np.zeros(3, dtype=np.float32)
        else:
            source_delta = source_pos - self.prev_source_pos
        self._step_kernel(
            float(dt),
            source_pos,
            source_delta.astype(np.float32),
            source_angle,
            float(self.source_inner_radius),
            float(self.source_bottom_z),
            float(self.source_lip_z),
            float(self.particle_radius),
        )
        self._compute_neighbor_counts(float(self.smoothing_radius))
        sub_dt = float(dt) / float(self.fluid_substeps)
        for _ in range(self.fluid_substeps):
            self._integrate_sph_forces(
                sub_dt,
                float(self.smoothing_radius),
                float(self.pressure_stiffness),
                float(self.near_pressure_stiffness),
                float(self.viscosity),
                float(self.surface_tension),
                float(self.gravity),
                float(self.particle_radius),
                float(self.table_z),
                float(self.rest_density),
                float(self.max_speed),
                int(self.target_box_count),
            )
        self.prev_source_pos = source_pos.copy()
        self.sync_to_mujoco()

    def get_particle_positions(self) -> np.ndarray:
        return self.x.to_numpy()

    def count(self, target: TargetBounds) -> FluidCounts:
        positions = self.get_particle_positions()
        in_target = 0
        spilled = 0
        for pos in positions:
            inside_xy = np.linalg.norm(pos[:2] - target.xy) < target.radius
            inside_z = target.bottom_z < pos[2] < target.top_z
            if inside_xy and inside_z:
                in_target += 1
            elif pos[2] < target.spill_z:
                spilled += 1
        return FluidCounts(in_target=in_target, spilled=spilled)

    def sync_to_mujoco(self) -> None:
        positions = self.get_particle_positions()
        for mocap_id, pos in zip(self.mocap_ids, positions):
            self.data.mocap_pos[mocap_id] = pos
            self.data.mocap_quat[mocap_id] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(self.model, self.data)

    def _extract_target_box_colliders(
        self,
    ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        box_pos = np.zeros((TARGET_BOX_CAPACITY, 3), dtype=np.float32)
        box_rot = np.zeros((TARGET_BOX_CAPACITY, 3, 3), dtype=np.float32)
        box_inv_rot = np.zeros((TARGET_BOX_CAPACITY, 3, 3), dtype=np.float32)
        box_half = np.zeros((TARGET_BOX_CAPACITY, 3), dtype=np.float32)

        target_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "target_container",
        )
        if target_body_id < 0:
            return 0, box_pos, box_rot, box_inv_rot, box_half

        mujoco.mj_forward(self.model, self.data)
        count = 0
        for geom_id in range(self.model.ngeom):
            is_target_geom = self.model.geom_bodyid[geom_id] == target_body_id
            is_collision_box = (
                self.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX
                and self.model.geom_group[geom_id] == 0
            )
            if not (is_target_geom and is_collision_box):
                continue
            if count >= TARGET_BOX_CAPACITY:
                break

            rot = self.data.geom_xmat[geom_id].reshape(3, 3).astype(np.float32)
            box_pos[count] = self.data.geom_xpos[geom_id].astype(np.float32)
            box_rot[count] = rot
            box_inv_rot[count] = rot.T
            box_half[count] = self.model.geom_size[geom_id, :3].astype(np.float32)
            count += 1

        return count, box_pos, box_rot, box_inv_rot, box_half

    @ti.kernel  # type: ignore[name-defined]
    def _step_kernel(
        self: ti.template(),  # type: ignore[union-attr]
        dt: ti.f32,  # type: ignore[union-attr]
        source_pos: ti.types.vector(3, ti.f32),  # type: ignore[union-attr]
        source_delta: ti.types.vector(3, ti.f32),  # type: ignore[union-attr]
        source_angle: ti.f32,  # type: ignore[union-attr]
        source_inner_radius: ti.f32,  # type: ignore[union-attr]
        source_bottom_z: ti.f32,  # type: ignore[union-attr]
        source_lip_z: ti.f32,  # type: ignore[union-attr]
        radius: ti.f32,  # type: ignore[union-attr]
    ):
        for i in range(self.particle_count):
            p = self.x[i]
            vel = self.v[i]
            c = ti.cos(source_angle)
            s = ti.sin(source_angle)
            local = self.local_x[i]

            if self.released[i] == 0:
                # Water inside the cup follows the moving rigid cup boundary.
                p.x = source_pos.x + c * local.x + s * local.z
                p.y = source_pos.y + local.y
                p.z = source_pos.z - s * local.x + c * local.z
                vel = source_delta / max(dt, 1e-5)

                radial = ti.sqrt(local.x * local.x + local.y * local.y + 1e-8)
                spill_progress = ti.max(0.0, source_angle) / 1.25
                spill_radius = source_inner_radius - radius * (2.4 + 3.2 * spill_progress)
                escaped_side = (
                    local.x > spill_radius
                    and local.z > source_lip_z - radius * (2.5 + 3.0 * spill_progress)
                )
                escaped_lip = (
                    radial > source_inner_radius - 1.15 * radius
                    and local.z > source_lip_z - 2.25 * radius
                )
                if escaped_side or escaped_lip:
                    self.released[i] = 1
                    out_dir = ti.Vector([c, 0.0, -s])
                    out_dir = out_dir / ti.sqrt(out_dir.dot(out_dir) + 1e-8)
                    vel += out_dir * 0.14
            self.x[i] = p
            self.v[i] = vel

    @ti.kernel  # type: ignore[name-defined]
    def _compute_neighbor_counts(
        self: ti.template(),  # type: ignore[union-attr]
        h: ti.f32,  # type: ignore[union-attr]
    ):
        for i in range(self.particle_count):
            count = 0.0
            if self.released[i] == 1:
                pi = self.x[i]
                for j in range(self.particle_count):
                    if i != j and self.released[j] == 1:
                        rij = pi - self.x[j]
                        dist = ti.sqrt(rij.dot(rij) + 1e-8)
                        if dist < h:
                            q = 1.0 - dist / h
                            count += q * q * q
            self.neighbor_count[i] = count

    @ti.kernel  # type: ignore[name-defined]
    def _integrate_sph_forces(
        self: ti.template(),  # type: ignore[union-attr]
        dt: ti.f32,  # type: ignore[union-attr]
        h: ti.f32,  # type: ignore[union-attr]
        pressure_stiffness: ti.f32,  # type: ignore[union-attr]
        near_pressure_stiffness: ti.f32,  # type: ignore[union-attr]
        viscosity: ti.f32,  # type: ignore[union-attr]
        surface_tension: ti.f32,  # type: ignore[union-attr]
        gravity: ti.f32,  # type: ignore[union-attr]
        radius: ti.f32,  # type: ignore[union-attr]
        table_z: ti.f32,  # type: ignore[union-attr]
        rest_density: ti.f32,  # type: ignore[union-attr]
        max_speed: ti.f32,  # type: ignore[union-attr]
        target_box_count: ti.i32,  # type: ignore[union-attr]
    ):
        for i in range(self.particle_count):
            if self.released[i] == 1:
                p = self.x[i]
                vel = self.v[i]
                old_p = p
                acc = ti.Vector([0.0, 0.0, gravity])

                for j in range(self.particle_count):
                    if i != j and self.released[j] == 1:
                        rij = p - self.x[j]
                        dist = ti.sqrt(rij.dot(rij) + 1e-8)
                        if dist < h:
                            q = 1.0 - dist / h
                            n = rij / dist
                            density_i = ti.max(self.neighbor_count[i] - rest_density, 0.0)
                            density_j = ti.max(self.neighbor_count[j] - rest_density, 0.0)
                            pressure = pressure_stiffness * (density_i + density_j)
                            near_pressure = near_pressure_stiffness * q * q * q
                            acc += n * (pressure * q * q + near_pressure)
                            acc += (self.v[j] - vel) * viscosity * q
                            acc -= n * surface_tension * q * (1.0 - q)

                vel += acc * dt
                speed = ti.sqrt(vel.dot(vel) + 1e-8)
                if speed > max_speed:
                    vel *= max_speed / speed
                p += vel * dt
                step_delta = p - old_p
                step_dist = ti.sqrt(step_delta.dot(step_delta) + 1e-8)
                max_step = ti.max(0.006, max_speed * dt)
                if step_dist > max_step:
                    p = old_p + step_delta * (max_step / step_dist)
                    vel *= 0.55

                # Table collision.
                if p.z < table_z:
                    p.z = table_z
                    vel.z = max(0.0, -vel.z * 0.08)
                    vel.x *= 0.50
                    vel.y *= 0.50

                for box_i in range(target_box_count):
                    center = self.target_box_pos[box_i]
                    inv_rot = self.target_box_inv_rot[box_i]
                    rot = self.target_box_rot[box_i]
                    half = self.target_box_half[box_i]

                    local = inv_rot @ (p - center)
                    closest = ti.Vector(
                        [
                            ti.min(half.x, ti.max(-half.x, local.x)),
                            ti.min(half.y, ti.max(-half.y, local.y)),
                            ti.min(half.z, ti.max(-half.z, local.z)),
                        ]
                    )
                    delta = local - closest
                    dist_sq = delta.dot(delta)

                    if dist_sq < radius * radius:
                        normal_local = ti.Vector([0.0, 0.0, 1.0])
                        penetration = radius
                        if dist_sq > 1e-10:
                            dist = ti.sqrt(dist_sq)
                            normal_local = delta / dist
                            penetration = radius - dist
                        else:
                            face = half - ti.abs(local)
                            normal_local = ti.Vector([1.0, 0.0, 0.0])
                            penetration = radius + face.x
                            if face.y < penetration - radius:
                                normal_local = ti.Vector([0.0, 1.0, 0.0])
                                penetration = radius + face.y
                            if face.z < penetration - radius:
                                normal_local = ti.Vector([0.0, 0.0, 1.0])
                                penetration = radius + face.z
                            if normal_local.x != 0.0:
                                normal_local.x = ti.select(local.x >= 0.0, 1.0, -1.0)
                            if normal_local.y != 0.0:
                                normal_local.y = ti.select(local.y >= 0.0, 1.0, -1.0)
                            if normal_local.z != 0.0:
                                normal_local.z = ti.select(local.z >= 0.0, 1.0, -1.0)

                        normal = rot @ normal_local
                        p += normal * (penetration + 1e-4)
                        normal_v = vel.dot(normal)
                        if normal_v < 0.0:
                            vel -= (1.0 + 0.08) * normal_v * normal
                            vel *= 0.68

                self.x[i] = p
                self.v[i] = vel
