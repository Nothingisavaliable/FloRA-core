"""Project-local LIBERO registration for Panda, UR5e, and Sawyer.

LIBERO tabletop problems turn a requested name such as ``Sawyer`` into
``MountedSawyer``.  The installed LIBERO tree supplies MountedPanda and
MountedUR5e, but not MountedSawyer.  This module adds the missing model without
editing the external LIBERO checkout and makes all three runtime mappings
explicit and auditable.
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_LIBERO_ROOT = Path(os.environ.get("AEXT_LIBERO_ROOT", "/home/wucheng/LIBERO"))


@dataclass(frozen=True)
class RobotProfile:
    name: str
    mounted_name: str
    dof: int
    default_gripper: str
    controller: str
    fk_backend: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ROBOT_PROFILES: dict[str, RobotProfile] = {
    "Panda": RobotProfile(
        name="Panda",
        mounted_name="MountedPanda",
        dof=7,
        default_gripper="PandaGripper",
        controller="OSC_POSE",
        fk_backend="mujoco",
    ),
    "UR5e": RobotProfile(
        name="UR5e",
        mounted_name="MountedUR5e",
        dof=6,
        default_gripper="Robotiq85Gripper",
        controller="OSC_POSE",
        fk_backend="mujoco",
    ),
    "Sawyer": RobotProfile(
        name="Sawyer",
        mounted_name="MountedSawyer",
        dof=7,
        default_gripper="RethinkGripper",
        controller="OSC_POSE",
        fk_backend="mujoco",
    ),
}


def _add_libero_to_path(libero_root: Path) -> None:
    if not libero_root.exists():
        raise FileNotFoundError(
            f"LIBERO root does not exist: {libero_root}. "
            "Set AEXT_LIBERO_ROOT to the checkout path."
        )
    root_text = str(libero_root.resolve())
    if root_text not in sys.path:
        sys.path.insert(0, root_text)


def register_three_arm_models(
    libero_root: Path | str = DEFAULT_LIBERO_ROOT,
) -> dict[str, dict[str, Any]]:
    """Register the three mounted robot models and return an audit snapshot.

    The operation is idempotent within a process.  Defining ``MountedSawyer``
    triggers robosuite's model metaclass registration; the runtime robot class
    mapping is updated separately because the two registries serve different
    layers.
    """

    root = Path(libero_root)
    _add_libero_to_path(root)

    from robosuite.models.robots.robot_model import REGISTERED_ROBOTS
    from robosuite.robots import ROBOT_CLASS_MAPPING
    from robosuite.robots.single_arm import SingleArm

    # Importing LIBERO's robot package registers its mounted Panda and UR5e.
    import libero.libero.envs.robots  # noqa: F401

    if "MountedSawyer" not in REGISTERED_ROBOTS:
        from robosuite.models.robots import Sawyer

        class MountedSawyer(Sawyer):
            """Sawyer with the tabletop offsets expected by LIBERO domains."""

            def __init__(self, idn=0):
                super().__init__(idn=idn)
                # robosuite 1.4's Sawyer assets predate MuJoCo 3's stricter
                # solid-volume check.  The meshes are visual geometry and the
                # links already carry explicit inertials, so shell inertia is
                # the correct compatibility interpretation.
                for mesh in self.asset.findall("mesh"):
                    mesh.set("inertia", "shell")

            @property
            def base_xpos_offset(self):
                return {
                    "bins": (-0.5, -0.1, 0),
                    "empty": (-0.6, 0, 0),
                    "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
                    "study_table": lambda table_length: (-0.25 - table_length / 2, 0, 0),
                    "kitchen_table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
                }

        # Keep the class reachable for debuggers; metaclass registration has
        # already happened at class creation time.
        globals()["MountedSawyer"] = MountedSawyer

    for profile in ROBOT_PROFILES.values():
        if profile.mounted_name not in REGISTERED_ROBOTS:
            raise RuntimeError(
                f"model registration failed for {profile.mounted_name}; "
                f"registered={sorted(REGISTERED_ROBOTS)}"
            )
        ROBOT_CLASS_MAPPING[profile.mounted_name] = SingleArm

    return {
        name: {
            **profile.to_dict(),
            "model_registered": profile.mounted_name in REGISTERED_ROBOTS,
            "runtime_registered": profile.mounted_name in ROBOT_CLASS_MAPPING,
        }
        for name, profile in ROBOT_PROFILES.items()
    }


def require_robot(name: str) -> RobotProfile:
    try:
        return ROBOT_PROFILES[name]
    except KeyError as exc:
        raise ValueError(
            f"unsupported experiment robot {name!r}; choose {sorted(ROBOT_PROFILES)}"
        ) from exc
