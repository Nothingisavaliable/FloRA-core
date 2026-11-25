"""Predict → evaluate → commit → check, without a robot or simulator adapter."""
from dataclasses import dataclass
from collections.abc import Callable, Iterable
import math

import numpy as np

from .execution.global_contract import (
    EvaluatedGlobalPlan, GlobalLocalContract, StructuralEnvelope,
)


@dataclass(frozen=True)
class Realization:
    """Normalized outlet, cyclic target coordinate and spatial channel."""

    u_o: float
    u_g: float
    channel: str

    def __post_init__(self):
        if not math.isfinite(self.u_o) or not 0 <= self.u_o <= 1:
            raise ValueError("u_o must be finite and in [0,1]")
        if not math.isfinite(self.u_g) or not 0 <= self.u_g < 1:
            raise ValueError("u_g must be finite and in [0,1)")
        if self.channel not in {"c_left", "c_right", "c_over"}:
            raise ValueError("unknown spatial channel")

    @property
    def key(self) -> str:
        return f"outlet={float(self.u_o).hex()}|target={float(self.u_g).hex()}|{self.channel}"

    def condition(self) -> dict:
        return {
            "source_parameter": np.asarray([self.u_o], dtype=np.float32),
            "target_parameter": np.asarray([self.u_g], dtype=np.float32),
            "channel_id": ("c_left", "c_right", "c_over").index(self.channel),
        }


def candidate_grid(
    outlets=(0.25, 0.5, 0.75), targets=(0.0, 1 / 3, 2 / 3),
    channels=("c_left", "c_right", "c_over"),
) -> tuple[Realization, ...]:
    """Deterministic Cartesian candidates, with no experiment identifiers."""
    outlets, targets, channels = tuple(outlets), tuple(targets), tuple(channels)
    return tuple(Realization(float(o), float(g), c)
                 for o in outlets for g in targets for c in channels)


@dataclass(frozen=True)
class Evaluation:
    """Adapter result. Higher score wins, but only feasible plans may commit.

    The adapter must perform the checks required by its deployment (including
    robot feasibility). A geometric fidelity score alone is not a safety proof.
    """

    feasible: bool
    score: float
    evaluation_id: str

    def __post_init__(self):
        if not math.isfinite(self.score) or not self.evaluation_id:
            raise ValueError("evaluation requires finite score and nonempty ID")


class NoFeasiblePlan(RuntimeError):
    pass


class FloRA:
    """Application supplies model/scene and feasibility adapters explicitly.

    Predictor returns a task-frame [L,9] position + row-major Rot6D array.
    Calling select retires the previous plan, even when selection fails.
    Local control must call check before using the committed reference.
    """

    def __init__(
        self, predictor: Callable[[Realization], np.ndarray],
        evaluator: Callable[[Realization, np.ndarray], Evaluation],
        envelope: StructuralEnvelope = StructuralEnvelope(),
    ):
        self.predictor = predictor
        self.evaluator = evaluator
        self.contract = GlobalLocalContract(envelope)

    def select(self, candidates: Iterable[Realization]) -> EvaluatedGlobalPlan:
        self.contract.clear()
        best = None
        best_score = -math.inf
        for realization in candidates:
            # Own the candidate memory before calling the evaluator.
            trajectory = np.array(self.predictor(realization), dtype=float, copy=True)
            provisional = EvaluatedGlobalPlan(trajectory, realization.key)
            rotations = provisional.trajectory[:, 3:9]
            first, second = rotations[:, :3], rotations[:, 3:]
            if (np.any(np.linalg.norm(first, axis=1) < 1e-8)
                    or np.any(np.linalg.norm(np.cross(first, second), axis=1) < 1e-8)):
                raise ValueError("degenerate Rot6D trajectory")
            provisional.trajectory.setflags(write=False)
            result = self.evaluator(realization, provisional.trajectory)
            if result.feasible and result.score > best_score:
                best = EvaluatedGlobalPlan(
                    provisional.trajectory, realization.key, result.evaluation_id,
                )
                best_score = result.score
        if best is None:
            raise NoFeasiblePlan("no candidate passed evaluation")
        best.trajectory.setflags(write=False)
        self.contract.commit(best)
        return best

    def check(self, current_pose7, progress: float, realization: Realization):
        pose = np.asarray(current_pose7, dtype=float)
        if (pose.shape != (7,) or not np.isfinite(pose).all()
                or np.linalg.norm(pose[3:]) < 1e-8):
            raise ValueError("current pose must be finite pose7 with a nonzero quaternion")
        if not math.isfinite(progress) or not 0 <= progress <= 1:
            raise ValueError("progress must be in [0,1]")
        status = self.contract.check(pose, progress, realization.key)
        if status.needs_reevaluation:
            self.contract.invalidate("reevaluation_required")
        return status
