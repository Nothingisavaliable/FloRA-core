"""Damped-least-squares site IK shared by the supported robosuite arms."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


def solve_site_ik(
    sim: Any,
    site_id: int,
    site_name: str,
    qpos_indexes: list[int],
    qvel_indexes: list[int],
    q_lower: np.ndarray,
    q_upper: np.ndarray,
    target_pos: np.ndarray,
    target_rot: np.ndarray,
    *,
    seed_q: np.ndarray,
    max_iters: int = 180,
    damping: float = 0.05,
    rot_weight: float = 0.35,
    max_step: float = 0.08,
    pos_tol: float = 0.004,
    rot_tol: float = 0.08,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Numerically solve a single-arm TCP pose using MuJoCo site Jacobians."""
    lower = np.asarray(q_lower, dtype=np.float64)
    upper = np.asarray(q_upper, dtype=np.float64)
    q = np.clip(np.asarray(seed_q, dtype=np.float64), lower, upper)
    best_q = q.copy()
    best_score = best_pos = best_rot = float("inf")
    iterations = 0
    for iterations in range(1, int(max_iters) + 1):
        sim.data.qpos[qpos_indexes] = q
        sim.data.qvel[qvel_indexes] = 0.0
        sim.forward()
        current_pos = np.asarray(sim.data.site_xpos[site_id], dtype=np.float64).copy()
        current_rot = np.asarray(sim.data.site_xmat[site_id], dtype=np.float64).reshape(3, 3).copy()
        dp = np.asarray(target_pos, dtype=np.float64) - current_pos
        dr = Rotation.from_matrix(np.asarray(target_rot, dtype=np.float64) @ current_rot.T).as_rotvec()
        pos_error, rot_error = float(np.linalg.norm(dp)), float(np.linalg.norm(dr))
        score = pos_error + float(rot_weight) * rot_error
        if score < best_score:
            best_q, best_score, best_pos, best_rot = q.copy(), score, pos_error, rot_error
        if pos_error <= pos_tol and rot_error <= rot_tol:
            break
        jacp = np.asarray(sim.data.get_site_jacp(site_name), dtype=np.float64).reshape(3, -1)
        jacr = np.asarray(sim.data.get_site_jacr(site_name), dtype=np.float64).reshape(3, -1)
        jac = np.vstack((jacp[:, qvel_indexes], float(rot_weight) * jacr[:, qvel_indexes]))
        error = np.r_[dp, float(rot_weight) * dr]
        lhs = jac @ jac.T + float(damping) ** 2 * np.eye(6)
        delta = jac.T @ np.linalg.solve(lhs, error)
        norm = float(np.linalg.norm(delta))
        if norm > max_step:
            delta *= float(max_step) / max(norm, 1e-12)
        q = np.clip(q + delta, lower, upper)
    sim.data.qpos[qpos_indexes] = best_q
    sim.data.qvel[qvel_indexes] = 0.0
    sim.forward()
    return best_q, {
        "iters": iterations, "pos_error_m": best_pos, "rot_error_rad": best_rot,
        "converged": bool(best_pos <= pos_tol and best_rot <= rot_tol),
    }
