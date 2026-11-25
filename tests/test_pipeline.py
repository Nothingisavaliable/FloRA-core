import numpy as np
import pytest

from flora_core.pipeline import (
    Evaluation, FloRA, NoFeasiblePlan, Realization, candidate_grid,
)
from flora_core.geometry.frames import (
    make_cup_to_bowl_task_frame, world_positions_to_task,
)
from flora_core.execution.global_contract import interpolate_global_reference


def trajectory(realization):
    path = np.zeros((8, 9))
    path[:, 0] = np.linspace(-0.4, 0, 8)
    path[:, 3:9] = [1, 0, 0, 0, 1, 0]
    return path


def test_candidates_and_conditions():
    candidates = candidate_grid()
    assert len(candidates) == len({c.key for c in candidates}) == 27
    assert candidates[0].condition()["source_parameter"].shape == (1,)
    assert Realization(0, 0, "c_left").key
    assert len(candidate_grid(iter([0.5]), iter([0, 0.5]), iter(["c_left"]))) == 2


@pytest.mark.parametrize("o,g,c", [(float('nan'), 0, 'c_left'),
    (0.5, 1, 'c_left'), (0.5, 0, 'unknown'), (-1, 0, 'c_left')])
def test_invalid_realization(o, g, c):
    with pytest.raises(ValueError):
        Realization(o, g, c)


def test_selection_and_latched_reevaluation():
    engine = FloRA(trajectory, lambda m, t: Evaluation(
        m.channel != "c_over", m.u_o, "geometry-check"))
    plan = engine.select(candidate_grid())
    selected = next(m for m in candidate_grid() if m.key == plan.realization_id)
    assert selected.u_o == 0.75
    assert not plan.trajectory.flags.writeable
    pose = interpolate_global_reference(plan.trajectory, 0)
    assert engine.check(pose, 0, selected).consistent
    pose[1] += 1
    assert engine.check(pose, 0, selected).needs_reevaluation
    pose[1] -= 1
    assert engine.check(pose, 0, selected).needs_reevaluation
    engine.select([selected])
    assert engine.check(pose, 0, selected).consistent


def test_no_feasible_plan_retires_previous_plan():
    engine = FloRA(trajectory, lambda m, t: Evaluation(True, 1, "checked"))
    engine.select(candidate_grid())
    engine.evaluator = lambda m, t: Evaluation(False, 100, "rejected")
    with pytest.raises(NoFeasiblePlan):
        engine.select(candidate_grid())
    assert not engine.contract.has_valid_plan
    with pytest.raises(NoFeasiblePlan):
        engine.select([])


def test_invalid_prediction_never_reaches_evaluator():
    def forbidden(*args):
        pytest.fail("invalid trajectory reached evaluator")
    engine = FloRA(lambda m: np.zeros((4, 9)), forbidden)
    with pytest.raises(ValueError, match="degenerate"):
        engine.select(candidate_grid())


def test_task_frame():
    frame, rotation = make_cup_to_bowl_task_frame(
        np.array([0, 0, 0.2]), np.array([0.4, 0, 0.1]))
    np.testing.assert_allclose(rotation, np.eye(3))
    np.testing.assert_allclose(world_positions_to_task(
        np.array([[0.4, 0, 0.1]]), frame), [[0, 0, 0]], atol=1e-7)


@pytest.mark.parametrize("progress", [-1, 2, float('nan')])
def test_invalid_progress(progress):
    engine = FloRA(trajectory, lambda m, t: Evaluation(True, 1, "checked"))
    m = candidate_grid()[0]
    engine.select([m])
    with pytest.raises(ValueError):
        engine.check([0, 0, 0, 1, 0, 0, 0], progress, m)
