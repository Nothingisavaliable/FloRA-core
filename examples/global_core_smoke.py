"""Untrained Global Predictor + selection API smoke test (not a paper reproduction).

Run after: pip install -e '.[models]'
Then:      python examples/global_core_smoke.py
"""
from __future__ import annotations

import numpy as np
import torch

from flora_core.data import SCENE_PARAMETER_DIM
from flora_core.models import GlobalLocalDiffusionConfig, GlobalTrajectoryPredictor
from flora_core.pipeline import Evaluation, FloRA, candidate_grid


def main() -> None:
    torch.manual_seed(7)
    cfg = GlobalLocalDiffusionConfig(
        global_scene_representation="parameterized",
        model_dim=32, num_heads=4, scene_slots_per_entity=1,
        channel_tokens=4, global_horizon=8,
        fusion_layers=1, global_decoder_layers=1,
    )
    model = GlobalTrajectoryPredictor(cfg).eval()
    scene = torch.zeros(1, SCENE_PARAMETER_DIM)

    @torch.no_grad()
    def predictor(realization):
        inputs = realization.condition()
        result = model(
            scene_parameters=scene,
            source_parameter=torch.tensor(inputs["source_parameter"]).reshape(1, 1),
            target_parameter=torch.tensor(inputs["target_parameter"]).reshape(1, 1),
            channel_id=torch.tensor([inputs["channel_id"]]),
            progress=torch.tensor([[0.0]]),
        )
        return result.squeeze(0).cpu().numpy()

    def evaluator(realization, trajectory):
        # Synthetic score only: this is NOT a collision or robot-feasibility check.
        return Evaluation(
            feasible=bool(np.isfinite(trajectory).all()),
            score=-float(np.square(trajectory[:, :3]).mean()),
            evaluation_id="synthetic-smoke-only",
        )

    plan = FloRA(predictor, evaluator).select(candidate_grid()[:3])
    print(f"Smoke test OK: selected {plan.realization_id} with {plan.trajectory.shape} plan")
    print("WARNING: untrained network, synthetic evaluator, no Local DP3 or robot execution")


if __name__ == "__main__":
    main()
