from dataclasses import asdict

import pytest

torch = pytest.importorskip("torch")

from flora_core.models import (
    GlobalLocalDiffusionConfig,
    GlobalTrajectoryPredictor,
    NormalizedInteractionQueryEncoder,
    PeriodicTargetQueryEncoder,
    load_global_predictor_checkpoint,
)


def small_config(**overrides):
    values = dict(
        model_dim=32,
        num_heads=4,
        scene_slots_per_entity=1,
        channel_tokens=4,
        global_horizon=8,
        fusion_layers=1,
        global_decoder_layers=1,
        global_scene_representation="parameterized",
        realization_input="abstract",
    )
    values.update(overrides)
    return GlobalLocalDiffusionConfig(**values)


def model_inputs(batch=2):
    return {
        "scene_parameters": torch.randn(batch, 25),
        "source_parameter": torch.tensor([[0.25], [0.75]])[:batch],
        "target_parameter": torch.tensor([[0.0], [1.0 / 3.0]])[:batch],
        "channel_id": torch.tensor([0, 2])[:batch],
        "progress": torch.tensor([[0.0], [0.5]])[:batch],
    }


def test_uo_encoder_validates_range_and_is_differentiable():
    encoder = NormalizedInteractionQueryEncoder(small_config(), 1)
    uo = torch.tensor([[0.25], [0.75]], requires_grad=True)
    token = encoder(uo)
    assert token.shape == (2, 1, 32)
    token.square().mean().backward()
    assert uo.grad is not None and torch.isfinite(uo.grad).all()
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        encoder(torch.tensor([[1.1]]))


def test_scene_state_mode_anchors_current_pose():
    model = GlobalTrajectoryPredictor(small_config(global_scene_source="scene_state")).eval()
    inputs = model_inputs()
    state = torch.zeros(2, 11)
    state[:, :3] = torch.tensor([-0.3, 0.1, 0.2])
    state[:, 3] = 1.0
    inputs["fused_state"] = state
    with torch.no_grad():
        trajectory = model(**inputs)
    assert trajectory.shape == (2, 8, 9)
    assert torch.isfinite(trajectory).all()
    assert torch.allclose(trajectory[:, 0, :3], state[:, :3])


def test_ug_encoder_is_periodic_at_zero_and_one():
    encoder = PeriodicTargetQueryEncoder(small_config()).eval()
    with torch.no_grad():
        zero = encoder(torch.tensor([[0.0]]))
        one = encoder(torch.tensor([[1.0]]))
    assert torch.allclose(zero, one, atol=1e-6)


def test_uo_changes_plan_and_global_trajectory():
    torch.manual_seed(7)
    model = GlobalTrajectoryPredictor(small_config()).eval()
    inputs = model_inputs(batch=1)
    with torch.no_grad():
        low_plan = model.encode_realization_plan(None, None, **inputs)
        low_trajectory = model(**inputs)
        inputs["source_parameter"] = torch.tensor([[0.75]])
        high_plan = model.encode_realization_plan(None, None, **inputs)
        high_trajectory = model(**inputs)
    assert not torch.allclose(low_plan, high_plan)
    assert not torch.allclose(low_trajectory, high_trajectory)


def test_uo_outlet_loss_backpropagates():
    torch.manual_seed(8)
    model = GlobalTrajectoryPredictor(small_config(lambda_u_o=1.0))
    predicted = model(**model_inputs())
    target = torch.randn_like(predicted)
    losses = model.training_loss(
        predicted,
        target,
        rim_local=torch.tensor([[0.03, 0.0, 0.08], [0.02, 0.01, 0.07]]),
        k_star=torch.tensor([4, 6]),
    )
    assert losses["u_o_loss"].item() > 0.0
    losses["loss"].backward()
    assert model.source_query_encoder.parameter_mlp[0].weight.grad is not None


def test_aext_checkpoint_layout_round_trip(tmp_path):
    torch.manual_seed(9)
    config = small_config()
    original = GlobalTrajectoryPredictor(config).eval()
    path = tmp_path / "aext_global.pt"
    torch.save(
        {
            "schema": "core_global_predictor_checkpoint_v1",
            "config": asdict(config),
            "model_state_dict": original.state_dict(),
        },
        path,
    )
    restored, payload = load_global_predictor_checkpoint(path)
    restored.eval()
    inputs = model_inputs()
    with torch.no_grad():
        expected = original(**inputs)
        actual = restored(**inputs)
    assert payload["schema"] == "core_global_predictor_checkpoint_v1"
    assert torch.equal(expected, actual)


def test_global_mask_hides_all_factor_paths_and_is_disabled_in_eval():
    torch.manual_seed(42)
    cfg = small_config(global_scene_source="scene_state", object_point_interaction=True,
                       grounded_source_bottleneck=True, global_realization_mask_prob=1.0)
    model = GlobalTrajectoryPredictor(cfg).train()
    inputs = model_inputs()
    cloud = torch.randn(2, 48, 6)
    cloud[..., 3:] = 0
    cloud[:, :16, 3] = 1
    cloud[:, 16:32, 4] = 1
    cloud[:, 32:, 5] = 1
    state = torch.zeros(2, 11)
    state[:, 3] = 1
    inputs.update(fused_point_cloud=cloud, fused_state=state,
                  source_context_local=torch.randn(2, 128, 3),
                  source_patch_grounded=torch.randn(2, 32, 6))
    changed = dict(inputs, source_parameter=1 - inputs["source_parameter"],
                   target_parameter=torch.full((2, 1), .7), channel_id=torch.tensor([1, 1]),
                   source_context_local=torch.randn(2, 128, 3),
                   source_patch_grounded=torch.randn(2, 32, 6))
    with torch.no_grad():
        torch.manual_seed(3)
        first = model(**inputs)
        torch.manual_seed(3)
        hidden = model(**changed)
        torch.testing.assert_close(first, hidden, rtol=0, atol=0)
        model.eval()
        baseline = GlobalTrajectoryPredictor(small_config(
            global_scene_source="scene_state", object_point_interaction=True,
            grounded_source_bottleneck=True)).eval()
        baseline.load_state_dict(model.state_dict())
        torch.testing.assert_close(model(**inputs), baseline(**inputs), rtol=0, atol=0)
        assert not torch.allclose(model(**inputs), model(**changed))
