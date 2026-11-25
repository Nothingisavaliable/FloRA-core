import pytest

torch = pytest.importorskip("torch")
from torch import nn

import flora_core.models.local_dp3 as local_module
from flora_core.models import GlobalLocalDiffusionConfig


def small_config(**overrides):
    values = dict(
        model_dim=16,
        num_heads=4,
        global_horizon=8,
        local_horizon=8,
        local_obs_steps=2,
        local_action_steps=4,
        local_inference_steps=2,
        base_channels=8,
        trajectory_encoder_layers=1,
    )
    values.update(overrides)
    return GlobalLocalDiffusionConfig(**values)


class _FakeLocalBranch(nn.Module):
    """Fast stand-in; official-component construction has a separate smoke test."""

    def __init__(
        self,
        cfg,
        *,
        use_plan_condition=True,
        use_realization_condition=True,
    ):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.cfg = cfg
        self.use_plan_condition = use_plan_condition
        self.use_realization_condition = use_realization_condition

    def encode_condition(
        self, point_cloud, state, trajectory_tokens=None, realization_feature=None
    ):
        del state
        result = point_cloud.new_zeros(point_cloud.shape[0], self.cfg.model_dim)
        if trajectory_tokens is not None:
            result = result + trajectory_tokens.mean(dim=1)
        if realization_feature is not None:
            result = result + realization_feature
        return result

    def forward(self, noisy_action, timestep, global_condition):
        del timestep
        return noisy_action * self.scale + global_condition.mean(dim=-1)[:, None, None]

    def add_noise(self, clean_action, noise, timestep):
        del timestep
        # Mimic "normalize then diffuse" with an observable factor.
        return 2.0 * clean_action + noise

    def conditional_sample(
        self,
        point_cloud,
        state,
        trajectory_tokens=None,
        realization_feature=None,
        *,
        generator=None,
    ):
        del state, realization_feature, generator
        batch = point_cloud.shape[0]
        prediction = point_cloud.new_zeros(
            batch, self.cfg.local_horizon, self.cfg.action_dim
        )
        start = self.cfg.local_obs_steps - 1
        end = start + self.cfg.local_action_steps
        return {
            "action": prediction[:, start:end],
            "action_pred": prediction,
            "normalized_action": prediction[:, start:end],
            "normalized_action_pred": prediction,
        }

    def fit_normalizer(self, **kwargs):
        del kwargs

    def set_normalizer(self, normalizer):
        del normalizer


def _identity_trajectory(batch, horizon):
    trajectory = torch.zeros(batch, horizon, 9)
    trajectory[..., 3] = 1.0
    trajectory[..., 7] = 1.0
    return trajectory


def _state_history(batch=2, history=2):
    state = torch.zeros(batch, history, 11)
    state[..., 3] = 1.0
    return state


def _realization(batch=2):
    return (
        torch.full((batch, 1), 0.25),
        torch.full((batch, 1), 1.0 / 3.0),
        torch.tensor([0, 2][:batch]),
    )


def test_local_policy_detaches_and_keeps_first_reference_error(monkeypatch):
    monkeypatch.setattr(local_module, "OfficialDP3LocalBranch", _FakeLocalBranch)
    cfg = small_config()
    policy = local_module.LocalTrajectoryDP3Policy(cfg)

    trajectory = _identity_trajectory(2, cfg.global_horizon)
    trajectory[:, :, 0] = torch.linspace(0.0, 0.7, cfg.global_horizon)
    trajectory.requires_grad_()
    state = _state_history()
    state[:, -1, 0] = torch.tensor([0.25, 0.25])
    source, target, channel = _realization()

    outputs = policy(
        current_point_cloud=torch.randn(2, 2, 24, 6),
        current_state=state,
        source_parameter=source,
        target_parameter=target,
        channel_id=channel,
        global_trajectory=trajectory,
        global_reference_index=torch.tensor([3, 3]),
        noisy_action=torch.randn(2, cfg.local_horizon, cfg.action_dim),
        diffusion_timestep=torch.tensor([3, 7]),
    )

    local_trajectory = outputs["local_condition_trajectory"]
    assert not local_trajectory.requires_grad
    # Waypoint 3 is x=0.3 while the current object is at x=0.25. The local
    # condition must preserve this 5-cm correction instead of forcing identity.
    assert torch.allclose(
        local_trajectory[:, 0, 0],
        torch.tensor([0.05, 0.05]),
        atol=1e-6,
    )
    assert torch.equal(
        outputs["global_reference_index"],
        torch.tensor([3, 3]),
    )

    policy.training_loss(
        outputs,
        target_noise=torch.zeros_like(outputs["predicted_noise"]),
    )["loss"].backward()
    assert trajectory.grad is None
    assert policy.trajectory_encoder.input[0].weight.grad is not None
    assert policy.realization_encoder.source[0].weight.grad is not None
    assert policy.local_dp3.scale.grad is not None


def test_realization_mask_removes_only_requested_feature(monkeypatch):
    monkeypatch.setattr(local_module, "OfficialDP3LocalBranch", _FakeLocalBranch)
    cfg = small_config()
    policy = local_module.LocalTrajectoryDP3Policy(cfg).eval()
    source, target, channel = _realization()
    inputs = dict(
        current_point_cloud=torch.randn(2, 2, 24, 6),
        current_state=_state_history(), source_parameter=source,
        target_parameter=target, channel_id=channel,
        global_trajectory=_identity_trajectory(2, cfg.global_horizon),
        global_reference_index=torch.tensor([0, 0]),
        noisy_action=torch.randn(2, cfg.local_horizon, cfg.action_dim),
        diffusion_timestep=torch.tensor([3, 7]),
    )
    full = policy(**inputs)
    kept = policy(**inputs, realization_keep_mask=torch.ones(2, dtype=torch.bool))
    assert torch.equal(full["predicted_noise"], kept["predicted_noise"])
    factor_kept = policy(**inputs, realization_keep_mask=torch.ones(2, 3, dtype=torch.bool))
    torch.testing.assert_close(full["predicted_noise"], factor_kept["predicted_noise"])
    factor_hidden = policy(**inputs, realization_keep_mask=torch.tensor([[False]*3, [True]*3]))
    assert torch.count_nonzero(factor_hidden["realization_feature"][0]) == 0
    torch.testing.assert_close(factor_hidden["trajectory_tokens"], full["trajectory_tokens"])
    masked = policy(**inputs, realization_keep_mask=torch.tensor([False, True]))
    assert torch.equal(masked["trajectory_tokens"], full["trajectory_tokens"])
    assert torch.count_nonzero(masked["realization_feature"][0]) == 0
    assert torch.equal(masked["realization_feature"][1], full["realization_feature"][1])
    torch.testing.assert_close(
        masked["local_global_condition"][0],
        full["local_global_condition"][0] - full["realization_feature"][0],
    )
    assert torch.equal(masked["predicted_noise"][1], full["predicted_noise"][1])
    with pytest.raises(ValueError, match="boolean"):
        policy(**inputs, realization_keep_mask=torch.ones(2))


def test_local_policy_q_sample_normalizes_before_diffusion(monkeypatch):
    monkeypatch.setattr(local_module, "OfficialDP3LocalBranch", _FakeLocalBranch)
    policy = local_module.LocalTrajectoryDP3Policy(small_config())
    clean = torch.ones(2, 8, 6)
    noise = torch.full_like(clean, 0.5)
    noisy = policy.q_sample(clean, noise, torch.tensor([2, 5]))
    assert torch.allclose(noisy, 2.5 * torch.ones_like(noisy))


def test_local_policy_sampling_shape_and_explicit_realization(monkeypatch):
    monkeypatch.setattr(local_module, "OfficialDP3LocalBranch", _FakeLocalBranch)
    cfg = small_config()
    policy = local_module.LocalTrajectoryDP3Policy(cfg)
    source, target, channel = _realization()
    result = policy.sample_actions(
        current_point_cloud=torch.randn(2, 2, 24, 6),
        current_state=_state_history(),
        source_parameter=source,
        target_parameter=target,
        channel_id=channel,
        global_trajectory=_identity_trajectory(2, cfg.global_horizon),
        global_reference_index=torch.tensor([0, 0]),
    )
    assert result["action"].shape == (
        2,
        cfg.local_action_steps,
        cfg.action_dim,
    )
    assert result["normalized_action"].shape == (
        2,
        cfg.local_action_steps,
        cfg.action_dim,
    )
    assert result["trajectory_tokens"].shape == (
        2,
        cfg.global_horizon,
        cfg.model_dim,
    )
    assert result["realization_feature"].shape == (2, cfg.model_dim)


@pytest.mark.parametrize(
    ("mode", "has_plan", "has_realization"),
    [
        ("vanilla", False, False),
        ("realization_only", False, True),
        ("global_only", True, False),
        ("flora", True, True),
    ],
)
def test_condition_modes_expose_only_requested_features(
    monkeypatch, mode, has_plan, has_realization
):
    monkeypatch.setattr(local_module, "OfficialDP3LocalBranch", _FakeLocalBranch)
    cfg = small_config()
    policy = local_module.LocalTrajectoryDP3Policy(cfg, condition_mode=mode)
    source, target, channel = _realization()
    outputs = policy(
        current_point_cloud=torch.randn(2, 2, 24, 6),
        current_state=_state_history(),
        source_parameter=source,
        target_parameter=target,
        channel_id=channel,
        global_trajectory=_identity_trajectory(2, cfg.global_horizon),
        global_reference_index=torch.tensor([0, 0]),
        noisy_action=torch.randn(2, cfg.local_horizon, cfg.action_dim),
        diffusion_timestep=torch.tensor([3, 7]),
    )
    assert ("trajectory_tokens" in outputs) is has_plan
    assert ("realization_feature" in outputs) is has_realization
    assert policy.local_dp3.use_plan_condition is has_plan
    assert policy.local_dp3.use_realization_condition is has_realization


def test_reference_index_fallback_uses_pose_and_orientation():
    cfg = small_config()
    trajectory = _identity_trajectory(1, cfg.global_horizon)
    trajectory[0, :, 0] = torch.linspace(0.0, 0.7, cfg.global_horizon)
    state = _state_history(batch=1)
    state[0, -1, 0] = 0.51
    index = local_module._infer_reference_index(
        trajectory,
        state[:, -1],
    )
    assert index.item() == 5


def test_standard_branch_rejects_a2_cross_attention_before_import():
    with pytest.raises(ValueError, match="A2 extension"):
        local_module.OfficialDP3LocalBranch(
            small_config(local_trajectory_cross_attention=True)
        )


def test_future_action_chunk_starts_at_current_action():
    physical = torch.arange(1 * 8 * 2, dtype=torch.float32).reshape(1, 8, 2)
    normalized = physical + 1000.0
    action, normalized_action = local_module._select_future_action_chunk(
        physical,
        normalized,
        action_steps=4,
    )
    assert torch.equal(action, physical[:, :4])
    assert torch.equal(normalized_action, normalized[:, :4])
    # Regression guard: FloRA must not use the official-DP3 history-aligned
    # slice physical[:, 1:5] for obs_steps=2.
    assert not torch.equal(action, physical[:, 1:5])


def test_factor_masks_hide_values_and_preserve_full_condition():
    import itertools
    encoder = local_module.LocalRealizationEncoder(small_config()).eval()
    source = torch.full((8, 1), 0.25)
    target = torch.full((8, 1), 0.0)
    channel = torch.zeros(8, dtype=torch.long)
    masks = torch.tensor(list(itertools.product([False, True], repeat=3)))
    masked = encoder(source, target, channel, keep_mask=masks)
    changed = encoder(
        torch.where(masks[:, :1], source, torch.full_like(source, 0.75)),
        torch.where(masks[:, 1:2], target, torch.full_like(target, 0.5)),
        torch.where(masks[:, 2], channel, torch.ones_like(channel)),
        keep_mask=masks,
    )
    torch.testing.assert_close(masked, changed)
    torch.testing.assert_close(masked[-1], encoder(source, target, channel)[-1])
    assert torch.count_nonzero(masked[0]) == 0
