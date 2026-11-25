"""Official SimpleDP3 local policy conditioned on a FloRA global trajectory.

This module deliberately imports 3D-Diffusion-Policy lazily.  Geometry, data,
and global-only users therefore do not need the DP3 environment.  Local policy
users can either install ``diffusion_policy_3d`` or point
``FLORA_OFFICIAL_DP3_ROOT`` at an official 3D-Diffusion-Policy checkout.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F

from .global_predictor import (
    GlobalLocalDiffusionConfig,
    SemanticSceneEncoder,
    TrajectoryTokenEncoder,
    current_relative_trajectory,
    rotation_6d_to_matrix,
)


def _load_official_dp3_components():
    """Load the official DP3 components without introducing an AEXT dependency."""
    configured_root = os.environ.get("FLORA_OFFICIAL_DP3_ROOT")
    if configured_root:
        root = Path(configured_root).expanduser().resolve()
        if not root.is_dir():
            raise RuntimeError(
                f"FLORA_OFFICIAL_DP3_ROOT is not a directory: {root}"
            )
        root_string = str(root)
        if root_string not in sys.path:
            sys.path.insert(0, root_string)
    try:
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        from diffusion_policy_3d.policy.simple_dp3 import SimpleDP3
        from diffusion_policy_3d.model.diffusion.simple_conditional_unet1d import (
            ConditionalUnet1D,
        )
    except (ImportError, RuntimeError) as exc:
        hint = (
            "install the official 3D-Diffusion-Policy dependencies or set "
            "FLORA_OFFICIAL_DP3_ROOT to its repository root"
        )
        raise RuntimeError(f"official DP3 components are unavailable; {hint}") from exc
    return DDPMScheduler, SimpleDP3, ConditionalUnet1D


class _DP3ConfigDict(dict):
    """Mapping with attribute access, matching DP3's Hydra config contract."""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class _DP3ObsSpec:
    """Shape leaf that official DP3's recursive ``dict_apply`` treats atomically."""

    __slots__ = ("shape",)

    def __init__(self, shape: list[int]) -> None:
        self.shape = shape

    def __getitem__(self, key: str):
        return getattr(self, key)


class LocalRealizationEncoder(nn.Module):
    """Encode abstract m=(u_o,u_g,c) as one local-policy condition vector."""

    def __init__(self, cfg: GlobalLocalDiffusionConfig) -> None:
        super().__init__()
        dim = cfg.model_dim
        self.source = nn.Sequential(
            nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.target = nn.Sequential(
            nn.Linear(2, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.channel = nn.Embedding(3, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        source_parameter: torch.Tensor,
        target_parameter: torch.Tensor,
        channel_id: torch.Tensor,
        keep_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if source_parameter.ndim != 2 or source_parameter.shape[-1] != 1:
            raise ValueError("source_parameter must be [B,1]")
        if target_parameter.ndim != 2 or target_parameter.shape[-1] != 1:
            raise ValueError("target_parameter must be [B,1]")
        if channel_id.ndim != 1:
            raise ValueError("channel_id must be [B]")
        if not torch.isfinite(source_parameter).all() or not torch.isfinite(target_parameter).all():
            raise ValueError("realization parameters must be finite")
        if torch.any((source_parameter < 0.0) | (source_parameter > 1.0)):
            raise ValueError("source_parameter must lie in [0,1]")
        if torch.any((target_parameter < 0.0) | (target_parameter > 1.0)):
            raise ValueError("target_parameter must lie in [0,1]")
        channel_id = channel_id.long()
        if torch.any((channel_id < 0) | (channel_id > 2)):
            raise ValueError("channel_id values must be 0, 1, or 2")
        phase = 2.0 * math.pi * target_parameter.float()
        target_periodic = torch.cat((phase.sin(), phase.cos()), dim=-1)
        features = (self.source(source_parameter.float()),
                    self.target(target_periodic), self.channel(channel_id))
        if keep_mask is None:
            return self.norm(features[0] + features[1] + features[2])
        if keep_mask.shape != (source_parameter.shape[0], 3) or keep_mask.dtype != torch.bool:
            raise ValueError("factor keep mask must be boolean [B,3] in u_o,u_g,c order")
        keep_mask = keep_mask.to(device=features[0].device)
        combined = sum(feature * keep_mask[:, i:i + 1].to(feature.dtype)
                       for i, feature in enumerate(features))
        # LayerNorm bias must not reintroduce an absent whole condition.
        return self.norm(combined) * keep_mask.any(dim=1, keepdim=True).to(combined.dtype)


def _infer_reference_index(
    global_trajectory: torch.Tensor,
    current_state: torch.Tensor,
) -> torch.Tensor:
    """Infer the closest SE(3) waypoint for training/fallback alignment."""
    if global_trajectory.ndim != 3 or global_trajectory.shape[-1] != 9:
        raise ValueError("global_trajectory must be [B,L,9]")
    if current_state.ndim != 2 or current_state.shape[-1] != 11:
        raise ValueError("current_state must be [B,11]")

    current_position = current_state[:, :3]
    current_rotation = SemanticSceneEncoder._quaternion_wxyz_to_matrix(
        current_state[:, 3:7]
    )
    waypoint_position = global_trajectory[..., :3]
    waypoint_rotation = rotation_6d_to_matrix(global_trajectory[..., 3:9])
    position_error = torch.linalg.vector_norm(
        waypoint_position - current_position[:, None, :], dim=-1
    )
    relative = torch.matmul(
        current_rotation[:, None, :, :],
        waypoint_rotation.transpose(-1, -2),
    )
    trace = relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    orientation_error = torch.acos(cosine)
    cost = position_error / 0.05 + orientation_error / math.radians(20.0)
    return cost.argmin(dim=1)


def _resample_remaining_trajectory(
    global_trajectory: torch.Tensor,
    reference_index: torch.Tensor,
) -> torch.Tensor:
    """Resample the remaining suffix to the original fixed token length."""
    if global_trajectory.ndim != 3 or global_trajectory.shape[-1] != 9:
        raise ValueError("global_trajectory must be [B,L,9]")
    batch, horizon, dim = global_trajectory.shape
    if reference_index.shape != (batch,):
        raise ValueError("reference_index must be [B]")
    reference_index = reference_index.to(
        device=global_trajectory.device, dtype=torch.long
    ).clamp(0, horizon - 1)
    alpha = torch.linspace(
        0.0, 1.0, horizon,
        device=global_trajectory.device,
        dtype=global_trajectory.dtype,
    )
    start = reference_index.to(global_trajectory.dtype)[:, None]
    last = global_trajectory.new_tensor(float(horizon - 1))
    sample_index = start + alpha[None, :] * (last - start)
    sample_index = sample_index.round().long().clamp(0, horizon - 1)
    return torch.gather(
        global_trajectory,
        1,
        sample_index[..., None].expand(batch, horizon, dim),
    )


def _select_future_action_chunk(
    action_pred: torch.Tensor,
    normalized_action_pred: torch.Tensor,
    *,
    action_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the executable future chunk under FloRA's action alignment.

    FloRA training windows are aligned as obs=[o_{t-1},o_t] and action
    target=[a_t,a_{t+1},...,a_{t+H-1}], where a_t drives state t -> t+1.
    Therefore denoised trajectory index 0 is already the first future action
    to execute. The common official-DP3 slice from n_obs_steps-1 assumes a
    different horizon alignment and must not be used for FloRA's dataset.
    """
    if action_pred.ndim != 3 or normalized_action_pred.ndim != 3:
        raise ValueError("predicted action trajectories must be [B,H,A]")
    if action_pred.shape != normalized_action_pred.shape:
        raise ValueError("physical and normalized action trajectories must match")
    if action_steps < 1 or action_steps > action_pred.shape[1]:
        raise ValueError("action_steps must lie within the predicted horizon")
    return (
        action_pred[:, :action_steps],
        normalized_action_pred[:, :action_steps],
    )

class OfficialDP3LocalBranch(nn.Module):
    """Official SimpleDP3 encoder and FiLM U-Net with FloRA conditions.

    Observation and action normalization follows official SimpleDP3. FloRA
    appends one attended remaining-global feature and one explicit realization
    feature to the U-Net global condition.
    """

    def __init__(
        self,
        cfg: GlobalLocalDiffusionConfig,
        *,
        use_plan_condition: bool = True,
        use_realization_condition: bool = True,
    ) -> None:
        super().__init__()
        self.use_plan_condition = bool(use_plan_condition)
        self.use_realization_condition = bool(use_realization_condition)
        if cfg.local_trajectory_cross_attention:
            raise ValueError(
                "local_trajectory_cross_attention requires the optional A2 "
                "extension; the standard FloRA local branch uses official FiLM DP3"
            )
        DDPMScheduler, SimpleDP3, ConditionalUnet1D = (
            _load_official_dp3_components()
        )
        pointcloud_cfg = _DP3ConfigDict(
            in_channels=cfg.point_dim,
            out_channels=cfg.model_dim,
            use_layernorm=True,
            final_norm="layernorm",
            normal_channel=False,
        )
        shape_meta = {
            "action": {"shape": [cfg.action_dim]},
            "obs": {
                "point_cloud": _DP3ObsSpec([1024, cfg.point_dim]),
                "agent_pos": _DP3ObsSpec([cfg.state_dim]),
            },
        }
        scheduler = DDPMScheduler(
            num_train_timesteps=cfg.diffusion_steps,
            beta_start=1e-4,
            beta_end=2e-2,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
        official = SimpleDP3(
            shape_meta=shape_meta,
            noise_scheduler=scheduler,
            horizon=cfg.local_horizon,
            n_action_steps=cfg.local_action_steps,
            n_obs_steps=cfg.local_obs_steps,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=cfg.model_dim,
            down_dims=(
                cfg.base_channels,
                2 * cfg.base_channels,
                4 * cfg.base_channels,
            ),
            kernel_size=5,
            n_groups=min(8, cfg.base_channels),
            condition_type="film",
            encoder_output_dim=cfg.model_dim,
            use_pc_color=True,
            pointnet_type="pointnet",
            pointcloud_encoder_cfg=pointcloud_cfg,
        )
        self.obs_encoder = official.obs_encoder
        self.normalizer = official.normalizer
        self.noise_scheduler = official.noise_scheduler
        self.obs_feature_dim = int(self.obs_encoder.output_shape())
        self.obs_steps = int(cfg.local_obs_steps)
        self.action_steps = int(cfg.local_action_steps)
        self.inference_steps = int(cfg.local_inference_steps)
        self.horizon = int(cfg.local_horizon)
        self.action_dim = int(cfg.action_dim)
        self.model_dim = int(cfg.model_dim)

        if self.use_plan_condition:
            self.plan_query = nn.Linear(self.obs_feature_dim, cfg.model_dim)
            self.plan_attention = nn.MultiheadAttention(
                cfg.model_dim, cfg.num_heads, batch_first=True
            )
            self.plan_norm = nn.LayerNorm(cfg.model_dim)
        else:
            self.plan_query = None
            self.plan_attention = None
            self.plan_norm = None

        global_cond_dim = self.obs_feature_dim * self.obs_steps
        if self.use_plan_condition:
            global_cond_dim += cfg.model_dim
        if self.use_realization_condition:
            global_cond_dim += cfg.model_dim
        self.model = ConditionalUnet1D(
            input_dim=cfg.action_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=cfg.model_dim,
            down_dims=(
                cfg.base_channels,
                2 * cfg.base_channels,
                4 * cfg.base_channels,
            ),
            kernel_size=5,
            n_groups=min(8, cfg.base_channels),
            condition_type="film",
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True,
        )
        root = os.environ.get("FLORA_OFFICIAL_DP3_ROOT", "installed package")
        self.provenance = {
            "repository": root,
            "policy_class": "diffusion_policy_3d.policy.simple_dp3.SimpleDP3",
            "observation_encoder": (
                self.obs_encoder.__class__.__module__
                + "."
                + self.obs_encoder.__class__.__name__
            ),
            "denoiser": self.model.__class__.__module__
            + "."
            + self.model.__class__.__name__,
            "extension": (
                "append attended remaining-global feature and explicit "
                "realization feature to global_cond"
                if self.use_plan_condition and self.use_realization_condition
                else (
                    "vanilla DP3 observation-only global_cond"
                    if not self.use_plan_condition and not self.use_realization_condition
                    else (
                        "append attended remaining-global feature to global_cond"
                        if self.use_plan_condition
                        else "append explicit realization feature to global_cond"
                    )
                )
            ),
        }

    def get_extra_state(self) -> dict[str, str]:
        return dict(self.provenance)

    def set_extra_state(self, state: dict[str, str]) -> None:
        self.provenance = dict(state)

    def _require_normalizer(self) -> None:
        try:
            self.normalizer["point_cloud"]
            self.normalizer["agent_pos"]
            self.normalizer["action"]
        except (KeyError, RuntimeError) as exc:
            raise RuntimeError(
                "Local DP3 normalizer is not initialized. Fit it from the "
                "training split or call set_normalizer() before training/inference."
            ) from exc

    def fit_normalizer(
        self,
        *,
        point_cloud: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
        mode: str = "limits",
    ) -> None:
        """Fit official DP3 normalization statistics from training data."""
        self.normalizer.fit(
            {
                "point_cloud": point_cloud,
                "agent_pos": state,
                "action": action,
            },
            last_n_dims=1,
            mode=mode,
        )

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        self._require_normalizer()
        return self.normalizer["action"].normalize(action)

    def unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        self._require_normalizer()
        return self.normalizer["action"].unnormalize(action)

    def _encode_observation(
        self,
        point_cloud_history: torch.Tensor,
        state_history: torch.Tensor,
    ) -> torch.Tensor:
        if point_cloud_history.ndim != 4 or state_history.ndim != 3:
            raise ValueError("observation history must be [B,T,N,C] and [B,T,S]")
        if point_cloud_history.shape[:2] != state_history.shape[:2]:
            raise ValueError("point-cloud and state histories must share [B,T]")
        if point_cloud_history.shape[1] < self.obs_steps:
            raise ValueError(
                f"official DP3 requires at least {self.obs_steps} observation steps"
            )
        self._require_normalizer()
        point_cloud_history = point_cloud_history[:, -self.obs_steps :]
        state_history = state_history[:, -self.obs_steps :]
        normalized = self.normalizer.normalize(
            {
                "point_cloud": point_cloud_history,
                "agent_pos": state_history,
            }
        )
        batch = point_cloud_history.shape[0]
        return self.obs_encoder(
            {
                "point_cloud": normalized["point_cloud"].reshape(
                    batch * self.obs_steps, *point_cloud_history.shape[2:]
                ),
                "agent_pos": normalized["agent_pos"].reshape(
                    batch * self.obs_steps, state_history.shape[-1]
                ),
            }
        ).reshape(batch, self.obs_steps, self.obs_feature_dim)

    def encode_condition(
        self,
        point_cloud_history: torch.Tensor,
        state_history: torch.Tensor,
        trajectory_tokens: torch.Tensor | None = None,
        realization_feature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        observation = self._encode_observation(
            point_cloud_history, state_history
        )
        features = [observation.reshape(observation.shape[0], -1)]

        if self.use_plan_condition:
            if trajectory_tokens is None or trajectory_tokens.ndim != 3:
                raise ValueError("trajectory_tokens must be [B,L,D]")
            if trajectory_tokens.shape[0] != observation.shape[0]:
                raise ValueError("trajectory and observation batches must match")
            if (
                self.plan_query is None
                or self.plan_attention is None
                or self.plan_norm is None
            ):
                raise RuntimeError("plan conditioning modules are unavailable")
            query = self.plan_query(observation[:, -1:])
            plan_feature, _ = self.plan_attention(
                query, trajectory_tokens, trajectory_tokens, need_weights=False
            )
            features.append(self.plan_norm(query + plan_feature).squeeze(1))

        if self.use_realization_condition:
            if realization_feature is None or realization_feature.shape != (
                observation.shape[0],
                self.model_dim,
            ):
                raise ValueError(
                    f"realization_feature must be [B,{self.model_dim}]"
                )
            features.append(realization_feature)

        return torch.cat(features, dim=-1) if len(features) > 1 else features[0]

    def forward(
        self,
        noisy_action: torch.Tensor,
        diffusion_timestep: torch.Tensor,
        global_condition: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            sample=noisy_action,
            timestep=diffusion_timestep,
            local_cond=None,
            global_cond=global_condition,
        )

    def add_noise(
        self,
        clean_action: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        normalized_action = self.normalize_action(clean_action)
        return self.noise_scheduler.add_noise(
            normalized_action, noise, timestep.long()
        )

    @torch.no_grad()
    def conditional_sample(
        self,
        point_cloud_history: torch.Tensor,
        state_history: torch.Tensor,
        trajectory_tokens: torch.Tensor | None = None,
        realization_feature: torch.Tensor | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        global_condition = self.encode_condition(
            point_cloud_history,
            state_history,
            trajectory_tokens,
            realization_feature,
        )
        batch = point_cloud_history.shape[0]
        trajectory = torch.randn(
            (batch, self.horizon, self.action_dim),
            device=point_cloud_history.device,
            dtype=point_cloud_history.dtype,
            generator=generator,
        )
        self.noise_scheduler.set_timesteps(self.inference_steps)
        for timestep in self.noise_scheduler.timesteps:
            predicted_noise = self(
                trajectory,
                torch.full(
                    (batch,),
                    int(timestep),
                    device=trajectory.device,
                    dtype=torch.long,
                ),
                global_condition,
            )
            trajectory = self.noise_scheduler.step(
                predicted_noise,
                timestep,
                trajectory,
                generator=generator,
            ).prev_sample
        action_pred = self.unnormalize_action(trajectory)
        action, normalized_action = _select_future_action_chunk(
            action_pred,
            trajectory,
            action_steps=self.action_steps,
        )
        return {
            "action": action,
            "action_pred": action_pred,
            "normalized_action": normalized_action,
            "normalized_action_pred": trajectory,
        }

    def set_normalizer(self, normalizer: nn.Module) -> None:
        self.normalizer.load_state_dict(normalizer.state_dict())


class LocalTrajectoryDP3Policy(nn.Module):
    """Local DP3 policy consuming an evaluated FloRA global trajectory.

    The policy keeps observation-driven receding-horizon control while adding
    two soft conditions: the explicit realization m=(u_o,u_g,c) and the
    remaining evaluated global motion. The global trajectory is detached before
    encoding so local diffusion gradients cannot repurpose it as an opaque
    latent channel.
    """

    def __init__(
        self,
        cfg: GlobalLocalDiffusionConfig = GlobalLocalDiffusionConfig(),
        *,
        current_relative_guidance: bool = True,
        remaining_global_guidance: bool = True,
        condition_mode: str = "flora",
    ) -> None:
        super().__init__()
        condition_flags = {
            "vanilla": (False, False),
            "realization_only": (False, True),
            "global_only": (True, False),
            "flora": (True, True),
        }
        if condition_mode not in condition_flags:
            raise ValueError(f"unsupported Local-DP3 condition mode: {condition_mode!r}")
        use_plan_condition, use_realization_condition = condition_flags[condition_mode]
        self.cfg = cfg
        self.condition_mode = condition_mode
        self.use_plan_condition = use_plan_condition
        self.use_realization_condition = use_realization_condition
        self.current_relative_guidance = bool(current_relative_guidance)
        self.remaining_global_guidance = bool(remaining_global_guidance)
        self.trajectory_encoder = (
            TrajectoryTokenEncoder(cfg) if use_plan_condition else None
        )
        self.realization_encoder = (
            LocalRealizationEncoder(cfg) if use_realization_condition else None
        )
        # Keep the default construction path byte-for-byte compatible with
        # existing FloRA checkpoints and lightweight test doubles.
        self.local_dp3 = (
            OfficialDP3LocalBranch(cfg)
            if condition_mode == "flora"
            else OfficialDP3LocalBranch(
                cfg,
                use_plan_condition=use_plan_condition,
                use_realization_condition=use_realization_condition,
            )
        )

    def _prepare_trajectory(
        self,
        global_trajectory: torch.Tensor,
        current_state: torch.Tensor,
        global_reference_index: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        trajectory = global_trajectory.detach()
        if global_reference_index is None:
            global_reference_index = _infer_reference_index(
                trajectory, current_state[:, -1]
            )
        else:
            global_reference_index = global_reference_index.to(
                device=trajectory.device,
                dtype=torch.long,
            ).reshape(trajectory.shape[0])

        if self.remaining_global_guidance:
            trajectory = _resample_remaining_trajectory(
                trajectory, global_reference_index
            )
        if self.current_relative_guidance:
            trajectory = current_relative_trajectory(
                trajectory,
                current_state[:, -1],
                force_identity_waypoint0=False,
            )
        return trajectory, global_reference_index

    def _encode_realization(
        self,
        source_parameter: torch.Tensor,
        target_parameter: torch.Tensor,
        channel_id: torch.Tensor,
    ) -> torch.Tensor:
        if self.realization_encoder is None:
            raise RuntimeError("realization encoder is disabled for this condition mode")
        return self.realization_encoder(
            source_parameter,
            target_parameter,
            channel_id,
        )

    def forward(
        self,
        *,
        current_point_cloud: torch.Tensor,
        current_state: torch.Tensor,
        source_parameter: torch.Tensor,
        target_parameter: torch.Tensor,
        channel_id: torch.Tensor,
        global_trajectory: torch.Tensor,
        noisy_action: torch.Tensor,
        diffusion_timestep: torch.Tensor,
        global_reference_index: torch.Tensor | None = None,
        realization_keep_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        local_trajectory = None
        reference_index = None
        trajectory_tokens = None
        realization_feature = None
        if self.use_plan_condition:
            local_trajectory, reference_index = self._prepare_trajectory(
                global_trajectory,
                current_state,
                global_reference_index,
            )
            if self.trajectory_encoder is None:
                raise RuntimeError("trajectory encoder is unavailable")
            trajectory_tokens = self.trajectory_encoder(local_trajectory)
        if self.use_realization_condition:
            factor_mask = realization_keep_mask is not None and realization_keep_mask.ndim == 2
            if factor_mask:
                realization_feature = self.realization_encoder(
                    source_parameter, target_parameter, channel_id,
                    keep_mask=realization_keep_mask,
                )
            else:
                realization_feature = self._encode_realization(
                    source_parameter, target_parameter, channel_id,
                )
            if realization_keep_mask is not None and not factor_mask:
                if realization_keep_mask.shape != (current_state.shape[0],):
                    raise ValueError("realization_keep_mask must be [B]")
                if realization_keep_mask.dtype != torch.bool:
                    raise ValueError("realization_keep_mask must be boolean")
                # Zero denotes an absent encoded condition, not a physical
                # realization with zero-valued parameters. Keep observations
                # and the complete remaining Global condition intact.
                realization_feature = realization_feature * realization_keep_mask.to(
                    device=realization_feature.device,
                    dtype=realization_feature.dtype,
                )[:, None]
        global_condition = self.local_dp3.encode_condition(
            current_point_cloud,
            current_state,
            trajectory_tokens,
            realization_feature,
        )
        predicted_noise = self.local_dp3(
            noisy_action,
            diffusion_timestep,
            global_condition,
        )
        result = {
            "predicted_noise": predicted_noise,
            "local_global_condition": global_condition,
        }
        if trajectory_tokens is not None:
            result.update(
                trajectory_tokens=trajectory_tokens,
                global_reference_index=reference_index,
                local_condition_trajectory=local_trajectory,
            )
        if realization_feature is not None:
            result["realization_feature"] = realization_feature
        return result

    def training_loss(
        self,
        outputs: dict[str, torch.Tensor],
        *,
        target_noise: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        diffusion = F.mse_loss(outputs["predicted_noise"], target_noise)
        return {"loss": diffusion, "diffusion_loss": diffusion}

    def q_sample(
        self,
        clean_action: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize raw task-space actions, then apply forward diffusion."""
        return self.local_dp3.add_noise(clean_action, noise, timestep)

    def fit_normalizer(
        self,
        *,
        point_cloud: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
        mode: str = "limits",
    ) -> None:
        self.local_dp3.fit_normalizer(
            point_cloud=point_cloud,
            state=state,
            action=action,
            mode=mode,
        )

    def set_normalizer(self, normalizer: nn.Module) -> None:
        self.local_dp3.set_normalizer(normalizer)

    @torch.no_grad()
    def sample_actions(
        self,
        *,
        current_point_cloud: torch.Tensor,
        current_state: torch.Tensor,
        source_parameter: torch.Tensor,
        target_parameter: torch.Tensor,
        channel_id: torch.Tensor,
        global_trajectory: torch.Tensor,
        global_reference_index: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        local_trajectory = None
        reference_index = None
        trajectory_tokens = None
        realization_feature = None
        if self.use_plan_condition:
            local_trajectory, reference_index = self._prepare_trajectory(
                global_trajectory,
                current_state,
                global_reference_index,
            )
            if self.trajectory_encoder is None:
                raise RuntimeError("trajectory encoder is unavailable")
            trajectory_tokens = self.trajectory_encoder(local_trajectory)
        if self.use_realization_condition:
            realization_feature = self._encode_realization(
                source_parameter,
                target_parameter,
                channel_id,
            )
        result = self.local_dp3.conditional_sample(
            current_point_cloud,
            current_state,
            trajectory_tokens,
            realization_feature,
            generator=generator,
        )
        if trajectory_tokens is not None:
            result["trajectory_tokens"] = trajectory_tokens
            result["global_reference_index"] = reference_index
            result["local_condition_trajectory"] = local_trajectory
        if realization_feature is not None:
            result["realization_feature"] = realization_feature
        return result


