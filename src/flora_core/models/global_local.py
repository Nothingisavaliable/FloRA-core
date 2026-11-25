"""Composition of FloRA's global predictor and official Local DP3 policy."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from .global_predictor import GlobalLocalDiffusionConfig, GlobalTrajectoryPredictor
from .local_dp3 import LocalTrajectoryDP3Policy


class GlobalLocalRealizationDiffusion(nn.Module):
    """Trainable Global--Local composition with an explicit optimization boundary.

    Runtime execution should evaluate and commit the predicted global trajectory
    before calling :meth:`sample_committed_trajectory`.  This class does not
    silently bypass the evaluator during local action sampling.
    """

    def __init__(
        self,
        cfg: GlobalLocalDiffusionConfig = GlobalLocalDiffusionConfig(),
        *,
        current_relative_guidance: bool = True,
        remaining_global_guidance: bool = True,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.global_predictor = GlobalTrajectoryPredictor(cfg)
        self.local_policy = LocalTrajectoryDP3Policy(
            cfg,
            current_relative_guidance=current_relative_guidance,
            remaining_global_guidance=remaining_global_guidance,
        )

    def predict_global_trajectory(
        self, global_inputs: Mapping[str, object]
    ) -> torch.Tensor:
        """Run the existing FloRA global predictor without duplicating it."""
        return self.global_predictor(**dict(global_inputs))

    @staticmethod
    def _realization_inputs(
        global_inputs: Mapping[str, object],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values = []
        for key in ("source_parameter", "target_parameter", "channel_id"):
            value = global_inputs.get(key)
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"global_inputs[{key!r}] must be a tensor for Local DP3"
                )
            values.append(value)
        return values[0], values[1], values[2]

    def forward(
        self,
        *,
        global_inputs: Mapping[str, object],
        current_point_cloud: torch.Tensor,
        current_state: torch.Tensor,
        noisy_action: torch.Tensor,
        diffusion_timestep: torch.Tensor,
        teacher_global_trajectory: torch.Tensor | None = None,
        teacher_forcing_probability: float | None = None,
        teacher_forcing_mask: torch.Tensor | None = None,
        global_reference_index: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        global_trajectory = self.predict_global_trajectory(global_inputs)
        if teacher_forcing_probability is None:
            teacher_forcing_probability = (
                1.0 if teacher_global_trajectory is not None else 0.0
            )
        if not 0.0 <= float(teacher_forcing_probability) <= 1.0:
            raise ValueError("teacher_forcing_probability must lie in [0,1]")
        batch = global_trajectory.shape[0]
        if teacher_global_trajectory is None:
            if teacher_forcing_probability > 0.0 or teacher_forcing_mask is not None:
                raise ValueError("teacher forcing requires teacher_global_trajectory")
            teacher_forcing_mask = torch.zeros(
                batch, dtype=torch.bool, device=global_trajectory.device
            )
            selected_trajectory = global_trajectory
        else:
            if teacher_global_trajectory.shape != global_trajectory.shape:
                raise ValueError(
                    "teacher_global_trajectory must match predicted global trajectory"
                )
            if teacher_forcing_mask is None:
                teacher_forcing_mask = (
                    torch.rand(batch, device=global_trajectory.device)
                    < float(teacher_forcing_probability)
                )
            else:
                teacher_forcing_mask = teacher_forcing_mask.to(
                    device=global_trajectory.device, dtype=torch.bool
                ).reshape(batch)
            selected_trajectory = torch.where(
                teacher_forcing_mask[:, None, None],
                teacher_global_trajectory,
                global_trajectory,
            )
        source_parameter, target_parameter, channel_id = self._realization_inputs(
            global_inputs
        )
        local_outputs = self.local_policy(
            current_point_cloud=current_point_cloud,
            current_state=current_state,
            source_parameter=source_parameter,
            target_parameter=target_parameter,
            channel_id=channel_id,
            global_trajectory=selected_trajectory,
            noisy_action=noisy_action,
            diffusion_timestep=diffusion_timestep,
            global_reference_index=global_reference_index,
        )
        return {
            "global_trajectory": global_trajectory,
            "teacher_forcing_mask": teacher_forcing_mask,
            **local_outputs,
        }

    def training_loss(
        self,
        outputs: dict[str, torch.Tensor],
        *,
        target_global_trajectory: torch.Tensor,
        target_noise: torch.Tensor,
        global_loss_kwargs: Mapping[str, object] | None = None,
    ) -> dict[str, torch.Tensor]:
        global_losses = self.global_predictor.training_loss(
            outputs["global_trajectory"],
            target_global_trajectory,
            **dict(global_loss_kwargs or {}),
        )
        diffusion = self.local_policy.training_loss(
            outputs, target_noise=target_noise
        )["diffusion_loss"]
        total = global_losses["loss"] + self.cfg.lambda_diffusion * diffusion
        return {
            "loss": total,
            "trajectory_loss": global_losses["loss"],
            "position_loss": global_losses["position_loss"],
            "rotation_loss": global_losses["rotation_loss"],
            "u_o_loss": global_losses["u_o_loss"],
            "diffusion_loss": diffusion,
        }

    def q_sample(
        self,
        clean_action: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        return self.local_policy.q_sample(clean_action, noise, timestep)

    def fit_local_normalizer(
        self,
        *,
        point_cloud: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
        mode: str = "limits",
    ) -> None:
        self.local_policy.fit_normalizer(
            point_cloud=point_cloud,
            state=state,
            action=action,
            mode=mode,
        )

    def set_local_normalizer(self, normalizer: nn.Module) -> None:
        self.local_policy.set_normalizer(normalizer)

    @torch.no_grad()
    def sample_committed_trajectory(
        self,
        *,
        current_point_cloud: torch.Tensor,
        current_state: torch.Tensor,
        source_parameter: torch.Tensor,
        target_parameter: torch.Tensor,
        channel_id: torch.Tensor,
        committed_global_trajectory: torch.Tensor,
        global_reference_index: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Sample actions only after the caller has evaluated/committed Global."""
        return self.local_policy.sample_actions(
            current_point_cloud=current_point_cloud,
            current_state=current_state,
            source_parameter=source_parameter,
            target_parameter=target_parameter,
            channel_id=channel_id,
            global_trajectory=committed_global_trajectory,
            global_reference_index=global_reference_index,
            generator=generator,
        )
