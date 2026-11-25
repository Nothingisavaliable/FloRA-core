"""Hierarchical global--local realization-conditioned diffusion network.

The implementation follows the paper contract

    m -> remaining global task-space trajectory -> local 6-DoF action chunk.

The global trajectory is an explicit, interpretable condition rather than an
opaque latent.  It is supervised as an SE(3) sequence, encoded as ordered
tokens, and pooled into the global condition of the official SimpleDP3
ConditionalUnet1D.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from os import PathLike

import torch
from torch import nn
import torch.nn.functional as F

from flora_core.data.scene_features import SCENE_PARAMETER_DIM, SCENE_PARAMETER_SLICES


@dataclass(frozen=True)
class GlobalLocalDiffusionConfig:
    point_dim: int = 6                 # xyz + source/target/environment one-hot
    scene_coordinate_representation: str = "absolute"
    # Global-only core scene path. Kept pointcloud by dataclass default so old
    # checkpoints remain loadable; the core trainer defaults to parameterized.
    global_scene_representation: str = "pointcloud"
    # Explicit semantic contract for the Global branch. ``legacy`` keeps
    # archived core checkpoints/inputs unchanged. ``scene_state`` separates a
    # stable, refreshable scene snapshot S from the current query state x_t.
    global_scene_source: str = "legacy"
    scene_parameter_dim: int = SCENE_PARAMETER_DIM
    # Pilot-C: replace Go/Gg parameter tokens with u-conditioned object point-cloud reads.
    object_point_interaction: bool = False
    # core source-only representation intervention. Replace the two active
    # abstract/collapsed source tokens with two grounded tokens while keeping
    # the 20-token fusion/decoder contract unchanged.
    grounded_source_bottleneck: bool = False
    scene_factor_masking: bool = False
    global_realization_mask_prob: float = 0.0
    full_condition_prob: float = 0.70
    one_factor_mask_prob: float = 0.25
    two_factor_mask_prob: float = 0.05
    state_dim: int = 11                # pose7 + phase one-hot3 + progress1
    action_dim: int = 6                # task-frame linear/angular velocity
    source_parameter_dim: int = 1      # normalized outlet coordinate u_o
    target_parameter_dim: int = 1      # normalized cyclic target coordinate u_g
    model_dim: int = 128
    num_heads: int = 4
    scene_slots_per_entity: int = 4
    channel_tokens: int = 16
    # "abstract" (u_o/u_g/c), "structured" (u_o/u_g/gamma_c), or
    # "instantiated" (p_o/p_g/gamma_c).
    realization_input: str = "abstract"
    # B2 observation-grounded realization conditioning (local branch only):
    # factor tokens query point-level geometry instead of the observation
    # feature querying realization tokens.
    observation_grounded_conditioning: bool = False
    # B2 scene-grounded realization conditioning (local branch only): encode
    # each factor as its abstract scalar plus the scene-instantiated task-frame
    # interaction pose (position + object-orientation Rot6D) rather than the
    # scalar alone.  No cross-attention to point geometry is used.
    scene_grounded_conditioning: bool = False
    # A2 local trajectory cross-attention (local branch only): keep the full
    # 32-token Global guidance Z_G and let each local denoiser residual block
    # attend to it (H_a' = H_a + CrossAttn(H_a, Z_G, Z_G)) instead of pooling
    # the trajectory to a single feature.  Only the A2 branch sets this.
    local_trajectory_cross_attention: bool = False
    global_horizon: int = 32
    local_horizon: int = 16
    local_obs_steps: int = 2
    local_action_steps: int = 4
    local_inference_steps: int = 10
    base_channels: int = 64
    fusion_layers: int = 2
    global_decoder_layers: int = 2
    trajectory_encoder_layers: int = 2
    diffusion_steps: int = 100
    lambda_position: float = 1.0
    lambda_rotation: float = 0.25
    lambda_diffusion: float = 1.0
    lambda_u_o: float = 0.0            # interaction loss weight on the source-outlet position (0 = off)
    # --- Orientation anchor (core Stage-1 Global, structural projection) ---
    # Fix the two directional DoFs of the interaction orientation at the
    # *trajectory-dependent* tilt-max interaction frame k*: tilt axis (from u_g)
    # + yaw/alignment (from u_o & u_g).  Tilt magnitude theta is extracted from
    # the network's own prediction at k* and kept; translation and the rest of
    # the trajectory are untouched.  k* is computed per-sample as the last
    # waypoint within 1e-4 of the maximum tilt, matching the evaluator's frame
    # selection exactly (see anchor_kstar_stability_gate.py).
    anchor_enabled: bool = False        # master switch; False = baseline
    # --- Cross-scene support conditioning (global branch only) ---
    support_enabled: bool = False       # master switch; False = baseline
    support_traj_length: int = 32       # resampled support trajectory length
    n_support: int = 4                  # support slots per query
    min_family_support: int = 1         # one other scene makes an SR correspondence
    support_motion_hidden: int = 128
    support_motion_out: int = 128       # aggregator value_dim
    aggregator_hidden: int = 128        # must be divisible by aggregator_heads
    aggregator_heads: int = 4
    aggregator_out: int = 128           # motion_prior dim before projection
    support_dropout_prob: float = 0.30
    # --- Pseudo-missing SR context conditioning (Stage II, global branch only) ---
    context_enabled: bool = False       # master switch; False = no Stage II context
    context_traj_length: int = 32       # resampled context trajectory length
    n_context: int = 4                  # partial-overlap context slots per query
    context_dropout_prob: float = 0.10


class SemanticSceneEncoder(nn.Module):
    """Turn a semantic point cloud into ordered entity scene tokens.

    ``absolute`` preserves the archived cup-to-bowl task-frame XYZ.  ``local``
    expresses source points in the current source-object frame and centers
    target points at their target-cloud centroid.  ``local_scale`` uses the
    same metric local coordinates and appends each entity's metric bounding-box
    diagonal as a seventh point feature.  Environment points remain in the
    task frame in all variants.  This deliberately keeps the ablation narrow:
    the trajectory target, state, token count, and network depth are unchanged.
    """

    def __init__(self, cfg: GlobalLocalDiffusionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        dim = cfg.model_dim
        self.dim = dim
        self.slots = cfg.scene_slots_per_entity
        self.point_encoder = nn.Sequential(
            nn.Linear(cfg.point_dim, dim), nn.LayerNorm(dim), nn.SiLU(),
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.SiLU(),
        )
        self.slot_queries = nn.Parameter(torch.randn(3, self.slots, dim) * 0.02)
        self.entity_embedding = nn.Parameter(torch.randn(3, 1, dim) * 0.02)
        self.attention = nn.MultiheadAttention(dim, cfg.num_heads, batch_first=True)
        self.slot_norm = nn.LayerNorm(dim)
        self.slot_ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.SiLU(), nn.Linear(4 * dim, dim)
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(cfg.state_dim, 2 * dim), nn.SiLU(),
            nn.Linear(2 * dim, dim), nn.LayerNorm(dim),
        )

    @staticmethod
    def _quaternion_wxyz_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
        quaternion = F.normalize(quaternion, dim=-1)
        w, x, y, z = quaternion.unbind(dim=-1)
        return torch.stack(
            (
                1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
            ),
            dim=-1,
        ).reshape(quaternion.shape[:-1] + (3, 3))

    def _represent_points(
        self, point_cloud: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        mode = self.cfg.scene_coordinate_representation
        if mode == "absolute":
            represented = point_cloud
        elif mode in {"local", "local_scale"}:
            xyz = point_cloud[..., :3]
            semantic = point_cloud[..., 3:6]
            source_mask = semantic[..., 0] > 0.5
            target_mask = semantic[..., 1] > 0.5

            source_rotation = self._quaternion_wxyz_to_matrix(state[..., 3:7])
            source_center = state[..., :3]
            source_local = torch.einsum(
                "bij,bnj->bni", source_rotation.transpose(-1, -2), xyz - source_center[:, None]
            )
            target_count = target_mask.sum(dim=1, keepdim=True).clamp_min(1)
            target_center = (
                xyz * target_mask[..., None]
            ).sum(dim=1) / target_count
            target_local = xyz - target_center[:, None]

            local_xyz = xyz.clone()
            local_xyz = torch.where(source_mask[..., None], source_local, local_xyz)
            local_xyz = torch.where(target_mask[..., None], target_local, local_xyz)
            represented = torch.cat((local_xyz, semantic), dim=-1)

            if mode == "local_scale":
                scale_feature = torch.zeros_like(xyz[..., :1])
                for entity, mask in enumerate((source_mask, target_mask)):
                    entity_xyz = source_local if entity == 0 else target_local
                    positive_inf = torch.full_like(entity_xyz, torch.inf)
                    negative_inf = torch.full_like(entity_xyz, -torch.inf)
                    minimum = torch.where(mask[..., None], entity_xyz, positive_inf).amin(dim=1)
                    maximum = torch.where(mask[..., None], entity_xyz, negative_inf).amax(dim=1)
                    scale = torch.linalg.vector_norm(maximum - minimum, dim=-1)
                    scale = torch.where(torch.isfinite(scale), scale, torch.zeros_like(scale))
                    scale_feature = torch.where(
                        mask[..., None], scale[:, None, None], scale_feature
                    )
                represented = torch.cat((represented, scale_feature), dim=-1)
        else:
            raise ValueError(f"unknown scene coordinate representation: {mode!r}")
        if represented.shape[-1] != self.cfg.point_dim:
            raise ValueError(
                f"representation {mode!r} produced point_dim={represented.shape[-1]}, "
                f"but config expects {self.cfg.point_dim}"
            )
        return represented

    def forward(
        self, point_cloud: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if point_cloud.ndim != 3 or point_cloud.shape[-1] < 6:
            raise ValueError("point_cloud must be [B,N,6+] with semantic one-hot channels")
        if state.ndim != 2:
            raise ValueError("state must be [B,state_dim]")
        batch = point_cloud.shape[0]
        represented = self._represent_points(point_cloud, state)
        point_features = self.point_encoder(represented)
        semantic = point_cloud[..., 3:6]
        entity_tokens = []
        for entity in range(3):
            valid = semantic[..., entity] > 0.5
            # MultiheadAttention cannot accept a row with every key masked.
            missing = ~valid.any(dim=1)
            if missing.any():
                valid = valid.clone()
                valid[missing, 0] = True
            query = (
                self.slot_queries[entity].unsqueeze(0).expand(batch, -1, -1)
                + self.entity_embedding[entity]
            )
            attended, _ = self.attention(
                query, point_features, point_features, key_padding_mask=~valid,
                need_weights=False,
            )
            attended = self.slot_norm(query + attended)
            entity_tokens.append(self.slot_norm(attended + self.slot_ffn(attended)))
        # Strict Global adds its current-state token separately. Its stable
        # scene snapshot must not encode the current query state a second time.
        if self.cfg.global_scene_source == "scene_state":
            tokens = torch.cat(entity_tokens, dim=1)
        else:
            state_token = self.state_encoder(state).unsqueeze(1)
            tokens = torch.cat((*entity_tokens, state_token), dim=1)
        pooled = tokens.mean(dim=1)
        return tokens, pooled


class ParameterizedSceneEncoder(nn.Module):
    """Encode five normalized, physical core scene-factor groups as tokens."""

    def __init__(self, cfg: GlobalLocalDiffusionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.scene_parameter_dim != SCENE_PARAMETER_DIM:
            raise ValueError(
                f"scene_parameter_dim must be {SCENE_PARAMETER_DIM}, got {cfg.scene_parameter_dim}"
            )
        dim = cfg.model_dim
        self.factor_names = list(SCENE_PARAMETER_SLICES)
        if cfg.object_point_interaction:
            self.factor_names = self.factor_names[:3]  # d, psi, obstacle only
        self.encoders = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(end - start, dim), nn.SiLU(),
                nn.Linear(dim, dim), nn.LayerNorm(dim),
            )
            for name, (start, end) in SCENE_PARAMETER_SLICES.items()
            if name in self.factor_names
        })
        self.type_embedding = nn.Parameter(torch.randn(len(self.factor_names), 1, dim) * 0.02)
        # Instantiate this parameter in both Mask and NoMask controls so all
        # downstream shared layers consume the same initialization RNG stream.
        # NoMask never reads it in forward(), hence its functional path remains
        # unchanged while paired initialization stays exact.
        self.mask_tokens = nn.Parameter(
            torch.randn(len(self.factor_names), 1, dim) * 0.02
        )

    def forward(
        self, scene_parameters: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if scene_parameters.ndim != 2 or scene_parameters.shape[-1] != SCENE_PARAMETER_DIM:
            raise ValueError(
                f"scene_parameters must be [B,{SCENE_PARAMETER_DIM}], "
                f"got {tuple(scene_parameters.shape)}"
            )
        tokens = []
        for index, name in enumerate(self.factor_names):
            start, end = SCENE_PARAMETER_SLICES[name]
            token = self.encoders[name](scene_parameters[:, start:end]).unsqueeze(1)
            tokens.append(token + self.type_embedding[index])
        result = torch.cat(tokens, dim=1)
        if self.training and self.cfg.scene_factor_masking:
            probabilities = (
                self.cfg.full_condition_prob,
                self.cfg.one_factor_mask_prob,
                self.cfg.two_factor_mask_prob,
            )
            if abs(sum(probabilities) - 1.0) > 1e-6 or min(probabilities) < 0.0:
                raise ValueError(f"invalid scene-factor mask probabilities: {probabilities}")
            batch = result.shape[0]
            draw = torch.rand(batch, device=result.device)
            counts = torch.zeros(batch, dtype=torch.long, device=result.device)
            counts[draw >= probabilities[0]] = 1
            counts[draw >= probabilities[0] + probabilities[1]] = 2
            # Independent random ranks give a uniform subset without replacement.
            factor_count = len(self.factor_names)
            ranks = torch.rand(batch, factor_count, device=result.device).argsort(dim=-1)
            mask = torch.zeros(batch, factor_count, dtype=torch.bool, device=result.device)
            selected = torch.arange(2, device=result.device).unsqueeze(0) < counts.unsqueeze(1)
            mask.scatter_(1, ranks[:, :2], selected)
            learned = self.mask_tokens.squeeze(1).unsqueeze(0).expand(batch, -1, -1)
            result = torch.where(mask.unsqueeze(-1), learned, result)
        return result, result.mean(dim=1)


class NormalizedInteractionQueryEncoder(nn.Module):
    """Encode a normalized interaction variable without re-encoding geometry."""

    def __init__(self, cfg: GlobalLocalDiffusionConfig, parameter_dim: int) -> None:
        super().__init__()
        dim = cfg.model_dim
        self.parameter_dim = int(parameter_dim)
        self.parameter_mlp = nn.Sequential(
            nn.Linear(self.parameter_dim, dim), nn.SiLU(),
            nn.Linear(dim, dim), nn.LayerNorm(dim),
        )
        self.type_embedding = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

    def forward(self, parameter: torch.Tensor) -> torch.Tensor:
        if parameter.ndim != 2 or parameter.shape[-1] != self.parameter_dim:
            raise ValueError(
                f"normalized parameter must be [B,{self.parameter_dim}], got {tuple(parameter.shape)}"
            )
        if not torch.isfinite(parameter).all() or torch.any((parameter < 0.0) | (parameter > 1.0)):
            raise ValueError("normalized interaction parameters must be finite and lie in [0,1]")
        return self.parameter_mlp(parameter.float()).unsqueeze(1) + self.type_embedding


class PeriodicTargetQueryEncoder(nn.Module):
    """Encode one-dimensional cyclic u_g without a discontinuity at 0/1."""

    def __init__(self, cfg: GlobalLocalDiffusionConfig) -> None:
        super().__init__()
        dim = cfg.model_dim
        self.parameter_mlp = nn.Sequential(
            nn.Linear(2, dim), nn.SiLU(), nn.Linear(dim, dim), nn.LayerNorm(dim)
        )
        self.type_embedding = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

    def forward(self, parameter: torch.Tensor) -> torch.Tensor:
        if parameter.ndim != 2 or parameter.shape[-1] != 1:
            raise ValueError(
                f"normalized target parameter must be [B,1], got {tuple(parameter.shape)}"
            )
        if not torch.isfinite(parameter).all() or torch.any((parameter < 0.0) | (parameter > 1.0)):
            raise ValueError("normalized target parameters must be finite and lie in [0,1]")
        phase = 2.0 * math.pi * parameter.float()
        periodic = torch.cat((phase.sin(), phase.cos()), dim=-1)
        return self.parameter_mlp(periodic).unsqueeze(1) + self.type_embedding


class CanonicalChannelQueryEncoder(nn.Module):
    """Encode c from a scene-independent canonical task-frame skeleton.

    The skeleton only represents the normalized left/right/over identity.  It
    is not the collision-aware scene instantiation gamma_c(s), which remains a
    stored supervision/visualization object outside this encoder.
    """

    def __init__(self, cfg: GlobalLocalDiffusionConfig) -> None:
        super().__init__()
        dim = cfg.model_dim
        self.token_count = cfg.channel_tokens
        self.point_mlp = nn.Sequential(
            nn.Linear(4, dim), nn.SiLU(), nn.Linear(dim, dim), nn.LayerNorm(dim)
        )
        self.channel_embedding = nn.Embedding(3, dim)
        self.type_embedding = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        # Source and target are fixed at (-1,0,0) and (0,0,0).  Only the
        # normalized coarse passage differs; metric obstacle avoidance is not
        # encoded here.
        canonical = torch.tensor(
            [
                [[-1.00, 0.00, 0.12], [-0.78, 0.28, 0.22], [-0.50, 0.48, 0.25],
                 [-0.22, 0.28, 0.22], [0.00, 0.00, 0.12]],
                [[-1.00, 0.00, 0.12], [-0.78, -0.28, 0.22], [-0.50, -0.48, 0.25],
                 [-0.22, -0.28, 0.22], [0.00, 0.00, 0.12]],
                [[-1.00, 0.00, 0.12], [-0.78, 0.00, 0.50], [-0.50, 0.00, 0.78],
                 [-0.22, 0.00, 0.50], [0.00, 0.00, 0.12]],
            ],
            dtype=torch.float32,
        )
        self.register_buffer("canonical_skeletons", canonical, persistent=True)

    def forward(self, channel_id: torch.Tensor) -> torch.Tensor:
        if channel_id.ndim != 1:
            raise ValueError("channel_id must be [B], with 0=left, 1=right, 2=over")
        channel_id = channel_id.long()
        if torch.any((channel_id < 0) | (channel_id > 2)):
            raise ValueError("channel_id values must be 0=left, 1=right, or 2=over")
        channel = self.canonical_skeletons[channel_id]
        sampled = F.interpolate(
            channel.transpose(1, 2), size=self.token_count,
            mode="linear", align_corners=True,
        ).transpose(1, 2)
        order = torch.linspace(0.0, 1.0, self.token_count, device=channel.device)
        order = order.reshape(1, -1, 1).expand(channel.shape[0], -1, -1)
        tokens = self.point_mlp(torch.cat((sampled, order), dim=-1))
        return tokens + self.channel_embedding(channel_id).unsqueeze(1) + self.type_embedding


class PositionQueryEncoder(nn.Module):
    """Encode a 3D scene-instantiated interaction position (p_o or p_g)."""

    def __init__(self, cfg: GlobalLocalDiffusionConfig, position_dim: int = 3) -> None:
        super().__init__()
        dim = cfg.model_dim
        self.position_dim = int(position_dim)
        self.position_mlp = nn.Sequential(
            nn.Linear(self.position_dim, dim), nn.SiLU(),
            nn.Linear(dim, dim), nn.LayerNorm(dim),
        )
        self.type_embedding = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

    def forward(self, position: torch.Tensor) -> torch.Tensor:
        if position.ndim != 2 or position.shape[-1] != self.position_dim:
            raise ValueError(
                f"position must be [B,{self.position_dim}], got {tuple(position.shape)}"
            )
        if not torch.isfinite(position).all():
            raise ValueError("interaction positions must be finite")
        return self.position_mlp(position.float()).unsqueeze(1) + self.type_embedding


class SceneGroundedSourceQueryEncoder(nn.Module):
    """Encode the scene-grounded source factor ``[u_o, p_o, src_rot6d]`` -> one token.

    The scalar outlet coordinate is concatenated with its scene-instantiated
    task-frame interaction position and source-object orientation (first two
    rows of the task-frame rotation matrix), so the token carries the concrete
    pose the scalar realises in this scene rather than the scalar alone.
    """

    def __init__(self, cfg: GlobalLocalDiffusionConfig) -> None:
        super().__init__()
        dim = cfg.model_dim
        self.input_dim = 1 + 3 + 6
        self.parameter_mlp = nn.Sequential(
            nn.Linear(self.input_dim, dim), nn.SiLU(),
            nn.Linear(dim, dim), nn.LayerNorm(dim),
        )
        self.type_embedding = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

    def forward(self, grounded: torch.Tensor) -> torch.Tensor:
        if grounded.ndim != 2 or grounded.shape[-1] != self.input_dim:
            raise ValueError(
                f"scene-grounded source factor must be [B,{self.input_dim}], "
                f"got {tuple(grounded.shape)}"
            )
        if not torch.isfinite(grounded).all():
            raise ValueError("scene-grounded source factor must be finite")
        u_o = grounded[:, 0:1]
        if torch.any((u_o < 0.0) | (u_o > 1.0)):
            raise ValueError("u_o must lie in [0,1]")
        return self.parameter_mlp(grounded.float()).unsqueeze(1) + self.type_embedding


class SceneGroundedTargetQueryEncoder(nn.Module):
    """Encode the scene-grounded target factor ``[sin,cos,p_g,tgt_rot6d]`` -> one token."""

    def __init__(self, cfg: GlobalLocalDiffusionConfig) -> None:
        super().__init__()
        dim = cfg.model_dim
        self.input_dim = 2 + 3 + 6
        self.parameter_mlp = nn.Sequential(
            nn.Linear(self.input_dim, dim), nn.SiLU(),
            nn.Linear(dim, dim), nn.LayerNorm(dim),
        )
        self.type_embedding = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

    def forward(self, grounded: torch.Tensor) -> torch.Tensor:
        if grounded.ndim != 2 or grounded.shape[-1] != self.input_dim:
            raise ValueError(
                f"scene-grounded target factor must be [B,{self.input_dim}], "
                f"got {tuple(grounded.shape)}"
            )
        if not torch.isfinite(grounded).all():
            raise ValueError("scene-grounded target factor must be finite")
        return self.parameter_mlp(grounded.float()).unsqueeze(1) + self.type_embedding


class SpatialChannelEncoder(nn.Module):
    """Encode the scene-instantiated corridor polyline γ_c ∈ R^{N_c×3}.

    Each ordered waypoint is projected through a shared MLP, then summed with a
    learned order embedding (position along the polyline) and a channel type
    embedding.  The polyline is resampled to ``token_count`` first, so the output
    stays [B, token_count, dim] regardless of the archive's ``N_c``.
    """

    def __init__(self, cfg: GlobalLocalDiffusionConfig) -> None:
        super().__init__()
        dim = cfg.model_dim
        self.token_count = cfg.channel_tokens
        self.point_mlp = nn.Sequential(
            nn.Linear(3, dim), nn.SiLU(), nn.Linear(dim, dim), nn.LayerNorm(dim)
        )
        self.order_embedding = nn.Parameter(torch.randn(1, self.token_count, dim) * 0.02)
        self.type_embedding = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

    def forward(self, channel_waypoints: torch.Tensor) -> torch.Tensor:
        if channel_waypoints.ndim != 3 or channel_waypoints.shape[-1] != 3:
            raise ValueError(
                f"channel_waypoints must be [B,N_c,3], got {tuple(channel_waypoints.shape)}"
            )
        if not torch.isfinite(channel_waypoints).all():
            raise ValueError("channel waypoints must be finite")
        sampled = F.interpolate(
            channel_waypoints.transpose(1, 2), size=self.token_count,
            mode="linear", align_corners=True,
        ).transpose(1, 2)
        tokens = self.point_mlp(sampled)
        return tokens + self.order_embedding + self.type_embedding


class ObjectInteractionGeometryEncoder(nn.Module):
    """Use one realization token to query point-level source/target geometry."""

    def __init__(self, cfg: GlobalLocalDiffusionConfig, semantic_index: int) -> None:
        super().__init__()
        self.semantic_index = int(semantic_index)
        dim = cfg.model_dim
        self.point_encoder = nn.Sequential(
            nn.Linear(cfg.point_dim, dim), nn.LayerNorm(dim), nn.SiLU(),
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.SiLU(),
        )
        self.attention = nn.MultiheadAttention(dim, cfg.num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.SiLU(), nn.Linear(4 * dim, dim))
        self.norm2 = nn.LayerNorm(dim)
        self.type_embedding = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

    def forward(self, query: torch.Tensor, point_cloud: torch.Tensor) -> torch.Tensor:
        if query.ndim != 3 or query.shape[1] != 1:
            raise ValueError("object interaction query must be [B,1,D]")
        if point_cloud.ndim != 3 or point_cloud.shape[-1] < 6:
            raise ValueError("object interaction point cloud must be [B,N,6+]")
        valid = point_cloud[..., 3 + self.semantic_index] > 0.5
        # The archived core semantic cloud must contain both object entities.
        missing = ~valid.any(dim=1)
        if missing.any():
            raise ValueError(f"point cloud is missing semantic entity {self.semantic_index}")
        features = self.point_encoder(point_cloud[..., : self.point_encoder[0].in_features])
        update, _ = self.attention(
            query, features, features, key_padding_mask=~valid, need_weights=False
        )
        token = self.norm1(query + update) + self.type_embedding
        return self.norm2(token + self.ffn(token))


class PointSetTokenEncoder(nn.Module):
    """Encode a selected point set as one permutation-invariant token."""

    def __init__(self, cfg: GlobalLocalDiffusionConfig, input_dim: int) -> None:
        super().__init__()
        dim = cfg.model_dim
        self.input_dim = int(input_dim)
        self.point_mlp = nn.Sequential(
            nn.Linear(self.input_dim, dim), nn.LayerNorm(dim), nn.SiLU(),
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.SiLU(),
        )
        self.pool_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.SiLU(), nn.Linear(4 * dim, dim)
        )
        self.output_norm = nn.LayerNorm(dim)
        self.type_embedding = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3 or points.shape[-1] != self.input_dim:
            raise ValueError(
                f"point-set input must be [B,N,{self.input_dim}], got {tuple(points.shape)}"
            )
        if points.shape[1] == 0 or not torch.isfinite(points).all():
            raise ValueError("point-set input must be non-empty and finite")
        features = self.point_mlp(points.float())
        pooled = self.pool_norm(features.mean(dim=1) + features.max(dim=1).values)
        token = pooled.unsqueeze(1) + self.type_embedding
        return self.output_norm(token + self.ffn(token))


def _build_realization_query_encoders(
    cfg: GlobalLocalDiffusionConfig,
) -> tuple[nn.Module, nn.Module, nn.Module]:
    """Construct the three realization query encoders for the selected input mode."""
    if cfg.realization_input == "instantiated":
        return PositionQueryEncoder(cfg), PositionQueryEncoder(cfg), SpatialChannelEncoder(cfg)
    if cfg.realization_input == "structured":
        return (
            NormalizedInteractionQueryEncoder(cfg, cfg.source_parameter_dim),
            PeriodicTargetQueryEncoder(cfg),
            SpatialChannelEncoder(cfg),
        )
    if cfg.realization_input != "abstract":
        raise ValueError(f"unknown realization_input: {cfg.realization_input!r}")
    return (
        NormalizedInteractionQueryEncoder(cfg, cfg.source_parameter_dim),
        PeriodicTargetQueryEncoder(cfg),
        CanonicalChannelQueryEncoder(cfg),
    )


class CrossConditionFusion(nn.Module):
    def __init__(self, cfg: GlobalLocalDiffusionConfig) -> None:
        super().__init__()
        dim = cfg.model_dim
        self.self_attention = nn.MultiheadAttention(dim, cfg.num_heads, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(dim, cfg.num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.SiLU(), nn.Linear(4 * dim, dim))

    def forward(self, realization: torch.Tensor, scene: torch.Tensor) -> torch.Tensor:
        update, _ = self.self_attention(realization, realization, realization, need_weights=False)
        realization = self.norm1(realization + update)
        update, _ = self.cross_attention(realization, scene, scene, need_weights=False)
        realization = self.norm2(realization + update)
        return self.norm3(realization + self.ffn(realization))


class GlobalTrajectoryDecoder(nn.Module):
    def __init__(self, cfg: GlobalLocalDiffusionConfig) -> None:
        super().__init__()
        dim = cfg.model_dim
        self.horizon = cfg.global_horizon
        self.query = nn.Parameter(torch.randn(1, self.horizon, dim) * 0.02)
        self.progress_encoder = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
        layer = nn.TransformerDecoderLayer(
            d_model=dim, nhead=cfg.num_heads, dim_feedforward=4 * dim,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, cfg.global_decoder_layers)
        self.pose_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 9))
        progress = torch.linspace(0.0, 1.0, self.horizon).reshape(1, self.horizon, 1)
        self.register_buffer("query_progress", progress, persistent=False)

    def forward(self, plan_tokens: torch.Tensor) -> torch.Tensor:
        batch = plan_tokens.shape[0]
        query = self.query.expand(batch, -1, -1)
        query = query + self.progress_encoder(self.query_progress.expand(batch, -1, -1))
        return self.pose_head(self.decoder(query, plan_tokens))


class TrajectoryTokenEncoder(nn.Module):
    def __init__(self, cfg: GlobalLocalDiffusionConfig) -> None:
        super().__init__()
        dim = cfg.model_dim
        self.input = nn.Sequential(nn.Linear(10, dim), nn.LayerNorm(dim), nn.SiLU())
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=cfg.num_heads, dim_feedforward=4 * dim,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, cfg.trajectory_encoder_layers)

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        batch, horizon = trajectory.shape[:2]
        order = torch.linspace(0.0, 1.0, horizon, device=trajectory.device)
        order = order.reshape(1, horizon, 1).expand(batch, -1, -1)
        return self.encoder(self.input(torch.cat((trajectory, order), dim=-1)))


def rotation_6d_to_matrix(rotation: torch.Tensor) -> torch.Tensor:
    first = F.normalize(rotation[..., 0:3], dim=-1)
    second_raw = rotation[..., 3:6]
    second = F.normalize(second_raw - (first * second_raw).sum(-1, keepdim=True) * first, dim=-1)
    third = torch.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-2)


def rotation_geodesic_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_matrix = rotation_6d_to_matrix(predicted)
    target_matrix = rotation_6d_to_matrix(target)
    relative = pred_matrix @ target_matrix.transpose(-1, -2)
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-0.9999, 0.9999)
    return torch.acos(cosine)


def outlet_position(trajectory: torch.Tensor, rim_local: torch.Tensor) -> torch.Tensor:
    """Lift the source-outlet rim point through each trajectory waypoint.

    ``trajectory`` is [B,L,9] = [position(3), Rot6D(6)] in the task frame;
    ``rim_local`` is the requested u_o's cup-local rim point [B,3].  Returns the
    outlet position [B,L,3] = position + R(rot6d) @ rim_local, differentiable in
    both the position and rotation of every waypoint.
    """
    if trajectory.ndim != 3 or trajectory.shape[-1] != 9:
        raise ValueError(
            f"trajectory must be [B,L,9], got {tuple(trajectory.shape)}"
        )
    if rim_local.ndim != 2 or rim_local.shape[-1] != 3:
        raise ValueError(f"rim_local must be [B,3], got {tuple(rim_local.shape)}")
    if rim_local.shape[0] != trajectory.shape[0]:
        raise ValueError("rim_local batch must match trajectory batch")
    position = trajectory[..., :3]
    rotation = rotation_6d_to_matrix(trajectory[..., 3:9])  # [B,L,3,3]
    return position + torch.einsum("blij,bj->bli", rotation, rim_local)


def rotation_matrix_from_axis_angle(
    axis: torch.Tensor, angle: torch.Tensor
) -> torch.Tensor:
    """Batch Rodrigues rotation matrix from a unit axis [B,3] and angle [B]."""
    axis = F.normalize(axis, dim=-1)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = torch.zeros_like(x)
    skew = torch.stack(
        (
            torch.stack((zero, -z, y), dim=-1),
            torch.stack((z, zero, -x), dim=-1),
            torch.stack((-y, x, zero), dim=-1),
        ),
        dim=-2,
    )
    c = torch.cos(angle)
    s = torch.sin(angle)
    identity = torch.eye(3, dtype=axis.dtype, device=axis.device).expand(axis.shape[0], 3, 3)
    return identity + s[..., None, None] * skew + (1.0 - c)[..., None, None] * (skew @ skew)


def rotation_matrix_z(angle: torch.Tensor) -> torch.Tensor:
    """Batch rotation about the task-frame z-axis by ``angle`` [B] -> [B,3,3]."""
    c = torch.cos(angle)
    s = torch.sin(angle)
    zero = torch.zeros_like(angle)
    one = torch.ones_like(angle)
    return torch.stack(
        (
            torch.stack((c, -s, zero), dim=-1),
            torch.stack((s, c, zero), dim=-1),
            torch.stack((zero, zero, one), dim=-1),
        ),
        dim=-2,
    )


def interaction_orientation_anchor_project(
    trajectory: torch.Tensor,
    *,
    anchor_tilt_axis: torch.Tensor,
    anchor_yaw: torch.Tensor,
    anchor_initial_rotation: torch.Tensor,
) -> torch.Tensor:
    """Structural orientation anchor at the trajectory-dependent tilt-max frame k*.

    ``trajectory`` is [B,L,9] = [position(3), Rot6D(6)] in the task frame (with
    waypoint 0 already carrying the query-pose boundary condition for scene_state).
    The interaction frame k* is computed per-sample as the *last* waypoint whose
    tilt is within 1e-4 of the maximum tilt, matching the evaluator's frame
    selection exactly.  The network's tilt *magnitude* theta is extracted from its
    own predicted orientation at k* (as the angle between the yawed initial up and
    the predicted up), then the orientation at k* is rebuilt as

        R*[k*] = R_tilt(a_tilt*, theta) @ R_yaw(phi*) @ R_initial,

    pinning the tilt axis (u_g) and yaw/alignment (u_o & u_g) while preserving the
    learned tilt magnitude.  Position and every other waypoint are untouched.
    k* is non-differentiable (the gradient flows through theta, not the index).

    Multiplication order is validated by ``anchor_geometry_sanity_test.py`` against
    the unchanged evaluator (100%/100% matching).
    """
    if trajectory.ndim != 3 or trajectory.shape[-1] != 9:
        raise ValueError(f"trajectory must be [B,L,9], got {tuple(trajectory.shape)}")
    batch = trajectory.shape[0]
    for name, tensor, expected in (
        ("anchor_tilt_axis", anchor_tilt_axis, (batch, 3)),
        ("anchor_yaw", anchor_yaw, (batch,)),
        ("anchor_initial_rotation", anchor_initial_rotation, (batch, 3, 3)),
    ):
        if tensor.shape != expected:
            raise ValueError(f"{name} must be {expected}, got {tuple(tensor.shape)}")

    matrices = rotation_6d_to_matrix(trajectory[..., 3:9])  # [B,L,3,3]
    e_z = trajectory.new_tensor([0.0, 0.0, 1.0])
    up0 = torch.einsum("bij,j->bi", anchor_initial_rotation, e_z)  # R_initial @ e_z

    # Dynamic interaction frame k* (last waypoint within 1e-4 of the max tilt).
    # arccos here is only used for the argmax-style index and never feeds the
    # gradient, so its unbounded slope at +-1 is harmless; wrap in no_grad.
    length = trajectory.shape[1]
    with torch.no_grad():
        ups = torch.einsum("blij,j->bli", matrices, e_z)  # [B,L,3]
        tilt = torch.arccos(torch.clamp((ups * up0[:, None, :]).sum(-1), -1.0, 1.0))
        maximum = tilt.max(dim=1, keepdim=True).values  # [B,1]
        within_max = tilt >= (maximum - 1e-4)  # [B,L]
        frame_idx = torch.arange(length, device=trajectory.device, dtype=torch.long)
        k_star = (within_max.long() * frame_idx).max(dim=1).values  # [B] last max frame

    row = torch.arange(batch, device=trajectory.device, dtype=torch.long)
    Rk_up = torch.einsum("bij,j->bi", matrices[row, k_star], e_z)  # predicted up at k*

    # Tilt magnitude from the network's own prediction at k*.  Use the
    # atan2(||cross||, dot) form rather than acos(dot): acos has an unbounded
    # gradient at dot = +-1, which diverges the decoder when the predicted up
    # aligns with the yawed initial up mid-training.
    R_yaw = rotation_matrix_z(anchor_yaw)
    up_yawed = torch.einsum("bij,bj->bi", R_yaw, up0)
    dot = (up_yawed * Rk_up).sum(-1)
    cross = torch.linalg.cross(up_yawed, Rk_up, dim=-1)
    cross_norm = torch.sqrt((cross * cross).sum(-1) + 1e-12)
    theta = torch.atan2(cross_norm, dot)

    R_tilt = rotation_matrix_from_axis_angle(anchor_tilt_axis, theta)
    R_star = R_tilt @ R_yaw @ anchor_initial_rotation  # [B,3,3]

    matrices = matrices.clone()
    matrices[row, k_star] = R_star
    rot6d = matrices[..., :2, :].reshape(*matrices.shape[:-2], 6)
    return torch.cat((trajectory[..., :3], rot6d), dim=-1)


def current_relative_trajectory(
    trajectory: torch.Tensor,
    state: torch.Tensor,
    *,
    force_identity_waypoint0: bool = True,
) -> torch.Tensor:
    """Convert absolute task-frame Global waypoints to current-relative guidance.

    ``trajectory`` is [B,L,9] = [position(3), Rot6D(6)] in the task frame; the
    Rot6D encodes the first two rows of the rotation matrix.  ``state`` is the
    core current query state [B,11] = pose7 + phase one-hot3 + progress1.

    The result is ``Delta T_k = T_t^{-1} . T_hat_k`` in the same
    [position(3), Rot6D(6)] layout. By default waypoint 0 is forced to the
    identity to preserve the archived Global contract. Local remaining-path
    guidance may pass ``force_identity_waypoint0=False`` so the actual
    tracking error to the first remaining reference is retained.
    """
    if trajectory.ndim != 3 or trajectory.shape[-1] != 9:
        raise ValueError(
            f"trajectory must be [B,L,9], got {tuple(trajectory.shape)}"
        )
    if state.ndim != 2 or state.shape[-1] != 11:
        raise ValueError(
            f"state must be [B,11] (core x_t), got {tuple(state.shape)}"
        )
    current_rotation = SemanticSceneEncoder._quaternion_wxyz_to_matrix(
        state[..., 3:7]
    )
    current_position = state[..., :3]
    inv_rotation = current_rotation.transpose(-1, -2)

    waypoint_position = trajectory[..., :3]
    waypoint_rotation = rotation_6d_to_matrix(trajectory[..., 3:9])

    rel_position = torch.einsum(
        "bij,blj->bli",
        inv_rotation,
        waypoint_position - current_position[:, None, :],
    )
    rel_rotation = torch.einsum("bij,bljk->blik", inv_rotation, waypoint_rotation)
    rel_rot6d = rel_rotation[..., :2, :].reshape(*rel_rotation.shape[:-2], 6)
    relative = torch.cat((rel_position, rel_rot6d), dim=-1)

    if force_identity_waypoint0:
        identity_rot6d = trajectory.new_tensor(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        )
        relative = relative.clone()
        relative[..., 0, :3] = 0.0
        relative[..., 0, 3:] = identity_rot6d
    return relative


class SupportTrajectoryEncoder(nn.Module):
    """Encode a 9-dim global support trajectory into one compact motion feature.

    Mirrors the cross-scene reference ``SupportMotionEncoder`` (three stride-2
    1D convolutions followed by global-average pooling and an output MLP), but
    parameterized for the global branch's 9-dim SE(3) trajectory space and an
    arbitrary hidden width.  GroupNorm group counts are derived from the actual
    channel counts so the module stays valid for a smoke test's small ``hidden``.
    """

    def __init__(
        self,
        pose_dim: int = 9,
        hidden: int = 128,
        output_dim: int = 128,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()

        def group_norm(channels: int, target_groups: int) -> nn.GroupNorm:
            groups = target_groups
            while groups > 1 and channels % groups != 0:
                groups -= 1
            return nn.GroupNorm(groups, channels)

        self.conv1 = nn.Sequential(
            nn.Conv1d(pose_dim, hidden // 4, kernel_size, stride=2, padding=kernel_size // 2),
            group_norm(hidden // 4, 4),
            nn.SiLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(hidden // 4, hidden // 2, kernel_size, stride=2, padding=kernel_size // 2),
            group_norm(hidden // 2, 8),
            nn.SiLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(hidden // 2, hidden, kernel_size, stride=2, padding=kernel_size // 2),
            group_norm(hidden, 16),
            nn.SiLU(),
        )
        self.output_proj = nn.Sequential(
            nn.Linear(hidden, output_dim), nn.SiLU(),
            nn.Linear(output_dim, output_dim), nn.LayerNorm(output_dim),
        )

    def forward(self, trajectories: torch.Tensor) -> torch.Tensor:
        if trajectories.ndim != 3 or trajectories.shape[-1] != 9:
            raise ValueError("support trajectories must be [B,T,9]")
        features = trajectories.transpose(1, 2)  # [B,9,T]
        features = self.conv1(features)
        features = self.conv2(features)
        features = self.conv3(features)
        pooled = features.mean(dim=-1)  # [B,hidden]
        return self.output_proj(pooled)


class CrossSceneSupportAggregator(nn.Module):
    """Cross-attention that pools K support motion features under one query.

    query  = [scene_pooled, realization_vec]         -> [B, query_dim]
    keys   = [support_scene_pooled, realization_vec] -> [B, K, key_dim]
    values = support motion features                  -> [B, K, value_dim]

    Unlike the cross-scene reference implementation, ``support_mask`` is
    honoured here: it is passed as ``key_padding_mask`` so zero-padded support
    slots cannot be attended to.
    """

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        value_dim: int,
        hidden: int = 128,
        output_dim: int = 128,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        if hidden % num_heads != 0:
            raise ValueError(
                f"aggregator hidden ({hidden}) must be divisible by heads ({num_heads})"
            )
        self.output_dim = int(output_dim)
        self.query_proj = nn.Sequential(
            nn.Linear(query_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden),
        )
        self.key_proj = nn.Sequential(
            nn.Linear(key_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden),
        )
        self.value_proj = nn.Sequential(
            nn.Linear(value_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden),
        )
        self.attention = nn.MultiheadAttention(hidden, num_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden, self.output_dim), nn.SiLU(),
            nn.Linear(self.output_dim, self.output_dim), nn.LayerNorm(self.output_dim),
        )

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        support_mask: torch.Tensor,
    ) -> torch.Tensor:
        q = self.query_proj(query).unsqueeze(1)      # [B,1,hidden]
        k = self.key_proj(keys)                       # [B,K,hidden]
        v = self.value_proj(values)                   # [B,K,hidden]
        key_padding_mask = ~(support_mask > 0.5)      # [B,K] True=ignore
        row_valid = (support_mask > 0.5).any(dim=-1)  # [B]
        # Rows with no valid support would mask every key and NaN the softmax;
        # unmask one (zero-padded) slot, then zero the row's output afterwards
        # so it degrades exactly to the no-support baseline.
        fully_masked = key_padding_mask.all(dim=-1)
        if fully_masked.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[fully_masked, 0] = False
        attended, _ = self.attention(
            q, k, v, key_padding_mask=key_padding_mask, need_weights=False
        )
        out = self.attn_norm(q.squeeze(1) + attended.squeeze(1))
        out = out * row_valid.unsqueeze(-1).float()   # [B,hidden] (zero rows without support)
        return self.output_proj(out) * row_valid.unsqueeze(-1).to(out.dtype)


class RealizationVectorEncoder(nn.Module):
    """Pool the shared realization encoders into one [B,dim] vector.

    Reuses the *same* encoder instances that build the plan tokens (passed in,
    not re-instantiated) so the support query/key live in the identical semantic
    space as the realization plan rather than a separate embedding table.
    """

    def __init__(
        self,
        source: nn.Module,
        target: nn.Module,
        channel: nn.Module,
        dim: int,
    ) -> None:
        super().__init__()
        self.source = source
        self.target = target
        self.channel = channel
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        source_parameter: torch.Tensor,
        target_parameter: torch.Tensor,
        channel_id: torch.Tensor,
    ) -> torch.Tensor:
        source = self.source(source_parameter).squeeze(1)   # [B,dim]
        target = self.target(target_parameter).squeeze(1)   # [B,dim]
        channel = self.channel(channel_id).mean(dim=1)      # [B,dim]
        return self.norm(source + target + channel)


class GlobalTrajectoryPredictor(nn.Module):
    """Global-only network used by core legacy and core protocols.

    This is exactly the global branch of :class:`GlobalLocalRealizationDiffusion`
    but intentionally does not construct the official DP3 local branch.  It
    ``global_scene_source`` explicitly preserves the archived core behavior or
    enables core's separated scene/state semantics.
    """

    def __init__(self, cfg: GlobalLocalDiffusionConfig = GlobalLocalDiffusionConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        dim = cfg.model_dim
        if not 0.0 <= cfg.global_realization_mask_prob <= 1.0:
            raise ValueError("global_realization_mask_prob must lie in [0,1]")
        if cfg.global_realization_mask_prob and (
            not cfg.grounded_source_bottleneck or cfg.support_enabled or cfg.context_enabled
        ):
            raise ValueError("Global realization masking requires the grounded source bottleneck without support/context")
        if cfg.global_scene_source not in {"legacy", "scene_state"}:
            raise ValueError(
                "global_scene_source must be 'legacy' or 'scene_state'"
            )
        if (cfg.global_scene_source == "scene_state"
                and cfg.global_scene_representation == "pointcloud"
                and cfg.scene_coordinate_representation != "absolute"):
            raise ValueError(
                "Strict pointcloud Global requires absolute task-frame coordinates"
            )
        if cfg.global_scene_representation == "parameterized":
            self.scene_encoder = ParameterizedSceneEncoder(cfg)
        elif cfg.global_scene_representation == "pointcloud":
            self.scene_encoder = SemanticSceneEncoder(cfg)
        else:
            raise ValueError(
                f"unknown global_scene_representation: {cfg.global_scene_representation!r}"
            )
        (
            self.source_query_encoder,
            self.target_query_encoder,
            self.channel_query_encoder,
        ) = _build_realization_query_encoders(cfg)
        if cfg.object_point_interaction:
            if cfg.realization_input == "instantiated":
                raise ValueError(
                    "object_point_interaction requires normalized u_o/u_g queries, "
                    "not instantiated p_o/p_g"
                )
            self.source_interaction_encoder = ObjectInteractionGeometryEncoder(cfg, 0)
            self.target_interaction_encoder = ObjectInteractionGeometryEncoder(cfg, 1)
        if cfg.global_scene_source == "scene_state":
            # x_t contains current object pose (7D) and phase one-hot (3D).
            # phase-local progress is encoded separately below, so it is not
            # duplicated inside this token.
            self.current_state_encoder = nn.Sequential(
                nn.Linear(10, 2 * dim), nn.SiLU(),
                nn.Linear(2 * dim, dim), nn.LayerNorm(dim),
            )
            self.current_state_type_embedding = nn.Parameter(
                torch.randn(1, 1, dim) * 0.02
            )
        self.progress_encoder = nn.Sequential(
            nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.fusion = nn.ModuleList(
            [CrossConditionFusion(cfg) for _ in range(cfg.fusion_layers)]
        )
        self.global_decoder = GlobalTrajectoryDecoder(cfg)

        self.realization_vec_encoder = RealizationVectorEncoder(
            self.source_query_encoder,
            self.target_query_encoder,
            self.channel_query_encoder,
            dim,
        )
        # Instantiate after every archived shared module so enabling the new
        # path cannot perturb their seeded initialization stream.
        if cfg.grounded_source_bottleneck:
            if not cfg.object_point_interaction or cfg.realization_input != "abstract":
                raise ValueError(
                    "grounded_source_bottleneck requires abstract core object-point conditioning"
                )
            if cfg.support_enabled or cfg.context_enabled:
                raise ValueError(
                    "the source-bottleneck intervention excludes support/context conditioning"
                )
            self.source_context_encoder = PointSetTokenEncoder(cfg, 3)
            self.source_ground_encoder = PointSetTokenEncoder(cfg, 6)
        if cfg.support_enabled:
            self.support_motion_encoder = SupportTrajectoryEncoder(
                pose_dim=9,
                hidden=cfg.support_motion_hidden,
                output_dim=cfg.support_motion_out,
            )
            self.support_aggregator = CrossSceneSupportAggregator(
                query_dim=2 * dim,
                key_dim=2 * dim,
                value_dim=cfg.support_motion_out,
                hidden=cfg.aggregator_hidden,
                output_dim=cfg.aggregator_out,
                num_heads=cfg.aggregator_heads,
            )
            # Bias-free so a zeroed (dropped) motion prior contributes nothing.
            self.motion_prior_proj = nn.Linear(cfg.aggregator_out, dim, bias=False)
        if cfg.context_enabled:
            self.context_motion_encoder = SupportTrajectoryEncoder(
                pose_dim=9,
                hidden=cfg.support_motion_hidden,
                output_dim=cfg.support_motion_out,
            )
            self.context_aggregator = CrossSceneSupportAggregator(
                query_dim=2 * dim,
                key_dim=2 * dim,
                value_dim=cfg.support_motion_out,
                hidden=cfg.aggregator_hidden,
                output_dim=cfg.aggregator_out,
                num_heads=cfg.aggregator_heads,
            )
            # Bias-free so a zeroed (dropped) motion prior contributes nothing.
            self.context_prior_proj = nn.Linear(cfg.aggregator_out, dim, bias=False)

    def _encode_plan_scene_realization(
        self,
        fused_point_cloud: torch.Tensor | None,
        fused_state: torch.Tensor | None,
        scene_parameters: torch.Tensor | None,
        source_parameter: torch.Tensor,
        target_parameter: torch.Tensor,
        channel_id: torch.Tensor,
        progress: torch.Tensor,
        source_context_local: torch.Tensor | None = None,
        source_patch_grounded: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.cfg.global_scene_representation == "parameterized":
            if scene_parameters is None:
                raise ValueError("parameterized global scene path requires scene_parameters")
            scene_tokens, scene_pooled = self.scene_encoder(scene_parameters, fused_state)
        else:
            if fused_point_cloud is None or fused_state is None:
                raise ValueError("pointcloud global scene path requires point cloud and state")
            scene_tokens, scene_pooled = self.scene_encoder(fused_point_cloud, fused_state)
        if self.cfg.global_scene_source == "scene_state":
            if fused_state is None or fused_state.ndim != 2 or fused_state.shape[-1] != 11:
                raise ValueError("core Global requires current x_t as [B,11]")
            state_token = (
                self.current_state_encoder(fused_state[:, :10]).unsqueeze(1)
                + self.current_state_type_embedding
            )
            scene_tokens = torch.cat((scene_tokens, state_token), dim=1)
            scene_pooled = scene_tokens.mean(dim=1)
        target_query = self.target_query_encoder(target_parameter)
        channel_tokens = self.channel_query_encoder(channel_id)
        if self.cfg.grounded_source_bottleneck:
            if source_context_local is None or source_patch_grounded is None:
                raise ValueError(
                    "grounded source bottleneck requires source_context_local and source_patch_grounded"
                )
            source_context = self.source_context_encoder(source_context_local)
            source_ground = self.source_ground_encoder(source_patch_grounded)
            if fused_point_cloud is None:
                raise ValueError("target interaction still requires the semantic point cloud")
            target_interaction = self.target_interaction_encoder(target_query, fused_point_cloud)
            realization = torch.cat(
                (source_context, source_ground, target_query, target_interaction, channel_tokens),
                dim=1,
            )
        elif self.cfg.object_point_interaction:
            if fused_point_cloud is None:
                raise ValueError("object_point_interaction requires the semantic point cloud")
            source_query = self.source_query_encoder(source_parameter)
            source_interaction = self.source_interaction_encoder(source_query, fused_point_cloud)
            target_interaction = self.target_interaction_encoder(target_query, fused_point_cloud)
            realization = torch.cat(
                (source_query, source_interaction, target_query, target_interaction, channel_tokens),
                dim=1,
            )
        else:
            source_query = self.source_query_encoder(source_parameter)
            realization = torch.cat((source_query, target_query, channel_tokens), dim=1)
        # Group every encoded path for a factor together: source context/patch,
        # target query/interaction, and all channel tokens. Keep scene/state.
        if self.training and self.cfg.global_realization_mask_prob:
            keep = torch.rand(realization.shape[0], 3, device=realization.device) >= self.cfg.global_realization_mask_prob
            token_keep = torch.cat((keep[:, :1].expand(-1, 2),
                                    keep[:, 1:2].expand(-1, 2),
                                    keep[:, 2:3].expand(-1, channel_tokens.shape[1])), dim=1)
            realization = realization * token_keep.unsqueeze(-1).to(realization.dtype)
        realization = realization + self.progress_encoder(progress).unsqueeze(1)
        for layer in self.fusion:
            realization = layer(realization, scene_tokens)
        realization_vec = (
            torch.zeros_like(scene_pooled)
            if self.cfg.grounded_source_bottleneck
            else self.realization_vec_encoder(source_parameter, target_parameter, channel_id)
        )
        return realization, scene_pooled, realization_vec

    def encode_realization_plan(
        self,
        fused_point_cloud: torch.Tensor,
        fused_state: torch.Tensor,
        source_parameter: torch.Tensor,
        target_parameter: torch.Tensor,
        channel_id: torch.Tensor,
        progress: torch.Tensor,
        scene_parameters: torch.Tensor | None = None,
    ) -> torch.Tensor:
        plan, _, _ = self._encode_plan_scene_realization(
            fused_point_cloud, fused_state, scene_parameters, source_parameter, target_parameter,
            channel_id, progress,
        )
        return plan

    def _condition_with_support(
        self,
        plan: torch.Tensor,
        scene_pooled: torch.Tensor,
        realization_vec: torch.Tensor,
        *,
        support_trajectory: torch.Tensor,
        support_point_cloud: torch.Tensor | None,
        support_state: torch.Tensor | None,
        support_scene_parameters: torch.Tensor | None,
        support_mask: torch.Tensor,
        support_dropout_prob: float,
    ) -> torch.Tensor:
        """Blend a support-aggregated motion prior into the plan tokens.

        Rows whose ``support_mask`` has no positive entry (no other-scene cell
        for that realization) are zeroed by the aggregator, so the per-row
        motion prior is exactly zero and the plan is unchanged.  Support dropout
        zeroes the aggregated prior post-hoc (not the mask), which — combined
        with the bias-free ``motion_prior_proj`` — makes full dropout degrade
        exactly to the no-support baseline.
        """
        batch, n_support = support_trajectory.shape[:2]
        if not bool((support_mask > 0.5).any()):
            return plan
        values = self.support_motion_encoder(
            support_trajectory.reshape(batch * n_support, *support_trajectory.shape[2:])
        ).reshape(batch, n_support, -1)
        if self.cfg.global_scene_representation == "parameterized":
            if support_scene_parameters is None:
                raise ValueError("parameterized support requires support_scene_parameters")
            support_scene_pooled = self.scene_encoder(
                support_scene_parameters.reshape(batch * n_support, -1), None
            )[1].reshape(batch, n_support, -1)
        else:
            if support_point_cloud is None or support_state is None:
                raise ValueError("pointcloud support requires support point cloud and state")
            support_scene_pooled = self.scene_encoder(
                support_point_cloud.reshape(batch * n_support, *support_point_cloud.shape[2:]),
                support_state.reshape(batch * n_support, support_state.shape[-1]),
            )[1].reshape(batch, n_support, -1)
        keys = torch.cat(
            (
                support_scene_pooled,
                realization_vec.unsqueeze(1).expand(-1, n_support, -1),
            ),
            dim=-1,
        )
        query = torch.cat((scene_pooled, realization_vec), dim=-1)
        motion_prior = self.support_aggregator(query, keys, values, support_mask)
        # CFG-style condition dropout: applied whenever requested (not gated on
        # training mode) so an inference-time caller can degrade to the
        # no-support baseline by passing a support set that is fully dropped.
        if support_dropout_prob > 0.0:
            dropped = torch.rand(batch, device=plan.device) < support_dropout_prob
            if dropped.any():
                motion_prior = motion_prior * (~dropped).float().unsqueeze(-1)
        return plan + self.motion_prior_proj(motion_prior).unsqueeze(1)

    def _condition_with_context(
        self,
        plan: torch.Tensor,
        scene_pooled: torch.Tensor,
        realization_vec: torch.Tensor,
        *,
        context_trajectory: torch.Tensor,
        context_point_cloud: torch.Tensor,
        context_state: torch.Tensor,
        context_source_parameter: torch.Tensor,
        context_target_parameter: torch.Tensor,
        context_channel_id: torch.Tensor,
        context_mask: torch.Tensor,
        context_dropout_prob: float,
    ) -> torch.Tensor:
        """Blend a partial-overlap context motion prior into the plan tokens.

        Unlike ``_condition_with_support`` (which reuses the query realization
        vector for every key because all slots share the realization), each
        context slot carries its *own* realization (u_o/u_g/c) and scene, so the
        cross-attention keys encode the actual combinatorial relation to the
        query.  Rows whose ``context_mask`` has no positive entry are zeroed by
        the aggregator and degrade exactly to the no-context baseline.
        """
        batch, n_context = context_trajectory.shape[:2]
        if not bool((context_mask > 0.5).any()):
            return plan
        values = self.context_motion_encoder(
            context_trajectory.reshape(batch * n_context, *context_trajectory.shape[2:])
        ).reshape(batch, n_context, -1)
        context_scene_pooled = self.scene_encoder(
            context_point_cloud.reshape(batch * n_context, *context_point_cloud.shape[2:]),
            context_state.reshape(batch * n_context, context_state.shape[-1]),
        )[1].reshape(batch, n_context, -1)
        context_realization_vec = self.realization_vec_encoder(
            context_source_parameter.reshape(batch * n_context, -1),
            context_target_parameter.reshape(batch * n_context, -1),
            context_channel_id.reshape(batch * n_context),
        ).reshape(batch, n_context, -1)
        keys = torch.cat((context_scene_pooled, context_realization_vec), dim=-1)
        query = torch.cat((scene_pooled, realization_vec), dim=-1)
        motion_prior = self.context_aggregator(query, keys, values, context_mask)
        if context_dropout_prob > 0.0:
            dropped = torch.rand(batch, device=plan.device) < context_dropout_prob
            if dropped.any():
                motion_prior = motion_prior * (~dropped).float().unsqueeze(-1)
        return plan + self.context_prior_proj(motion_prior).unsqueeze(1)

    def forward(
        self,
        *,
        fused_point_cloud: torch.Tensor | None = None,
        fused_state: torch.Tensor | None = None,
        scene_parameters: torch.Tensor | None = None,
        source_parameter: torch.Tensor,
        target_parameter: torch.Tensor,
        channel_id: torch.Tensor,
        progress: torch.Tensor,
        support_trajectory: torch.Tensor | None = None,
        support_point_cloud: torch.Tensor | None = None,
        support_state: torch.Tensor | None = None,
        support_scene_parameters: torch.Tensor | None = None,
        support_mask: torch.Tensor | None = None,
        support_dropout_prob: float = 0.0,
        context_trajectory: torch.Tensor | None = None,
        context_point_cloud: torch.Tensor | None = None,
        context_state: torch.Tensor | None = None,
        context_source_parameter: torch.Tensor | None = None,
        context_target_parameter: torch.Tensor | None = None,
        context_channel_id: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        context_dropout_prob: float = 0.0,
        anchor_tilt_axis: torch.Tensor | None = None,
        anchor_yaw: torch.Tensor | None = None,
        anchor_initial_rotation: torch.Tensor | None = None,
        source_context_local: torch.Tensor | None = None,
        source_patch_grounded: torch.Tensor | None = None,
    ) -> torch.Tensor:
        plan, scene_pooled, realization_vec = self._encode_plan_scene_realization(
            fused_point_cloud, fused_state, scene_parameters, source_parameter, target_parameter,
            channel_id, progress, source_context_local, source_patch_grounded,
        )
        if support_trajectory is not None:
            if not self.cfg.support_enabled:
                raise RuntimeError(
                    "support tensors were passed but support_enabled=False; "
                    "construct the model with a config that enables support"
                )
            if support_mask is None:
                raise ValueError(
                    "support conditioning requires support_mask"
                )
            plan = self._condition_with_support(
                plan, scene_pooled, realization_vec,
                support_trajectory=support_trajectory,
                support_point_cloud=support_point_cloud,
                support_state=support_state,
                support_scene_parameters=support_scene_parameters,
                support_mask=support_mask,
                support_dropout_prob=support_dropout_prob,
            )
        if context_trajectory is not None:
            if not self.cfg.context_enabled:
                raise RuntimeError(
                    "context tensors were passed but context_enabled=False; "
                    "construct the model with a config that enables context"
                )
            if (
                context_point_cloud is None or context_state is None
                or context_source_parameter is None or context_target_parameter is None
                or context_channel_id is None or context_mask is None
            ):
                raise ValueError(
                    "context conditioning requires context_point_cloud, context_state, "
                    "context_source_parameter, context_target_parameter, "
                    "context_channel_id, and context_mask"
                )
            plan = self._condition_with_context(
                plan, scene_pooled, realization_vec,
                context_trajectory=context_trajectory,
                context_point_cloud=context_point_cloud,
                context_state=context_state,
                context_source_parameter=context_source_parameter,
                context_target_parameter=context_target_parameter,
                context_channel_id=context_channel_id,
                context_mask=context_mask,
                context_dropout_prob=context_dropout_prob,
            )
        trajectory = self.global_decoder(plan)
        if self.cfg.global_scene_source == "scene_state":
            if fused_state is None:
                raise ValueError("core Global requires fused_state for waypoint-0 anchoring")
            # x_q is already expressed in the task frame.  Enforce the known
            # boundary condition structurally instead of asking the network to
            # learn it: waypoint 0 is the query pose, and only waypoints 1:T
            # are predictions.  Rot6D follows this module's row-major
            # convention (the first two rows of the rotation matrix).
            query_rotation = SemanticSceneEncoder._quaternion_wxyz_to_matrix(
                fused_state[..., 3:7]
            )
            query_pose = torch.cat(
                (fused_state[..., :3], query_rotation[..., :2, :].reshape(-1, 6)),
                dim=-1,
            )
            trajectory = torch.cat((query_pose[:, None, :], trajectory[:, 1:, :]), dim=1)
        if self.cfg.anchor_enabled:
            if (
                anchor_tilt_axis is None
                or anchor_yaw is None
                or anchor_initial_rotation is None
            ):
                raise ValueError(
                    "anchor_enabled requires anchor_tilt_axis, anchor_yaw, and "
                    "anchor_initial_rotation"
                )
            trajectory = interaction_orientation_anchor_project(
                trajectory,
                anchor_tilt_axis=anchor_tilt_axis,
                anchor_yaw=anchor_yaw,
                anchor_initial_rotation=anchor_initial_rotation,
            )
        return trajectory

    def training_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        *,
        rim_local: torch.Tensor | None = None,
        k_star: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if predicted.shape != target.shape or predicted.shape[-1] != 9:
            raise ValueError("global trajectories must have matching [B,L,9] shapes")
        if self.cfg.global_scene_source == "scene_state":
            if predicted.shape[1] < 2:
                raise ValueError("core Global requires at least one future waypoint")
            # Waypoint 0 is the exact x_q boundary condition inserted in
            # forward(); it is not a learned target.
            predicted = predicted[:, 1:, :]
            target = target[:, 1:, :]
        position = F.mse_loss(predicted[..., :3], target[..., :3])
        rotation = rotation_geodesic_loss(
            predicted[..., 3:9], target[..., 3:9]
        ).mean()
        total = self.cfg.lambda_position * position + self.cfg.lambda_rotation * rotation
        u_o_loss = predicted.new_zeros(())
        if self.cfg.lambda_u_o > 0.0:
            if rim_local is None or k_star is None:
                raise ValueError(
                    "lambda_u_o > 0 requires rim_local and k_star for the interaction loss"
                )
            if k_star.shape[0] != predicted.shape[0]:
                raise ValueError("k_star batch must match trajectory batch")
            # k_star is an original 32-waypoint index; after dropping waypoint 0
            # above, gather at k_star - 1, clamped into the future-waypoint range.
            idx = (k_star - 1).clamp(0, predicted.shape[1] - 1)
            pred_outlet = outlet_position(predicted, rim_local)
            tgt_outlet = outlet_position(target, rim_local)
            pred_k = pred_outlet[torch.arange(predicted.shape[0], device=predicted.device), idx]
            tgt_k = tgt_outlet[torch.arange(target.shape[0], device=target.device), idx]
            u_o_loss = F.mse_loss(pred_k, tgt_k)
            total = total + self.cfg.lambda_u_o * u_o_loss
        return {
            "loss": total,
            "position_loss": position,
            "rotation_loss": rotation,
            "u_o_loss": u_o_loss,
        }


def load_global_predictor_checkpoint(
    path: str | PathLike[str],
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> tuple[GlobalTrajectoryPredictor, dict]:
    """Load a trusted AEXT/FloRA global-predictor checkpoint.

    The migrated module preserves the AEXT parameter names, so historical
    core and core global-predictor state dictionaries load directly.
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("global-predictor checkpoint must be a dictionary")
    config = checkpoint.get("config")
    state = checkpoint.get("model_state_dict")
    if not isinstance(config, dict) or not isinstance(state, dict):
        raise ValueError("checkpoint lacks config/model_state_dict")
    model = GlobalTrajectoryPredictor(GlobalLocalDiffusionConfig(**config))
    model.load_state_dict(state, strict=strict)
    return model, checkpoint
