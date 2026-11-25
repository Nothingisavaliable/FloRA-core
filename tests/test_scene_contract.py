"""Regression checks for missing scene contract and masked support rows."""

import pytest

torch = pytest.importorskip("torch")

from flora_core.data.scene_features import (
    SCENE_FACTOR_WIDTHS,
    SCENE_PARAMETER_DIM,
    SCENE_PARAMETER_SLICES,
)
from flora_core.models.global_predictor import CrossSceneSupportAggregator, GlobalTrajectoryPredictor, GlobalLocalDiffusionConfig


def test_scene_slices_cover_25_dimensions_without_overlap():
    assert tuple(SCENE_FACTOR_WIDTHS) == ("d", "psi", "O", "G_o", "G_g")
    assert SCENE_PARAMETER_DIM == 25
    cursor = 0
    for name, (start, end) in SCENE_PARAMETER_SLICES.items():
        assert start == cursor
        assert end - start == SCENE_FACTOR_WIDTHS[name]
        cursor = end
    assert cursor == SCENE_PARAMETER_DIM


def test_parameterized_predictor_imports_and_runs():
    cfg = GlobalLocalDiffusionConfig(
        global_scene_representation="parameterized",
        model_dim=32,
        num_heads=4,
        scene_slots_per_entity=1,
        channel_tokens=4,
        global_horizon=8,
        fusion_layers=1,
        global_decoder_layers=1,
    )
    model = GlobalTrajectoryPredictor(cfg).eval()
    with torch.no_grad():
        result = model(
            scene_parameters=torch.zeros(1, SCENE_PARAMETER_DIM),
            source_parameter=torch.tensor([[0.25]]),
            target_parameter=torch.tensor([[0.0]]),
            channel_id=torch.tensor([0]),
            progress=torch.tensor([[0.0]]),
        )
    assert result.shape == (1, 8, 9)
    assert torch.isfinite(result).all()


def test_mixed_batch_all_masked_support_is_finite_and_zero():
    torch.manual_seed(22)
    aggregator = CrossSceneSupportAggregator(
        query_dim=16, key_dim=16, value_dim=8,
        hidden=16, output_dim=8, num_heads=4,
    ).eval()
    query = torch.randn(2, 16)
    keys = torch.randn(2, 3, 16)
    values = torch.randn(2, 3, 8)
    support_mask = torch.tensor([[0., 0., 0.], [1., 0., 0.]])
    with torch.no_grad():
        result = aggregator(query, keys, values, support_mask)
    assert result.shape == (2, 8)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result[0], torch.zeros_like(result[0]), atol=0, rtol=0)
