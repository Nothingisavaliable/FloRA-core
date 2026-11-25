"""Named layout for the 25-D parameterized scene condition.

The frozen collection design has five scene factors, in this order:
distance (d), relative yaw (psi), obstacle (O), source geometry (G_o),
and target geometry (G_g).  This repository did not originally include
the feature-extraction module.  The widths below are a *provisional*
25-dimensional layout to restore a runnable model, NOT an assertion that
these are the exact widths used to train archival checkpoints.

Before publishing trained weights or claiming RQ1/RQ2 reproducibility,
compare the original scene-parameter builder, normalization, and the
per-factor encoder layer shapes against these slices.
"""

SCENE_FACTOR_WIDTHS = {
    "d": 1,
    "psi": 2,
    "O": 6,
    "G_o": 8,
    "G_g": 8,
}

SCENE_PARAMETER_SLICES: dict[str, tuple[int, int]] = {}
_offset = 0
for _name, _width in SCENE_FACTOR_WIDTHS.items():
    SCENE_PARAMETER_SLICES[_name] = (_offset, _offset + _width)
    _offset += _width
SCENE_PARAMETER_DIM = _offset
assert SCENE_PARAMETER_DIM == 25
