"""Robot embodiment evaluation interfaces and reusable motion evaluators."""

from .global_motion import (
    GlobalMotionBackend,
    GlobalMotionConfig,
    evaluate_global_motion,
    evaluate_stage1,
    evaluate_stage2,
)
from .robosuite_backend import (
    RobosuiteOSCBackend,
    create_libero_osc_backend,
    register_libero_robots,
)

__all__ = [
    "GlobalMotionBackend", "GlobalMotionConfig", "evaluate_global_motion",
    "evaluate_stage1", "evaluate_stage2",
    "RobosuiteOSCBackend", "create_libero_osc_backend", "register_libero_robots",
]
