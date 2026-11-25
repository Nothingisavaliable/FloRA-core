"""Helpers for locating assets installed with the LIBERO Python package."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def find_libero_root() -> Path:
    """Return the LIBERO repository/package root from its installed package.

    LIBERO is commonly installed in editable mode. Its importable package lives
    at ``<root>/libero/libero``; locating that module avoids a machine-specific
    root environment variable while still allowing callers to pass an explicit
    root for nonstandard asset layouts.
    """
    try:
        spec = importlib.util.find_spec("libero.libero")
    except (ModuleNotFoundError, ValueError):
        spec = None
    if spec is None or spec.origin is None:
        raise FileNotFoundError(
            "LIBERO is not importable; install the LIBERO package or pass its "
            "asset root explicitly"
        )
    package_file = Path(spec.origin).expanduser().resolve()
    root = package_file.parents[2]
    if not (root / "libero").is_dir():
        raise FileNotFoundError(
            f"Could not infer the LIBERO asset root from {package_file}; "
            "pass libero_root explicitly"
        )
    return root
