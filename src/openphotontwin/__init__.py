"""Compatibility import for the former ``openphotontwin`` package name.

TBL 2.x is canonically imported as ``tbl``. This shim preserves existing
applications and serialized references while keeping one implementation.
"""

from __future__ import annotations

import importlib
import sys
import warnings

import tbl as _tbl
from tbl import *  # noqa: F403

_SUBMODULES = (
    "adapters",
    "backends",
    "calibration",
    "circuit",
    "components",
    "detectors",
    "errors",
    "experiments",
    "models",
    "simulator",
    "timebin",
)
for _submodule in _SUBMODULES:
    _module = importlib.import_module(f"tbl.{_submodule}")
    sys.modules[f"{__name__}.{_submodule}"] = _module
    globals()[_submodule] = _module

__all__ = _tbl.__all__
__version__ = _tbl.__version__

warnings.warn(
    "'openphotontwin' is now named 'tbl'; update imports to 'import tbl'",
    DeprecationWarning,
    stacklevel=2,
)
