# pchandler - Toolbox for point-cloud handling, processing and analysis
#
# Copyright (c) 2025, Nicholas Meyer, Geosensors and Engineering Geodesy,
# Institute of Geodesy and Photogrammetry, ETH Zurich, Switzerland
# SPDX-License-Identifier: BSD-3-Clause
#
# Author: Nicholas Meyer (meyernic@ethz.ch)

"""
Public API for pchandler.geometry.
Re-exports key classes and functions from the submodules.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__all__ = [
    "splitter",
    "util",
    "coordinates",
    "transforms",
    "spherical",
]

_lazy_map = {
    # "get_outline_polygon": "util",
    "OptimizedShiftManager": "optimal_shift",
    "OptimizedShift": "optimal_shift",
}

__all__ = __all__ + list(_lazy_map)

if TYPE_CHECKING:
    from . import coordinates, spherical, splitter, transforms, util
    from .optimal_shift import OptimizedShift, OptimizedShiftManager

    # from .util import get_outline_polygon


def __getattr__(name: str):
    if name in _lazy_map:
        module = importlib.import_module(f"{__name__}.{_lazy_map[name]}")
        val = getattr(module, name)
    else:
        try:
            val = importlib.import_module(f"{__name__}.{name}")
        except ModuleNotFoundError:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = val
    return val


def __dir__():
    # so tab-completion / introspection shows the lazy names
    return sorted(set(__all__) | set(globals().keys()))
