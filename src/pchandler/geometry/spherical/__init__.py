# pchandler - Toolbox for point-cloud handling, processing and analysis
#
# Copyright (c) 2025, Nicholas Meyer, Geosensors and Engineering Geodesy,
# Institute of Geodesy and Photogrammetry, ETH Zurich, Switzerland
# SPDX-License-Identifier: BSD-3-Clause
#
# Author: Nicholas Meyer (meyernic@ethz.ch)

"""
Contains spherical geometry classes and functions.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__all__ = []

_lazy_map = {"Angle": "angle", "AngleArray": "angle", "FoV": "fov", "FoVTree": "fov"}

__all__ = __all__ + list(_lazy_map)

if TYPE_CHECKING:
    from .angle import Angle as Angle
    from .angle import AngleArray as AngleArray
    from .fov import FoV as FoV
    from .fov import FoVTree as FoVTree


def __getattr__(name: str):
    if name in _lazy_map:
        module = importlib.import_module(f"{__name__}.{_lazy_map[name]}")
        val = getattr(module, name)
    else:
        try:
            val = importlib.import_module(f"{__name__}.{name}")
        except ModuleNotFoundError:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    globals()[name] = val
    return val


def __dir__():
    # so tab-completion / introspection shows the lazy names
    return sorted(set(__all__) | set(globals().keys()))
