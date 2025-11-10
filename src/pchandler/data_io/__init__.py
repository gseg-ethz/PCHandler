# pchandler - Toolbox for point-cloud handling, processing and analysis
#
# Copyright (c) 2025, Nicholas Meyer, Geosensors and Engineering Geodesy,
# Institute of Geodesy and Photogrammetry, ETH Zurich, Switzerland
# SPDX-License-Identifier: BSD-3-Clause
#
# Author: Nicholas Meyer (meyernic@ethz.ch)

"""
Public API for pchandler.data_io.
Re-exports key classes for easy IO access
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__all__ = [
    "core", "util"
]

_lazy_map: dict[str, str | tuple[str, str]] = {
    "Csv": ("csv", "CsvHandler"),
    "E57": ("e57", "E57Handler"),
    "Las": ("las", "LasHandler"),
    "Ply": ("ply", "PlyHandler"),
    # "Pcd": ("pcd", "PcdHandler"),
    "find_point_cloud_in_directory": "core",
    "load_file": "util"
}

__all__ = __all__ + list(_lazy_map)

if TYPE_CHECKING:
    from . import core, util

    # from .pcd import PcdHandler as Pcd
    from .core import find_point_cloud_in_directory
    from .csv import CsvHandler as Csv
    from .e57 import E57Handler as E57
    from .las import LasHandler as Las
    from .ply import PlyHandler as Ply


def __getattr__(name: str):
    if name in _lazy_map:
        if isinstance(_lazy_map[name], str):
            module = importlib.import_module(f"{__name__}.{_lazy_map[name]}")
            val = getattr(module, name)
        else:  # Case Tuple
            module_name, real_name = _lazy_map[name]
            module = importlib.import_module(f"{__name__}.{module_name}")
            val = getattr(module, real_name)
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
