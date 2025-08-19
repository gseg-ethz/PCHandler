# pchandler - Toolbox for point-cloud handling, processing and analysis
#
# Copyright (c) 2025, Nicholas Meyer, Geosensors and Engineering Geodesy,
# Institute of Geodesy and Photogrammetry, ETH Zurich, Switzerland
# SPDX-License-Identifier: BSD-3-Clause
#
# Author: Nicholas Meyer (meyernic@ethz.ch)

from types import ModuleType

# Auto-generated stub for lazy exports
from typing import Final, Literal, NoReturn, overload

from . import base_types as base_types
from . import constants as constants
from . import data_io as data_io
from . import filters as filters
from . import geometry as geometry
from . import scalar_fields as scalar_fields
from ._version import __version__, __version_tuple__, version, version_tuple
from .core import PointCloudData

__author__: str
__email__: str
__all__: Final[list[str]] = [
    "data_io",
    "geometry",
    "filters",
    "scalar_fields",
    "base_types",
    "constants",
    "__author__",
    "__email__",
    "PointCloudData",
    "__version__",
    "version",
    "__version_tuple__",
    "version_tuple",
]

@overload
def __getattr__(name: Literal["PointCloudData"]) -> PointCloudData: ...
@overload
def __getattr__(name: Literal["__version__", "version"]) -> str: ...
@overload
def __getattr__(name: Literal["__version_tuple__", "version_tuple"]) -> tuple[int | str, ...]: ...
@overload
def __getattr__(
    name: Literal["data_io", "geometry", "filters", "scalar_fields", "base_types", "constants"],
) -> ModuleType: ...
def __getattr__(name: str) -> NoReturn: ...
