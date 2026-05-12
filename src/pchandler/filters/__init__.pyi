# pchandler - Toolbox for point-cloud handling, processing and analysis
#
# Copyright (c) 2025, Nicholas Meyer, Geosensors and Engineering Geodesy,
# Institute of Geodesy and Photogrammetry, ETH Zurich, Switzerland
# SPDX-License-Identifier: BSD-3-Clause
#
# Author: Nicholas Meyer (meyernic@ethz.ch)

# Auto-generated stub for lazy exports
from typing import Final, NoReturn

from .cartesian_filters import BoxFilter, PolygonFilter, SphereFilter
from .core import GenericFieldFilter, PointCloudFilter, ValidatedPolygonT
from .downsample import AngleBinDownsample, RandomDownsampleFilter, VoxelDownsample
from .gpu import PolygonFilterGPU, SphericalPolygonFilterGPU
from .outlier_filter import CartesianOutlierFilter, SphericalOutlierFilter
from .scalar_field_filters import ScalarFieldFilter, ScalarFieldPercentileFilter
from .spherical_coordinate_filters import FoVFilter, RangeFilter, SphericalPolygonFilter

__all__: Final[list[str]] = [
    "BoxFilter",
    "PolygonFilter",
    "SphereFilter",
    "GenericFieldFilter",
    "PointCloudFilter",
    "ValidatedPolygonT",
    "AngleBinDownsample",
    "RandomDownsampleFilter",
    "VoxelDownsample",
    "CartesianOutlierFilter",
    "SphericalOutlierFilter",
    "ScalarFieldFilter",
    "ScalarFieldPercentileFilter",
    "FoVFilter",
    "RangeFilter",
    "SphericalPolygonFilter",
    "PolygonFilterGPU",
    "SphericalPolygonFilterGPU",
]

def __getattr__(name: str) -> NoReturn: ...
