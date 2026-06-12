# pchandler – Toolbox for point-cloud handling, processing and analysis
#
# Copyright (c) 2022–2026 ETH Zurich
# Department of Civil, Environmental and Geomatic Engineering (D-BAUG)
# Institute of Geodesy and Photogrammetry
# Geosensors and Engineering Geodesy
#
# Authors:
#   Nicholas Meyer
#   Jon Allemand
#
# SPDX-License-Identifier: BSD-3-Clause

# pchandler - Toolbox for point-cloud handling, processing and analysis
#
# Copyright (c) 2025, Nicholas Meyer, Geosensors and Engineering Geodesy,
# Institute of Geodesy and Photogrammetry, ETH Zurich, Switzerland
# SPDX-License-Identifier: BSD-3-Clause
#
# Author: Nicholas Meyer (meyernic@ethz.ch)

"""Public API for :mod:`pchandler.filters`.

Re-exports key filter classes (cartesian, spherical, downsample, outlier,
scalar-field, GPU) via the lazy ``__getattr__`` mechanism so heavy or
optional dependencies (e.g. ``cudf`` / ``cuspatial``) are not imported at
package-import time.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

# Map public names to the submodule that actually defines them.
_lazy_map = {
    "BoxFilter": "cartesian_filters",
    "PolygonFilter": "cartesian_filters",
    "SphereFilter": "cartesian_filters",
    "GenericFieldFilter": "core",
    "PointCloudFilter": "core",
    "ValidatedPolygonT": "core",
    "AngleBinDownsample": "downsample",
    "RandomDownsampleFilter": "downsample",
    "VoxelDownsample": "downsample",
    "CartesianOutlierFilter": "outlier_filter",
    "SphericalOutlierFilter": "outlier_filter",
    "ScalarFieldFilter": "scalar_field_filters",
    "ScalarFieldPercentileFilter": "scalar_field_filters",
    "FoVFilter": "spherical_coordinate_filters",
    "RangeFilter": "spherical_coordinate_filters",
    "SphericalPolygonFilter": "spherical_coordinate_filters",
    "PolygonFilterGPU": "gpu",
    "SphericalPolygonFilterGPU": "gpu",
}

__all__ = list(_lazy_map)

if TYPE_CHECKING:
    from .cartesian_filters import BoxFilter as BoxFilter
    from .cartesian_filters import PolygonFilter as PolygonFilter
    from .cartesian_filters import SphereFilter as SphereFilter
    from .core import GenericFieldFilter as GenericFieldFilter
    from .core import PointCloudFilter as PointCloudFilter
    from .core import ValidatedPolygonT as ValidatedPolygonT
    from .downsample import AngleBinDownsample as AngleBinDownsample
    from .downsample import RandomDownsampleFilter as RandomDownsampleFilter
    from .downsample import VoxelDownsample as VoxelDownsample
    from .gpu import PolygonFilterGPU as PolygonFilterGPU
    from .gpu import SphericalPolygonFilterGPU as SphericalPolygonFilterGPU
    from .outlier_filter import CartesianOutlierFilter as CartesianOutlierFilter
    from .outlier_filter import SphericalOutlierFilter as SphericalOutlierFilter
    from .scalar_field_filters import ScalarFieldFilter as ScalarFieldFilter
    from .scalar_field_filters import ScalarFieldPercentileFilter as ScalarFieldPercentileFilter
    from .spherical_coordinate_filters import FoVFilter as FoVFilter
    from .spherical_coordinate_filters import RangeFilter as RangeFilter
    from .spherical_coordinate_filters import SphericalPolygonFilter as SphericalPolygonFilter


def __getattr__(name: str):
    if name in _lazy_map:
        module = importlib.import_module(f"{__name__}.{_lazy_map[name]}")
        val = getattr(module, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    # so tab-completion / introspection shows the lazy names
    return sorted(set(__all__) | set(globals().keys()))
