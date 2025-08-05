"""
Public API for pchandler.filters
Re-exports key classes and functions from the submodules.
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
    from .cartesian_filters import BoxFilter, PolygonFilter, SphereFilter
    from .core import GenericFieldFilter, PointCloudFilter, ValidatedPolygonT
    from .downsample import AngleBinDownsample, RandomDownsampleFilter, VoxelDownsample
    from .gpu import PolygonFilterGPU, SphericalPolygonFilterGPU
    from .outlier_filter import CartesianOutlierFilter, SphericalOutlierFilter
    from .scalar_field_filters import ScalarFieldFilter, ScalarFieldPercentileFilter
    from .spherical_coordinate_filters import (
        FoVFilter,
        RangeFilter,
        SphericalPolygonFilter,
    )


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
