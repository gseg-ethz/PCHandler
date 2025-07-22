"""
Public API for pchandler.geometry.filters
Re-exports key classes and functions from the submodules.
"""

from .cartesian_filters import (
    BoxFilter,
    PolygonFilter,
    SphereFilter,
)
from .core import GenericFieldFilter, PointCloudFilter
from .downsample import AngleBinDownsample, RandomDownsampleFilter, VoxelDownsample
from .outlier_filter import CartesianOutlierFilter, SphericalOutlierFilter
from .scalar_field_filters import ScalarFieldFilter, ScalarFieldPercentileFilter
from .spherical_coordinate_filters import FoVFilter, RangeFilter, SphericalPolygonFilter

__all__ = [
    "PointCloudFilter",
    "BoxFilter",
    "SphereFilter",
    "PolygonFilter",
    "RandomDownsampleFilter",
    "VoxelDownsample",
    "AngleBinDownsample",
    "SphericalOutlierFilter",
    "CartesianOutlierFilter",
    "ScalarFieldFilter",
    "ScalarFieldPercentileFilter",
    "FoVFilter",
    "RangeFilter",
    "GenericFieldFilter",
    "SphericalPolygonFilter",
]

# optional GPU support
try:
    from . import gpu

    __all__.append("gpu")
except ImportError:
    # either no GPU build or missing dependencies;
    # we simply don’t expose that sub-module
    pass
