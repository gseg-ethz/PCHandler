# Auto-generated stub for lazy exports
from typing import Any, Final, NoReturn

from .cartesian_filters import BoxFilter, PolygonFilter, SphereFilter
from .core import GenericFieldFilter, PointCloudFilter, ValidatedPolygonT
from .downsample import AngleBinDownsample, RandomDownsampleFilter, VoxelDownsample
from .gpu import PolygonFilterGPU, SphericalPolygonFilterGPU
from .outlier_filter import CartesianOutlierFilter, SphericalOutlierFilter
from .scalar_field_filters import ScalarFieldFilter, ScalarFieldPercentileFilter
from .spherical_coordinate_filters import FoVFilter, RangeFilter, SphericalPolygonFilter

__all__: Final[list[str]] = ['BoxFilter', 'PolygonFilter', 'SphereFilter', 'GenericFieldFilter', 'PointCloudFilter', 'ValidatedPolygonT', 'AngleBinDownsample', 'RandomDownsampleFilter', 'VoxelDownsample', 'CartesianOutlierFilter', 'SphericalOutlierFilter', 'ScalarFieldFilter', 'ScalarFieldPercentileFilter', 'FoVFilter', 'RangeFilter', 'SphericalPolygonFilter', 'PolygonFilterGPU', 'SphericalPolygonFilterGPU']

def __getattr__(name: str) -> NoReturn: ...
