"""
Public API for pchandler.geometry.filters
Re-exports key classes and functions from the submodules.
"""

from .core import (
    PointCloudFilter,
    GenericFieldFilter
)

from .cartesian_filters import (
    BoxFilter,
    SphereFilter,
    PolygonFilter,
)

from .downsample import (
    RandomDownsampleFilter,
    VoxelDownsample
)

from .outlier_filter import (
    SphericalOutlierFilter,
    CartesianOutlierFilter
)

from .scalar_field_filters import (
    ScalarFieldFilter,
    ScalarFieldPercentileFilter
)

from .spherical_coordinate_filters import (
    FoVFilter,
    RangeFilter
)

# GPU filters (PolygonFilterGPU/SphericalPolygonFilterGPU) require RAPIDS
# (cudf/cuspatial), which are absent on CPU-only machines and not pip-installable
# into isolated venvs. Import lazily so `import pchandler.geometry` — and
# PointCloudData + all CPU filters — works without a GPU. The GPU module stays
# importable as `pchandler.geometry.filters.gpu` wherever cudf IS present.
# (CPU-01 surgical fix.)
try:
    from . import gpu
except ImportError:  # cudf/cuspatial unavailable — CPU-only environment
    gpu = None  # type: ignore[assignment]

__all__ = ["PointCloudFilter","BoxFilter","SphereFilter","PolygonFilter", "RandomDownsampleFilter", "VoxelDownsample",
           "SphericalOutlierFilter", "CartesianOutlierFilter", "ScalarFieldFilter", "ScalarFieldPercentileFilter",
           "FoVFilter", "RangeFilter", "GenericFieldFilter", "gpu"]
