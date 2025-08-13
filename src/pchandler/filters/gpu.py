"""
GPU supported filters
"""

import copy
import gc
import logging
from typing import Optional

_HAS_GPU: bool = False
_GPU_IMPORT_ERROR: Optional[Exception] = None

try:
    import cudf
    import cuspatial
    import geopandas as gpd
except (ImportError, RuntimeError) as e:
    _GPU_IMPORT_ERROR = e
else:
    _HAS_GPU = True

from GSEGUtils.constants import validate_variables
from numpy.typing import NDArray
from shapely.geometry import Polygon

from pchandler import PointCloudData
from pchandler.filters.cartesian_filters import PlaneStrings
from pchandler.filters.core import PointCloudFilter, ValidatedPolygonT

logger = logging.getLogger(__name__.split(".")[0])


def is_available() -> bool:
    """Check if GPU support is available.
    Checks the availability of a GPU for computation.

    This function determines if a GPU is accessible in the environment.

    Returns
    -------
    bool
        True if a GPU is available, False otherwise.
    """
    return _HAS_GPU


def ensure_available():
    """Check if GPU supporting is available and libraries installed

    Raises
    ------
    ImportError
    """
    if not _HAS_GPU:
        msg = (
            "GPU support is not available. "
            "Install with `pip install pchandler[cuda11]` or `pip install pchandler[cuda12]`."
        )
        if _GPU_IMPORT_ERROR is not None:
            raise ImportError(msg) from _GPU_IMPORT_ERROR
        else:
            raise ImportError(msg)


class PolygonFilterGPU(PointCloudFilter):
    @validate_variables
    def __init__(self, polygon: ValidatedPolygonT, plane: PlaneStrings = "xy") -> None:
        """Filters points based on a polygon projected on a specified plane

        Parameters
        ----------
        polygon: ValidatedPolygonT
        plane: PlaneStrings, default="xy"
        """
        ensure_available()
        self.polygon: Polygon = polygon
        self.plane = plane

    def mask(self, pcd: PointCloudData) -> NDArray:
        """Create a boolean mask from the points inside the projected polygon.

        Parameters
        ----------
        pcd : PointCloudData

        Returns
        -------
        NDArray[np.bool_]
        """
        # if self.plane not in ('xy', 'xz', 'yz'): # Handled by validation
        #     raise ValueError("Invalid plane. Choose 'xy', 'xz', or 'yz'.")

        proj_pts = cudf.DataFrame(
            {axis_char: getattr(pcd, axis_char).astype(float) for axis_char in self.plane}
        ).interleave_columns()

        polygon = copy.deepcopy(self.polygon)
        if pcd.numerical_optimization_shift is not None:
            if self.plane == "xy":
                global_shift = -pcd.numerical_optimization_shift.value[:2]
            elif self.plane == "xz":
                global_shift = -pcd.numerical_optimization_shift.value[[0, 2]]
            else:
                global_shift = -pcd.numerical_optimization_shift.value[1:]
            polygon = gpd.GeoSeries([polygon]).translate(*global_shift).iloc[0]

        polygon_gpu = cuspatial.GeoSeries(gpd.GeoSeries([polygon]))
        proj_pts_gs = cuspatial.GeoSeries.from_points_xy(proj_pts)
        proj_pts_in = cuspatial.point_in_polygon(proj_pts_gs, polygon_gpu)
        mask = proj_pts_in[0].to_numpy()

        del polygon_gpu, proj_pts, proj_pts_gs, proj_pts_in
        gc.collect()

        return mask


class SphericalPolygonFilterGPU(PointCloudFilter):

    @validate_variables
    def __init__(self, polygon: ValidatedPolygonT):
        """Filters points based on a polygon defined in spherical angle coordinates

        Parameters
        ----------
        polygon: ValidatedPolygonT
        """
        ensure_available()
        self.polygon: Polygon = polygon

    def mask(self, pcd: PointCloudData) -> NDArray:
        """Creates mask from points inside the spherical angle defined polygon.

        Parameters
        ----------
        pcd : PointCloudData
            The input point cloud data containing points to be processed.

        Returns
        -------
        NDArray[np.bool_]
        """
        proj_pts = cudf.DataFrame({"x": pcd.hz.astype(float), "y": pcd.v.astype(float)}).interleave_columns()

        polygon_gpu = cuspatial.GeoSeries(gpd.GeoSeries([self.polygon]))
        proj_pts_gs = cuspatial.GeoSeries.from_points_xy(proj_pts)
        proj_pts_in = cuspatial.point_in_polygon(proj_pts_gs, polygon_gpu)
        mask = proj_pts_in[0].to_numpy()

        del polygon_gpu, proj_pts, proj_pts_gs, proj_pts_in
        gc.collect()

        return mask
