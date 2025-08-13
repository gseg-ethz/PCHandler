"""
GPU module for pchandler.filters

Provides functions that use GPU acceleration (via cudf and cuspatial) to filter point clouds.
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
    """
    Checks the availability of a GPU for computation.

    This function determines if a GPU is accessible in the environment.

    Returns
    -------
    bool
        True if a GPU is available, False otherwise.
    """
    return _HAS_GPU


def ensure_available():
    """
    Checks if GPU support is available.

    Raises
    ------
    ImportError
        If GPU support is not available or there is an import error related
        to GPU dependencies, an ImportError is raised with a detailed message
        on how to enable GPU support.
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
    """
    Filters point cloud data based on a planar polygon projection.

    The class leverages GPU-based spatial computations to filter points in a
    point cloud that lie within a user-defined polygon on a specified plane.
    It supports handling point cloud data with potential numerical optimization
    shifts for aligning points globally before applying the polygon mask.

    Parameters
    ----------
    polygon : Polygon
        The validated polygon used for point cloud data filtering.
    plane : PlaneStrings
        The planar axis type (`"xy"`, `"xz"`, or `"yz"`) specifying the target
        plane for projection and masking operations.
    """
    @validate_variables
    def __init__(self, polygon: ValidatedPolygonT, plane: PlaneStrings = "xy") -> None:
        """
        Initializes an object with a validated polygon and a specified plane.

        Parameters
        ----------
        polygon : ValidatedPolygonT
            The validated polygon to initialize.
        plane : PlaneStrings, optional
            The plane type in which the"""
        ensure_available()
        self.polygon: Polygon = polygon
        self.plane = plane

    def mask(self, pcd: PointCloudData) -> NDArray:
        """
        Computes a mask for point cloud data based on a planar polygon projection.

        This method projects the points of a `PointCloudData` onto the specified
        plane and determines whether points lie within a given polygon. It supports
        numerical optimization shifts for global alignment before applying the
        polygon mask.

        Parameters
        ----------
        pcd : PointCloudData
            The point cloud data object containing point coordinates, plane information,
            and optional numerical optimization shifts.

        Returns
        -------
        numpy.ndarray
            A boolean mask where `True` indicates that a point is inside the polygon,
            and `False` otherwise.
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
    """
    Performs filtering of point cloud data based on a polygon using GPU acceleration.

    The SphericalPolygonFilterGPU class provides functionality to mask points from a
    PointCloudData object based on their inclusion inside a given polygon. It leverages
    GPU-based processing for efficient computation on large datasets.

    Parameters
    ----------
    polygon : Polygon
        A validated polygon used for masking points in point cloud data.
    """
    @validate_variables
    def __init__(self, polygon: ValidatedPolygonT):
        """
        Initializes an instance with a validated polygon.

        Parameters
        ----------
        polygon : ValidatedPolygonT
            A polygon object that is validated and used during initialization.
        """
        ensure_available()
        self.polygon: Polygon = polygon

    def mask(self, pcd: PointCloudData) -> NDArray:
        """
        Masks points based on their inclusion in a polygon.

        Takes a PointCloudData object and determines which points fall within the defined
        polygon. Returns a NumPy array representing a mask where points within the
        polygon are marked.

        Parameters
        ----------
        pcd : PointCloudData
            The input point cloud data containing points to be processed.

        Returns
        -------
        numpy.ndarray
            A binary mask as a NumPy array where included points are marked.
        """
        proj_pts = cudf.DataFrame({"x": pcd.hz.astype(float), "y": pcd.v.astype(float)}).interleave_columns()

        polygon_gpu = cuspatial.GeoSeries(gpd.GeoSeries([self.polygon]))
        proj_pts_gs = cuspatial.GeoSeries.from_points_xy(proj_pts)
        proj_pts_in = cuspatial.point_in_polygon(proj_pts_gs, polygon_gpu)
        mask = proj_pts_in[0].to_numpy()

        del polygon_gpu, proj_pts, proj_pts_gs, proj_pts_in
        gc.collect()

        return mask
