import logging

import numpy as np
from numpy.typing import NDArray
from shapely import contains_xy
from shapely.geometry import Polygon

from pchandler import PointCloudData
from pchandler.filters import PointCloudFilter
from pchandler.geometry.spherical import FoV

logger = logging.getLogger(__name__.split(".")[0])


class FoVFilter(PointCloudFilter):
    """
    Filters a point cloud based on a given field of view (FoV).

    This class is used to create a mask for a point cloud data instance by limiting
    its data points to a specific field of view defined by top, bottom, left, and
    right boundaries.

    Parameters
    ----------
    fov : FoV
        Field of view object defining the top, bottom, left, and right boundaries.
    """
    def __init__(self, fov: FoV):
        """
        Initializes the class with the given field of view (FoV).

        Parameters
        ----------
        fov : FoV
            The field of view object to associate with the instance.
        """
        self.fov = fov

    def mask(self, pcd: PointCloudData) -> NDArray[np.bool_]:
        """
        Applies a logical mask to determine if points in a point cloud data fall within a
        specified field of view (FOV).

        Parameters
        ----------
        pcd : PointCloudData
            The point cloud data containing vertical (`v`) and horizontal (`hz`)
            coordinate values.

        Returns
        -------
        numpy.ndarray of numpy.bool_
            A boolean array indicating whether each point lies within the specified FOV.
        """
        mask = np.logical_and(
            np.logical_and(pcd.v >= self.fov.top, pcd.v <= self.fov.bottom),
            np.logical_and(pcd.hz >= self.fov.left, pcd.hz <= self.fov.right),
        )

        return mask


class RangeFilter(PointCloudFilter):
    """
    Filters points in a point cloud based on a range threshold.

    The class is used to filter point cloud data by only keeping points that fall
    within a specified range of values. The range is defined with a lower and an
    upper bound, both inclusive. This is useful for preprocessing point cloud data
    to exclude outliers or focus on specific regions.

    Parameters
    ----------
    low : float
        The lower bound of the range filter. Points with values below this are
        excluded.
    high : float
        The upper bound of the range filter. Points with values above this are
        excluded.
    """
    def __init__(self, low: float = 0.0, high: float = np.inf):
        """
        Initializes bounds for a range.

        This constructor sets the lower and upper bounds for a range. The default
        lower bound is 0.0, and the default upper bound is infinity.

        Parameters
        ----------
        low : float, optional
            The lower boundary of the range. Default is 0.0.
        high : float, optional
            The upper boundary of the range. Default is infinity.

        """
        self.low = low
        self.high = high

    def mask(self, pcd: PointCloudData) -> NDArray[np.bool_]:
        """
        Generate a boolean mask for a point cloud dataset based on a range filter.

        Parameters
        ----------
        pcd : PointCloudData
            The input point cloud dataset containing the attribute `r`.

        Returns
        -------
        NDArray[np.bool_]
            Boolean mask indicating whether each point in the point cloud satisfies
            the range filter criteria.
        """
        mask = np.logical_and(pcd.r >= self.low, pcd.r <= self.high)
        return mask


class SphericalPolygonFilter(PointCloudFilter):
    """
    A class managing operations on polygons.

    Provides methods to apply polygon-based filtering operations on
    point cloud data.

    Parameters
    ----------
    polygon : Polygon
        The polygon used as a filter.
    """
    def __init__(self, polygon: Polygon):
        """
        A class managing operations on polygons.

        Parameters
        ----------
        polygon : Polygon
            The polygon object to be managed.
        """
        self.polygon = polygon

    def mask(self, pcd: PointCloudData) -> NDArray[np.bool_]:
        """
        Applies a polygon mask to the given point cloud data.

        This method checks which points from the input point cloud data fall
        inside the specified polygon and returns a boolean mask.

        Parameters
        ----------
        pcd : PointCloudData
            Input point cloud data.

        Returns
        -------
        numpy.ndarray of bool
            A boolean array where each entry corresponds to whether the respective
            point in the input lies within the polygon.
        """
        mask = contains_xy(self.polygon, pcd.hz, pcd.v)
        return mask
