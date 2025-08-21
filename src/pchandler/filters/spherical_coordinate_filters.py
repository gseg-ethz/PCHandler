"""
Spherical coordinate based filters
"""
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
    def __init__(self, fov: FoV):
        """Filters points based on a given field of view (FoV).

        Parameters
        ----------
        fov : FoV
            Field of view object defining the top, bottom, left, and right boundaries.
        """
        self.fov = fov

    def mask(self, pcd: PointCloudData) -> NDArray[np.bool_]:
        """Create a boolean mask for a point cloud based on a given field of view (FoV).

        Parameters
        ----------
        pcd : PointCloudData

        Returns
        -------
        NDArray[np.bool_]
        """
        v_indices = np.logical_and(pcd.v >= self.fov.top, pcd.v <= self.fov.bottom)

        if self.fov.crosses_pi:
            # Combines the ranges [left, π] and [-π, right]
            hz_indices = np.logical_or(pcd.hz >= self.fov.left, pcd.hz <= self.fov.right)
        else:
            # Range of [left, right]
            hz_indices = np.logical_and(pcd.hz >= self.fov.left, pcd.hz <= self.fov.right)

        mask = np.logical_and(v_indices, hz_indices)

        return mask


class RangeFilter(PointCloudFilter):
    def __init__(self, low: float = 0.0, high: float = np.inf):
        """Filters points based on a range minimum and maximum threshold

        Parameters
        ----------
        low : float, optional
        high : float, optional

        """
        self.low = low
        self.high = high

    def mask(self, pcd: PointCloudData) -> NDArray[np.bool_]:
        """Create a boolean mask for a point cloud based on given range limits.

        Parameters
        ----------
        pcd : PointCloudData

        Returns
        -------
        NDArray[np.bool_]
        """
        mask = np.logical_and(pcd.r >= self.low, pcd.r <= self.high)
        return mask


class SphericalPolygonFilter(PointCloudFilter):
    def __init__(self, polygon: Polygon):
        """Filters points based on a polygon defined in spherical coordinates.

        Parameters
        ----------
        polygon : Polygon
        """
        self.polygon = polygon

    def mask(self, pcd: PointCloudData) -> NDArray[np.bool_]:
        """Create a boolean mask for a point cloud based on a polygon defined in spherical coordinates.

        Points inside the polygon are marked as `True`.

        Parameters
        ----------
        pcd : PointCloudData

        Returns
        -------
        NDArray[np.bool_]
        """
        mask = contains_xy(self.polygon, pcd.hz, pcd.v)
        return mask
