"""
Statistical outlier removal filters.
"""
import logging
from typing import Annotated

import numpy as np
import open3d as o3d
from numpy.typing import NDArray
from pydantic import Field, PositiveInt

from pchandler import PointCloudData
from pchandler.filters import PointCloudFilter

logger = logging.getLogger(__name__.split(".")[0])


class BaseOutlierFilter(PointCloudFilter):
    def __init__(self, std_ratio: Annotated[float, Field(gt=0, le=1)] = 0.95, number_of_neighbours: PositiveInt = 13):
        """Base class for filters using a statistical outlier removal method

        This uses algorithm is based on a standard deviation ratio and a number of neighbors considered.

        Parameters
        ----------
        std_ratio : float
        number_of_neighbours : int
        """
        self.std_ratio = std_ratio
        self.number_of_neighbours = number_of_neighbours


    def mask(self, pcd: o3d.geometry.PointCloud) -> NDArray[np.bool_]:
        """Create a boolean mask from the non-outlier points

        Parameters
        ----------
        pcd : o3d.geometry.PointCloud


        Returns
        -------
        NDArray[np.bool_]
        """
        mask = np.zeros(len(pcd.points), dtype=np.bool_)
        _, inliers = pcd.remove_statistical_outlier(self.number_of_neighbours, self.std_ratio, True)
        mask[inliers] = True
        return mask


class SphericalOutlierFilter(BaseOutlierFilter):
    """Outlier filter for point clouds in spherical coordinates"""

    def mask(self, pcd: PointCloudData) -> NDArray[np.bool_]:
        """Create a boolean mask from the non-outlier points in spherical coordinates.

        Parameters
        ----------
        pcd : PointCloudData

        Returns
        -------
        NDArray[np.bool_]
        """
        sp_pcd = o3d.geometry.PointCloud()
        sp_pcd.points = o3d.utility.Vector3dVector(
            np.hstack((pcd.spher[:, [1, 2]], np.zeros((len(pcd), 1), dtype=np.float32)))
        )
        return super().mask(sp_pcd)


class CartesianOutlierFilter(BaseOutlierFilter):
    """Outlier filter for point clouds in cartesian coordinates"""
    def mask(self, pcd: PointCloudData) -> NDArray[np.bool_]:
        """Create a boolean mask from the non-outlier points in cartesian coordinates

        Parameters
        ----------
        pcd : PointCloudData

        Returns
        -------
        NDArray[np.bool_]
        """
        return super().mask(pcd.to_o3d())
