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
    """
    Filter point clouds based on statistical outlier removal.

    This class provides functionality to filter point clouds using statistical
    outlier removal methodology, taking into account a set ratio of standard
    deviation and a defined number of neighbors around each point.

    Parameters
    ----------
    std_ratio : float
        Standard deviation ratio used for filtering. Must be greater than 0 and less
        than or equal to 1.
    number_of_neighbours : int
        Number of neighbors considered around each point during statistical
        computation.
    """
    def __init__(self, std_ratio: Annotated[float, Field(gt=0, le=1)] = 0.95, number_of_neighbours: PositiveInt = 13):
        """
        Represents a configuration for a model with standard deviation ratio and
        number of neighbors as parameters.

        Parameters
        ----------
        std_ratio : float
            The standard deviation ratio for the model. Must be greater than 0 and
            less than or equal to 1.
        number_of_neighbours : int
            The number of neighbors to consider. Must be a positive integer.
        """
        self.std_ratio = std_ratio
        self.number_of_neighbours = number_of_neighbours

    def mask(self, pcd: o3d.geometry.PointCloud) -> NDArray[np.bool_]:
        """
        Generate a boolean mask from a point cloud based on statistical outlier removal.

        Parameters
        ----------
        pcd : o3d.geometry.PointCloud
            The input point cloud from which a mask will be generated.

        Returns
        -------
        numpy.ndarray of numpy.bool_
            Boolean array where `True` represents inliers and `False` represents outliers.
        """
        mask = np.zeros(len(pcd.points), dtype=np.bool_)
        _, inliers = pcd.remove_statistical_outlier(self.number_of_neighbours, self.std_ratio, True)
        mask[inliers] = True
        return mask


class SphericalOutlierFilter(BaseOutlierFilter):
    """
    Removes outliers in the spherical coordinate space using statistical filtering.

    Parameters
    ----------
    std_ratio : float, default=0.95
        The standard deviation ratio for identifying outliers.
    number_of_neighbours : int, default=13
        The number of neighbors to consider for statistical outlier removal.
    """

    def mask(self, pcd: PointCloudData) -> NDArray[np.bool_]:
        """
        Masks the given PointCloudData object based on its spherical coordinates.

        Parameters
        ----------
        pcd : PointCloudData
            The PointCloudData object to be masked.

        Returns
        -------
        numpy.ndarray of bool
            A boolean array representing the mask.
        """
        sp_pcd = o3d.geometry.PointCloud()
        sp_pcd.points = o3d.utility.Vector3dVector(
            np.hstack((pcd.spher[:, [1, 2]], np.zeros((len(pcd), 1), dtype=np.float32)))
        )
        return super().mask(sp_pcd)


class CartesianOutlierFilter(BaseOutlierFilter):
    """
    Filters outliers from a point cloud using Cartesian coordinates.

    Applies a filtering process to identify and remove outliers based on
    their spatial properties in Cartesian space.

    """
    def mask(self, pcd: PointCloudData) -> NDArray[np.bool_]:
        """
        Applies a mask to the input point cloud data.

        Parameters
        ----------
        pcd : PointCloudData
            The input point cloud data to which the mask will be applied.

        Returns
        -------
        NDArray[np.bool_]
            A boolean array indicating the mask applied to the point cloud data.
        """
        return super().mask(pcd.to_o3d())
