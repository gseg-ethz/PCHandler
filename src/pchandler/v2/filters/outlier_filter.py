import logging
from typing import Annotated

import numpy as np
from numpy.typing import NDArray
import open3d as o3d
from pydantic import Field, PositiveInt


from ..geometry.core import PointCloudData
from .core import PointCloudFilter

logger = logging.getLogger(__name__.split(".")[0])

# TODO fix functions and add tests
class SphericalOutlierFilter(PointCloudFilter):
    """
    Removes outliers in the spherical coordinate space using statistical filtering.

    Parameters
    ----------
    std_ratio : float, default=0.95
        The standard deviation ratio for identifying outliers.
    number_of_neighbours : int, default=13
        The number of neighbors to consider for statistical outlier removal.
    """
    def __init__(self,
                 std_ratio: Annotated[float, Field(gt=0, le=1)]  = 0.95,
                 number_of_neighbours: PositiveInt = 13):
        self.std_ratio = std_ratio
        self.number_of_neighbours = number_of_neighbours

    def mask(self, pcd: PointCloudData) -> NDArray[np.bool_]:
        sp_pcd = o3d.geometry.PointCloud()
        sp_pcd.points = o3d.utility.Vector3dVector(
            np.hstack((pcd.spher[:, [1, 2]], np.zeros((len(pcd), 1), dtype=np.float32)))
        )

        mask = np.zeros(len(pcd), dtype=np.bool_)
        _, inliers = sp_pcd.remove_statistical_outlier(self.number_of_neighbours, self.std_ratio, True)
        mask[inliers] = True
        return mask


class CartesianOutlierFilter(PointCloudFilter):
    std_ratio: float = Field(gt=0, le=1.0)
    number_of_neighbours: PositiveInt

    def __init__(self,
                 std_ratio: Annotated[float, Field(gt=0, le=1)]  = 0.95,
                 number_of_neighbours: PositiveInt = 13):

        self.std_ratio = std_ratio
        self.number_of_neighbours = number_of_neighbours

    def mask(self, pcd: PointCloudData) -> NDArray[np.bool_]:
        pcd_o3d = pcd.to_o3d()

        mask = np.zeros(len(pcd), dtype=np.bool_)
        _, inliers = pcd_o3d.remove_statistical_outlier(self.number_of_neighbours, self.std_ratio, True)
        mask[inliers] = True
        return mask
