import logging

import open3d as o3d
import numpy as np

from .core import PointCloudFilter
from ..core import PointCloudData

logger = logging.getLogger(__name__.split(".")[0])


class SphericalOutlierFilter(PointCloudFilter):
    """
    Removes outliers in the spherical coordinate space using statistical filtering.

    Parameters
    ----------
    std_ratio : float, default=0.95
        The standard deviation ratio for identifying outliers.
    nb_neighbors : int, default=13
        The number of neighbors to consider for statistical outlier removal.
    """
    def __init__(self, std_ratio: float = 0.95, nb_neighbors: int = 13):
        self.std_ratio = std_ratio
        self.nb_neighbors = nb_neighbors

    def mask(self, pcd: PointCloudData) -> PointCloudData:
        sp_pcd = o3d.geometry.PointCloud()
        sp_pcd.points = o3d.utility.Vector3dVector(np.hstack((pcd.spherical_coordinates[:,1:],
                                                              np.zeros((pcd.nbPoints,1), dtype=np.float32))))

        mask = np.zeros(pcd.nbPoints, dtype=np.bool_)
        _, inliers = sp_pcd.remove_statistical_outlier(self.nb_neighbors, self.std_ratio, True)
        mask[inliers] = True
        return mask


class CartesianOutlierFilter(PointCloudFilter):
    def __init__(self, std_ratio: float = 0.95, nb_neighbors: int = 13):
        self.std_ratio = std_ratio
        self.nb_neighbors = nb_neighbors

    def mask(self, pcd: PointCloudData) -> PointCloudData:
        pcd_o3d = pcd.to_o3d()

        mask = np.zeros(pcd.nbPoints, dtype=np.bool_)
        _, inliers = pcd_o3d.remove_statistical_outlier(self.nb_neighbors, self.std_ratio, True)
        mask[inliers] = True
        return mask

