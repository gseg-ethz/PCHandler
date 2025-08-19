# pchandler - Toolbox for point-cloud handling, processing and analysis
#
# Copyright (c) 2025, Nicholas Meyer, Geosensors and Engineering Geodesy,
# Institute of Geodesy and Photogrammetry, ETH Zurich, Switzerland
# SPDX-License-Identifier: BSD-3-Clause
#
# Author: Nicholas Meyer (meyernic@ethz.ch)

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
        self.std_ratio = std_ratio
        self.number_of_neighbours = number_of_neighbours

    def mask(self, pcd: o3d.geometry.PointCloud) -> NDArray[np.bool_]:
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
        sp_pcd = o3d.geometry.PointCloud()
        sp_pcd.points = o3d.utility.Vector3dVector(
            np.hstack((pcd.spher[:, [1, 2]], np.zeros((len(pcd), 1), dtype=np.float32)))
        )
        return super().mask(sp_pcd)


class CartesianOutlierFilter(BaseOutlierFilter):
    def mask(self, pcd: PointCloudData) -> NDArray[np.bool_]:
        return super().mask(pcd.to_o3d())
