# pchandler - Toolbox for point-cloud handling, processing and analysis
#
# Copyright (c) 2025, Nicholas Meyer, Geosensors and Engineering Geodesy,
# Institute of Geodesy and Photogrammetry, ETH Zurich, Switzerland
# SPDX-License-Identifier: BSD-3-Clause
#
# Author: Nicholas Meyer (meyernic@ethz.ch)

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
        self.fov = fov

    def mask(self, pcd: PointCloudData) -> NDArray[np.bool_]:
        mask = np.logical_and(
            np.logical_and(pcd.v >= self.fov.top, pcd.v <= self.fov.bottom),
            np.logical_and(pcd.hz >= self.fov.left, pcd.hz <= self.fov.right),
        )

        return mask


class RangeFilter(PointCloudFilter):
    def __init__(self, low: float = 0.0, high: float = np.inf):
        self.low = low
        self.high = high

    def mask(self, pcd: PointCloudData) -> NDArray[np.bool_]:
        mask = np.logical_and(pcd.r >= self.low, pcd.r <= self.high)
        return mask


class SphericalPolygonFilter(PointCloudFilter):
    def __init__(self, polygon: Polygon):
        self.polygon = polygon

    def mask(self, pcd: PointCloudData) -> NDArray[np.bool_]:
        mask = contains_xy(self.polygon, pcd.hz, pcd.v)
        return mask
