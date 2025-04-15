import logging

import numpy as np
from numpy.typing import NDArray
from shapely import contains_xy
from shapely.geometry import Polygon

from ..core import PointCloudData
from ...fov import FoV
from .core import PointCloudFilter

logger = logging.getLogger(__name__.split(".")[0])



class FoVFilter(PointCloudFilter):
    def __init__(self, fov: FoV):
        self.fov = fov

    def mask(self, pcd: PointCloudData) -> NDArray[np.bool_]:
        spc = pcd.spherical_coordinates
        el_min = self.fov.elevation_min
        el_max = self.fov.elevation_max
        hor_min = self.fov.horizontal_min
        hor_max = self.fov.horizontal_max


        mask = np.logical_and(np.logical_and(spc[:, 1] >= el_min, spc[:, 1] <= el_max),
                              np.logical_and(spc[:, 2] >= hor_min, spc[:, 2] <= hor_max))

        return mask


class RangeFilter(PointCloudFilter):
    def __init__(self, low: float = 0.0, high: float = np.inf):
        self.low = low
        self.high = high

    def mask(self, pcd: PointCloudData) -> NDArray[np.bool_]:
        spc = pcd.spherical_coordinates
        mask = np.logical_and(spc[:, 0] >= self.low, spc[:, 0] <= self.high)
        return mask

class SphericalPolygonFilter(PointCloudFilter):
    def __init__(self, polygon: Polygon):
        self.polygon = polygon

    def mask(self, pcd: PointCloudData) -> NDArray[np.bool_]:
        mask = contains_xy(self.polygon, pcd.spherical_coordinates[:, 1], pcd.spherical_coordinates[:, 2])
        return mask