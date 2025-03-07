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

        if pcd._spherical_coordinates_represented_0_to_2pi:
            hor_min = hor_min + np.pi if hor_min < 0 else hor_min
            hor_max = hor_max - np.pi if hor_max > 0 else hor_max
            if hor_min > hor_max:
                hor_min, hor_max = hor_max, hor_min

        mask = np.logical_and(np.logical_and(spc[:, 1] >= el_min, spc[:, 1] <= el_max),
                              np.logical_and(spc[:, 2] >= hor_min, spc[:, 2] <= hor_max))

        return mask


class RangeFilter(PointCloudFilter):
    def __init__(self, low: float = 0.0, high: float = np.inf):
        self.low = low
        self.high = high

    def mask(self, pcd: PointCloudData) -> NDArray[np.bool_]:
        spc = pcd.spherical_coordinates
        return np.logical_and(spc >= self.low, spc <= self.high)

class SphericalPolygonFilter(PointCloudFilter):
    def __init__(self, polygon: Polygon):
        self.polygon = polygon

    def mask(self, pcd: PointCloudData) -> NDArray[np.bool_]:
        # Todo: Solve for -pi to pi and 0 to 2pi inconsistency
        mask = contains_xy(self.polygon, pcd.spherical_coordinates[:, 1], pcd.spherical_coordinates[:, 2])
        return mask