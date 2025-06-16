import logging
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from shapely import contains_xy
from shapely.affinity import translate
from shapely.geometry import Polygon

from pchandler.v2.geometry.core import PointCloudData

from .core import PointCloudFilter

logger = logging.getLogger(__name__.split(".")[0])


class BoxFilter(PointCloudFilter):
    def __init__(self, minimum_corner: Tuple[float, float, float], maximum_corner: Tuple[float, float, float]):
        self.min_corner = np.array(minimum_corner, dtype=float)
        self.max_corner = np.array(maximum_corner, dtype=float)

    def mask(self, pcd: PointCloudData) -> NDArray[np.bool_]:
        if pcd.global_coordinate_shift is not None:
            min_corner = self.min_corner - pcd.global_coordinate_shift
            max_corner = self.max_corner - pcd.global_coordinate_shift
        else:
            min_corner = self.min_corner
            max_corner = self.max_corner

        span = max_corner - min_corner
        min_corner[span == 0] = -np.inf
        max_corner[span == 0] = np.inf

        return np.all((pcd.xyz >= min_corner) & (pcd.xyz <= max_corner), axis=1)


class SphereFilter(PointCloudFilter):
    def __init__(self, sphere_center_point: NDArray[np.floating], radius: float):
        assert sphere_center_point.shape == (3,)
        assert radius > 0
        self.sphere_center_point = sphere_center_point
        self.radius = radius

    def mask(self, pcd: PointCloudData) -> NDArray[np.bool_]:
        point = (
            self.sphere_center_point
            if pcd.global_coordinate_shift is None
            else self.sphere_center_point - pcd.global_coordinate_shift
        )

        distances_to_point = np.linalg.norm(pcd.xyz - point, axis=1)
        return distances_to_point <= self.radius


class PolygonFilter(PointCloudFilter):
    def __init__(self, polygon: Polygon, plane: str = "xy"):
        assert plane in ["xy", "xz", "yz"]

        self.polygon = polygon
        self.plane = plane

    def mask(self, pcd: PointCloudData) -> NDArray[np.bool_]:
        if self.plane == "xy":
            dims = [0, 1]
        elif self.plane == "xz":
            dims = [0, 2]
        else:
            dims = [1, 2]

        polygon = (
            self.polygon
            if pcd.global_coordinate_shift is None
            else (translate(self.polygon, *(-1 * pcd.global_coordinate_shift[dims])))
        )

        mask = contains_xy(polygon, pcd.xyz[:, dims[0]], pcd.xyz[:, dims[1]])
        return mask
