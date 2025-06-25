import logging
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from shapely import contains_xy
from shapely.affinity import translate
from shapely.geometry import Polygon

from pydantic import PositiveFloat, validate_call

from ..constants import DEFAULT_CONFIG

from ..geometry.core import PointCloudData
from .core import PointCloudFilter
from ..base_types import Array_Nx3_T, Vector_3_T


logger = logging.getLogger(__name__.split(".")[0])


class BoxFilter(PointCloudFilter):
    def __init__(self, minimum: Vector_3_T, maximum: Vector_3_T):
        if np.any(minimum >= maximum):
            raise ValueError(f"Cannot create box filter where minimum corner is greater than the maximum corner"
                             f"\n {minimum=} vs {maximum=}")

        self.minimum = Vector_3_T(minimum)
        self.maximum = Vector_3_T(maximum)

    @property
    def extents(self) -> Vector_3_T:
        return self.maximum - self.minimum

    def mask(self, pcd: PointCloudData) -> NDArray[np.bool_]:
        if pcd.optimized_shift is not None:
            min_corner = self.minimum - pcd.optimized_shift.optimal_shift
            max_corner = self.maximum - pcd.optimized_shift.optimal_shift
        else:
            min_corner = self.minimum
            max_corner = self.maximum

        min_corner[self.extents == 0] = -np.inf
        max_corner[self.extents == 0] = np.inf

        return np.all((pcd.xyz >= min_corner) & (pcd.xyz <= max_corner), axis=1)


class SphereFilter(PointCloudFilter):
    def __init__(self, sphere_center: Vector_3_T, radius: PositiveFloat):
        self.sphere_center = Vector_3_T(sphere_center)
        self.radius = radius

    def mask(self, pcd: PointCloudData) -> NDArray[np.bool_]:
        point = (
            self.sphere_center
            if pcd.optimized_shift is None
            else self.sphere_center - pcd.optimized_shift
        )

        distances_to_point: np.ndarray = np.linalg.norm(pcd.xyz - point, axis=1)
        return distances_to_point <= self.radius


class PolygonFilter(PointCloudFilter):
    @validate_call(config=DEFAULT_CONFIG)
    def __init__(self, polygon: Polygon, plane: str = "xy"):
        # TODO check if function should be able to clip along a normal direction

        if plane not in ['nx', 'ny', 'nz']:
            raise ValueError(f"plane value string or normal vector does not exist")

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
            if pcd.optimized_shift is None
            else (translate(self.polygon, *(-1 * pcd.optimized_shift.optimal_shift[dims])))
        )

        mask = contains_xy(polygon, pcd.xyz[:, dims[0]], pcd.xyz[:, dims[1]])
        return mask
