import logging
from typing import Tuple, Annotated, Literal

import numpy as np
import numpy.typing as npt
from shapely import contains_xy         # type: ignore[import-untyped]
from shapely.affinity import translate  # type: ignore[import-untyped]
from shapely.geometry import Polygon    # type: ignore[import-untyped]

from pydantic import PositiveFloat, BeforeValidator, model_validator

from ..constants import DEFAULT_CONFIG

from ..geometry.core import PointCloudData
from .core import PointCloudFilter
from ..base_types import Array_Nx3_T, Vector_3_T


logger = logging.getLogger(__name__.split(".")[0])


PlaneStrings = Literal["xy", "xz", "yz"]


class BoxFilter(PointCloudFilter):
    minimum: Vector_3_T
    maximum: Vector_3_T

    def __init__(self, minimum: Vector_3_T, maximum: Vector_3_T):
        super().__init__(minimum=minimum, maximum=maximum)

    @model_validator(mode="after")
    def post_validation(self) -> None:
        if np.any(self.minimum >= self.maximum):
            raise ValueError(f"Cannot create box filter where minimum corner is greater than the maximum corner"
                             f"\n {self.minimum=} vs {self.maximum=}")

    @property
    def extents(self) -> Vector_3_T:
        return self.maximum - self.minimum

    def mask(self, pcd: PointCloudData) -> npt.NDArray[np.bool_]:
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
    sphere_center: Vector_3_T
    radius: PositiveFloat

    def __init__(self, sphere_center: Vector_3_T, radius: float) -> None:
        super().__init__(sphere_center=sphere_center, radius=radius)

    def mask(self, pcd: PointCloudData) -> npt.NDArray[np.bool_]:
        point = (
            self.sphere_center
            if pcd.optimized_shift is None
            else self.sphere_center - pcd.optimized_shift.optimal_shift
        )

        distances_to_point: npt.NDArray[np.float64|np.float32] = np.linalg.norm(pcd.xyz - point, axis=1)
        return distances_to_point <= self.radius


class PolygonFilter(PointCloudFilter):
    polygon: Annotated[Polygon, BeforeValidator(lambda x: Polygon(x))]
    plane: PlaneStrings

    def __init__(self, polygon: Polygon, plane: str = "xy") -> None:
        super().__init__(polygon=polygon, plane=plane)

    def mask(self, pcd: PointCloudData) -> npt.NDArray[np.bool_]:
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

        mask: npt.NDArray[np.bool_] = contains_xy(polygon, pcd.xyz[:, dims[0]], pcd.xyz[:, dims[1]])
        return mask
