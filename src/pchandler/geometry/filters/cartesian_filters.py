import logging
from typing import Tuple

import alphashape
import numpy as np
from numpy.typing import NDArray
from shapely.geometry import Polygon, MultiPolygon
from shapely.affinity import scale, translate
from shapely import contains_xy

from .core import PointCloudFilter
from ..core import PointCloudData

logger = logging.getLogger(__name__.split(".")[0])

class BoxFilter(PointCloudFilter):
    def __init__(self, minimum_corner: Tuple[float, float, float], maximum_corner: Tuple[float, float, float]):
        self.min_corner = np.array(minimum_corner, dtype=float)
        self.max_corner = np.array(maximum_corner, dtype=float)

    def mask(self, pcd: PointCloudData) -> NDArray[np.bool_]:
        if pcd.global_coordinate_shift is not None:
            min_corner = self.min_corner - pcd.global_coordinate_shift
            max_corner = self.max_corner - pcd.global_coordinate_shift

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
        point = self.sphere_center_point if pcd.global_coordinate_shift is None else self.sphere_center_point - pcd.global_coordinate_shift

        distances_to_point = np.linalg.norm(self.xyz - point, axis=1)
        return distances_to_point <= self.radius

class PolygonFilter(PointCloudFilter):
    def __init__(self, polygon: Polygon, plane: str = "xy"):
        assert plane in ["xy", "xz", "yz"]

        self.polygon = polygon
        self.plane = plane

    def mask(self, pcd: PointCloudData) -> NDArray[np.bool_]:
        if self.plane == 'xy':
            dims = [0, 1]
        elif self.plane == 'xz':
            dims = [0, 2]
        else:
            dims = [1, 2]

        polygon = self.polygon if pcd.global_coordinate_shift is None else (
            translate(self.polygon, *(-1*pcd.global_coordinate_shift[dims])))

        mask = contains_xy(polygon, pcd.xyz[:, dims[0]], pcd.xyz[:, dims[1]])
        return mask


def get_outline_polygon(pcd: PointCloudData, plane: str, alpha_value: float = 10.0, nb_points: int = -1) -> Polygon:
    """
    Computes the outline of the point cloud as a polygon in a specific 2D projection.

    Parameters
    ----------
    pcd : PointCloudData
        The point cloud data used to define the outline.
    plane : str
        The plane of projection ('xy', 'xz', or 'yz').
    alpha_value : float, default=10.0
        The alpha value for the alpha shape algorithm, controlling the detail of the outline.
    nb_points : int, default=-1
        The number of points to use for the computation. If -1, all points are used.

    Returns
    -------
    Polygon
        A Shapely Polygon representing the outline of the point cloud.

    Raises
    ------
    ValueError
        If the specified plane is invalid.
    NotImplementedError
        If the outline computation results in an unsupported geometry type.
    """
    match plane:
        case 'xy':
            proj_pts = pcd.xyz[:, :2]
        case 'xz':
            proj_pts = pcd.xyz[:, [0, 2]]
        case 'yz':
            proj_pts = pcd.xyz[:, 1:]
        case _:
            raise ValueError

    # Normalize the points
    proj_pts_mean = proj_pts.mean(axis=0)
    proj_pts_scale = proj_pts.max(axis=0) - proj_pts.min(axis=0)
    proj_pts_norm = (proj_pts - proj_pts_mean) / proj_pts_scale
    if nb_points > 0:
        proj_pts_norm = proj_pts_norm[np.random.permutation(proj_pts_norm.shape[0])[:nb_points], :]

    # Add noise to reduce risk of underdefined dimensionality
    noise = np.random.normal(scale=1e-6, size=proj_pts_norm.shape)
    proj_pts_norm = proj_pts_norm + noise

    als_norm = alphashape.alphashape(proj_pts_norm, alpha=alpha_value)
    als = translate(scale(als_norm, *proj_pts_scale), *proj_pts_mean)

    if isinstance(als, MultiPolygon):
        als = max(als.geoms, key=lambda polygon: polygon.area)

    if not isinstance(als, Polygon):
        raise NotImplementedError

    if pcd.global_coordinate_shift is not None:
        match plane:
            case 'xy':
                gs = pcd.global_coordinate_shift[:2]
            case 'xz':
                gs = pcd.global_coordinate_shift[[0, 2]]
            case 'yz':
                gs = pcd.global_coordinate_shift[1:]
            case _:
                raise ValueError
        als = translate(als, *gs)

    return als