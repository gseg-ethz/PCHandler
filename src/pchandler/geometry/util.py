import copy
import logging
from functools import wraps

import alphashape
import numpy as np
from shapely.affinity import scale, translate
from shapely.geometry import MultiPolygon, Polygon

from .core import PointCloudData

logger = logging.getLogger(__name__.split(".")[0])


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
        case "xy":
            proj_pts = pcd.xyz[:, :2]
        case "xz":
            proj_pts = pcd.xyz[:, [0, 2]]
        case "yz":
            proj_pts = pcd.xyz[:, 1:]
        case _:
            raise ValueError

    # Normalize the points
    proj_pts_mean = proj_pts.mean(axis=0)
    proj_pts_scale = proj_pts.max(axis=0) - proj_pts.min(axis=0)
    proj_pts_norm = (proj_pts - proj_pts_mean) / proj_pts_scale
    if nb_points > 0:
        proj_pts_norm = proj_pts_norm[np.random.permutation(proj_pts_norm.shape[0])[:nb_points], :]

    # Add noise to reduce the risk of underdefined dimensionality
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
            case "xy":
                gs = pcd.global_coordinate_shift[:2]
            case "xz":
                gs = pcd.global_coordinate_shift[[0, 2]]
            case "yz":
                gs = pcd.global_coordinate_shift[1:]
            case _:
                raise ValueError
        als = translate(als, *gs)

    return als


def bypass_immutable(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        original_state: bool = getattr(self, '_immutable', False)
        self.set_immutability(not original_state)
        try:
            return method(self, *args, **kwargs)
        finally:
            self.set_immutability(original_state)
    return wrapper


def return_copy(deep=True):
    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            result = method(self, *args, **kwargs)
            if not self._immutable:
                return result
            return copy.deepcopy(result) if deep else copy.copy(result)
        return wrapper
    return decorator


