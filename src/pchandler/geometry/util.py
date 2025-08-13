from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterable, NamedTuple, Optional, Self

import alphashape
import numpy as np
from shapely.affinity import scale, translate
from shapely.geometry import MultiPolygon, Polygon

from GSEGUtils.base_types import Array_Nx3_T, Vector_3_T

if TYPE_CHECKING:
    from pchandler import PointCloudData

logger = logging.getLogger(__name__.split(".")[0])


def get_outline_polygon(pcd: PointCloudData, plane: str, alpha_value: float = 10.0, nb_points: int = -1) -> Polygon:
    """Computes the outline of the point cloud as a polygon in a specific 2D projection.

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

    # Normalise the points
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

    if pcd.numerical_optimization_shift is not None:
        match plane:
            case "xy":
                gs = pcd.numerical_optimization_shift.value[:2]
            case "xz":
                gs = pcd.numerical_optimization_shift.value[[0, 2]]
            case "yz":
                gs = pcd.numerical_optimization_shift.value[1:]
            case _:
                raise ValueError
        als = translate(als, *gs)

    return als


# TODO could change this to a frozen basemodel with validation
class MinMaxPoints(NamedTuple):
    """
    Named tuple for determining the minimum and maximum points in a given
    3-dimensional coordinate space.

    This class provides utility methods for deriving minimum and maximum points from
    a set of points, constructing instances from existing minimum and maximum points,
    and calculating derived attributes such as the central point and extents.

    Parameters
    ----------
    minimum : Vector_3_T
        The minimum point in the 3D space.
    maximum : Vector_3_T
        The maximum point in the 3D space.
    """
    minimum: Vector_3_T
    maximum: Vector_3_T

    @classmethod
    def from_points(cls, points: Array_Nx3_T, already_applied_shift_vec: Optional[Vector_3_T] = None) -> Self:
        """
        Creates an instance of the class from a given set of points and an optional shift vector.

        Parameters
        ----------
        points : Array_Nx3_T
            Array (N, 3)
        already_applied_shift_vec : Optional[Vector_3_T], optional
            Size 3 vector that has already been applied to the points. If not provided, it defaults to a zero vector.

        Returns
        -------
        Self
            An instance of the class initialized with the calculated minimum and maximum points
            based on the given points and the shift vector.
        """

        if len(points) == 0:
            return cls(np.zeros(3), np.zeros(3))

        if already_applied_shift_vec is None:
            already_applied_shift_vec = np.zeros((3,))

        min_point = np.min(points, axis=0) + already_applied_shift_vec
        max_point = np.max(points, axis=0) + already_applied_shift_vec

        return cls(min_point, max_point)

    @classmethod
    def from_minmax_points(cls, minmax_points: Iterable[Self | Array_Nx3_T]) -> Self:
        """
        Creates an instance of the class using a collection of minimum and maximum points.

        Transform the provided iterable of minimum and maximum points into a 3-dimensional
        array. These points are then used to create the desired instance of the class via
        `from_points`.

        Parameters
        ----------
        minmax_points : Iterable[Self | Array_Nx3_T]
            Collection of points, where each element can either be an instance of the class
            or an `Array_Nx3_T`. The points represent the minimum and maximum bounds that
            will be processed and used to construct the instance.

        Returns
        -------
        Self
            An instance of the class constructed based on the processed points.

        """
        arr = Array_Nx3_T(np.vstack(tuple(minmax_points)))
        return cls.from_points(arr)

    @property
    def central_point(self) -> Vector_3_T:
        """
        Get the central point of the bounding box.

        The central point is calculated as the mean of the minimum and maximum
        coordinates along each axis.

        Returns
        -------
        Vector_3_T
            The central point of the bounding box as a 3-dimensional vector.
        """
        return np.mean(np.vstack((self.minimum, self.maximum)), axis=0)

    @property
    def extents(self) -> Vector_3_T:
        """
        Computes the extents of a bounding box based on its minimum and maximum values.

        Returns
        -------
        Vector_3_T
            Difference between the maximum and minimum values representing the
            extents of the bounding box.
        """
        return self.maximum - self.minimum

    def __array__(self) -> Array_Nx3_T:
        """
        Returns the bounds of the object as a NumPy array.

        The method constructs a 2D array where the first row represents the minimum
        bounds and the second row represents the maximum bounds.

        Returns
        -------
        Array_Nx3_T
            A 2D array with the minimum bounds in the first row and maximum bounds
            in the second row.
        """
        return np.vstack((self.minimum, self.maximum))
