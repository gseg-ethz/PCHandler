# pchandler - Toolbox for point-cloud handling, processing and analysis
#
# Copyright (c) 2025, Nicholas Meyer, Geosensors and Engineering Geodesy,
# Institute of Geodesy and Photogrammetry, ETH Zurich, Switzerland
# SPDX-License-Identifier: BSD-3-Clause
#
# Author: Nicholas Meyer (meyernic@ethz.ch)

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
    """Countains the minimum and maximum points of a set of coordinates to represent it's bounding box.

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
        """Create an instance from a given set of points

        Parameters
        ----------
        points : Array_Nx3_T
        already_applied_shift_vec : Optional[Vector_3_T], optional
            If provided, a vector that represents a shift applied to the points before

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
        """Create an instance from a set of MinMaxPoints or Array_Nx3_T objects.

        Parameters
        ----------
        minmax_points : Iterable[Self | Array_Nx3_T]
            Collection of point sets, either as point clouds or MinMaxPoints objects.

        Returns
        -------
        Self
        """
        arr = Array_Nx3_T(np.vstack(tuple(minmax_points)))
        return cls.from_points(arr)

    @property
    def central_point(self) -> Vector_3_T:
        """Return the center point of the bounding box

        Returns
        -------
        Vector_3_T
        """
        return np.mean(np.vstack((self.minimum, self.maximum)), axis=0)

    @property
    def extents(self) -> Vector_3_T:
        """Return the extents of the bounding box

        Returns
        -------
        Vector_3_T
        """
        return self.maximum - self.minimum

    def __array__(self) -> Array_Nx3_T:
        """Enables the use of the MinMaxPoints object as an array.

        Returns
        -------
        Array_Nx3_T
        """
        return np.vstack((self.minimum, self.maximum))
