# pchandler - Toolbox for point-cloud handling, processing and analysis
#
# Copyright (c) 2025, Nicholas Meyer, Geosensors and Engineering Geodesy,
# Institute of Geodesy and Photogrammetry, ETH Zurich, Switzerland
# SPDX-License-Identifier: BSD-3-Clause
#
# Author: Nicholas Meyer (meyernic@ethz.ch)

"""
Cartesian coordinate-based filters.
"""

import logging
from typing import Literal

import numpy as np
import numpy.typing as npt
from GSEGUtils.constants import validate_variables
from pydantic import PositiveFloat
from shapely import contains_xy  # type: ignore[import-untyped]
from shapely.affinity import translate  # type: ignore[import-untyped]
from shapely.geometry import Polygon  # type: ignore[import-untyped]

from pchandler import PointCloudData
from GSEGUtils.base_types import Vector_3_T, Vector_Bool_T
from pchandler.filters import PointCloudFilter, ValidatedPolygonT

logger = logging.getLogger(__name__.split(".")[0])


PlaneStrings = Literal["xy", "xz", "yz"]


def _get_offset(pcd: PointCloudData, mode: Literal["local", "global"] = "local") -> Vector_3_T:
    if mode == "local":
        return np.zeros(shape=(3,))
    elif mode == "global":
        if pcd.numerical_optimization_shift is None:
            return np.zeros(shape=(3,))
        else:
            return pcd.numerical_optimization_shift.value
    else:
        raise ValueError(f"Invalid mode: {mode}")


class BoxFilter(PointCloudFilter):
    @validate_variables
    def __init__(self, minimum: Vector_3_T, maximum: Vector_3_T):
        """Filters points based on a 3D bounding box

        Parameters
        ----------
        minimum : Vector_3_T
        maximum : Vector_3_T
        """
        if np.any(minimum >= maximum):
            raise ValueError(
                f"Cannot create box filter where minimum corner is greater than the maximum corner"
                f"\n {minimum=} vs {maximum=}"
            )

        self.minimum = minimum
        self.maximum = maximum

    @property
    def extents(self) -> Vector_3_T:
        """Computes the extents of a 3D space.

        Returns
        -------
        Vector_3_T
        """
        return self.maximum - self.minimum

    def mask(self, pcd: PointCloudData, mode: Literal["local", "global"] = "local") -> Vector_Bool_T:
        """Create a boolean mask for points within a 3D bounding box.

        Parameters
        ----------
        pcd : PointCloudData
        mode : Literal["local", "global"], default="local"
            Defines the coordinate frame of reference for the bounding box.

        Returns
        -------
        Vector_Bool_T
        """
        offset = _get_offset(pcd, mode)

        min_corner = self.minimum - offset
        max_corner = self.maximum - offset

        min_corner[self.extents == 0] = -np.inf
        max_corner[self.extents == 0] = np.inf

        return np.all((pcd.xyz >= min_corner) & (pcd.xyz <= max_corner), axis=1)


class SphereFilter(PointCloudFilter):
    @validate_variables
    def __init__(self, sphere_center: Vector_3_T, radius: PositiveFloat) -> None:
        """Filters points based on a sphere with a defined center and radius

        Parameters
        ----------
        sphere_center : Vector_3_T
        radius : PositiveFloat
        """
        self.sphere_center = sphere_center
        self.radius = radius

    def mask(self, pcd: PointCloudData, mode: Literal["local", "global"] = "local") -> Vector_Bool_T:
        """Create a boolean mask for points within the sphere.
        Parameters
        ----------
        pcd : PointCloudData
        mode : Literal["local", "global"], default="local"
            Defines the coordinate frame of reference for the bounding box.

        Returns
        -------
        Vector_Bool_T
        """
        offset = _get_offset(pcd, mode)

        point = self.sphere_center - offset

        distances_to_point: np.ndarray[np.floating] = np.linalg.norm(pcd.xyz - point, axis=1)
        return distances_to_point <= self.radius


class PolygonFilter(PointCloudFilter):
    @validate_variables
    def __init__(self, polygon: ValidatedPolygonT, plane: PlaneStrings = "xy") -> None:
        """Filters points based on a polygon projected on a specified plane

        Parameters
        ----------
        polygon: ValidatedPolygonT
        plane: PlaneStrings, default="xy"
        """

        self.polygon: Polygon = polygon
        self.plane = plane

    # DECISION should the Polygon Filter match the GPU variant?
    def mask(self, pcd: PointCloudData, mode: Literal["local", "global"] = "local") -> Vector_Bool_T:
        """Create a boolean mask from the points inside the projected polygon.

        Parameters
        ----------
        pcd : PointCloudData
        mode : Literal["local", "global"], default="local"
            Defines the coordinate frame of reference for the bounding box.

        Returns
        -------
        Vector_Bool_T
        """
        if self.plane == "xy":
            dims = [0, 1]
        elif self.plane == "xz":
            dims = [0, 2]
        else:
            dims = [1, 2]

        offset = _get_offset(pcd, mode)

        polygon = translate(self.polygon, * (-1 * offset[dims]))

        mask: npt.NDArray[np.bool_] = contains_xy(polygon, pcd.xyz[:, dims[0]], pcd.xyz[:, dims[1]])
        return mask
