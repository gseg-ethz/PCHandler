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
from GSEGUtils.base_types import Vector_3_T
from pchandler.filters import PointCloudFilter, ValidatedPolygonT

logger = logging.getLogger(__name__.split(".")[0])


PlaneStrings = Literal["xy", "xz", "yz"]


def _get_offset(pcd: PointCloudData, mode: Literal["local", "global"] = "local") -> Vector_3_T:
    """
    Compute the offset vector based on the provided mode and point cloud data.

    Parameters
    ----------
    pcd : PointCloudData
        The point cloud data object used to determine the offset.
    mode : {"local", "global"}, optional
        Specifies the type of offset to compute. Defaults to "local".

    Returns
    -------
    Vector_3_T
        The computed offset vector for the specified mode.

    Raises
    ------
    ValueError
        If the provided mode is neither "local" nor "global".
    """
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
    """
    Filters point cloud data based on a 3D box volume.

    The BoxFilter class defines a filter for selecting points within a specific
    3D rectangular box defined by a minimum and maximum corner. It provides methods
    to create filters for point cloud data and compute corresponding masks.

    Parameters
    ----------
    minimum : Vector_3_T
        Minimum corner coordinates of the box.
    maximum : Vector_3_T
        Maximum corner coordinates of the box.
    """
    @validate_variables
    def __init__(self, minimum: Vector_3_T, maximum: Vector_3_T):
        """
        Initializes a bounding box filter with specified minimum and maximum corners.

        Parameters
        ----------
        minimum : Vector_3_T
            The minimum corner of the bounding box.
        maximum : Vector_3_T
            The maximum corner of the bounding box.

        Raises
        ------
        ValueError
            If any component of the minimum corner is greater than or equal to the
            corresponding component of the maximum corner.
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
        """
        Computes the extents of a 3D space.

        This property calculates the difference between the maximum and minimum
        bounds in a 3D space, resulting in the extents. Extents represent the
        size of the bounding box along each axis.

        Returns
        -------
        Vector_3_T
            The extents of the 3D bounding box calculated as
            (maximum - minimum).
        """
        return self.maximum - self.minimum

    def mask(self, pcd: PointCloudData, mode: Literal["local", "global"] = "local") -> npt.NDArray[np.bool_]:
        """
        Computes a mask for points in the point cloud based on their spatial location
        relative to a defined bounding box. The mask indicates whether each point is
        inside the bounding box or not.

        Parameters
        ----------
        pcd : PointCloudData
            The point cloud data containing 3D points.
        mode : Literal["local", "global"], optional
            The reference mode for calculating the offset, either "local" or "global".
            Default is "local".

        Returns
        -------
        numpy.ndarray
            A boolean array where each element is True if the corresponding point in
            the point cloud is within the bounding box, and False otherwise.
        """
        offset = _get_offset(pcd, mode)

        min_corner = self.minimum - offset
        max_corner = self.maximum - offset

        min_corner[self.extents == 0] = -np.inf
        max_corner[self.extents == 0] = np.inf

        return np.all((pcd.xyz >= min_corner) & (pcd.xyz <= max_corner), axis=1)


class SphereFilter(PointCloudFilter):
    """
    A filter for determining which 3D points lie within a spherical region.

    This class applies a spherical filter on point cloud data to identify points
    inside the defined sphere based on a center point and radius. It supports
    both local and global coordinate frames.

    Parameters
    ----------
    sphere_center : Vector_3_T
        The 3D vector representing the center point of the sphere.
    radius : PositiveFloat
        Positive floating-point value indicating the radius of the sphere.
    """
    @validate_variables
    def __init__(self, sphere_center: Vector_3_T, radius: PositiveFloat) -> None:
        """
        Initializes a sphere with a defined center and radius.

        Parameters
        ----------
        sphere_center : Vector_3_T
            The 3D vector representing the center of the sphere.
        radius : PositiveFloat
            The positive floating-point value representing the radius of the sphere.
        """
        self.sphere_center = sphere_center
        self.radius = radius

    def mask(self, pcd: PointCloudData, mode: Literal["local", "global"] = "local") -> npt.NDArray[np.bool_]:
        """
        Computes a boolean mask for points within a specified radius from a sphere's center.

        Parameters
        ----------
        pcd : PointCloudData
            The input point cloud data containing points in 3D space.
        mode : Literal["local", "global"], optional
            Specifies the coordinate frame of reference. Default is "local".

        Returns
        -------
        npt.NDArray[np.bool_]
            A boolean array where each entry indicates whether a point is within
            the radius of the sphere center.
        """
        offset = _get_offset(pcd, mode)

        point = self.sphere_center - offset

        distances_to_point: npt.NDArray[np.float64 | np.float32] = np.linalg.norm(pcd.xyz - point, axis=1)
        return distances_to_point <= self.radius


class PolygonFilter(PointCloudFilter):
    """
    Filters points within a point cloud based on a polygon in a specific 2D plane.

    The PolygonFilter class is used to mask points in a point cloud that lie within a
    defined polygon projected onto a specified 2D plane. It supports filtering
    on three orthogonal planes: "xy", "xz", and "yz".

    Parameters
    ----------
    polygon : Polygon
        The polygon object used for filtering in the specified plane.
    plane : PlaneStrings
        The 2D plane used for projection, specifying which dimensions (e.g., "xy", "xz", "yz").
    """
    @validate_variables
    def __init__(self, polygon: ValidatedPolygonT, plane: PlaneStrings = "xy") -> None:
        """
        Initialize the class with a polygon and plane.

        Parameters
        ----------
        polygon : ValidatedPolygonT
            The validated polygon object.
        plane : PlaneStrings, optional
            The plane in which the polygon exists. Default is "xy".
        """
        self.polygon: Polygon = polygon
        self.plane = plane

    def mask(self, pcd: PointCloudData, mode: Literal["local", "global"] = "local") -> npt.NDArray[np.bool_]:
        """
        Generates a mask for the points in the point cloud based on a defined polygon
        and specified projection plane.

        Parameters
        ----------
        pcd : PointCloudData
            The input point cloud data to be masked.
        mode : Literal["local", "global"], optional
            Mode of operation determining the origin for offset calculation.
            Defaults to "local".

        Returns
        -------
        npt.NDArray[np.bool_]
            Boolean array where True indicates that the corresponding point in
            the point cloud lies within the polygon projected on the specified plane.
        """
        if self.plane == "xy":
            dims = [0, 1]
        elif self.plane == "xz":
            dims = [0, 2]
        else:
            dims = [1, 2]

        offset = _get_offset(pcd, mode)

        polygon = translate(self.polygon, *(-1 * offset[dims]))

        mask: npt.NDArray[np.bool_] = contains_xy(polygon, pcd.xyz[:, dims[0]], pcd.xyz[:, dims[1]])
        return mask
