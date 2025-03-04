"""
Filtering module for pchandler.geometry.

Provides functions that filter or extract subsets from a PointCloudData instance.
"""
from abc import ABC, abstractmethod
import logging
import warnings

import numpy as np
from typing import Callable, Tuple, Optional
from shapely.geometry import Polygon, Point
from numpy.typing import NDArray

from .core import PointCloudData
from ..fov import FoV

logger = logging.getLogger(__name__.split(".")[0])


class PointCloudFilter(ABC):
    """
    Abstract base class for filters on a PointCloudData.

    Subclasses should implement the mask() method to return a boolean mask
    that selects the desired points.
    """

    @abstractmethod
    def mask(self, pcd: PointCloudData) -> NDArray[np.bool_]:
        """
        Compute and return a boolean mask for the provided point cloud.

        Parameters:
            pcd (PointCloudData): The point cloud to filter.

        Returns:
            A numpy boolean array with shape (N,) where N is the number of points.
        """
        pass

    def reduce(self, pcd: PointCloudData) -> None:
        """
        Reduces the point cloud in-place to only the points where mask() is True.

        Parameters:
            pcd (PointCloudData): The point cloud to reduce.

        Returns:
            The modified point cloud.
        """
        m = self.mask(pcd)
        pcd.reduce(m)

    def extract(self, pcd: PointCloudData) -> PointCloudData:
        """
        Extracts points where mask() is True: returns a new point cloud with those points,
        and removes them from the original.

        Parameters:
            pcd (PointCloudData): The point cloud to extract points from.

        Returns:
            A new PointCloudData instance containing the extracted points.
        """
        m = self.mask(pcd)
        new_pcd = pcd.extract(m)
        return new_pcd

    def sample(self, pcd: PointCloudData) -> PointCloudData:
        """
        Returns a new point cloud with only the points where mask() is True, leaving
        the original point cloud untouched.

        Parameters:
            pcd (PointCloudData): The point cloud to sample from.

        Returns:
            A new PointCloudData instance containing only the sampled points.
        """
        m = self.mask(pcd)
        return pcd.sample(m)


class FoVFilter(PointCloudFilter):
    def __init__(self, fov: FoV):
        self.fov = fov

    def mask(self, pcd: PointCloudData) -> NDArray[np.bool_]:
        spc = pcd.spherical_coordinates
        mask = np.logical_and(np.logical_and(spc[:, 1] >= self.fov.elevation_min,
                                             spc[:, 1] <= self.fov.elevation_max),
                              np.logical_and(spc[:, 2] >= self.fov.horizontal_min,
                                             spc[:, 2] <= self.fov.horizontal_max, ))

        return mask


class RangeFilter(PointCloudFilter):
    def __init__(self, low: float = 0.0, high: float = np.inf):
        self.low = low
        self.high = high

    def mask(self, pcd: PointCloudData) -> NDArray[np.bool_]:
        spc = pcd.spherical_coordinates
        return np.logical_and(spc >= self.low, spc <= self.high)


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
    def __init__(self, point: NDArray[np.floating], radius: float):
        assert point.shape == (3,)
        assert radius > 0
        self.point = point
        self.radius = radius

    def mask(self, pcd: PointCloudData) -> NDArray[np.bool_]:
        distances_to_point = np.linalg.norm(self.xyz - self.point, axis=1)
        return distances_to_point <= self.radius

class PolygonFilter(PointCloudFilter):
    def __init__(self, polygon: Polygon, plane: str = "xy"):
        assert plane in ["xy", "xz", "yz"]

        self.polygon = polygon
        self.plane = plane

    def mask(self, pcd: PointCloudData) -> NDArray[np.bool_]:
        if self.plane == 'xy':
            proj_pts = pcd.xyz[:, :2]
        elif self.plane == 'xz':
            proj_pts = pcd.xyz[:, [0, 2]]
        else:
            proj_pts = pcd.xyz[:, 1:]


class ScalarFieldFilter(PointCloudFilter):
    def __init__(self, field_label: str, lower_bound: float = -np.inf, upper_bound: float = np.inf):
        self.field_label = field_label
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def mask(self, pcd: PointCloudData) -> NDArray[np.bool_]:
        assert self.field_label in pcd.scalar_fields.keys(), f"Field {self.field_label} is not defined."
        scalar_field_data = pcd.scalar_fields[self.field_label].data

        return np.logical_and(scalar_field_data >= self.lower_bound,
                              scalar_field_data <= self.upper_bound)

class ScalarFieldPercentileFilter(PointCloudFilter):
    def __init__(self, field_label: str, lower_percentile: float = 0.0, upper_percentile: float = 100.0):
        assert 0.0 <= lower_percentile <= upper_percentile <= 100.0
        self.field_label = field_label
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def mask(self, pcd: PointCloudData) -> NDArray[np.bool_]:
        assert self.field_label in pcd.scalar_fields.keys(), f"Field {self.field_label} is not defined."
        lower_bound, upper_bound = np.percentile(pcd, [self.lower_percentile, self.upper_percentile])
        scalar_field_data = pcd.scalar_fields[self.field_label].data

        return np.logical_and(scalar_field_data >= self.lower_bound,
                              scalar_field_data <= self.upper_bound)


class GenericFieldFilter(PointCloudFilter):
    """
    A generic filter that uses a user-supplied function to generate a mask
    from a given field.
    """
    def __init__(self, field_label: str, filter_func: Callable):
        """
        Parameters:
            field_label: The field (attribute or scalar field key) on which to operate.
            filter_func: A callable that takes the field data and returns a boolean mask.
        """
        self.field = field_label
        self.filter_func = filter_func

    def mask(self, pcd: PointCloudData) -> NDArray[np.bool_]:
        """
        Retrieves the field data from the point cloud, applies the filter function,
        and returns the resulting boolean mask.
        """
        if self.field == "spherical_coordinates":
            _ = pcd.spherical_coordinates  # Ensure it's computed.
            data = pcd.spherical_coordinates
        elif self.field in pcd.scalar_fields:
            data = pcd.scalar_fields[self.field].data
        elif hasattr(pcd, self.field):
            data = getattr(pcd, self.field)
        else:
            raise ValueError(f"Field '{self.field}' does not exist in the point cloud.")
        return self.filter_func(data)
        return np.array([self.polygon.contains(Point(pt)) for pt in proj_pts])


class RandomFilter(PointCloudFilter):
    def __init__(self, size: float | int):
        self.size = size

    def mask(self, pcd: PointCloudData) -> NDArray[np.bool_]:
        nb_points = pcd.nbPoints
        size = self.size
        if isinstance(size, float) and 0 < size < 1:
            size = int(np.ceil(size * nb_points))
        if size >= nb_points:
            warnings.warn("Subsampling ratio above 1; returning original point cloud.")
            return np.ones((nb_points,), dtype=np.bool_)
        indices = np.sort(np.random.choice(np.arange(nb_points), size=size, replace=False))
        mask = np.zeros(nb_points, dtype=np.bool_)
        mask[indices] = True
        return mask

#
#
#
# def random_subsample(pcd: PointCloudData, size: float | int, in_place: bool = True) -> PointCloudData:
#     """
#     Randomly subsamples the point cloud.
#
#     Parameters
#     ----------
#     pcd : PointCloudData
#         The point cloud to subsample.
#     size : float or int
#         If a float (0 < size < 1), specifies the proportion of points to retain.
#         If an int, specifies the exact number of points to retain.
#     in_place : bool, optional
#         If True, modifies the original point cloud; otherwise, returns a new one.
#
#     Returns
#     -------
#     PointCloudData
#         The subsampled point cloud.
#     """
#     nb_points = pcd.nbPoints
#     if isinstance(size, float) and 0 < size < 1:
#         size = int(np.ceil(size * nb_points))
#     if size >= nb_points:
#         return pcd if in_place else pcd.copy()
#
#     selection = np.sort(np.random.choice(np.arange(nb_points), size=size, replace=False))
#     if in_place:
#         return pcd._reduce_points_to(selection)
#     else:
#         return pcd._copy_selection(selection)