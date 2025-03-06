"""
Filtering module for pchandler.geometry.

Provides functions that filter or extract subsets from a PointCloudData instance.
"""
from abc import ABC, abstractmethod
import logging
from typing import Callable, Tuple, Optional
import warnings

import alphashape
import numpy as np
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.affinity import scale, translate
from shapely.prepared import prep
from shapely import contains_xy
# from shapely.vectorized import contains
from numpy.typing import NDArray

from .core import PointCloudData
from .scalar_fields import ScalarFieldManager
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
        el_min = self.fov.el_min
        el_max = self.fov.el_max
        hor_min = self.fov.hor_min
        hor_max = self.fov.hor_max

        if pcd._spherical_coordinates_represented_0_to_2pi:
            hor_min = hor_min + np.pi if hor_min < 0 else hor_min
            hor_max = hor_max - np.pi if hor_max > 0 else hor_max
            if hor_min > hor_max:
                hor_min, hor_max = hor_max, hor_min

        mask = np.logical_and(np.logical_and(spc[:, 1] >= el_min, spc[:, 1] <= el_max),
                              np.logical_and(spc[:, 2] >= hor_min, spc[:, 2] <= hor_max))

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
        point = self.point if pcd.global_coordinate_shift is None else self.point - pcd.global_coordinate_shift

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


class RandomDownsamplingFilter(PointCloudFilter):
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
class VoxelDownsamplingFilter:
    _possible_weighting_method: list[str] = ["nearest", "constant", "linear"]


    def __init__(self, voxel_size: float, weigthing_method: str = "linear"):
        if weigthing_method not in self._possible_weighting_method:
            raise ValueError(f"Weighing method '{weigthing_method}' is not supported.")
        if voxel_size <= 0:
            raise ValueError(f"Voxel_size '{voxel_size}' must be positive.")
        self.voxel_size = voxel_size
        self.weigthing_method = weigthing_method

    def sample(self, pcd: PointCloudData) -> PointCloudData:
        unique, unique_inverse = np.unique(np.round(pcd.xyz / self.voxel_size).astype(np.int32), axis=0,
                                           return_inverse=True)

        # Calculate centroids for each voxel
        centroids = np.zeros((unique.shape[0], 3), dtype=np.float32)
        for i in range(3):  # x, y, z dimensions
            centroids[:, i] = np.bincount(unique_inverse, weights=self.xyz[:, i], minlength=unique.shape[0])

        counts = np.bincount(unique_inverse, minlength=unique.shape[0])
        centroids /= counts[:, None]  # Normalize to get centroids

        # Compute distances of points to their respective voxel centroids
        match self.weigthing_method:
            case "nearest":
                raise NotImplementedError
            case "constant":
                weights = np.ones_like(counts, dtype=np.float32)[unique_inverse]
            case "linear":
                distances = np.linalg.norm(pcd.xyz - centroids[unique_inverse], axis=1)
                weights = np.reciprocal(np.where(distances > 1e-6, distances, 1.0))  # Avoid division by zero
                weight_sums = np.bincount(unique_inverse, weights=weights, minlength=unique.shape[0])

                # Normalize weights per voxel
                weights /= np.where(weight_sums[unique_inverse] > 0, weight_sums[unique_inverse], 1)  # Avoid NaNs

        sfm = ScalarFieldManager()
        for field_name, field_values in self.scalar_fields.items():
            # Compute weighted sum of scalar values within each voxel
            scalar_sum = np.bincount(unique_inverse, weights=field_values * weights, minlength=unique.shape[0])
            weight_sum = np.bincount(unique_inverse, weights=weights, minlength=unique.shape[0])
            sfm.create_field(field_name, scalar_sum / weight_sum)
            # self.scalar_fields[field_name] = (scalar_sum / weight_sum).astype(field_values.dtype)

        # # Average scalar fields
        # averaged_scalar_fields = {}
        # for field_name, field_values in self.scalar_fields.items():
        #     # Compute the sum of scalar values within each voxel
        #     scalar_sum = np.bincount(unique_inverse, weights=field_values, minlength=unique.shape[0])
        #     # Compute the average
        #     averaged_scalar_fields[field_name] = (scalar_sum / counts).astype(field_values.dtype)

        # object.__setattr__(self, "xyz", centroids)
        # if self._spherical_coordinates_calculated:
        #     object.__delattr__(self, "spherical_coordinates")
        #     object.__setattr__(self, "_spherical_coordinates_calculated", False)
        #
        # # Todo: Add functionality
        # warnings.warn("Normals, and colors are not retained during `voxel_downsample`!")
        # object.__setattr__(self, 'color', None)
        # object.__setattr__(self, 'normals', None)
        # # for field_name in self.scalar_fields.keys():
        # #     del self.scalar_fields[field_name]
        # # self.scalar_fields.clear()
        #
        # return
        warnings.warn("Normals, and colors are not retained during `voxel_downsample`!")
        return PointCloudData(centroids, scalar_fields=sfm.scalar_fields,
                              spherical_coordinates_origin=pcd.spherical_coordinates_origin,
                              global_coordinate_shift=pcd.global_coordinate_shift,
                              _global_shift_already_applied=True)



def get_outline_polygon(pcd: PointCloudData, plane: str, alpha_value: float = 10.0, nb_points: int = -1) -> Polygon:
    """
    Computes the outline of the point cloud as a polygon in a specific 2D projection.

    Parameters
    ----------
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