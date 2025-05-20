import logging
import warnings

import numpy as np
from numpy.typing import NDArray

from ..core import PointCloudData
from ..scalar_fields import ScalarFieldManager
from ...util import unique_rows_fast
from .core import PointCloudFilter

logger = logging.getLogger(__name__.split(".")[0])


class RandomDownsampleFilter(PointCloudFilter):
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


class VoxelDownsample:
    _possible_weighting_method: list[str] = ["nearest", "constant", "linear"]

    def __init__(self, voxel_size: float, weigthing_method: str = "linear"):
        if weigthing_method not in self._possible_weighting_method:
            raise ValueError(f"Weighing method '{weigthing_method}' is not supported.")
        if voxel_size <= 0:
            raise ValueError(f"Voxel_size '{voxel_size}' must be positive.")
        self.voxel_size = voxel_size
        self.weigthing_method = weigthing_method

    def sample(self, pcd: PointCloudData) -> PointCloudData:
        unique, unique_inverse = np.unique(
            np.round(pcd.xyz / self.voxel_size).astype(np.int32), axis=0, return_inverse=True
        )

        # Calculate centroids for each voxel
        centroids = np.zeros((unique.shape[0], 3), dtype=np.float32)
        for i in range(3):  # x, y, z dimensions
            centroids[:, i] = np.bincount(unique_inverse, weights=pcd.xyz[:, i], minlength=unique.shape[0])

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
        for field_name, field_values in pcd.scalar_fields.items():
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
        return PointCloudData(
            centroids,
            scalar_fields=sfm,
            spherical_coordinates_origin=pcd.spherical_coordinates_origin,
            global_coordinate_shift=pcd.global_coordinate_shift,
            _global_shift_already_applied=True,
        )

class AngleBinDownsample:
    _possible_weighting_method: list[str] = ["nearest", "constant", "linear"]

    def __init__(self, angle_bin_size: float, weighting_method: str = "linear"):
        if weighting_method not in self._possible_weighting_method:
            raise ValueError(f"Weighing method '{weighting_method}' is not supported.")
        if angle_bin_size <= 0:
            raise ValueError(f"Voxel_size '{angle_bin_size}' must be positive.")
        self.angle_bin_size = angle_bin_size
        self.weighting_method = weighting_method

    def sample(self, pcd: PointCloudData) -> PointCloudData:
        pcd_angles = pcd.spherical_coordinates[:,1:]
        # unique, unique_inverse = np.unique(
        #     np.round(pcd_angles / self.angle_bin_size).astype(np.int32), axis=0, return_inverse=True
        # )
        unique, unique_inverse = unique_rows_fast( #np.unique
            ((pcd_angles + self.angle_bin_size/2) // self.angle_bin_size).astype(np.int32)
        )

        # Calculate centroids for each voxel
        centroids = np.zeros((unique.shape[0], 2), dtype=np.float32)
        for i in range(2):  # x, y, z dimensions
            centroids[:, i] = np.bincount(unique_inverse, weights=pcd_angles[:, i], minlength=unique.shape[0])

        counts = np.bincount(unique_inverse, minlength=unique.shape[0])
        centroids /= counts[:, None]  # Normalize to get centroids

        # Compute distances of points to their respective voxel centroids
        match self.weighting_method:
            case "nearest":
                raise NotImplementedError
            case "constant":
                weights = np.ones_like(counts, dtype=np.float32)[unique_inverse]
            case "linear":
                distances = np.linalg.norm(pcd_angles - centroids[unique_inverse], axis=1)
                weights = np.reciprocal(np.where(distances > 1e-6, distances, 1.0))  # Avoid division by zero
                weight_sums = np.bincount(unique_inverse, weights=weights, minlength=unique.shape[0])

                # Normalize weights per voxel
                weights /= np.where(weight_sums[unique_inverse] > 0, weight_sums[unique_inverse], 1)  # Avoid NaNs

        weight_sum = np.bincount(unique_inverse, weights=weights, minlength=unique.shape[0])
        sfm = ScalarFieldManager()
        for field_name, field_values in pcd.scalar_fields.items():
            # Compute weighted sum of scalar values within each voxel
            scalar_sum = np.bincount(unique_inverse, weights=field_values * weights, minlength=unique.shape[0])
            sfm.create_field(field_name, scalar_sum / weight_sum)
            # self.scalar_fields[field_name] = (scalar_sum / weight_sum).astype(field_values.dtype)

        ranges = np.bincount(unique_inverse, weights=pcd.spherical_coordinates[:,0] * weights, minlength=unique.shape[0]) / weight_sum

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
        return PointCloudData.from_spherical_coordinates(
            spherical_coordinates=np.hstack((ranges[:, np.newaxis], centroids)),
            scalar_fields=sfm,
            spherical_coordinates_origin=pcd.spherical_coordinates_origin,
            global_coordinate_shift=pcd.global_coordinate_shift,
        )