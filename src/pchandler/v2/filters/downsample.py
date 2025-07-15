import logging
import warnings
from abc import ABC, abstractmethod
from typing import Literal, Annotated, Any

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field, PositiveFloat

from pchandler.v2.util import unique_rows_fast
from pchandler.v2.geometry.core import PointCloudData
from pchandler.v2.geometry.scalar_field_manager import ScalarFieldManager

from .core import PointCloudFilter
from ..constants import validate_variables
from ..geometry.coordinates import SphericalCoordinates
from ..geometry.scalar_fields import AbstractScalarField

logger = logging.getLogger(__name__.split(".")[0])

WeightingMethods = Literal["nearest", "constant", "linear"]


class RandomDownsampleFilter(PointCloudFilter):
    @validate_variables
    def __init__(self, size: Annotated[PositiveFloat, Field(lt=1)]):
        self.size = size

    def mask(self, pcd: PointCloudData) -> npt.NDArray[np.bool_]:
        indices = np.sort(np.random.choice(len(pcd), size=int(np.ceil(self.size * len(pcd))), replace=False))
        mask = np.zeros(len(pcd), dtype=np.bool_)
        mask[indices] = True
        return mask


class VoxelDownsample:
    @validate_variables
    def __init__(self, voxel_size: PositiveFloat, weighting_method: WeightingMethods = "linear"):
        self.voxel_size = voxel_size
        self.weighting_method = weighting_method

    # TODO remove duplication code
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
        match self.weighting_method:
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
            case _:
                raise ValueError(f"Unrecognised 'weighting_method' passed = {self.weighting_method}")

        weight_sum = np.bincount(unique_inverse, weights=weights, minlength=unique.shape[0])
        sfm = ScalarFieldManager()

        for field_name, field_values in pcd.scalar_fields.items():
            # Compute weighted sum of scalar values within each voxel
            if field_name in ('rgb', 'normals'):
                scalar_sum = np.zeros((unique.shape[0],3))

                for i in range(3):
                    scalar_sum[:, i] = np.bincount(unique_inverse, weights=field_values.arr[:, i] * weights, minlength=unique.shape[0])

                if len(weight_sum.shape) == 1:
                    weight_sum = weight_sum[:, None]

                if field_name == 'rgb':
                    warnings.warn(f'RGB colours are not retained. '
                                  f'A weighted value is taken using {self.weighting_method=}')

                elif field_name == 'normals':
                    warnings.warn(f'Normals are not retained. '
                                  f'A weighted value is taken using {self.weighting_method=}'
                                  f'These values may not be very representative of the data')

            else:
                scalar_sum = np.bincount(unique_inverse, weights=field_values * weights, minlength=unique.shape[0])
                weight_sum = weight_sum[:, 0]

            sfm[field_name] = type(field_values)(scalar_sum / weight_sum,
                                                 name=field_name,
                                                 origin_dtype=field_values.origin_dtype)

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

        return PointCloudData(centroids, scalar_fields=sfm, optimized_shift=pcd.optimized_shift)


class AngleBinDownsample:
    @validate_variables
    def __init__(self, angle_bin_size: PositiveFloat, weighting_method: WeightingMethods = "linear"):
        self.angle_bin_size = angle_bin_size
        self.weighting_method = weighting_method

    def sample(self, pcd: PointCloudData) -> PointCloudData:
        pcd_angles = pcd.spher[:, 1:]
        # unique, unique_inverse = np.unique(
        #     np.round(pcd_angles / self.angle_bin_size).astype(np.int32), axis=0, return_inverse=True
        # )
        unique, unique_inverse = unique_rows_fast(  # np.unique
            ((pcd_angles + self.angle_bin_size / 2) // self.angle_bin_size).astype(np.int32)
        )

        # Calculate centroids for each voxel
        centroids = np.zeros((unique.shape[0], 2), dtype=np.float32)
        for i in range(2):  # x, y dimensions
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
            case _:
                raise ValueError(f"Unrecognised 'weighting_method' passed = {self.weighting_method}")

        weight_sum = np.bincount(unique_inverse, weights=weights, minlength=unique.shape[0])
        sfm = ScalarFieldManager()

        for field_name, field_values in pcd.scalar_fields.items():
            # Compute weighted sum of scalar values within each voxel
            if field_name in ('rgb', 'normals'):
                scalar_sum = np.zeros((unique.shape[0],3))

                for i in range(3):
                    scalar_sum[:, i] = np.bincount(unique_inverse, weights=field_values.arr[:, i] * weights, minlength=unique.shape[0])

                if len(weight_sum.shape) == 1:
                    weight_sum = weight_sum[:, None]

                if field_name == 'rgb':
                    warnings.warn(f'RGB colours are not retained. '
                                  f'A weighted value is taken using {self.weighting_method=}')

                elif field_name == 'normals':
                    warnings.warn(f'Normals are not retained. '
                                  f'A weighted value is taken using {self.weighting_method=}'
                                  f'These values may not be very representative of the data')
            else:
                scalar_sum = np.bincount(unique_inverse, weights=field_values * weights, minlength=unique.shape[0])
                weight_sum = weight_sum[:, 0]

            sfm[field_name] = type(field_values)(scalar_sum / weight_sum,
                                   name=field_name,
                                   origin_dtype=field_values.origin_dtype)


        ranges = (
            np.bincount(unique_inverse, weights=pcd.spher[:, 0] * weights, minlength=unique.shape[0])
            / weight_sum
        )

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

        coords = SphericalCoordinates(arr=np.hstack((ranges[:, np.newaxis], centroids)))
        return PointCloudData(coords.xyz, scalar_fields=sfm, optimized_shift=pcd.optimized_shift)
