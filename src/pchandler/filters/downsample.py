# pchandler – Toolbox for point-cloud handling, processing and analysis
#
# Copyright (c) 2022–2026 ETH Zurich
# Department of Civil, Environmental and Geomatic Engineering (D-BAUG)
# Institute of Geodesy and Photogrammetry
# Geosensors and Engineering Geodesy
#
# Authors:
#   Nicholas Meyer
#   Jon Allemand
#
# SPDX-License-Identifier: BSD-3-Clause

# pchandler - Toolbox for point-cloud handling, processing and analysis
#
# Copyright (c) 2025, Nicholas Meyer, Geosensors and Engineering Geodesy,
# Institute of Geodesy and Photogrammetry, ETH Zurich, Switzerland
# SPDX-License-Identifier: BSD-3-Clause
#
# Author: Nicholas Meyer (meyernic@ethz.ch)

"""Downsampling filters for point clouds (random, voxel-grid, angle-bin)."""

from __future__ import annotations

import logging
from typing import Annotated, Literal, cast

import numpy as np
import numpy.typing as npt
from GSEGUtils.base_types import Vector_Bool_T
from GSEGUtils.constants import validate_variables
from GSEGUtils.util import unique_rows_fast
from pydantic import Field, PositiveFloat

from pchandler import PointCloudData
from pchandler.filters import PointCloudFilter
from pchandler.geometry.coordinates import rhv2xyz
from pchandler.scalar_fields import ScalarFieldManager

logger = logging.getLogger(__name__.split(".")[0])

WeightingMethods = Literal["nearest", "constant", "linear"]


def _computed_weighted_values(
    obj: VoxelDownsample | AngleBinDownsample,
    pcd: PointCloudData,
    unique_inverse: npt.NDArray[np.int32 | np.int64],
    weights: npt.NDArray[np.float32],
    unique,
) -> tuple[ScalarFieldManager, npt.NDArray[np.int32 | np.int64]]:
    weight_sum = np.bincount(unique_inverse, weights=weights, minlength=unique.shape[0])

    sfm: ScalarFieldManager = ScalarFieldManager()

    for field_name, field_values in pcd.scalar_fields.items():
        # Compute weighted-sum of scalar values within each voxel
        if field_name in ("rgb", "normals"):
            scalar_sum: np.ndarray = np.zeros((unique.shape[0], 3))

            for i in range(3):
                scalar_sum[:, i] = np.bincount(
                    unique_inverse, weights=field_values.arr[:, i] * weights, minlength=unique.shape[0]
                )

            if weight_sum.ndim == 1:
                weight_sum = weight_sum[:, None]

            if field_name == "rgb":
                logger.warning(f"RGB colours are not retained. A weighted value is taken using {obj.weighting_method=}")

            elif field_name == "normals":
                logger.warning(
                    f"Normals are not retained. "
                    f"A weighted value is taken using {obj.weighting_method=}"
                    f"These values may not be very representative of the data"
                )

        else:
            scalar_sum = np.bincount(
                unique_inverse, weights=cast(np.ndarray, field_values * weights), minlength=unique.shape[0]
            )
            weight_sum = weight_sum.reshape(-1)

        sfm[field_name] = type(field_values)(
            scalar_sum / weight_sum, name=field_name, origin_dtype=field_values.origin_dtype
        )

    return sfm, weight_sum


def _calculate_centroids_and_weights(obj, unique, ndim, unique_inverse, values, pcd):
    # Calculate centroids for each voxel
    centroids = np.zeros((unique.shape[0], ndim), dtype=np.float32)
    for i in range(ndim):  # x, y, z dimensions
        centroids[:, i] = np.bincount(unique_inverse, weights=values[:, i], minlength=unique.shape[0])

    counts = np.bincount(unique_inverse, minlength=unique.shape[0])
    centroids /= counts[:, None]  # Normalize to get centroids

    # Compute distances of points to their respective voxel centroids
    match obj.weighting_method:
        case "nearest":
            raise NotImplementedError
        case "constant":
            weights = np.ones_like(counts, dtype=np.float32)[unique_inverse]
        case "linear":
            distances = np.linalg.norm(values - centroids[unique_inverse], axis=1)
            weights = np.reciprocal(np.where(distances > 1e-6, distances, 1.0))  # Avoid division by zero
            weight_sums = np.bincount(unique_inverse, weights=weights, minlength=unique.shape[0])

            # Normalize weights per voxel
            weights /= np.where(weight_sums[unique_inverse] > 0, weight_sums[unique_inverse], 1)  # Avoid NaNs
        case _:
            raise ValueError(f"Unrecognised 'weighting_method' passed = {obj.weighting_method}")

    sfm, weight_sum = _computed_weighted_values(obj, pcd, unique_inverse, weights, unique)

    return centroids, sfm, weights, weight_sum


class RandomDownsampleFilter(PointCloudFilter):
    """Downsample a point cloud by uniformly random sampling a fixed ratio of points."""

    @validate_variables
    def __init__(self, size: Annotated[PositiveFloat, Field(lt=1)]):
        """Downsample the point cloud by randomly sampling a fixed ratio of points.

        Parameters
        ----------
        size : PositiveFloat
            Fraction of points to keep, in the open interval ``(0.0, 1.0)``.
        """
        self.size = size

    def mask(self, pcd: PointCloudData) -> Vector_Bool_T:
        """Create a mask based on a uniformly random sample of points.

        Parameters
        ----------
        pcd : PointCloudData
            Point cloud to be sampled.

        Returns
        -------
        Vector_Bool_T
            Boolean mask, ``True`` for sampled points.
        """
        indices = np.sort(np.random.choice(len(pcd), size=int(np.ceil(self.size * len(pcd))), replace=False))
        mask = np.zeros(len(pcd), dtype=np.bool_)
        mask[indices] = True
        return mask


class VoxelDownsample:
    """Downsample a point cloud onto a regular 3D voxel grid."""

    @validate_variables
    def __init__(self, voxel_size: PositiveFloat, weighting_method: WeightingMethods = "linear"):
        """Build a voxel-grid downsampler.

        Voxel centroids represent the downsampled point cloud.

        Parameters
        ----------
        voxel_size : PositiveFloat
            Edge length of each cubic voxel cell.
        weighting_method : WeightingMethods, default="linear"
            Per-point weighting strategy. One of ``"nearest"``, ``"constant"``,
            ``"linear"``.
        """
        self.voxel_size = voxel_size
        self.weighting_method = weighting_method

    def sample(self, pcd: PointCloudData) -> PointCloudData:
        """Return a downsampled copy of the point cloud built from voxel centroids.

        Parameters
        ----------
        pcd : PointCloudData
            Source point cloud.

        Returns
        -------
        PointCloudData
            New point cloud with one point per occupied voxel.
        """
        values = pcd.xyz
        ndim = values.shape[1]

        unique, unique_inverse = np.unique(
            np.round(values / self.voxel_size).astype(np.int32), axis=0, return_inverse=True
        )

        centroids, sfm, _, _ = _calculate_centroids_and_weights(self, unique, ndim, unique_inverse, values, pcd)

        new_pcd = pcd.copy(
            array=centroids,
            update={
                "scalar_fields": sfm,
                "unshifted_bbox": None,
            },
            link_to_same_NOS=True,
        )

        return new_pcd


class AngleBinDownsample:
    """Downsample a point cloud using spherical-angle binning in horizontal/vertical space."""

    @validate_variables
    def __init__(self, angle_bin_size: PositiveFloat, weighting_method: WeightingMethods = "linear"):
        """Build a spherical-angle-bin downsampler (2D over horizontal × vertical).

        Parameters
        ----------
        angle_bin_size : PositiveFloat
            Edge length of each angular bin (same units as the spherical
            coordinates being binned).
        weighting_method : WeightingMethods, default="linear"
            Per-point weighting strategy. One of ``"nearest"``, ``"constant"``,
            ``"linear"``.
        """
        self.angle_bin_size = angle_bin_size
        self.weighting_method = weighting_method

    def sample(self, pcd: PointCloudData) -> PointCloudData:
        """Return a downsampled copy of the point cloud built from angle-bin centroids.

        Parameters
        ----------
        pcd : PointCloudData
            Source point cloud.

        Returns
        -------
        PointCloudData
            New point cloud with one point per occupied angular bin.
        """
        values = pcd.spher[:, 1:]
        ndim = values.shape[1]

        unique, unique_inverse = unique_rows_fast(  # np.unique
            ((values + self.angle_bin_size / 2) // self.angle_bin_size).astype(np.int32)
        )

        centroids, sfm, weights, weight_sum = _calculate_centroids_and_weights(
            self, unique, ndim, unique_inverse, values, pcd
        )

        ranges = np.bincount(unique_inverse, weights=pcd.r * weights, minlength=unique.shape[0]) / weight_sum
        coords = rhv2xyz(np.hstack((ranges[:, np.newaxis], centroids)), pcd.socs_origin)

        # coords = SphericalCoordinates(arr=np.hstack((ranges[:, np.newaxis], centroids)))
        new_pcd = pcd.copy(
            array=coords,
            update={
                "scalar_fields": sfm,
                "unshifted_bbox": None,
            },
            link_to_same_NOS=True,
        )
        return new_pcd
