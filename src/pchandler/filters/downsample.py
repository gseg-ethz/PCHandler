from __future__ import annotations

import logging
from typing import Annotated, Literal, cast

import numpy as np
import numpy.typing as npt
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
    """
    Computes weighted scalar field values and sums weights across unique data points.

    Parameters
    ----------
    obj : VoxelDownsample or AngleBinDownsample
        The downsampling object determining the weighting method.
    pcd : PointCloudData
        The input point cloud data containing scalar fields.
    unique_inverse : numpy.ndarray of int32 or int64
        Indices mapping original elements to unique groups after downsampling.
    weights : numpy.ndarray of float32
        Weights corresponding to each element in the point cloud.
    unique
        Unique indices or values resulting from voxel downsampling.

    Returns
    -------
    sfm : ScalarFieldManager
        The manager containing the computed weighted scalar fields.
    weight_sum : numpy.ndarray
        Sum of the weights for each unique group.

    """
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
                logger.warning(
                    f"RGB colours are not retained. " f"A weighted value is taken using {obj.weighting_method=}"
                )

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
    """
    Calculates centroids and weights for voxels based on the provided data and weighting method.

    Parameters
    ----------
    obj : Any
        Object containing the weighting method attribute.
    unique : numpy.ndarray
        Array containing unique voxel identifiers.
    ndim : int
        Number of dimensions (e.g., 3 for x, y, z).
    unique_inverse : numpy.ndarray
        Array mapping each point to its corresponding voxel identifier.
    values : numpy.ndarray
        Coordinates or values associated with the points.
    pcd : Any
        Point cloud data input.

    Returns
    -------
    tuple
        A tuple containing:
        - centroids (numpy.ndarray): Centroids of each voxel.
        - sfm (Any): Computed weighted values based on the input and voxel data.
        - weights (numpy.ndarray): Weights for each point.
        - weight_sum (numpy.ndarray): Sum of weights for each voxel.
    """
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
    """
    RandomDownsampleFilter is a filter that randomly downsamples a point cloud data by selecting
    a subset of points based on a specified size ratio.

    This class is designed to process and downsample point cloud data for tasks where
    a reduced representation of the dataset is required.

    Parameters
    ----------
    size : PositiveFloat
        A value indicating the fraction of the points to retain in the downsampled
        point cloud. Must be less than 1.
    """
    @validate_variables
    def __init__(self, size: Annotated[PositiveFloat, Field(lt=1)]):
        """
        Initialize the class with the specified size parameter.

        Parameters
        ----------
        size : PositiveFloat
            A float value representing the size, which must be less than 1.
        """
        self.size = size

    def mask(self, pcd: PointCloudData) -> npt.NDArray[np.bool_]:
        """
        Generates a boolean mask for the given point cloud data, selecting a subset of points
        based on a randomly chosen set of indices.

        Parameters
        ----------
        pcd : PointCloudData
            Input point cloud data to generate the mask for.

        Returns
        -------
        numpy.ndarray[numpy.bool_]
            A boolean array where `True` indicates the selected points and `False`
            represents the excluded points.
        """
        indices = np.sort(np.random.choice(len(pcd), size=int(np.ceil(self.size * len(pcd))), replace=False))
        mask = np.zeros(len(pcd), dtype=np.bool_)
        mask[indices] = True
        return mask


class VoxelDownsample:
    """
    A class for voxel downsampling of point cloud data.

    This class provides functionality to reduce the number of points in point cloud
    data using voxel downsampling with specified voxel size and weighting methods.

    Parameters
    ----------
    voxel_size : PositiveFloat
        The size of the voxel used for downsampling operations.
    weighting_method : WeightingMethods
        The method employed for weighting during the downsampling process.
    """
    @validate_variables
    def __init__(self, voxel_size: PositiveFloat, weighting_method: WeightingMethods = "linear"):
        """
        Initializes a class instance with specified voxel size and weighting method.

        Parameters
        ----------
        voxel_size : PositiveFloat
            The size of the voxel for processing.
        weighting_method : WeightingMethods, optional
            The method to be used for weighting. Defaults to "linear".
        """
        self.voxel_size = voxel_size
        self.weighting_method = weighting_method

    def sample(self, pcd: PointCloudData) -> PointCloudData:
        """
        Samples a point cloud data using voxel downsampling.

        This method reduces the number of points in the input point cloud data by
        performing voxel downsampling based on the specified voxel size.

        Parameters
        ----------
        pcd : PointCloudData
            The point cloud data to sample.

        Returns
        -------
        PointCloudData
            A new point cloud data object containing the downsampled point cloud
            with updated centroids and scalar fields.
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
    """
    Processes and resamples point cloud data in spherical coordinates for improved spatial representation
    by downsampling based on angle bins and weighting methods.

    Detailed description of the class, its purpose, and usage.

    Parameters
    ----------
    angle_bin_size : PositiveFloat
        Size of the angle bin, must be a positive float.
    weighting_method : WeightingMethods
        Method used for weighting, default is "linear".
    """
    @validate_variables
    def __init__(self, angle_bin_size: PositiveFloat, weighting_method: WeightingMethods = "linear"):
        """
        Initializes the class with specified angle bin size and weighting method.

        Parameters
        ----------
        angle_bin_size : PositiveFloat
            Size of the angle bin, must be a positive float.
        weighting_method : WeightingMethods, optional
            Method to be used for weighting, default is "linear".
        """
        self.angle_bin_size = angle_bin_size
        self.weighting_method = weighting_method

    def sample(self, pcd: PointCloudData) -> PointCloudData:
        """
        Samples the input point cloud data based on spherical coordinates while maintaining
        spatial fidelity and calculating various centroids, weights, and corresponding ranges.

        Parameters
        ----------
        pcd : PointCloudData
            The input point cloud data.

        Returns
        -------
        PointCloudData
            A new point cloud data object based on the sampled spherical coordinates.
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
