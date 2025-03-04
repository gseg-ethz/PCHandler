"""
Core module for pchandler.geometry.

Defines the PointCloudData class and its essential methods.
"""
import sys
from dataclasses import dataclass, field, InitVar, KW_ONLY
from functools import cached_property
import logging
from typing import Optional, Dict, Iterable
if sys.version[0] == 3 and sys.version_info[1] >= 11:
    from typing import Self
else:
    from typing_extensions import Self

import numpy as np
from numpy.typing import NDArray
import open3d as o3d

from .scalar_fields import ScalarFieldManager

logger = logging.getLogger(__name__.split(".")[0])

@dataclass(frozen=True)
class PointCloudData:
    """
    Represents a 3D point cloud in a Cartesian coordinate system.

    Attributes
    ----------
    xyz : NDArray[np.float32]
        An (N x 3) array containing the x, y, and z coordinates.
    scalar_fields : dict[str, NDArray]
        A dictionary of named scalar fields.
    color : Optional[NDArray[np.uint8]]
        An (N x 3) array of colors.
    normals : Optional[NDArray[np.float32]]
        An (N x 3) array of normal vectors.
    global_coordinate_shift : Optional[NDArray[np.float_]]
        A (3,) array applied to center the coordinates.
    spherical_coordinates_origin : NDArray[np.float_]
        A (3,) array defining the origin for spherical coordinate calculations.
    """
    xyz: NDArray[np.float32]
    _: KW_ONLY
    scalar_fields: ScalarFieldManager = field(default_factory=ScalarFieldManager)
    color: Optional[NDArray[np.uint8]] = None
    normals: Optional[NDArray[np.float32]] = None
    global_coordinate_shift: Optional[NDArray[np.float_]] = None
    spherical_coordinates_origin: Optional[NDArray[np.float_]] = None
    _spherical_coordinates_calculated: bool = False
    _spherical_coordinates_represented_0_to_2pi: Optional[bool] = None
    _global_shift_already_applied: InitVar[bool] = False

    def __post_init__(self, _global_shift_already_applied: bool) -> None:
        # Set a default origin if none provided.
        if self.spherical_coordinates_origin is None:
            object.__setattr__(self, "spherical_coordinates_origin", np.zeros((3,), dtype=np.float_))

        # Validate inputs.
        if not isinstance(self.xyz, np.ndarray):
            msg = "xyz must be a numpy array"
            logger.error(msg)
            raise TypeError(msg)

        if self.color is not None and not isinstance(self.color, np.ndarray):
            msg = "color must be a numpy array"
            logger.error(msg)
            raise TypeError(msg)

        if self.normals is not None and not isinstance(self.normals, np.ndarray):
            msg = "normals must be a numpy array"
            logger.error(msg)
            raise TypeError(msg)

        if not isinstance(self.scalar_fields, ScalarFieldManager):
            msg = "scalar_fields must be a ScalarFieldManager"
            logger.error(msg)
            raise TypeError(msg)

        # Check dimensions.
        if self.xyz.ndim != 2 or self.xyz.shape[1] != 3:
            msg = "xyz should have shape (N, 3)"
            logger.error(msg)
            raise ValueError(msg)

        if self.color is not None and self.color.shape != (self.xyz.shape[0], 3):
            msg = "color must match xyz dimensions"
            logger.error(msg)
            raise ValueError(msg)

        if self.normals is not None and self.normals.shape != (self.xyz.shape[0], 3):
            msg = "normals must match xyz dimensions"
            logger.error(msg)
            raise ValueError(msg)

        for key, sf in self.scalar_fields.items():
            if len(sf) != self.xyz.shape[0]:
                msg = f"Scalar field '{key}' must have length equal to the number of points (N)"
                logger.error(msg)
                raise ValueError(msg)

        if self.spherical_coordinates_origin.shape != (3,):
            msg = "spherical_coordinates_origin must be (3,)"
            logger.error(msg)
            raise ValueError(msg)

        if self.global_coordinate_shift is not None and self.global_coordinate_shift.shape != (3,):
            msg = "global_coordinate_shift must be (3,)"
            logger.error(msg)
            raise ValueError(msg)

        # Apply a global coordinate shift if needed.
        if self.global_coordinate_shift is None and self._needs_global_shift(self.xyz):
            shift = self._calculate_optimal_global_shift(self.xyz)
            object.__setattr__(self, 'global_coordinate_shift', shift)
        if self.global_coordinate_shift is not None and not _global_shift_already_applied:
            shifted_xyz = (self.xyz - self.global_coordinate_shift).astype(np.float32)
            object.__setattr__(self, 'xyz', shifted_xyz)
        else:
            object.__setattr__(self, "xyz", self.xyz.astype(np.float32))

        # Adjust the spherical coordinates origin when a global shift is applied.
        if self.global_coordinate_shift is not None:
            new_origin = self.spherical_coordinates_origin - self.global_coordinate_shift
            object.__setattr__(self, "spherical_coordinates_origin", new_origin)

        logger.info(f"PCD created with {self.xyz.shape[0]:_d} number of points")

    @property
    def nbPoints(self) -> int:
        """Returns the number of points in the point cloud."""
        return self.xyz.shape[0]

    @cached_property
    def spherical_coordinates(self) -> NDArray[np.float32]:
        """
        Calculates and caches spherical coordinates (range, elevation, azimuth).
        """
        xyz_shifted = self.xyz - self.spherical_coordinates_origin
        if len(xyz_shifted) == 0:
            object.__setattr__(self, "_spherical_coordinates_calculated", True)
            object.__setattr__(self, "_spherical_coordinates_represented_0_to_2pi", False)
            return np.empty_like(xyz_shifted)

        sph = np.zeros_like(self.xyz, dtype=np.float32)
        xy_sq = xyz_shifted[:, 0] ** 2 + xyz_shifted[:, 1] ** 2
        sph[:, 0] = np.sqrt(xy_sq + xyz_shifted[:, 2] ** 2)  # range
        sph[:, 1] = np.arctan2(np.sqrt(xy_sq), xyz_shifted[:, 2])  # elevation
        sph[:, 2] = -np.arctan2(xyz_shifted[:, 1], xyz_shifted[:, 0])  # azimuth

        if self._spherical_coordinates_represented_0_to_2pi:
            sph[:, 2] = np.where(sph[:, 2] < 0, sph[:, 2] + 2 * np.pi, sph[:, 2])

        object.__setattr__(self, "_spherical_coordinates_calculated", True)
        return sph

    def __repr__(self) -> str:
        return f"PointCloudData with {self.nbPoints} points"

    def __getitem__(self, item):
        if isinstance(item, slice):
            indices = np.arange(self.nbPoints)[item]
        else:
            indices = item
        return self.sample(indices)

    def _convert_indexing_to_mask(self, selection: NDArray[np.bool_] | NDArray[np.int_] | slice | list[int]) -> NDArray[np.bool_]:
        """
        Converts selection types into a boolean mask of shape (self.nbPoints,).

        - If selection is a boolean mask, it is validated and returned as-is.
        - If selection is a slice, it is converted into a boolean mask.
        - If selection is a list of indices or an integer ndarray, it is converted into a boolean mask.
        - Raises an error for invalid input types.

        Returns:
            A boolean mask (`np.ndarray[np.bool_]`) of shape `(self.nbPoints,)` where `True` indicates selected points.
        """
        if isinstance(selection, np.ndarray) and selection.dtype == np.bool_:
            if selection.shape != (self.nbPoints,):
                raise ValueError(f"Boolean mask must have shape ({self.nbPoints},), but got {selection.shape}")
            return selection

        mask = np.zeros(self.nbPoints, dtype=np.bool_)  # Initialize all False
        if isinstance(selection, slice):
            mask[selection] = True
        elif isinstance(selection, list):
            mask[np.array(selection, dtype=np.int_)] = True
        elif isinstance(selection, np.ndarray):
            if selection.dtype == np.int_:
                mask[selection] = True
            else:
                raise TypeError(f"Expected selection dtype to be int or bool, but got {selection.dtype}")
        else:
            raise TypeError(f"Unsupported selection type: {type(selection)}. Must be slice, list[int], or np.ndarray.")

        return mask

    def reduce(self, selection: NDArray[np.bool_] | NDArray[np.int_] | slice | list[int]) -> None:
        """
        Reduces the point cloud to only the points specified by the mask.
        Operates in place.
        """

        mask = self._convert_indexing_to_mask(selection)

        object.__setattr__(self, "xyz", self.xyz[mask])
        object.__setattr__(self, "scalar_fields", self.scalar_fields[mask])

        if self.color is not None:
            object.__setattr__(self, "color", self.color[mask])
        if self.normals is not None:
            object.__setattr__(self, "normals", self.normals[mask])
        if self._spherical_coordinates_calculated:
            object.__setattr__(self, "spherical_coordinates", self.spherical_coordinates[mask])
        return

    def sample(self, selection: NDArray[np.bool_] | NDArray[np.int_] | slice | list[int]) -> Self:
        """
        Creates a copy of the point cloud with only the points specified by the mask.
        """
        mask = self._convert_indexing_to_mask(selection)

        new_xyz = self.xyz[mask].copy()
        new_color = self.color[mask].copy() if self.color is not None else None
        new_normals = self.normals[mask].copy() if self.normals is not None else None
        # new_sf = {k: v[indices].copy() for k, v in self.scalar_fields.items()}
        new_sf = self.scalar_fields[mask]
        new_gcs = self.global_coordinate_shift.copy() if self.global_coordinate_shift is not None else None
        new_origin = self.spherical_coordinates_origin.copy()
        new_pcd = PointCloudData(new_xyz, color=new_color, normals=new_normals,
                                 scalar_fields=new_sf, global_coordinate_shift=new_gcs,
                                 spherical_coordinates_origin=new_origin,
                                 _global_shift_already_applied=True,
                                 _spherical_coordinates_represented_0_to_2pi=self._spherical_coordinates_represented_0_to_2pi)
        if self._spherical_coordinates_calculated:
            object.__setattr__(new_pcd, "spherical_coordinates", self.spherical_coordinates[mask].copy())
            object.__setattr__(new_pcd, "_spherical_coordinates_calculated", True)
        return new_pcd

    def extract(self, selection: NDArray[np.bool_] | NDArray[np.int_] | slice | list[int]) -> Self:
        mask = self._convert_indexing_to_mask(selection)
        new_pcd = self.sample(mask)
        self.reduce(~mask)
        return new_pcd

    def copy(self) -> Self:
        """Creates a deep copy of the point cloud."""
        mask = np.ones(self.xyz.shape[0], dtype=np.bool_)
        return self.sample(mask)

    def transform(self, transformation_matrix: NDArray[np.floating]) -> None:
        """
        Applies a 4x4 transformation matrix to the point cloud.
        """
        points_h = np.hstack((self.xyz, np.ones((self.nbPoints, 1), dtype=self.xyz.dtype)))
        origin_h = np.hstack((self.spherical_coordinates_origin, np.array([1], dtype=self.xyz.dtype)))
        combined = np.vstack((points_h, origin_h)).T
        transformed = transformation_matrix @ combined
        w = transformed[-1, :]
        transformed = np.where(w != 0, transformed[:-1, :] / w, transformed[:-1, :]).T
        new_xyz = transformed[:-1, :].astype(self.xyz.dtype)
        object.__setattr__(self, "xyz", new_xyz)
        object.__setattr__(self, "spherical_coordinates_origin", transformed[-1, :])
        if self._spherical_coordinates_calculated:
            object.__setattr__(self, "_spherical_coordinates_calculated", False)
            if "spherical_coordinates" in self.__dict__:
                del self.__dict__["spherical_coordinates"]

    def change_spherical_coordinates_origin(self, new_origin: NDArray[np.float_]) -> None:
        """
        Changes the origin for spherical coordinate calculations.
        """
        assert new_origin.shape == (3,), "new_origin must be a 3-element array"
        object.__setattr__(self, "spherical_coordinates_origin", new_origin)
        if self._spherical_coordinates_calculated:
            object.__setattr__(self, "_spherical_coordinates_calculated", False)
            if "spherical_coordinates" in self.__dict__:
                del self.__dict__["spherical_coordinates"]

    def to_o3d(self) -> o3d.geometry.PointCloud:
        """
        Converts the point cloud to an Open3D PointCloud.
        """
        pcd_o3d = o3d.geometry.PointCloud()
        pts = (self.xyz + self.global_coordinate_shift).astype(
            np.float32) if self.global_coordinate_shift is not None else self.xyz
        pcd_o3d.points = o3d.utility.Vector3dVector(pts)
        return pcd_o3d

    @classmethod
    def from_o3d(cls, pcd_o3d: o3d.geometry.PointCloud,
                 scan_center: Optional[NDArray[np.float_]] = None) -> Self:
        """
        Creates a PointCloudData instance from an Open3D PointCloud.
        """
        return cls(np.asarray(pcd_o3d.points), spherical_coordinates_origin=scan_center)

    @classmethod
    def from_spherical_coordinates(cls, spherical_coords: NDArray[np.floating],
                                   scalar_fields: Optional[Dict[str, NDArray]] = None,
                                   spherical_coordinates_origin: Optional[
                                       NDArray[np.float_]] = None) -> Self:
        """
        Creates a PointCloudData instance from spherical coordinates (range, elevation, azimuth).
        """
        xyz = np.zeros((spherical_coords.shape[0], 3), dtype=np.float32)
        xyz[:, 0] = spherical_coords[:, 0] * np.sin(spherical_coords[:, 1]) * np.cos(spherical_coords[:, 2])
        xyz[:, 1] = -spherical_coords[:, 0] * np.sin(spherical_coords[:, 1]) * np.sin(spherical_coords[:, 2])
        xyz[:, 2] = spherical_coords[:, 0] * np.cos(spherical_coords[:, 1])
        if spherical_coordinates_origin is not None:
            xyz += spherical_coordinates_origin
        return cls(xyz, scalar_fields=scalar_fields, spherical_coordinates_origin=spherical_coordinates_origin)

    @classmethod
    def from_range_image(cls, range_data: NDArray[np.floating], fov,
                         scalar_fields: Optional[Dict[str, NDArray]] = None,
                         spherical_coordinates_origin: Optional[NDArray[np.float_]] = None) -> Self:
        """
        Creates a PointCloudData instance from a range image.
        (Assumes fov provides elevation_min, elevation_max, horizontal_min, and horizontal_max attributes.)
        """
        resolution = range_data.shape
        elevation_range = np.linspace(fov.elevation_min, fov.elevation_max, num=resolution[0], endpoint=True)
        horizontal_range = np.linspace(fov.horizontal_min, fov.horizontal_max, num=resolution[1], endpoint=True)
        elevation_mesh, horizontal_mesh = np.meshgrid(elevation_range, horizontal_range, indexing="ij")
        ranges = range_data.flatten()
        elevations = elevation_mesh.flatten()
        horizontals = horizontal_mesh.flatten()
        spherical_coordinates = np.vstack((ranges, elevations, horizontals)).T
        spherical_coordinates = spherical_coordinates[~np.isnan(ranges), :]
        if scalar_fields:
            for key, sf in scalar_fields.items():
                scalar_fields[key] = sf.flatten()[~np.isnan(ranges)]
        return cls.from_spherical_coordinates(spherical_coordinates, scalar_fields, spherical_coordinates_origin)

    @classmethod
    def merge_pcd(cls, pcd_list: Iterable[Self]) -> Self:
        """
        Merges multiple PointCloudData instances into one.
        """
        from collections import defaultdict
        all_xyz = []
        all_color = []
        all_normals = []
        scalar_fields = defaultdict(list)
        global_shifts = []
        origins = []

        for idx, pcd in enumerate(pcd_list):
            all_xyz.append(pcd.xyz)
            if pcd.color is not None:
                all_color.append(pcd.color)
            if pcd.normals is not None:
                all_normals.append(pcd.normals)
            for key, arr in pcd.scalar_fields.items():
                scalar_fields[key].append(arr)
            scalar_fields.setdefault("point_cloud_merge", []).append(
                np.ones((pcd.xyz.shape[0],), dtype=np.uint8) * (idx + 1))
            global_shifts.append(pcd.global_coordinate_shift)
            origins.append(pcd.spherical_coordinates_origin)

        merged_xyz = np.vstack(all_xyz)
        merged_color = np.vstack(all_color) if all_color else None
        merged_normals = np.vstack(all_normals) if all_normals else None
        merged_sf = {key: np.hstack(arr_list) for key, arr_list in scalar_fields.items()}
        gcs = global_shifts[0]
        origin = origins[0]
        return cls(merged_xyz, color=merged_color, normals=merged_normals,
                   scalar_fields=merged_sf, global_coordinate_shift=gcs,
                   spherical_coordinates_origin=origin, _global_shift_already_applied=(gcs is not None))

    @staticmethod
    def _needs_global_shift(xyz: NDArray[np.float_], decimal_magnitude: int = 4) -> bool:
        """Determines if coordinates require a global shift."""
        return np.any(np.abs(xyz) >= 10 ** decimal_magnitude)

    @staticmethod
    def _calculate_optimal_global_shift(xyz: NDArray[np.float_], decimal_magnitude: int = 4) -> NDArray[np.float_]:
        """Calculates an optimal median-based global shift."""
        return np.median(np.round(xyz, decimals=-(decimal_magnitude - 1)), axis=0)
