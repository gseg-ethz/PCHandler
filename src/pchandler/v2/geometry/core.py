"""
Core module for pchandler.geometry.

Defines the PointCloudData class and its essential methods.
"""

import gc
import logging
import sys
from dataclasses import KW_ONLY, InitVar, dataclass, field
from functools import cached_property
from typing import Dict, Iterable, Optional

if sys.version[0] == 3 and sys.version_info[1] >= 11:
    from typing import Self
else:
    from typing_extensions import Self

import numpy as np
import open3d as o3d
from numpy.typing import NDArray

from ..fov import FoV
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
    global_coordinate_shift : Optional[NDArray[np.float64]]
        A (3,) array applied to center the coordinates.
    spherical_coordinates_origin : NDArray[np.float64]
        A (3,) array defining the origin for spherical coordinate calculations.
    """

    xyz: NDArray[np.float32]
    _: KW_ONLY
    scalar_fields: ScalarFieldManager = field(default_factory=ScalarFieldManager)
    color: Optional[NDArray[np.uint8]] = None
    normals: Optional[NDArray[np.float32]] = None
    global_coordinate_shift: Optional[NDArray[np.float64]] = None
    spherical_coordinates_origin: Optional[NDArray[np.float64]] = None
    _spherical_coordinates_calculated: bool = False
    _global_shift_already_applied: InitVar[bool] = False

    def __post_init__(self, _global_shift_already_applied: bool) -> None:
        """
        Validates and processes input data after object initialization.

        Parameters
        ----------
        _global_shift_already_applied : bool
            Indicates whether the global coordinate shift has already been applied to the `xyz` coordinates prior.
        """
        # Handle scalar_field conversion from dict to ScalarFieldManager if necessary
        if isinstance(self.scalar_fields, dict):
            sfm = ScalarFieldManager()
            for sf_id, sf in self.scalar_fields.items():
                sfm.create_field(sf_id, sf)
            object.__setattr__(self, "scalar_fields", sfm)

        if self.scalar_fields is None or self.scalar_fields.shape[1] == 0:
            object.__setattr__(self, "scalar_fields", ScalarFieldManager(expected_length=self.nbPoints))

        self._validate_internal_state()

        if self.spherical_coordinates_origin is None:
            object.__setattr__(self, "spherical_coordinates_origin", np.zeros((3,), dtype=np.float64))

        # Apply a global coordinate shift if needed.
        if self.global_coordinate_shift is None and self._needs_global_shift(self.xyz):
            shift = self._calculate_optimal_global_shift(self.xyz)
            object.__setattr__(self, "global_coordinate_shift", shift)
        if self.global_coordinate_shift is not None and not _global_shift_already_applied:
            shifted_xyz = (self.xyz - self.global_coordinate_shift).astype(np.float32)
            object.__setattr__(self, "xyz", shifted_xyz)
        else:
            object.__setattr__(self, "xyz", self.xyz.astype(np.float32))

        # Adjust the spherical coordinates origin when a global shift is applied.
        if self.global_coordinate_shift is not None:
            new_origin = self.spherical_coordinates_origin - self.global_coordinate_shift
            object.__setattr__(self, "spherical_coordinates_origin", new_origin)

        logger.info(f"PCD created with {self.xyz.shape[0]:_d} number of points")

    def _validate_internal_state(self):
        # Validate inputs.
        if not isinstance(self.xyz, np.ndarray):
            raise TypeError(f"xyz must be a numpy array. Got type {type(self.xyz)}.")

        if self.color is not None and not isinstance(self.color, np.ndarray):
            raise TypeError(f"color must be a numpy array or None. Got type {type(self.color)}")

        if self.normals is not None and not isinstance(self.normals, np.ndarray):
            raise TypeError(f"normals must be a numpy array or None. Got type {type(self.normals)}")

        if self.global_coordinate_shift is not None and not isinstance(self.global_coordinate_shift, np.ndarray):
            raise TypeError(
                f"global_coordinate_shift must be a numpy array or None. "
                f"Got type {type(self.global_coordinate_shift)}"
            )

        if self.spherical_coordinates_origin is not None and not isinstance(
            self.spherical_coordinates_origin, np.ndarray
        ):
            raise TypeError(
                f"spherical_coordinates_origin must be a numpy array or None. "
                f"Got type {type(self.spherical_coordinates_origin)}"
            )

        if not isinstance(self.scalar_fields, ScalarFieldManager):
            raise TypeError(
                f"scalar_fields must be a ScalarFieldManager or a dict. Got type {type(self.scalar_fields)}"
            )

        # Check dimensions.
        if self.xyz.ndim != 2 or self.xyz.shape[1] != 3:
            raise ValueError(f"xyz must have shape (N, 3). Got {self.xyz.shape}")

        if self.color is not None and self.color.shape != (self.xyz.shape[0], 3):
            raise ValueError(f"color must have shape {self.xyz.shape}. Got shape {self.color.shape}")

        if self.normals is not None and self.normals.shape != (self.xyz.shape[0], 3):
            raise ValueError(f"normals must have shape {self.xyz.shape}. Got shape {self.normals.shape}")

        if self.scalar_fields.shape[1] != self.xyz.shape[0] and self.scalar_fields.shape[1] is not None:
            raise ValueError(f"Scalar fields must have length equal to the number of points (N)")

        if self.spherical_coordinates_origin is not None and self.spherical_coordinates_origin.shape != (3,):
            raise ValueError(
                f"spherical_coordinates_origin must be (3,). " f"Got shape {self.spherical_coordinates_origin.shape}"
            )

        if self.global_coordinate_shift is not None and self.global_coordinate_shift.shape != (3,):
            raise ValueError(f"global_coordinate_shift must be (3,). Got shape {self.global_coordinate_shift.shape}")

    @property
    def nbPoints(self) -> int:
        """
        Returns the number of points in the point cloud.

        Returns
        -------
        int
            The number of points (rows) in the `xyz` attribute.
        """
        return self.xyz.shape[0]

    @cached_property
    def spherical_coordinates(self) -> NDArray[np.float32]:
        """
        Calculates and caches the spherical coordinates of the points.

        Returns
        -------
        NDArray[np.float32]
            An (N x 3) array of spherical coordinates (range, elevation, azimuth).
        """
        if self.nbPoints == 0:
            object.__setattr__(self, "_spherical_coordinates_calculated", True)
            return np.empty_like(self.xyz, dtype=np.float32)

        xyz_shifted = self.xyz - self.spherical_coordinates_origin

        sph = np.zeros_like(self.xyz, dtype=np.float32)
        xy_sq = xyz_shifted[:, 0] ** 2 + xyz_shifted[:, 1] ** 2
        sph[:, 0] = np.sqrt(xy_sq + xyz_shifted[:, 2] ** 2)  # range
        sph[:, 1] = np.arctan2(np.sqrt(xy_sq), xyz_shifted[:, 2])  # elevation (defined from z-axis downward)
        sph[:, 2] = -np.arctan2(xyz_shifted[:, 1], xyz_shifted[:, 0])  # azimuth

        # # Check for continuous representation using a randomly sampled subset of 100 points
        # if self._spherical_coordinates_represented_0_to_2pi is None:
        #     hz_sampled = sph[np.random.choice(self.nbPoints, size=100, replace=False), 2] if self.nbPoints > 100 else sph[:, 2]
        #
        #     # Create a copy with negative angles shifted to [0, 2pi] and check which extent is smaller
        #     hz_adjusted = np.where(hz_sampled < 0, hz_sampled + 2 * np.pi, hz_sampled)
        #
        #     object.__setattr__(self, "_spherical_coordinates_represented_0_to_2pi",
        #                        np.ptp(hz_adjusted) < np.ptp(hz_sampled))
        #
        # if self._spherical_coordinates_represented_0_to_2pi:
        #     sph[:, 2] = np.where(sph[:, 2] < 0, sph[:, 2] + 2 * np.pi, sph[:, 2])

        object.__setattr__(self, "_spherical_coordinates_calculated", True)
        return sph

    @property
    def fov(self) -> FoV:
        """
        Returns
        -------
            FoV: Field of View object based on spherical coordinates.
        """
        return FoV(
            horizontal_min=self.spherical_coordinates[:, 2].min(),
            horizontal_max=self.spherical_coordinates[:, 2].max(),
            elevation_min=self.spherical_coordinates[:, 1].min(),
            elevation_max=self.spherical_coordinates[:, 1].max(),
            unit="rad",
        )

    def __repr__(self) -> str:
        nbPoints = self.nbPoints
        scalar_field_keys = [self.scalar_fields.keys()]
        return f"PointCloudData(): {self.__dict__}"

    def __str__(self) -> str:
        return f"PointCloudData with {self.nbPoints} points"

    def __getitem__(self, item):
        if isinstance(item, slice):
            indices = np.arange(self.nbPoints)[item]
        else:
            indices = item
        return self.sample(indices)

    def _convert_indexing_to_mask(
        self, selection: NDArray[np.bool_] | NDArray[np.int_] | slice | list[int]
    ) -> NDArray[np.bool_]:
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

    def set_color(self, color: NDArray[np.uint8] | NDArray[np.floating]) -> None:
        if color.shape != (self.nbPoints, 3):
            raise ValueError(f"Color shape must be ({self.nbPoints},3), but got {color.shape}")

        if np.issubdtype(color.dtype, np.floating):
            if not (np.iinfo(np.uint8).min <= np.min(color) <= np.max(color) <= np.iinfo(np.uint8).max):
                raise ValueError(f"Color Values must be between {np.iinfo(np.uint8).min} and {np.iinfo(np.uint8).max}.")
            if np.max(color) <= 1.0:
                color = (color * np.iinfo(np.uint8).max).astype(np.uint8)
                logger.debug(
                    f"Scaling color values from 0 to 1 to {np.iinfo(np.uint8).min} and " f"{np.iinfo(np.uint8).max}."
                )
        object.__setattr__(self, "_color", color)

    def delete_color(self):
        object.__setattr__(self, "_color", None)

    def reduce(self, selection: NDArray[np.bool_] | NDArray[np.integer] | slice | list[int]) -> None:
        """
        Reduces the point cloud to only the points specified by the selection.

        Parameters
        ----------
        selection : NDArray[np.bool_] | NDArray[np.int_] | slice | list[int]
            A boolean or integer array specifying the points to retain.

        Returns
        -------
        PointCloudData
            The modified point cloud instance with only the selected points.
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
        Creates a copy of the point cloud containing only the points specified by the selection.

        Parameters
        ----------
        selection : NDArray[np.bool_] | NDArray[np.int_] | slice | list[int]
            A boolean or integer array specifying the points to copy.

        Returns
        -------
        PointCloudData
            A new point cloud instance containing the selected points.
        """
        mask = self._convert_indexing_to_mask(selection)

        new_xyz = self.xyz[mask].copy()
        new_color = self.color[mask].copy() if self.color is not None else None
        new_normals = self.normals[mask].copy() if self.normals is not None else None
        # new_sf = {k: v[indices].copy() for k, v in self.scalar_fields.items()}
        new_sf = self.scalar_fields[mask]
        new_gcs = self.global_coordinate_shift.copy() if self.global_coordinate_shift is not None else None
        new_origin = self.spherical_coordinates_origin.copy()
        new_pcd = PointCloudData(
            new_xyz,
            color=new_color,
            normals=new_normals,
            scalar_fields=new_sf,
            global_coordinate_shift=new_gcs,
            spherical_coordinates_origin=new_origin,
            _global_shift_already_applied=True,
        )
        if self._spherical_coordinates_calculated:
            object.__setattr__(new_pcd, "spherical_coordinates", self.spherical_coordinates[mask].copy())
            object.__setattr__(new_pcd, "_spherical_coordinates_calculated", True)
        return new_pcd

    def extract(self, selection: NDArray[np.bool_] | NDArray[np.int_] | slice | list[int]) -> Self:
        """
        Creates a new point cloud by extracting the selection.

        Parameters
        ----------
        selection : NDArray[np.bool_] | NDArray[np.int_] | slice | list[int]
            A boolean or integer array specifying the points to extract.

        Returns
        -------
        PointCloudData
            A new point cloud instance containing the selected points.
        """
        mask = self._convert_indexing_to_mask(selection)
        new_pcd = self.sample(mask)
        self.reduce(~mask)
        return new_pcd

    def copy(self) -> Self:
        """
        Creates a copy of the current point cloud.

        Returns
        -------
        PointCloudData
            A new instance with the same attributes as the original.
        """
        mask = np.ones(self.xyz.shape[0], dtype=np.bool_)
        return self.sample(mask)

    def transform(self, transformation_matrix: NDArray[np.floating]) -> None:
        """
        Applies a transformation matrix to the point cloud.

        Parameters
        ----------
        transformation_matrix : NDArray[np.floating]
            A (4 x 4) transformation matrix.
        """
        assert transformation_matrix.shape == (4, 4)

        xyz_sco = np.vstack((self.xyz, self.spherical_coordinates_origin[np.newaxis, :]))
        if self.global_coordinate_shift is None:
            xyz_homogeneous = np.hstack((xyz_sco, np.ones((self.nbPoints + 1, 1), dtype=self.xyz.dtype))).transpose()
        else:
            xyz_homogeneous = np.hstack(
                (xyz_sco + self.global_coordinate_shift, np.ones((self.nbPoints + 1, 1), dtype=np.float64))
            ).transpose()

        transformed_xyz_homogeneous = transformation_matrix @ xyz_homogeneous
        w = transformed_xyz_homogeneous[-1]
        transformed_xyz = np.where(
            w != 0, transformed_xyz_homogeneous[:-1] / w, transformed_xyz_homogeneous[:-1]
        ).transpose()

        # Check if global coordinate shift has become unnecessary
        if self.global_coordinate_shift is not None and not self._needs_global_shift(transformed_xyz[:-1, :]):
            object.__setattr__(self, "global_coordinate_shift", None)
        # Check if old global shift still works, if so apply
        elif self.global_coordinate_shift is not None and not self._needs_global_shift(
            transformed_xyz[:-1, :] - self.global_coordinate_shift
        ):
            transformed_xyz = transformed_xyz - self.global_coordinate_shift
            # object.__setattr__(self, "xyz", transformed_xyz.astype(self.xyz.dtype, casting="same_kind"))
        # Old global shift doesn't work
        elif self.global_coordinate_shift is not None:
            object.__setattr__(
                self, "global_coordinate_shift", self._calculate_optimal_global_shift(transformed_xyz[:-1, :])
            )
            transformed_xyz = transformed_xyz - self.global_coordinate_shift
            # object.__setattr__(self, "xyz", transformed_xyz.astype(self.xyz.dtype, casting="same_kind"))

        elif self._needs_global_shift(transformed_xyz[:-1, :]):
            object.__setattr__(
                self, "global_coordinate_shift", self._calculate_optimal_global_shift(transformed_xyz[:-1, :])
            )
            transformed_xyz = transformed_xyz - self.global_coordinate_shift

        object.__setattr__(self, "xyz", transformed_xyz[:-1, :].astype(self.xyz.dtype, casting="same_kind"))
        object.__setattr__(self, "spherical_coordinates_origin", transformed_xyz[-1, :])
        if self._spherical_coordinates_calculated:
            object.__delattr__(self, "spherical_coordinates")
            object.__setattr__(self, "_spherical_coordinates_calculated", False)
        return

    def change_spherical_coordinates_origin(self, new_origin: NDArray[np.float64]) -> None:
        """
        Changes the origin used for spherical coordinate calculations.

        Parameters
        ----------
        new_origin : NDArray[np.float64]
            A (3,) array specifying the new origin for spherical coordinate calculations.
        """
        if new_origin is not isinstance(np.ndarray):
            raise TypeError(f"new_origin must be a numpy array. Got {type(new_origin)}")
        if new_origin.shape != (3,):
            raise ValueError(f"new_origin must be a numpy array of shape (3,). Got {new_origin.shape}")

        if self.global_coordinate_shift is not None:
            new_origin -= self.global_coordinate_shift

        object.__setattr__(self, "spherical_coordinates_origin", new_origin)
        if self._spherical_coordinates_calculated:
            object.__delattr__(self, "spherical_coordinates")
            object.__setattr__(self, "_spherical_coordinates_calculated", False)

    def to_o3d(self) -> o3d.geometry.PointCloud:
        """
        Converts the point cloud to an Open3D `PointCloud` object.

        Returns
        -------
        o3d.geometry.PointCloud
            An Open3D representation of the point cloud.
        """
        pcd_o3d = o3d.geometry.PointCloud()
        if self.global_coordinate_shift is None:
            pcd_o3d.points = o3d.utility.Vector3dVector(self.xyz)
        else:
            pcd_o3d.points = o3d.utility.Vector3dVector((self.xyz + self.global_coordinate_shift).astype(np.float64))
        return pcd_o3d

    @classmethod
    def from_o3d(cls, pcd_o3d: o3d.geometry.PointCloud, scan_center: Optional[NDArray[np.float64]] = None) -> Self:
        """
        Creates a `PointCloudData` instance from an Open3D `PointCloud`.

        Parameters
        ----------
        pcd_o3d : o3d.geometry.PointCloud
            An Open3D `PointCloud` object.
        scan_center : np.ndarray, optional
            The scan center for spherical coordinate calculations.

        Returns
        -------
        PointCloudData
            A new instance of the `PointCloudData` class.
        """
        return cls(np.asarray(pcd_o3d.points), spherical_coordinates_origin=scan_center)

    @classmethod
    def from_spherical_coordinates(
        cls,
        spherical_coordinates: NDArray[np.floating],
        scalar_fields: Optional[ScalarFieldManager | dict[str, NDArray]] = None,
        spherical_coordinates_origin: Optional[NDArray[np.float64]] = None,
        **kwargs
    ) -> Self:
        """
        Creates a `PointCloudData` instance from spherical coordinates.

        Parameters
        ----------
        spherical_coordinates : NDArray[np.floating]
            An (N x 3) array of spherical coordinates (range, elevation, azimuth).
        scalar_fields : ScalarFieldManager, optional
            Scalar fields associated with the spherical coordinates.
        spherical_coordinates_origin : NDArray[np.float64], optional
            The origin for spherical coordinate calculations.

        Returns
        -------
        PointCloudData
            A new instance of the `PointCloudData` class.
        """
        xyz = np.zeros((spherical_coordinates.shape[0], 3), dtype=spherical_coordinates.dtype)
        xyz[:, 0] = spherical_coordinates[:, 0] * np.sin(spherical_coordinates[:, 1]) * np.cos(spherical_coordinates[:, 2])
        xyz[:, 1] = -spherical_coordinates[:, 0] * np.sin(spherical_coordinates[:, 1]) * np.sin(spherical_coordinates[:, 2])
        xyz[:, 2] = spherical_coordinates[:, 0] * np.cos(spherical_coordinates[:, 1])
        if spherical_coordinates_origin is not None:
            xyz += spherical_coordinates_origin
        return cls(xyz, scalar_fields=scalar_fields, spherical_coordinates_origin=spherical_coordinates_origin, **kwargs)

    @classmethod
    def from_range_image(
        cls,
        range_data: NDArray[np.floating],
        fov: FoV,
        scalar_fields: Optional[dict[str, NDArray[np.generic]] | ScalarFieldManager] = None,
        spherical_coordinates_origin: Optional[NDArray[np.float64]] = None,
    ) -> Self:
        """
        Creates a `PointCloudData` instance from a range image.

        Parameters
        ----------
        range_data : NDArray[np.floating]
            A 2D array representing the range values.
        fov : FoV
            The field of view defining the angular limits of the range image.
        scalar_fields : dict[str, NDArray[np.generic]] | ScalarFieldManager, optional
            Scalar fields corresponding to the range data.
        spherical_coordinates_origin : NDArray[np.float64], optional
            The origin for spherical coordinate calculations.

        Returns
        -------
        PointCloudData
            A new instance of the `PointCloudData` class.
        """
        sfm = ScalarFieldManager() if scalar_fields is None else scalar_fields
        if not isinstance(sfm, ScalarFieldManager) and scalar_fields is not None:
            sfm = ScalarFieldManager()
            for sf_id, sf in scalar_fields.items():
                sfm.create_field(sf_id, sf.flatten())

        resolution = range_data.shape
        elevation_range = np.linspace(
            fov.elevation_min, fov.elevation_max, num=resolution[0], endpoint=True, dtype=np.float32
        )
        horizontal_range = np.linspace(
            fov.horizontal_min, fov.horizontal_max, num=resolution[1], endpoint=True, dtype=np.float32
        )

        elevation_mesh, horizontal_mesh = np.meshgrid(elevation_range, horizontal_range, indexing="ij")

        ranges = range_data.flatten()
        elevations = elevation_mesh.flatten()
        horizontals = horizontal_mesh.flatten()

        spherical_coordinates = np.vstack((ranges, elevations, horizontals)).T
        spherical_coordinates = spherical_coordinates[~np.isnan(ranges), :]

        sfm_reduced = sfm[~np.isnan(ranges)]

        return cls.from_spherical_coordinates(spherical_coordinates, sfm_reduced, spherical_coordinates_origin)

    @classmethod
    def merge_pcd(cls, pcds: Iterable[Self]) -> Self:
        """
        Merges multiple `PointCloudData` instances into a single point cloud.

        Parameters
        ----------
        pcds : Iterable[PointCloudData]
            A list or iterable of `PointCloudData` instances to merge.

        Returns
        -------
        PointCloudData
            A new instance containing the merged point cloud data.
        """
        xyz = []
        color = []
        normals = []
        sfms = []
        global_coordinate_shift = []
        spherical_coordinates_origin = []
        merge_id = []

        # Build lists of all elements
        for i, pcd in enumerate(pcds):
            xyz.append(pcd.xyz)
            color.append(pcd.color)
            normals.append(pcd.normals)
            sfms.append(pcd.scalar_fields)
            merge_id.append(np.ones((pcd.xyz.shape[0],), dtype=np.uint8) * (i + 1))
            global_coordinate_shift.append(pcd.global_coordinate_shift)
            spherical_coordinates_origin.append(pcd.spherical_coordinates_origin)

        scalar_fields = ScalarFieldManager.merge(sfms)
        scalar_fields.create_field("merge_id", np.concatenate(merge_id))

        if any(val is None for val in color):
            color = None

        if any(val is None for val in normals):
            normals = None

        # Check if all have same global coordinate shift, if not add gcs back
        gcs_pairs = zip(global_coordinate_shift[:-1], global_coordinate_shift[1:])
        if all(map(lambda gcs_pair: np.array_equal(*gcs_pair), gcs_pairs)):
            gcs = global_coordinate_shift[0]
            xyz_np = np.vstack(tuple(xyz))
            del xyz
            gc.collect()
        else:
            gcs = None
            xyz_64 = [x.astype(np.float64) for x in xyz]
            gcs_None_removed = [
                (
                    np.zeros(
                        3,
                    )
                    if g is None
                    else g
                )
                for g in global_coordinate_shift
            ]
            xyz_np = np.vstack(tuple(map(lambda x: np.add(*x), zip(xyz_64, gcs_None_removed))))
            del xyz, xyz_64
            gc.collect()

        color_np = np.vstack(tuple(color)) if color is not None else None  #
        del color
        gc.collect()

        normals_np = np.vstack(tuple(normals)) if normals is not None else None
        del normals
        gc.collect()

        sco_pairs = zip(spherical_coordinates_origin[:-1], spherical_coordinates_origin[1:])

        # Check if all spherical_coordinates_origin are equal and represented in the same system
        scs = None
        if all(map(lambda sco_pair: np.array_equal(*sco_pair), sco_pairs)):
            sco = spherical_coordinates_origin[0]
            if all([pcd._spherical_coordinates_calculated for pcd in pcds]):
                scs = np.vstack([pcd.spherical_coordinates for pcd in pcds])
        else:
            sco = None

        new_pcd = cls(
            xyz=xyz_np,
            color=color_np,
            normals=normals_np,
            scalar_fields=scalar_fields,
            global_coordinate_shift=gcs,
            _global_shift_already_applied=(gcs is not None),
            spherical_coordinates_origin=sco,
        )

        if scs is not None:
            object.__setattr__(new_pcd, "spherical_coordinates", scs)
            object.__setattr__(new_pcd, "_spherical_coordinates_calculated", True)

        return new_pcd

    @staticmethod
    def _needs_global_shift(xyz: NDArray[np.float64], decimal_magnitude: int = 4) -> bool:
        """
        Determines if a global coordinate shift is necessary.

        Parameters
        ----------
        xyz : NDArray[np.float64]
            The array of (N x 3) coordinates to check.
        decimal_magnitude : int, default=4
            The threshold magnitude for deciding if a shift is needed.

        Returns
        -------
        bool
            True if a global shift is necessary; otherwise, False.

        Todo
        ----
        Check if the span is too large to apply a shift --> Would consequently need a float64 representation!
        """
        return np.any(np.abs(xyz) >= 10**decimal_magnitude)

    @staticmethod
    def _calculate_optimal_global_shift(xyz: NDArray[np.float64], decimal_magnitude: int = 4) -> NDArray[np.float64]:
        """
        Calculates an optimal global shift based on the median of the coordinates.

        Parameters
        ----------
        xyz : NDArray[np.float64]
            The array of (N x 3) coordinates.
        decimal_magnitude : int, default=4
            The precision used to calculate the shift.

        Returns
        -------
        NDArray[np.float64]
            The calculated global shift as a (3,) array.
        """
        return np.median(np.round(xyz, decimals=-(decimal_magnitude - 1)), axis=0)
