"""
``pchandler.geometry``

This module provides a comprehensive set of tools for handling, manipulating, and processing 3D point cloud data.
It includes the `PointCloudData` class, which serves as a robust representation of point clouds, and a variety
of utility functions for tasks such as filtering, splitting, sampling, and merging point clouds.

The module integrates with popular libraries like Open3D, NumPy, Shapely, and cuSpatial for advanced
geometry processing, enabling GPU-accelerated workflows where supported.

Key Features:
-------------
- **PointCloudData Class**:
  - Represents 3D point clouds with attributes for coordinates, colors, normals, scalar fields, and more.
  - Provides methods for transformations, filtering, downsampling, and subsampling.
  - Supports conversion between Cartesian and spherical coordinate systems.

- **FoV and FoVTree Integration**:
  - Enables field-of-view-based splitting and extraction of point cloud data.
  - Compatible with hierarchical splitting for large datasets.

- **Advanced Filtering and Sampling**:
  - Filters points by range, angles, or polygonal boundaries.
  - Supports voxel grid downsampling and statistical outlier removal.

- **GPU Acceleration**:
  - Integrates cuSpatial and cuML for high-performance point cloud operations on supported GPUs.

Dependencies:
-------------
- ``numpy``: For numerical computations.
- ``open3d``: For 3D visualization and geometry operations.
- ``geopandas`` and ``shapely``: For working with polygons and geospatial data.
- ``alphashape``: For computing alpha shapes (convex hull approximations).
- ``joblib``: For parallel processing.
- ``cudf`` and ``cuspatial`` (optional): For GPU-accelerated operations.

Usage:
------
Example: Create a point cloud from spherical coordinates and apply a transformation:

.. code-block:: python

    import numpy as np
    from pchandler.geometry import PointCloudData

    # Generate spherical coordinates (range, elevation, azimuth)
    spherical_coords = np.array([[1.0, np.pi / 4, np.pi / 4],
                                  [2.0, np.pi / 6, np.pi / 3]])

    # Create a PointCloudData instance
    pcd = PointCloudData.from_spherical_coordinates(spherical_coords)

    # Apply a transformation matrix
    transformation = np.eye(4)  # Identity matrix for example
    pcd.transform(transformation)

    print(f"Transformed Point Cloud: {pcd.xyz}")

Example: Split a point cloud using a field-of-view tree:

.. code-block:: python

    from pchandler.fov import FoVTree
    from pchandler.geometry import split_pc_with_fov_tree

    # Create a FoVTree and split the point cloud
    split_pcds = split_pc_with_fov_tree(pcd, fov_tree=some_fov_tree)
"""


import sys

from collections import defaultdict
from dataclasses import dataclass, field, KW_ONLY, InitVar
from functools import cached_property
import gc
from itertools import compress, chain

from joblib import Parallel, delayed
from typing import Iterable, Callable, Any, Tuple, Optional
if sys.version[0] == 3 and sys.version_info[1] >= 11:
    from typing import Self
else:
    from typing_extensions import Self
import warnings

import alphashape
try:
    import cudf
    import cuspatial
    from cuml.neighbors import NearestNeighbors
    HAS_GPU_SUPPORT = True
except ImportError:
    HAS_GPU_SUPPORT = False
import geopandas as gpd
import numpy as np
from numpy.typing import NDArray
import open3d as o3d
from shapely.geometry import Polygon, MultiPolygon
from shapely.affinity import scale, translate

from pchandler.fov import FoV, FoVTree

# from pchandler.util import convert_angle, AngleUnit


# TODO: Implement global center shift and checks
# TODO: Add own init function
# TODO: Repair all that was broken by coordinate shift

@dataclass(frozen=True)
class PointCloudData:
    """
    Represents a 3D point cloud in a Cartesian coordinate system.

    Attributes
    ----------
    xyz : NDArray[np.float32]
        An (N x 3) float32 array containing the *x*, *y*, and *z* coordinates of the points.
    scalar_fields : dict[str, NDArray[np.generic]]
        A dictionary of named scalar fields, where each field is a 1D array of length N.
    color : Optional[NDArray[np.uint8]]
        An (N x 3) uint8 array containing the *r*, *g*, and *b* color values for each point.
    normals : Optional[NDArray[np.float32]]
        An (N x 3) array representing the normal vectors for each point.
    global_coordinate_shift : Optional[NDArray[np.float_]]
        A (3,) array specifying the global coordinate shift applied to the point cloud.
    spherical_coordinates_origin : NDArray[np.float_]
        A (3,) array specifying the origin for spherical coordinate calculations.
    _spherical_coordinates_calculated : bool
        A flag indicating if spherical coordinates have been calculated.
    _spherical_coordinates_represented_0_to_2pi : Optional[bool]
        A flag indicating whether spherical coordinates are represented in the range [0, 2π].
    """

    xyz: NDArray[np.float32]
    _: KW_ONLY
    scalar_fields: dict[str, NDArray[np.generic]] = field(default_factory=dict)
    color: Optional[NDArray[np.uint8]] = None
    normals: Optional[NDArray[np.float32]] = None
    global_coordinate_shift: Optional[NDArray[np.float_]] = None
    spherical_coordinates_origin: NDArray[np.float_] = None
    _spherical_coordinates_calculated: bool = False
    _spherical_coordinates_represented_0_to_2pi: Optional[bool] = None
    _global_shift_already_applied: InitVar[bool] = False

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

    def __post_init__(self, _global_shift_already_applied: bool) -> None:
        """
        Validates and processes input data after object initialization.

        Parameters
        ----------
        _global_shift_already_applied : bool
            Indicates whether the global coordinate shift has already been applied to the `xyz` coordinates prior.
        """
        if self.spherical_coordinates_origin is None:
            object.__setattr__(self, "spherical_coordinates_origin", np.zeros((3,), dtype=np.float_))

        assert isinstance(self.xyz, np.ndarray)
        assert self.color is None or isinstance(self.color, np.ndarray)
        assert self.normals is None or isinstance(self.normals, np.ndarray)
        assert isinstance(self.scalar_fields, dict)
        for key, value in self.scalar_fields.items():
            assert isinstance(key, str)
            assert isinstance(value, np.ndarray)

        # Check dimensions
        assert self.xyz.shape[1] == 3
        assert self.color is None or self.color.shape == (self.nbPoints, 3,)
        assert self.normals is None or self.normals.shape == (self.nbPoints, 3,)
        for sf in self.scalar_fields.values():
            assert sf.shape == (self.nbPoints,)
        assert self.spherical_coordinates_origin.shape == (3,)
        assert self.global_coordinate_shift is None or self.global_coordinate_shift.shape == (3,)

        if self.global_coordinate_shift is None and self.__check_for_need_of_global_shift(self.xyz):
            object.__setattr__(self, 'global_coordinate_shift', self.__calculate_optimal_global_shift(self.xyz))

        if self.global_coordinate_shift is not None and not _global_shift_already_applied:
            object.__setattr__(self, 'xyz', (self.xyz - self.global_coordinate_shift).astype(np.float32, casting="same_kind"))
        else:
            object.__setattr__(self, "xyz", self.xyz.astype(np.float32, casting="same_kind"))

        if self.global_coordinate_shift is not None:
            self.change_spherical_coordinates_origin(self.spherical_coordinates_origin - self.global_coordinate_shift)
                # object.__setattr__(self, 'spherical_coordinates_origin', self.global_coordinate_shift)


       # if "scalar_Intensity" in scalar_fields:
       #     if np.nanmax(scalar_fields["scalar_Intensity"]) > 1.0 or np.nanmin(scalar_fields["scalar_Intensity"]) < 0.0:
       #         pass
       #     scalar_fields["scalar_Intensity"] = scalar_fields["scalar_Intensity"].astype(np.float32)

    @staticmethod
    def __check_for_need_of_global_shift(xyz: NDArray[np.float_], _decimal_magnitude: int = 4) -> bool:
        """
        Determines if a global coordinate shift is necessary.

        Parameters
        ----------
        xyz : NDArray[np.float_]
            The array of (N x 3) coordinates to check.
        _decimal_magnitude : int, default=4
            The threshold magnitude for deciding if a shift is needed.

        Returns
        -------
        bool
            True if a global shift is necessary; otherwise, False.
        """
        return any((np.abs(xyz) >= 10 ** _decimal_magnitude).flatten())

    @staticmethod
    def __calculate_optimal_global_shift(xyz: NDArray[np.float_], _decimal_magnitude: int = 4) -> NDArray[np.float_]:
        """
        Calculates an optimal global shift based on the median of the coordinates.

        Parameters
        ----------
        xyz : NDArray[np.float_]
            The array of (N x 3) coordinates.
        _decimal_magnitude : int, default=4
            The precision used to calculate the shift.

        Returns
        -------
        NDArray[np.float_]
            The calculated global shift as a (3,) array.
        """
        return np.median(np.round(xyz, decimals=-(_decimal_magnitude - 1)), axis=0)


    @classmethod
    def from_range_image(cls, range_data: NDArray[np.floating], fov: FoV, scalar_fields: dict[str, NDArray[np.generic]] = None,
                         spherical_coordinates_origin: NDArray[np.float_] = None) -> Self:
        """
        Creates a `PointCloudData` instance from a range image.

        Parameters
        ----------
        range_data : NDArray[np.floating]
            A 2D array representing the range values.
        fov : FoV
            The field of view defining the angular limits of the range image.
        scalar_fields : dict[str, NDArray[np.generic]], optional
            Scalar fields corresponding to the range data.
        spherical_coordinates_origin : NDArray[np.float_], optional
            The origin for spherical coordinate calculations.

        Returns
        -------
        PointCloudData
            A new instance of the `PointCloudData` class.
        """
        resolution = range_data.shape

        ## TODO: Check endpoint consistency throughout
        elevation_range = np.linspace(fov.elevation_min, fov.elevation_max, num=resolution[0], endpoint=True)
        horizontal_range = np.linspace(fov.horizontal_min, fov.horizontal_max, num=resolution[1], endpoint=True)

        elevation_mesh, horizontal_mesh = np.meshgrid(elevation_range, horizontal_range, indexing="ij")

        ranges = range_data.flatten()
        elevations = elevation_mesh.flatten()
        horizontals = horizontal_mesh.flatten()

        spherical_coordinates = np.vstack((ranges, elevations, horizontals)).T

        spherical_coordinates = spherical_coordinates[~np.isnan(ranges), :]

        for key, sf in scalar_fields.items():
            scalar_fields[key] = sf.flatten()[~np.isnan(ranges)]

        return cls.from_spherical_coordinates(spherical_coordinates, scalar_fields, spherical_coordinates_origin)

    @classmethod
    def from_spherical_coordinates(cls, spherical_coordinates: NDArray[np.floating],
                                   scalar_fields: dict[str, NDArray[np.generic]] = None,
                                   spherical_coordinates_origin:  NDArray[np.float_] = None) -> Self:
        """
        Creates a `PointCloudData` instance from spherical coordinates.

        Parameters
        ----------
        spherical_coordinates : NDArray[np.floating]
            An (N x 3) array of spherical coordinates (range, elevation, azimuth).
        scalar_fields : dict[str, NDArray[np.generic]], optional
            Scalar fields associated with the spherical coordinates.
        spherical_coordinates_origin : NDArray[np.float_], optional
            The origin for spherical coordinate calculations.

        Returns
        -------
        PointCloudData
            A new instance of the `PointCloudData` class.
        """
        assert spherical_coordinates_origin is None or spherical_coordinates_origin.shape == (3,)

        xyz = np.zeros((len(spherical_coordinates), 3))
        xyz[:, 0] = spherical_coordinates[:, 0] * np.sin(spherical_coordinates[:, 1]) * np.cos(spherical_coordinates[:, 2])
        xyz[:, 1] = - spherical_coordinates[:, 0] * np.sin(spherical_coordinates[:, 1]) * np.sin(spherical_coordinates[:, 2])
        xyz[:, 2] = spherical_coordinates[:, 0] * np.cos(spherical_coordinates[:, 1])

        if spherical_coordinates_origin is not None:
            xyz = xyz + spherical_coordinates_origin

        return cls(xyz, scalar_fields=scalar_fields, spherical_coordinates_origin=spherical_coordinates_origin)

    def copy(self) -> Self:
        """
        Creates a copy of the current point cloud.

        Returns
        -------
        PointCloudData
            A new instance with the same attributes as the original.
        """
        mask = np.ones(self.xyz.shape[0], dtype=bool)
        return self._copy_selection(mask)

    @cached_property
    def spherical_coordinates(self) -> NDArray[np.float32]:
        """
        Calculates and caches the spherical coordinates of the points.

        Returns
        -------
        NDArray[np.float32]
            An (N x 3) array of spherical coordinates (range, elevation, azimuth).
        """
        #
        # if self.global_coordinate_shift is not None:
        #     xyz = self.xyz - (self.spherical_coordinates_origin - self.global_coordinate_shift)
        # elif self.spherical_coordinates_origin is not None:
        #     xyz = self.xyz - self.spherical_coordinates_origin
        # else:
        #     xyz = self.xyz

        xyz_shifted = self.xyz - self.spherical_coordinates_origin

        if len(xyz_shifted) == 0:
            object.__setattr__(self, "_spherical_coordinates_calculated", True)
            object.__setattr__(self, "_spherical_coordinates_represented_0_to_2pi", False)
            return np.empty_like(xyz)

        spherical_coordinates = np.zeros(self.xyz.shape, dtype=np.float32)
        xy = xyz_shifted[:, 0] ** 2 + xyz_shifted[:, 1] ** 2
        spherical_coordinates[:, 0] = np.sqrt(xy + xyz_shifted[:, 2] ** 2)
        spherical_coordinates[:, 1] = np.arctan2(np.sqrt(xy), xyz_shifted[:, 2])  # for elevation angle defined from Z-axis down
        spherical_coordinates[:, 2] = - np.arctan2(xyz_shifted[:, 1], xyz_shifted[:, 0])

        if self._spherical_coordinates_represented_0_to_2pi is None:
            # Check for continuous representation
            hz_shifted = spherical_coordinates[:,2].copy()
            hz_shifted[hz_shifted < 0] = 2*np.pi + hz_shifted[hz_shifted < 0]
            extent_shifted = hz_shifted.max() - hz_shifted.min()
            extent = spherical_coordinates[:, 2].max() - spherical_coordinates[:, 2].min()
            if extent_shifted < extent:
                object.__setattr__(self, "_spherical_coordinates_represented_0_to_2pi", True)
            else:
                object.__setattr__(self, "_spherical_coordinates_represented_0_to_2pi", False)

        if self._spherical_coordinates_represented_0_to_2pi:
            hz_shifted = spherical_coordinates[:, 2].copy()
            hz_shifted[hz_shifted < 0] = 2 * np.pi + hz_shifted[hz_shifted < 0]
            spherical_coordinates[:, 2] = hz_shifted

        object.__setattr__(self, "_spherical_coordinates_calculated", True)
        return spherical_coordinates

    def __repr__(self) -> str:
        return f"Point cloud with {self.xyz.shape[0]:,d} point(s)"

    def _reduce_points_to(self, mask: NDArray[np.bool_|np.integer]) -> Self:
        """
        Reduces the point cloud to only the points specified by the mask.

        Parameters
        ----------
        mask : NDArray[np.bool_] or NDArray[np.integer]
            A boolean or integer array specifying the points to retain.

        Returns
        -------
        PointCloudData
            The modified point cloud instance with only the selected points.
        """
        assert mask.dtype != np.bool_ or mask.shape == (self.nbPoints,), (
            f"If mask.dtype is np.bool_, its shape must be {(self.nbPoints,)}, but got {mask.shape}"
        )

        object.__setattr__(self, "xyz", self.xyz[mask])
        if self.color is not None:
            object.__setattr__(self, "color", self.color[mask])
        if self.normals is not None:
            object.__setattr__(self, "normals", self.normals[mask])
        if self._spherical_coordinates_calculated:
            object.__setattr__(self, "spherical_coordinates", self.spherical_coordinates[mask])
        for sf_key in self.scalar_fields.keys():
            self.scalar_fields[sf_key] = self.scalar_fields[sf_key][mask]
        return self

    def _copy_selection(self, mask: NDArray[np.bool_|np.integer]) -> Self:
        """
        Creates a copy of the point cloud containing only the points specified by the mask.

        Parameters
        ----------
        mask : NDArray[np.bool_] or NDArray[np.integer]
            A boolean or integer array specifying the points to copy.

        Returns
        -------
        PointCloudData
            A new point cloud instance containing the selected points.
        """
        assert mask.dtype != np.bool_ or mask.shape == (self.nbPoints,), (
            f"If mask.dtype is np.bool_, its shape must be {(self.nbPoints,)}, but got {mask.shape}"
        )

        xyz = self.xyz[mask].copy()
        color = self.color[mask].copy() if self.color is not None else None
        normals = self.normals[mask].copy() if self.normals is not None else None
        scalar_fields = dict()
        for sf_key, sf_value in self.scalar_fields.items():
            scalar_fields[sf_key] = sf_value[mask].copy()
        global_coordinate_shift = self.global_coordinate_shift.copy() if self.global_coordinate_shift is not None else None
        spherical_coordinates_origin = self.spherical_coordinates_origin.copy()
        new_pcd = PointCloudData(
            xyz, color=color, normals=normals, scalar_fields=scalar_fields,
            global_coordinate_shift=global_coordinate_shift, spherical_coordinates_origin=spherical_coordinates_origin,
            _global_shift_already_applied=True,
            _spherical_coordinates_represented_0_to_2pi=self._spherical_coordinates_represented_0_to_2pi
        )
        if self._spherical_coordinates_calculated:
            object.__setattr__(new_pcd, "spherical_coordinates", self.spherical_coordinates[mask].copy())
            object.__setattr__(new_pcd, "_spherical_coordinates_calculated", True)

        return new_pcd

    # def _sample(self,sf_sample: str,
    #            sample_func: Callable[[np.ndarray], np.ndarray[Any, np.dtype[bool]]],
    #             in_place: bool = False) -> "PointCloudData" | None:

    # def apply_math_to_xyz(self, math_func: Callable) -> None:
    #     """
    #     Applies a mathematical transformation to the `xyz` coordinates.
    #
    #     Parameters
    #     ----------
    #     math_func : Callable
    #         A function that takes an (N x 3) array and returns a transformed array.
    #     """
    #     object.__setattr__(self, "xyz", math_func(self.xyz))

    def transform(self, transformation_matrix: NDArray[np.floating]) -> None:
        """
        Applies a transformation matrix to the point cloud.

        Parameters
        ----------
        transformation_matrix : NDArray[np.floating]
            A (4 x 4) transformation matrix.
        """
        xyz_sco = np.vstack((self.xyz, self.spherical_coordinates_origin[np.newaxis,:]))
        if self.global_coordinate_shift is None:
            xyz_homogeneous = np.hstack((xyz_sco, np.ones((self.nbPoints + 1, 1), dtype=self.xyz.dtype))).transpose()
        else:
            xyz_homogeneous = np.hstack((xyz_sco + self.global_coordinate_shift,
                                         np.ones((self.nbPoints + 1, 1), dtype=self.global_coordinate_shift.dtype))).transpose()
        transformed_xyz_homogeneous = transformation_matrix @ xyz_homogeneous
        w = transformed_xyz_homogeneous[-1]
        transformed_xyz = np.where(w != 0, transformed_xyz_homogeneous[:-1] / w, transformed_xyz_homogeneous[:-1]).transpose()

        # Check if global coordinate shift has become unnecessary
        if self.global_coordinate_shift is not None and not self.__check_for_need_of_global_shift(transformed_xyz[:-1,:]):
            object.__setattr__(self, "global_coordinate_shift", None)
        # Check if old global shift still works, if so apply
        elif self.global_coordinate_shift is not None and not self.__check_for_need_of_global_shift(transformed_xyz[:-1,:] - self.global_coordinate_shift):
            transformed_xyz = transformed_xyz - self.global_coordinate_shift
            # object.__setattr__(self, "xyz", transformed_xyz.astype(self.xyz.dtype, casting="same_kind"))
        # Old global shift doesn't work
        elif self.global_coordinate_shift is not None:
            object.__setattr__(self, "global_coordinate_shift", self.__calculate_optimal_global_shift(transformed_xyz[:-1,:]))
            transformed_xyz = transformed_xyz - self.global_coordinate_shift
            # object.__setattr__(self, "xyz", transformed_xyz.astype(self.xyz.dtype, casting="same_kind"))

        elif self.__check_for_need_of_global_shift(transformed_xyz[:-1,:]):
            object.__setattr__(self, "global_coordinate_shift", self.__calculate_optimal_global_shift(transformed_xyz[:-1,:]))
            transformed_xyz = transformed_xyz - self.global_coordinate_shift

        object.__setattr__(self, "xyz", transformed_xyz[:-1,:].astype(self.xyz.dtype, casting="same_kind"))
        object.__setattr__(self, "spherical_coordinates_origin", transformed_xyz[-1,:])
        if self._spherical_coordinates_calculated:
            object.__delattr__(self, "spherical_coordinates")
            object.__setattr__(self, "_spherical_coordinates_calculated", False)
        return


    # def filter(self, sf_filter: str, truth_func: Callable[[np.ndarray], np.ndarray[Any, np.dtype[bool]]]) -> None:
    def filter(self, sf_filter: str, truth_func: Callable[[NDArray[np.generic]], NDArray[np.bool_]]) -> None:
        """
        Filters the point cloud based on a scalar field or property.

        Parameters
        ----------
        sf_filter : str
            The name of the scalar field or property to filter.
        truth_func : Callable[[NDArray[np.generic]], NDArray[np.bool_]]
            A function that returns a boolean mask for filtering points.
        """
        if sf_filter == "spherical_coordinates": # Needed to init caching
            self.spherical_coordinates
        if sf_filter in self.scalar_fields.keys():
            filter_mask = truth_func(self.scalar_fields[sf_filter])
            self._reduce_points_to(filter_mask)
        elif sf_filter in self.__dict__.keys():
            filter_mask = truth_func(self.__dict__[sf_filter])
            self._reduce_points_to(filter_mask)

    def box_cut(self, minimum_corner: Tuple[float, float, float], maximum_corner: Tuple[float, float, float]) -> None:
        """
        Removes points outside a specified bounding box.

        Parameters
        ----------
        minimum_corner : tuple[float, float, float]
            The minimum (x, y, z) corner of the bounding box.
        maximum_corner : tuple[float, float, float]
            The maximum (x, y, z) corner of the bounding box.
        """
        # If the dimension difference in an axis is 0, then it should be ignored
        minimum_corner = np.array(minimum_corner, dtype=float)
        maximum_corner = np.array(maximum_corner, dtype=float)

        if self.global_coordinate_shift is not None:
            minimum_corner -= self.global_coordinate_shift
            maximum_corner -= self.global_coordinate_shift

        span = maximum_corner - minimum_corner
        minimum_corner[span == 0] = -np.inf
        maximum_corner[span == 0] = np.inf

        mask = np.logical_and(np.all(self.xyz >= minimum_corner, axis=1),
                              np.all(self.xyz <= maximum_corner, axis=1))
        self._reduce_points_to(mask)

    def extract_sphere_around(self, point: NDArray[np.floating], radius: float):
        distances_to_point = np.linalg.norm(self.xyz - point, axis=1)
        mask = distances_to_point <= radius

        new_pcd = self._copy_selection(mask)
        self._reduce_points_to(np.logical_not(mask))

        return new_pcd

    def extract_angles(self, fov: FoV) -> Self:
        """
        Extracts points within a specified field of view (FoV) based on angles.

        Parameters
        ----------
        fov : FoV
            The field of view object defining angular limits.

        Returns
        -------
        PointCloudData
            A new point cloud instance containing only the points within the FoV.
        """
        angle_filter = ("spherical_coordinates",
                        lambda spc: np.logical_and(np.logical_and(spc[:, 1] >= fov.elevation_min,
                                                                  spc[:, 1] <= fov.elevation_max),
                                                   np.logical_and(spc[:, 2] >= fov.horizontal_min,
                                                                  spc[:, 2] <= fov.horizontal_max, )))
        return self.extract(*angle_filter)

    def sample_angles(self, fov: FoV) -> Self:
        """
        Samples points within a specified field of view (FoV) based on angles.

        Parameters
        ----------
        fov : FoV
            The field of view object defining angular limits.

        Returns
        -------
        PointCloudData
            A new point cloud instance containing only the sampled points within the FoV.
        """
        angle_filter = ("spherical_coordinates",
                        lambda spc: np.logical_and(np.logical_and(spc[:, 1] >= fov.elevation_min,
                                                                  spc[:, 1] <= fov.elevation_max),
                                                   np.logical_and(spc[:, 2] >= fov.horizontal_min,
                                                                  spc[:, 2] <= fov.horizontal_max, )))
        return self.sample(*angle_filter)

    def filter_range(self, *, low: float = None, high: float = None) -> None:
        """
        Filters the point cloud based on a range of distances from the spherical coordinates.

        Parameters
        ----------
        low : float, optional
            The lower bound of the range. Defaults to 0.
        high : float, optional
            The upper bound of the range. Defaults to infinity.
        """
        if low is None:
            low = 0
        if high is None:
            high = np.inf
        range_filter = ("spherical_coordinates",
                        lambda spc: np.logical_and(spc[:, 0] >= low, spc[:, 0] <= high))
        self.filter(*range_filter)
        return


    def extract_range(self, *, low: float = None, high: float = None) -> Self:
        """
        Extracts points within a specified range from the spherical coordinates.

        Parameters
        ----------
        low : float, optional
            The lower bound of the range. Defaults to 0.
        high : float, optional
            The upper bound of the range. Defaults to infinity.

        Returns
        -------
        PointCloudData
            A new point cloud instance containing only the extracted points.
        """
        if low is None:
            low = 0
        if high is None:
            high = np.inf
        range_filter = ("spherical_coordinates",
                        lambda spc: np.logical_and(spc[:, 0] >= low, spc[:, 0] <= high))
        return self.extract(*range_filter)

    def sample_range(self, *, low: float = None, high: float = None) -> Self:
        """
        Samples points within a specified range from the spherical coordinates.

        Parameters
        ----------
        low : float, optional
            The lower bound of the range. Defaults to 0.
        high : float, optional
            The upper bound of the range. Defaults to infinity.

        Returns
        -------
        PointCloudData
            A new point cloud instance containing only the sampled points.
        """
        if low is None:
            low = 0
        if high is None:
            high = np.inf
        range_filter = ("spherical_coordinates",
                        lambda spc: np.logical_and(spc[:, 0] >= low, spc[:, 0] <= high))
        return self.sample(*range_filter)

    def random_subsample(self, size: float | int, in_place: bool = True) -> Self:
        """
        Randomly subsamples the point cloud to reduce the number of points.

        Parameters
        ----------
        size : float or int
            If a float between 0 and 1, it specifies the proportion of points to retain.
            If an integer, it specifies the exact number of points to retain.
        in_place : bool, default=True
            If True, modifies the current point cloud in place. Otherwise, returns a new instance.

        Returns
        -------
        PointCloudData or None
            A new point cloud instance if `in_place` is False; otherwise, None.
        """
        if isinstance(size, float) and 0 < size < 1:
            size = np.ceil(size * self.nbPoints).astype(int)
        if size >= self.nbPoints:
            warnings.warn("Subsampling ratio above 1!")
            return self._copy_selection(np.arange(0, self.nbPoints)) if not in_place else None
        selection = np.sort(np.random.choice(np.arange(0, self.nbPoints), size=size, replace=False))
        if in_place:
            return self._reduce_points_to(selection)
        else:
            return self._copy_selection(selection)

    def sample(self, sf_sample: str, sample_func: Callable[[NDArray[np.generic]], NDArray[np.bool_]]) -> Self:
        """
        Samples a subset of points based on a scalar field or property.

        Parameters
        ----------
        sf_sample : str
            The name of the scalar field or property to use for sampling.
        sample_func : Callable[[NDArray[np.generic]], NDArray[np.bool_]]
            A function that returns a boolean mask for sampling points.

        Returns
        -------
        PointCloudData
            A new instance containing the sampled points.
        """
        if sf_sample == "spherical_coordinates":
            self.spherical_coordinates
        if sf_sample in self.scalar_fields.keys():
            filter_mask = sample_func(self.scalar_fields[sf_sample])
        elif sf_sample in self.__dict__.keys():
            filter_mask = sample_func(self.__dict__[sf_sample])
        else:
            raise ValueError("(Scalar) field does not exist")
        return self._copy_selection(filter_mask)

    def extract(self, sf_sample: str, sample_func: Callable[[NDArray[np.generic]], NDArray[np.bool_]]) -> Self:
        """
        Extracts a subset of points based on a scalar field or property.

        Parameters
        ----------
        sf_sample : str
            The name of the scalar field or property to use for sampling.
        sample_func : Callable[[NDArray[np.generic]], NDArray[np.bool_]]
            A function that returns a boolean mask for selecting points.

        Returns
        -------
        PointCloudData
            A new instance containing the extracted points.
        """
        if sf_sample == "spherical_coordinates":
            self.spherical_coordinates
        if sf_sample in self.scalar_fields.keys():
            filter_mask = sample_func(self.scalar_fields[sf_sample])
        elif sf_sample in self.__dict__.keys():
            filter_mask = sample_func(self.__dict__[sf_sample])
        else:
            raise ValueError("(Scalar) field does not exist")
        new_pcd = self._copy_selection(filter_mask)
        self._reduce_points_to(np.logical_not(filter_mask))
        return new_pcd

    def change_spherical_coordinates_origin(self, spherical_coordinates_origin: NDArray[np.float_]) -> None:
        """
        Changes the origin used for spherical coordinate calculations.

        Parameters
        ----------
        spherical_coordinates_origin : NDArray[np.float_]
            A (3,) array specifying the new origin for spherical coordinate calculations.
        """
        assert spherical_coordinates_origin.shape == (3,)
        object.__setattr__(self, "spherical_coordinates_origin", spherical_coordinates_origin)
        if self._spherical_coordinates_calculated:
            object.__delattr__(self, "spherical_coordinates")
            object.__setattr__(self, "_spherical_coordinates_calculated", False)

    @property
    def fov(self) -> FoV:
        """
        Returns
        -------
            FoV: Field of View object based on spherical coordinates.
        """
        # TODO: Accommodate smaller pcd that cross the hz: 200 -> -200 gon border (same for elevation, less common)
        return FoV(horizontal_min=self.spherical_coordinates[:, 2].min(),
                   horizontal_max=self.spherical_coordinates[:, 2].max(),
                   elevation_min=self.spherical_coordinates[:, 1].min(),
                   elevation_max=self.spherical_coordinates[:, 1].max(),
                   unit="rad")


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
        scalar_fields = defaultdict(list)
        global_coordinate_shift = []
        spherical_coordinates_origin = []

        # Build lists of all elements
        for i, pcd in enumerate(pcds):
            xyz.append(pcd.xyz)
            color.append(pcd.color)
            normals.append(pcd.normals)
            for sf, pcd_sf in pcd.scalar_fields.items():
                scalar_fields[sf].append(pcd_sf)
            scalar_fields["point_cloud_merge"].append(np.ones((pcd.xyz.shape[0],), dtype=np.uint8) * (i + 1))
            global_coordinate_shift.append(pcd.global_coordinate_shift)
            spherical_coordinates_origin.append(pcd.spherical_coordinates_origin)


        # Find empty pcd
        # empty_mask = [False if val is None else True for val in xyz]
        # xyz = list(compress(xyz, empty_mask))
        # color = list(compress(color, empty_mask))
        # normals = list(compress(normals, empty_mask))
        # scalar_fields = {sf_key: list(compress(sf_filter, empty_mask)) for sf_key, sf_filter in scalar_fields.items()}

        nb_pcds = len(xyz)

        if any(val is None for val in color):
            color = None

        if any(val is None for val in normals):
            normals = None

        remove_keys = []
        for sf_key, sf in scalar_fields.items():
            if any(val is None for val in sf) or len(sf) < nb_pcds:
                remove_keys.append(sf_key)

        for rk in remove_keys:
            del (scalar_fields[rk])


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
            gcs_None_removed = [np.zeros(3,) if g is None else g for g in global_coordinate_shift]
            xyz_np = np.vstack(tuple(map(lambda x: np.add(*x), zip(xyz_64, gcs_None_removed))))
            del xyz, xyz_64
            gc.collect()

        color_np = np.vstack(tuple(color)) if color is not None else None  #
        del color
        gc.collect()

        normals_np = np.vstack(tuple(normals)) if normals is not None else None
        del normals
        gc.collect()

        # sf_keys = scalar_fields.keys()
        # scalar_fields_np = {}
        # for sf_key in sf_keys:
        #     scalar_fields_np[sf_key] = np.hstack(tuple(scalar_fields[sf_key]))
        #     del scalar_fields[sf_key]
        #     gc.collect()

        scalar_fields = {sf_key: np.hstack(tuple(sf)) for sf_key, sf in scalar_fields.items()}

        sco_pairs = zip(spherical_coordinates_origin[:-1], spherical_coordinates_origin[1:])

        # Check if all spherical_coordinates_origin are equal and represented in the same system
        scs = None
        if all(map(lambda sco_pair: np.array_equal(*sco_pair), sco_pairs)) and \
                len(set([pcd._spherical_coordinates_represented_0_to_2pi for pcd in pcds])) == 1:

            sco = spherical_coordinates_origin[0]
            if all([pcd._spherical_coordinates_calculated for pcd in pcds]):
                scs = np.vstack([pcd.spherical_coordinates for pcd in pcds])
                # Todo: Add a check if the merged PCD has introduced a discontinuity
            scr = pcds[0]._spherical_coordinates_represented_0_to_2pi
        else:
            sco = None
            scr = None

        new_pcd = cls(xyz=xyz_np, color=color_np, normals=normals_np, scalar_fields=scalar_fields,
                      global_coordinate_shift=gcs, _global_shift_already_applied=(gcs is not None),
                      spherical_coordinates_origin=sco, _spherical_coordinates_represented_0_to_2pi=scr)

        if scs is not None:
            object.__setattr__(new_pcd, "spherical_coordinates", scs)
            object.__setattr__(new_pcd, "_spherical_coordinates_calculated", True)

        return new_pcd

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
            pcd_o3d.points = o3d.utility.Vector3dVector((self.xyz + self.global_coordinate_shift).astype(np.float32))
        return pcd_o3d

    @classmethod
    def from_o3d(cls, pcd_o3d: o3d.geometry.PointCloud, scan_center: Optional[NDArray[np.float_]] = None):
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
        return cls(xyz=np.asarray(pcd_o3d.points), spherical_coordinates_origin=scan_center)

    def filter_to_polygon(self, poly: Polygon, plane: str) -> None:
        """
        Filters the point cloud to include only points within a given polygon in a specific 2D projection.

        Parameters
        ----------
        poly : Polygon
            A Shapely Polygon defining the region of interest.
        plane : str
            The plane of projection ('xy', 'xz', or 'yz').

        Raises
        ------
        NotImplementedError
            If GPU support is not available.
        """
        if HAS_GPU_SUPPORT:
            self._filter_to_polygon_gpu(poly, plane)
        else:
            raise NotImplementedError()

    def _filter_to_polygon_gpu(self, poly: Polygon, plane: str) -> None:
        """
        Filters the point cloud using GPU acceleration to include points within a given polygon in a specific 2D projection.

        Parameters
        ----------
        poly : Polygon
            A Shapely Polygon defining the region of interest.
        plane : str
            The plane of projection ('xy', 'xz', or 'yz').

        Raises
        ------
        ValueError
            If the specified plane is invalid.
        """
        match plane:
            case 'xy':
                proj_pts = cudf.DataFrame({"x": self.xyz[:, 0].astype(float),
                                           "y": self.xyz[:, 1].astype(float)}).interleave_columns()
            case 'xz':
                proj_pts = cudf.DataFrame({"x": self.xyz[:, 0].astype(float),
                                           "y": self.xyz[:, 2].astype(float)}).interleave_columns()
            case 'yz':
                proj_pts = cudf.DataFrame({"x": self.xyz[:, 1].astype(float),
                                           "y": self.xyz[:, 2].astype(float)}).interleave_columns()
            case _:
                raise ValueError

        if self.global_coordinate_shift is not None:
            match plane:
                case 'xy':
                    gs = -self.global_coordinate_shift[:2]
                case 'xz':
                    gs = -self.global_coordinate_shift[[0, 2]]
                case 'yz':
                    gs = -self.global_coordinate_shift[1:]
                case _:
                    raise ValueError
            poly = translate(poly, *gs)

        polygon_gpu = cuspatial.GeoSeries(gpd.GeoSeries(poly))
        # polygon_gpu = cuspatial.GeoSeries.from_polygons_xy(poly)
        proj_pts_gs = cuspatial.GeoSeries.from_points_xy(proj_pts)
        proj_pts_in = cuspatial.point_in_polygon(proj_pts_gs, polygon_gpu)

        pts_in_mask = proj_pts_in[0].to_numpy()
        self._reduce_points_to(pts_in_mask)

        del polygon_gpu, proj_pts, proj_pts_gs, proj_pts_in
        gc.collect()

    def filter_spherical_polygon(self, poly: Polygon) -> None:
        """
        Filters the point cloud to include only points within a given polygon in the spherical projection.

        Parameters
        ----------
        poly : Polygon
            A Shapely Polygon defining the region of interest.

        Raises
        ------
        NotImplementedError
            If GPU support is not available.
        """
        if HAS_GPU_SUPPORT:
            self._filter_spherical_polygon_gpu(poly)
        else:
            raise NotImplementedError()

    def _filter_spherical_polygon_gpu(self, poly: Polygon) -> None:
        """
        Filters the point cloud using GPU acceleration to include points within a given polygon in the spherical projection.

        Parameters
        ----------
        poly : Polygon
            A Shapely Polygon defining the region of interest.

        Raises
        ------
        ValueError
            If the specified plane is invalid.
        """
        proj_pts = cudf.DataFrame({"x": self.spherical_coordinates[:, 1].astype(float),
                                   "y": self.spherical_coordinates[:, 2].astype(float)}).interleave_columns()



        polygon_gpu = cuspatial.GeoSeries(gpd.GeoSeries(poly))
        proj_pts_gs = cuspatial.GeoSeries.from_points_xy(proj_pts)
        proj_pts_in = cuspatial.point_in_polygon(proj_pts_gs, polygon_gpu)

        pts_in_mask = proj_pts_in[0].to_numpy()
        self._reduce_points_to(pts_in_mask)

        del polygon_gpu, proj_pts, proj_pts_gs, proj_pts_in
        gc.collect()


    def get_outline_polygon(self, plane: str, alpha_value: float = 10.0, nb_points: int = -1) -> Polygon:
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
                proj_pts = self.xyz[:, :2]
            case 'xz':
                proj_pts = self.xyz[:, [0, 2]]
            case 'yz':
                proj_pts = self.xyz[:, 1:]
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
            NotImplementedError

        if self.global_coordinate_shift is not None:
            match plane:
                case 'xy':
                    gs = self.global_coordinate_shift[:2]
                case 'xz':
                    gs = self.global_coordinate_shift[[0, 2]]
                case 'yz':
                    gs = self.global_coordinate_shift[1:]
                case _:
                    raise ValueError
            als = translate(als, *gs)

        return als

    def voxel_downsample(self, voxel_size: float, weighted: bool = False) -> None:
        """
        Reduces the number of points in the point cloud using voxel grid downsampling.

        Parameters
        ----------
        voxel_size : float
            The size of the voxel grid.
        """
        unique, unique_inverse = np.unique(np.round(self.xyz / voxel_size).astype(np.int32), axis=0, return_inverse=True)

        # Calculate centroids for each voxel
        centroids = np.zeros((unique.shape[0], 3), dtype=np.float32)
        for i in range(3):  # x, y, z dimensions
            centroids[:, i] = np.bincount(unique_inverse, weights=self.xyz[:, i], minlength=unique.shape[0])

        counts = np.bincount(unique_inverse, minlength=unique.shape[0])
        centroids /= counts[:, None]  # Normalize to get centroids

        # Compute distances of points to their respective voxel centroids
        if weighted:
            distances = np.linalg.norm(self.xyz - centroids[unique_inverse], axis=1)
            weights = np.reciprocal(np.where(distances > 1e-6, distances, 1.0))  # Avoid division by zero
            weight_sums = np.bincount(unique_inverse, weights=weights, minlength=unique.shape[0])

            # Normalize weights per voxel
            weights /= np.where(weight_sums[unique_inverse] > 0, weight_sums[unique_inverse], 1)  # Avoid NaNs
        else:
            weights = np.ones_like(counts, dtype=np.float32)[unique_inverse]

        for field_name, field_values in self.scalar_fields.items():
            # Compute weighted sum of scalar values within each voxel
            scalar_sum = np.bincount(unique_inverse, weights=field_values * weights, minlength=unique.shape[0])
            weight_sum = np.bincount(unique_inverse, weights=weights, minlength=unique.shape[0])
            self.scalar_fields[field_name] = (scalar_sum / weight_sum).astype(field_values.dtype)

        # # Average scalar fields
        # averaged_scalar_fields = {}
        # for field_name, field_values in self.scalar_fields.items():
        #     # Compute the sum of scalar values within each voxel
        #     scalar_sum = np.bincount(unique_inverse, weights=field_values, minlength=unique.shape[0])
        #     # Compute the average
        #     averaged_scalar_fields[field_name] = (scalar_sum / counts).astype(field_values.dtype)

        object.__setattr__(self, "xyz", centroids)
        if self._spherical_coordinates_calculated:
            object.__delattr__(self, "spherical_coordinates")
            object.__setattr__(self, "_spherical_coordinates_calculated", False)

        # Todo: Add functionality
        warnings.warn("Normals, and colors are not retained during `voxel_downsample`!")
        object.__setattr__(self, 'color', None)
        object.__setattr__(self, 'normals', None)
        # for field_name in self.scalar_fields.keys():
        #     del self.scalar_fields[field_name]
        # self.scalar_fields.clear()

        return

    def filter_spherical_outliers(self, std_ratio: float = 0.95, nb_neighbors: int = 13):
        """
        Removes outliers in the spherical coordinate space using statistical filtering.

        Parameters
        ----------
        std_ratio : float, default=0.95
            The standard deviation ratio for identifying outliers.
        nb_neighbors : int, default=13
            The number of neighbors to consider for statistical outlier removal.
        """
        sp_pcd = o3d.geometry.PointCloud()
        sp_pcd.points = o3d.utility.Vector3dVector(np.hstack((self.spherical_coordinates[:,1:],
                                                              np.zeros((self.nbPoints,1), dtype=np.float32))))
        _, inliers = sp_pcd.remove_statistical_outlier(nb_neighbors,std_ratio,True)
        self._reduce_points_to(inliers)
        # sp_dist = sp_pcd.compute_nearest_neighbor_distance()
        # sp_dist_np = np.asarray(sp_dist)
        # threshold = np.percentile(sp_dist_np, percentile)
        # sp_mask = sp_dist_np <= threshold
        #
        # self._reduce_points_to(sp_mask)

        # if HAS_GPU_SUPPORT:
        #     self._filter_spherical_outliers_gpu(percentile)
        # else:
        #     raise NotImplementedError

    def filter_xyz_outliers(self, std_ratio: float = 0.95, nb_neighbors: int = 13):
        """
        Removes outliers in the `xyz` space using statistical filtering.

        Parameters
        ----------
        std_ratio : float, default=0.95
            The standard deviation ratio for identifying outliers.
        nb_neighbors : int, default=13
            The number of neighbors to consider for statistical outlier removal.
        """
        pcd_o3d = self.to_o3d()

        _, inliers = pcd_o3d.remove_statistical_outlier(nb_neighbors,std_ratio,True)
        self._reduce_points_to(np.array(inliers))

    # def _filter_spherical_outliers_gpu(self, percentile: int = 95, nb_neighbors: int = 13):
    #     raise NotImplementedError
    #     knn = NearestNeighbors(n_neighbors = nb_neighbors)
    #     knn.fit(self.spherical_coordinates[:,1:])
    #     distances, indices = knn.kneighbors(self.spherical_coordinates[:,1:])




def split_pc_with_fov_tree(pcd: PointCloudData, fov_tree: FoVTree, remove_empty: bool = True, n_jobs: int = -1) \
        -> dict[str, PointCloudData]:
        # -> list[tuple[str, FoV, PointCloudData]]:

    """
    Splits a PointCloudData instance using a FoVTree.

    Parameters
    ----------
    pcd : PointCloudData
        The point cloud data to be split.
    fov_tree : FoVTree
        A tree structure defining the field of view regions for splitting the point cloud.
    remove_empty : bool, default=True
        Whether to remove empty splits (i.e., regions with no points).
    n_jobs : int, default=-1
        The number of parallel jobs to use. If -1, all available cores are used.

    Returns
    -------
    dict[str, PointCloudData]
        A dictionary where keys are FoV identifiers and values are the split PointCloudData objects.
    """
    # if fov_tree.is_leaf() or pcd.nbPoints <= self.minimum_nb_points:
    #     return [(fov_tree.identifier, fov_tree.node, pcd,)]
    if fov_tree.is_leaf():
        return {fov_tree.identifier: pcd}
        # return [(fov_tree.identifier, fov_tree.node, pcd,)]

    # Setup argumenets for call
    split_packages = [(pcd.extract_angles(child.node), child, remove_empty, n_jobs)
                      for child in fov_tree.children.values()]
    if remove_empty:
        split_packages = [sp for sp in split_packages if sp[0].nbPoints]
    # print(*[FoV(**sp[0].fov, unit="rad") for sp in split_packages], sep='\n')

    split = Parallel(n_jobs=n_jobs, prefer="processes", verbose=50, timeout=10 * 60)(delayed(
        split_pc_with_fov_tree)(*split_package) for split_package in split_packages)

    split_dict = {k: v for d in split for k, v in d.items()}
    return split_dict
