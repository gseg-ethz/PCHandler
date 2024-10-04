import sys


from collections import defaultdict
from dataclasses import dataclass, field, KW_ONLY, InitVar
from itertools import compress, chain
from functools import cached_property
import gc
from typing import Iterable, Callable, Any, Tuple, Optional
if sys.version[0] == 3 and sys.version_info[1] >= 11:
    from typing import Self
else:
    from typing_extensions import Self
import warnings

from joblib import Parallel, delayed
import numpy as np

from pchandler.fov import FoV, FoVTree

# from pchandler.util import convert_angle, AngleUnit


# TODO: Implement global center shift and checks
# TODO: Add own init function
# TODO: Repair all that was broken by coordinate shift

@dataclass(frozen=True)
class PointCloudData:
    """
    PointCloudData stores 3D points in a cartesian coordinate system.


    Attributes
    ----------
    xyz : np.ndarray
        nx3 float32 array with the *x*, *y* and *z* coordinates of the cloud.
    scalar_fields: dict[str, np.ndarray]
        dict of n 1D-arrays
    color : np.ndarray
        nx3 uint8 array of *r*, *g* and *b* colors.
    normals : np.ndarray
        nx3 float array


    """

    xyz: np.ndarray
    _: KW_ONLY
    scalar_fields: dict[str, np.ndarray] = field(default_factory=dict)
    color: Optional[np.ndarray] = None
    normals: Optional[np.ndarray] = None
    global_coordinate_shift: Optional[np.ndarray] = None
    spherical_coordinates_origin: Optional[np.ndarray] = None
    _spherical_coordinates_calculated: bool = False
    _spherical_coordinates_represented_0_to_2pi: Optional[bool] = None
    _global_shift_already_applied: InitVar[bool] = False

    @property
    def nbPoints(self):
        return self.xyz.shape[0]

    def __post_init__(self, _global_shift_already_applied: bool) -> None:
        """
        _global_shift_already_applied
        """
        # Check types
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
        assert self.spherical_coordinates_origin is None or self.spherical_coordinates_origin.shape == (3,)
        assert self.global_coordinate_shift is None or self.global_coordinate_shift.shape == (3,)

        if self.global_coordinate_shift is None and any((np.abs(self.xyz) >= 1e4).flatten()):
            object.__setattr__(self, 'global_coordinate_shift', np.median(np.round(self.xyz, decimals=-3), axis=0))

        if self.global_coordinate_shift is not None and not _global_shift_already_applied:
            object.__setattr__(self, 'xyz', (self.xyz - self.global_coordinate_shift).astype(np.float32, casting="same_kind"))
        else:
            object.__setattr__(self, "xyz", self.xyz.astype(np.float32, casting="same_kind"))

        if self.global_coordinate_shift is not None and self.spherical_coordinates_origin is None:
            self.change_spherical_coordinates_origin(self.global_coordinate_shift)
                # object.__setattr__(self, 'spherical_coordinates_origin', self.global_coordinate_shift)


       # if "scalar_Intensity" in scalar_fields:
       #     if np.nanmax(scalar_fields["scalar_Intensity"]) > 1.0 or np.nanmin(scalar_fields["scalar_Intensity"]) < 0.0:
       #         pass
       #     scalar_fields["scalar_Intensity"] = scalar_fields["scalar_Intensity"].astype(np.float32)

    @classmethod
    def from_range_image(cls, range_data: np.ndarray, fov: FoV, scalar_fields: dict[str, np.ndarray] = None,
                         spherical_coordinates_origin: np.ndarray = None) -> Self:

        # nbPts = np.count_nonzero(~np.isnan(range_data))
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
    def from_spherical_coordinates(cls, spherical_coordinates: np.ndarray, scalar_fields: dict[str, np.ndarray] = None,
                                   spherical_coordinates_origin: np.ndarray = None) -> Self:
        # object.__setattr__(self, "_spherical_coordinates_calculated", True)
        assert spherical_coordinates_origin is None or spherical_coordinates_origin.shape == (3,)

        xyz = np.zeros((len(spherical_coordinates), 3))
        xyz[:, 0] = spherical_coordinates[:, 0] * np.sin(spherical_coordinates[:, 1]) * np.cos(spherical_coordinates[:, 2])
        xyz[:, 1] = - spherical_coordinates[:, 0] * np.sin(spherical_coordinates[:, 1]) * np.sin(spherical_coordinates[:, 2])
        xyz[:, 2] = spherical_coordinates[:, 0] * np.cos(spherical_coordinates[:, 1])

        if spherical_coordinates_origin is not None:
            xyz = xyz + spherical_coordinates_origin

        return cls(xyz, scalar_fields=scalar_fields, spherical_coordinates_origin=spherical_coordinates_origin)

    def copy(self) -> Self:
        mask = np.ones(self.xyz.shape[0], dtype=bool)
        return self._copy_selection(mask)

    @cached_property
    def spherical_coordinates(self) -> np.ndarray:

        if self.global_coordinate_shift is not None:
            xyz = self.xyz - (self.spherical_coordinates_origin - self.global_coordinate_shift)
        elif self.spherical_coordinates_origin is not None:
            xyz = self.xyz - self.spherical_coordinates_origin
        else:
            xyz = self.xyz

        spherical_coordinates = np.zeros(self.xyz.shape)
        xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
        spherical_coordinates[:, 0] = np.sqrt(xy + xyz[:, 2] ** 2)
        spherical_coordinates[:, 1] = np.arctan2(np.sqrt(xy), xyz[:, 2])  # for elevation angle defined from Z-axis down
        spherical_coordinates[:, 2] = - np.arctan2(xyz[:, 1], xyz[:, 0])

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

    def _reduce_points_to(self, mask: np.ndarray) -> Self:
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

    def _copy_selection(self, mask: np.ndarray) -> Self:
        # TODO: check if mask is slice or truth array
        xyz = self.xyz[mask].copy()
        color = self.color[mask].copy() if self.color is not None else None
        normals = self.normals[mask].copy() if self.normals is not None else None
        scalar_fields = dict()
        for sf_key, sf_value in self.scalar_fields.items():
            scalar_fields[sf_key] = sf_value[mask].copy()
        global_coordinate_shift = self.global_coordinate_shift.copy() if self.global_coordinate_shift is not None else None
        spherical_coordinates_origin = self.spherical_coordinates_origin.copy() if self.spherical_coordinates_origin is not None else None
        return PointCloudData(xyz, color=color, normals=normals, scalar_fields=scalar_fields, global_coordinate_shift=global_coordinate_shift,
                              spherical_coordinates_origin=spherical_coordinates_origin, _global_shift_already_applied=True,
                              _spherical_coordinates_represented_0_to_2pi=self._spherical_coordinates_represented_0_to_2pi)

    # def _sample(self,sf_sample: str,
    #            sample_func: Callable[[np.ndarray], np.ndarray[Any, np.dtype[bool]]],
    #             in_place: bool = False) -> "PointCloudData" | None:

    def apply_math_to_xyz(self, math_func: Callable) -> None:
        object.__setattr__(self, "xyz", math_func(self.xyz))

    def filter(self, sf_filter: str, truth_func: Callable[[np.ndarray], np.ndarray[Any, np.dtype[bool]]]) -> None:
        """
        Filters the point cloud based on the function.

        Parameters
        ----------
        sf_filter : str
            Scalar field identifier.
        truth_func
            Callable that takes a number and returns a `bool`
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

    def extract_angles(self, fov: FoV) -> Self:
        angle_filter = ("spherical_coordinates",
                        lambda spc: np.logical_and(np.logical_and(spc[:, 1] >= fov.elevation_min,
                                                                  spc[:, 1] <= fov.elevation_max),
                                                   np.logical_and(spc[:, 2] >= fov.horizontal_min,
                                                                  spc[:, 2] <= fov.horizontal_max, )))
        return self.extract(*angle_filter)

    def extract_range(self, *, low: float = None, high: float = None) -> Self:
        if low is None:
            low = 0
        if high is None:
            high = np.inf
        range_filter = ("spherical_coordinates",
                        lambda spc: np.logical_and(spc[:, 0] >= low, spc[:, 0] <= high))
        return self.extract(*range_filter)

    def random_subsample(self, size: float | int, in_place: bool = True) -> Self:
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

    def sample(self, sf_sample: str,
               sample_func: Callable[[np.ndarray], np.ndarray[Any, np.dtype[bool]]]) -> Self:
        if sf_sample == "spherical_coordinates":
            self.spherical_coordinates
        if sf_sample in self.scalar_fields.keys():
            filter_mask = sample_func(self.scalar_fields[sf_sample])
        elif sf_sample in self.__dict__.keys():
            filter_mask = sample_func(self.__dict__[sf_sample])
        else:
            raise ValueError("(Scalar) field does not exist")
        return self._copy_selection(filter_mask)

    def extract(self, sf_sample: str,
                sample_func: Callable[[np.ndarray], np.ndarray[Any, np.dtype[bool]]]) -> Self:
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

    def change_spherical_coordinates_origin(self, spherical_coordinates_origin: np.ndarray) -> None:
        assert spherical_coordinates_origin.shape == (3,)
        object.__setattr__(self, "spherical_coordinates_origin", spherical_coordinates_origin)
        if self._spherical_coordinates_calculated:
            object.__delattr__(self, "spherical_coordinates")
            object.__setattr__(self, "_spherical_coordinates_calculated", False)

    @property
    def fov(self):
        # TODO: Accommodate smaller pcd that cross the hz: 200 -> -200 gon border (same for elevation, less common)
        return FoV(horizontal_min=self.spherical_coordinates[:, 2].min(),
                   horizontal_max=self.spherical_coordinates[:, 2].max(),
                   elevation_min=self.spherical_coordinates[:, 1].min(),
                   elevation_max=self.spherical_coordinates[:, 1].max(),
                   unit="rad")


    @classmethod
    def merge_pcd(cls, pcds: Iterable[Self]) -> Self:
        """
        Merge multiple point clouds.

        Merges two or more point clouds. The new point cloud will only retain scalar fields (and colors and normals) if
        TODO: Check for dtype (uint8 vs uint16 -> needs to be scaled)
        TODO:

        Parameters
        ----------
        pcds : iterable[pchandler.geometry.PointCloudData]

        Returns
        -------
        pcd : pchandler.geometry.PointCloudData
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

        # Check if all spherical_coordinates_origin are equal
        scs = None
        if all(map(lambda sco_pair: np.array_equal(*sco_pair), sco_pairs)):
            sco = spherical_coordinates_origin[0]
            if all([pcd._spherical_coordinates_calculated for pcd in pcds]):
                scs = np.vstack([pcd.spherical_coordinates_origin for pcd in pcds])
        else:
            sco = None

        new_pcd = cls(xyz=xyz_np, color=color_np, normals=normals_np, scalar_fields=scalar_fields,
                      global_coordinate_shift=gcs, _global_shift_already_applied=(gcs is not None),
                      spherical_coordinates_origin=sco)

        if scs is not None:
            object.__setattr__(new_pcd, "spherical_coordinates", scs)
            object.__setattr__(new_pcd, "_spherical_coordinates_calculated", True)

        return new_pcd


def split_pc_with_fov_tree(pcd: PointCloudData, fov_tree: FoVTree, remove_empty: bool = True, n_jobs: int = -1) \
        -> dict[str, PointCloudData]:
        # -> list[tuple[str, FoV, PointCloudData]]:

    """

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
