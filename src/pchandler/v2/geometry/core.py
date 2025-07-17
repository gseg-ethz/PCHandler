from __future__ import annotations

import warnings
from typing import Any, Mapping, Optional, Self, Annotated, MutableMapping, Union, Type, Callable, cast
from copy import deepcopy

import numpy as np
import numpy.typing as npt
import open3d as o3d
from pydantic import Field, model_validator, BeforeValidator

from ..base_types import Array_4x4_T, Array_Nx3_T, Vector_3_T, IndexLike
from .coordinates import CartesianCoordinates
from .optimal_shift import OptimizedShift

from .scalar_field_manager import ScalarFieldManager
from .scalar_fields import NormalFields, RGBFields, ScalarField, SF_T, NormalisedInt16ScalarField, ScalarFieldTriplet
from .transforms import TransformLedger


class PointCloudData(CartesianCoordinates):
    arr: Array_Nx3_T = Field(..., alias='xyz')
    # TODO: Rework Transform ledger
    # transform_ledger: Annotated[
    #     TransformLedger,
    #     Field(default_factory=TransformLedger),
    #     BeforeValidator(lambda value: value if not isinstance(value, TransformLedger) else TransformLedger(**value))
    # ]
    scalar_fields: ScalarFieldManager[ScalarField | ScalarFieldTriplet] = Field(default_factory=ScalarFieldManager)

    # “input‐only” fields that need to be folded into the ScalarFieldManager
    rgb_input: Optional[npt.NDArray[np.uint8 | np.float32 | np.float64]] = Field(
        None, alias="rgb", exclude=True, repr=False
    )
    normals_input: Optional[npt.NDArray[np.float32 | np.float64]] = Field(
        None, alias="normals", exclude=True, repr=False
    )
    intensity_input: Optional[npt.NDArray[np.uint16 | np.float32 | np.float64]] = Field(
        None, alias="intensity", exclude=True, repr=False
    )
    reflectance_input: Optional[npt.NDArray[np.uint16 | np.float32 | np.float64]] = Field(
        None, alias="reflectance", exclude=True, repr=False
    )

    def __init__(self, *args, **kwargs):
        # Accept xyz/arr as a positional argument
        if args:
            if len(args) > 1:
                raise TypeError(f"Expected at most 1 positional argument, got {len(args)}")
            if "xyz" in kwargs or "arr" in kwargs:
                raise TypeError("Cannot pass both positional and keyword for xyz/arr")
            kwargs["xyz"] = args[0]

        super().__init__(**kwargs)

    # def __init__(
    #     self,
    #     xyz: Array_Nx3_T | CartesianCoordinates,
    #     *,
    #     rgb: Optional[npt.NDArray[np.uint8|np.float32|np.float64] | RGBFields] = None,
    #     normals: Optional[npt.NDArray[np.float32|np.float64] | NormalFields] = None,
    #     intensity: Optional[npt.NDArray[np.uint16|np.float32|np.float64] | ScalarField] = None,
    #     reflectance: Optional[npt.NDArray[np.uint16|np.float32|np.float64] | ScalarField] = None,
    #     numerical_optimization_shift: Optional[OptimizedShift | ellipsis] = Ellipsis,
    #     socs_origin: Optional[npt.NDArray[np.float64]] = None,
    #     scalar_fields: Optional[ScalarFieldManager[ScalarField | ScalarFieldTriplet] | dict[str, SF_T|npt.NDArray] ] = None,
    #     project_transformation: Optional[Array_4x4_T] = None,
    #     transform_ledger: Optional[TransformLedger] = None,
    #     # frozen: bool = False
    # ):
    #     super().__init__(
    #         arr=xyz,
    #         numerical_optimization_shift = numerical_optimization_shift,
    #         socs_origin = socs_origin,
    #         project_transformation = project_transformation,
    #         transform_ledger = transform_ledger if transform_ledger else TransformLedger(),
    #     )
    #
    #     self.rgb = rgb
    #     self.normals = normals
    #     self.intensity = intensity
    #     self.reflectance = reflectance
    #
    #     if scalar_fields is not None:
    #         for key, value in scalar_fields:
    #             self.scalar_fields[key] = value

    @field_validator('scalar_fields', mode="before")
    @classmethod
    def _convert_scalar_fields(cls, value):
        if not isinstance(value, ScalarFieldManager):
            sfm = ScalarFieldManager()
            for key, value in value.items():
                sfm[key] = value
            return sfm
        return value


    @model_validator(mode="after")
    def _move_into_scalar_fields(self) -> Self:
        # map any provided “_input” into the scalar_fields manager
        for name in ("rgb", "normals", "intensity", "reflectance"):
            inp = getattr(self, f"{name}_input")
            if inp is not None:
                setattr(self, name, inp)
        return self


    @model_validator(mode="after")
    def ensure_scalar_field_manager_pointer(self) -> Self:
        """Revalidate model to ensure that the weakref points to the correct object"""

        self.scalar_fields.parent = self
        return self

    @property
    def normals(self: Self) -> Optional[NormalFields]:
        return self.scalar_fields.normals

    @normals.setter
    def normals(self, value: Optional[npt.NDArray | NormalFields]) -> None:
        self.scalar_fields.normals = value

    @property
    def rgb(self) -> Optional[RGBFields]:
        return self.scalar_fields.rgb

    @rgb.setter
    def rgb(self, value: Optional[npt.NDArray[np.floating|np.uint8] | RGBFields]) -> None:
        self.scalar_fields.rgb = value

    @property
    def intensity(self) -> Optional[NormalisedInt16ScalarField]:
        return self.scalar_fields.intensity

    @intensity.setter
    def intensity(self, value: Optional[npt.NDArray[np.uint16|np.floating] | ScalarField]) -> None:
        self.scalar_fields.intensity = value

    @property
    def reflectance(self: Self) -> Optional[NormalisedInt16ScalarField]:
        return self.scalar_fields.reflectance

    @reflectance.setter
    def reflectance(self, value: Optional[npt.NDArray | ScalarField]) -> None:
        self.scalar_fields.reflectance = value

    def __setitem__(self, key: IndexLike, value: npt.NDArray[Any] | PointCloudData) -> None:
        raise IndexError(
            f"Setting items in PointCloudData is not supported. Consider using the copy or "
            f"dump data to a dict and reinstantiate."
        )

    def __hash__(self) -> int:
        return id(self)

    def __reduce__(self):
        base_fn, base_args = super().__reduce__()

        state_dict, = base_args
        return (
            PointCloudData._reconstruct_scalar_field_link,
            (base_fn, state_dict),
        )

    @staticmethod
    def _reconstruct_scalar_field_link(
            base_fn: Callable[[dict], PointCloudData],
            state: dict,
      ):
        obj = base_fn(state)

        obj.scalar_fields.parent = obj
        return obj

    def copy(self,
             array: Optional[npt.NDArray[np.floating] | Self] = None,
             *,
             deep: bool = True,
             update: Optional[MutableMapping[str, Any]] = None,
             link_to_same_NOS: bool = True,
             **kwargs: dict[str, Any]) -> Self:
        """
        Produce a deep or shallow copy of the model. Updates the model also if parameter is parsed.
        """

        update = {} if update is None else update

        if array is not None and not isinstance(array, CartesianCoordinates | np.ndarray):
            raise TypeError(f"Invalid type of array passed: {type(array)}. Should be CartesianCoordinates or np.ndarray")

        if link_to_same_NOS:
            update["numerical_optimization_shift"] = self.numerical_optimization_shift

        return super().copy(array=array, deep=deep, update=update, **kwargs)

        # # Create a copy of the rest of the fields
        # data = self.model_dump(
        #     mode = "python",
        #     exclude=(set(override.keys()))
        # )
        # if array is not None:
        #     data.pop("arr")
        #
        # if link_to_same_NOS:
        #     numerical_optimization_shift = kwargs.pop("numerical_optimization_shift", None)
        #
        # if deep:
        #     data = deepcopy(data)
        #
        # if array is None:
        #     array = data.pop("arr")
        #
        # data.update(override)
        # if link_to_same_NOS:
        #     data["numerical_optimization_shift"] = numerical_optimization_shift
        #
        # return type(self)(array, **data)

    def sample(self, mask: npt.NDArray[np.bool_|np.integer]) -> PointCloudData:
        mask = self.create_mask(mask)
        return self.copy(self.arr[mask, :], update={"scalar_fields": self.scalar_fields.sample(mask)})

    def reduce(self, mask: npt.NDArray[np.bool_|np.integer]) -> None:
        super().reduce(mask)
        self.scalar_fields.reduce(mask)
        if 'spher' in self.__dict__:
            self.__dict__['spher'] = self.__dict__['spher'][mask]

    def extract(self, mask: npt.NDArray[np.bool_|np.integer]) -> Self:
        extracted = super().extract(mask)
        if 'spher' in extracted.__dict__:
            extracted.__dict__['spher'] = extracted.__dict__['spher'][mask]
        return extracted

    @staticmethod
    def merge(*pcds: PointCloudData) -> PointCloudData:
        scalar_fields = ScalarFieldManager.merge([pcd.scalar_fields for pcd in pcds])
        if not all([pcds[0].optimized == pcd.optimized for pcd in pcds[1:]]):
            raise ValueError('Can only merge point clouds if they are all optimized or unoptimized.')

        if isinstance(pcds[0].socs_origin, np.ndarray):
            if not all([np.all(pcds[0].socs_origin == pcd.socs_origin) for pcd in pcds[1:]]):
                raise ValueError("Cannot merge point clouds where some origins are known and some are ambiguous")

        for pcd in pcds:
            if pcd.socs_origin is not None:
                raise ValueError("Cannot merge point clouds where some origins are known and some are ambiguous")

        # TODO update when implementing the transformations
        if pcds[0].project_transformation is None:
            for pcd in pcds[1:]:
                if pcd.project_transformation is not None:
                    raise ValueError("Cannot merge point clouds where only some project transforms are defined")
        else:
            for pcd in pcds[1:]:
                if not isinstance(pcd.project_transformation, np.ndarray):
                    raise ValueError("Cannot merge point clouds where only some project transforms are defined")

        xyz = np.concatenate([pcd.xyz for pcd in pcds], axis=0)

        return PointCloudData(xyz, scalar_fields=scalar_fields)

    def to_o3d(self, as_tensor: bool = True) -> o3d.geometry.PointCloud | o3d.t.geometry.PointCloud:
        """
            Converts the point cloud to an Open3D `PointCloud` object.

            Returns
            -------
            o3d.geometry.PointCloud
                An Open3D representation of the point cloud.
            pcd_o3d = o3d.geometry.PointCloud()
            if self.global_coordinate_shift is None:
                pcd_o3d.points = o3d.utility.Vector3dVector(self.xyz)
            else:
                pcd_o3d.points = o3d.utility.Vector3dVector((self.xyz + self.global_coordinate_shift).astype(np.float64))
            return pcd_o3d
        """

        if as_tensor:
            pcd_o3d = o3d.t.geometry.PointCloud()

            if self.optimized_shift is None:
                pcd_o3d.point.positions = o3d.core.Tensor(self.xyz)
            else:
                pcd_o3d.point.positions = o3d.core.Tensor(
                    (self.xyz.astype(np.float64) + self.optimized_shift.value.astype(np.float64))
                )

            for sf_name in set(self.scalar_fields.keys()):
                setattr(pcd_o3d.point, sf_name, o3d.core.Tensor(self.scalar_fields[sf_name].arr))

        else:
            pcd_o3d = o3d.geometry.PointCloud()
            if self.optimized_shift is None:
                pcd_o3d.points = o3d.utility.Vector3dVector(self.xyz)
            else:
                pcd_o3d.points = o3d.utility.Vector3dVector(
                    (self.xyz.astype(np.float64) + self.optimized_shift.value.astype(np.float64))
                )

            if 'rgb' in self.scalar_fields:
                pcd_o3d.colors = o3d.utility.Vector3dVector(self.rgb.as_normalised_float32())

            if 'normals' in self.scalar_fields:
                pcd_o3d.normals = o3d.utility.Vector3dVector(self.normals)

            for sf_name in set(self.scalar_fields.keys()).difference({'rgb', 'normals'}):
                warnings.warn(f"Cannot add scalar field '{sf_name}' to the pcd_o3d object")

        return pcd_o3d



    @classmethod
    def from_o3d(cls, pcd_o3d: o3d.geometry.PointCloud | o3d.t.geometry.PointCloud) -> PointCloudData:
        """
            @classmethod
            def from_o3d(cls, pcd_o3d: o3d.geometry.PointCloud, scan_center: Optional[NDArray[np.float_]] = None) -> Self:
                Creates a `PointCloudData` instance from an Open3D `PointCloud`.

                Parameters
                ----------
                pcd_o3d : o3d.geometry.PointCloud | o3d.t.geometry.PointCloud
                    An Open3D `PointCloud` object.
                scan_center : np.ndarray, optional
                    The scan center for spherical coordinate calculations.

                Returns
                -------
                PointCloudData
                    A new instance of the `PointCloudData` class.
                return cls(np.asarray(pcd_o3d.points), spherical_coordinates_origin=scan_center)
        """

        if isinstance(pcd_o3d, o3d.t.geometry.PointCloud):
            pcd = PointCloudData(pcd_o3d.point.positions.numpy())

            for name, value in pcd_o3d.point.items():
                if name != 'positions':
                    setattr(pcd, name, value.numpy())

        elif isinstance(pcd_o3d, o3d.geometry.PointCloud):
            pcd = PointCloudData(np.asarray(pcd_o3d.points))
            if len(pcd_o3d.colors):
                pcd.rgb = pcd_o3d.colors

            if len(pcd_o3d.normals):
                pcd.normals = pcd_o3d.normals

        else:
            raise TypeError(f"Input point cloud is not an open 3d type but {type(pcd_o3d)=}")

        return pcd

    # # DECIDE Implement in PCHandler or in pc2image
    # @classmethod
    # def from_range_image(
    #         cls,
    #         range_data: NDArray[np.floating],
    #         fov: FoV,
    #         scalar_fields: Optional[dict[str, NDArray[np.generic]] | ScalarFieldManager] = None,
    #         spherical_coordinates_origin: Optional[NDArray[np.float_]] = None,
    # ) -> Self:
    #     """
    #     Creates a `PointCloudData` instance from a range image.
    #
    #     Parameters
    #     ----------
    #     range_data : NDArray[np.floating]
    #         A 2D array representing the range values.
    #     fov : FoV
    #         The field of view defining the angular limits of the range image.
    #     scalar_fields : dict[str, NDArray[np.generic]] | ScalarFieldManager, optional
    #         Scalar fields corresponding to the range data.
    #     spherical_coordinates_origin : NDArray[np.float_], optional
    #         The origin for spherical coordinate calculations.
    #
    #     Returns
    #     -------
    #     PointCloudData
    #         A new instance of the `PointCloudData` class.
    #     """
    #     sfm = ScalarFieldManager() if scalar_fields is None else scalar_fields
    #     if not isinstance(sfm, ScalarFieldManager) and scalar_fields is not None:
    #         sfm = ScalarFieldManager()
    #         for sf_id, sf in scalar_fields.items():
    #             sfm.create_field(sf_id, sf.flatten())
    #
    #     resolution = range_data.shape
    #     elevation_range = np.linspace(
    #         fov.elevation_min, fov.elevation_max, num=resolution[0], endpoint=True, dtype=np.float32
    #     )
    #     horizontal_range = np.linspace(
    #         fov.horizontal_min, fov.horizontal_max, num=resolution[1], endpoint=True, dtype=np.float32
    #     )
    #
    #     elevation_mesh, horizontal_mesh = np.meshgrid(elevation_range, horizontal_range, indexing="ij")
    #
    #     ranges = range_data.flatten()
    #     elevations = elevation_mesh.flatten()
    #     horizontals = horizontal_mesh.flatten()
    #
    #     spherical_coordinates = np.vstack((ranges, elevations, horizontals)).T
    #     spherical_coordinates = spherical_coordinates[~np.isnan(ranges), :]
    #
    #     sfm_reduced = sfm[~np.isnan(ranges)]
    #
    #     return cls.from_spherical_coordinates(spherical_coordinates, sfm_reduced, spherical_coordinates_origin)