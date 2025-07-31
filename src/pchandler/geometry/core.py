from __future__ import annotations

from typing import (Any, Optional, Self, MutableMapping, Callable,
                    overload, Literal, Unpack, Required, NotRequired)
import logging

import numpy as np
import numpy.typing as npt
import open3d as o3d
from pydantic import Field, field_validator, model_validator, AliasChoices

from pchandler.base_types import Array_Nx3_T, IndexLike, VectorT, Array_Nx3_Float_T, Array_Nx3_Uint8_T, ArrayT
from pchandler.geometry.coordinates import CartesianCoordinates, CartesianKw

from pchandler.geometry.scalar_field_manager import ScalarFieldManager
from pchandler.geometry.scalar_fields import (NormalFields, RGBFields, ScalarField, NormalisedInt16ScalarField,
                                              ScalarFieldTriplet)

logger = logging.getLogger(__name__)

class PointCloudDataKw(CartesianKw, total=False):
    scalar_fields: Optional[ScalarFieldManager[ScalarField | ScalarFieldTriplet]]
    rgb: Optional[npt.NDArray[np.uint8 | np.float32 | np.float64]]
    normals: Optional[npt.NDArray[np.float32 | np.float64]]
    intensity: Optional[npt.NDArray[np.uint16 | np.float32 | np.float64]]
    reflectance: Optional[npt.NDArray[np.uint16 | np.float32 | np.float64]]

class PointCloudData(CartesianCoordinates):

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

    def __init__(self, xyz=None, **kwargs: Unpack[PointCloudDataKw]):
        super().__init__(xyz=xyz, **kwargs)

    @property
    def nbPoints(self) -> int:
        return len(self)

    @field_validator('scalar_fields', mode="before")
    @classmethod
    def _convert_scalar_fields(cls, value):
        logger.debug(f"Running `_convert_scalar_fields` validator on {cls}")
        if not isinstance(value, ScalarFieldManager):
            value = ScalarFieldManager(fields=value)
        return value

    @model_validator(mode="after")
    def ensure_scalar_field_manager_pointer(self) -> Self:
        """Revalidate model to ensure that the weakref points to the correct object"""

        self.scalar_fields.parent = self
        return self

    @model_validator(mode="after")
    def _move_into_scalar_fields(self) -> Self:
        logger.debug(f"Running `_move_into_scalar_fields` validator on {self}")
        # map any provided “_input” into the scalar_fields manager
        for name in ("rgb", "normals", "intensity", "reflectance"):
            inp = getattr(self, f"{name}_input", None)
            if inp is not None:
                setattr(self, name, inp)
                delattr(self, f"{name}_input")
                # setattr(self, f"{name}_input", None)
        return self

    @property
    def normals(self: Self) -> Optional[NormalFields]:
        return self.scalar_fields.normals

    @normals.setter
    def normals(self, value: Optional[Array_Nx3_T | NormalFields]) -> None:
        self.scalar_fields.normals = value

    @property
    def rgb(self) -> Optional[RGBFields]:
        return self.scalar_fields.rgb

    @rgb.setter
    def rgb(self, value: Optional[Array_Nx3_Float_T | Array_Nx3_Uint8_T | RGBFields]) -> None:
        self.scalar_fields.rgb = value

    @property
    def intensity(self) -> Optional[ScalarField]:
        return self.scalar_fields.intensity

    @intensity.setter
    def intensity(self, value: Optional[VectorT | ScalarField]) -> None:
        self.scalar_fields.intensity = value

    @property
    def reflectance(self: Self) -> Optional[NormalisedInt16ScalarField]:
        return self.scalar_fields.reflectance

    @reflectance.setter
    def reflectance(self, value: Optional[VectorT | ScalarField]) -> None:
        self.scalar_fields.reflectance = value

    def __setitem__(self, key: IndexLike, value: ArrayT | PointCloudData) -> None:
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

    def copy(self: Self,
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

        if link_to_same_NOS and "numerical_optimization_shift" not in update:
            update["numerical_optimization_shift"] = self.numerical_optimization_shift
        update["_shift_applied_by"] = self._shift_applied_by # TODO: Rework structure!
        update["id"] = None

        return super().copy(array=array, deep=deep, update=update, **kwargs)


    def sample(self, mask: npt.NDArray[np.bool_|np.integer]) -> PointCloudData:
        mask = self.create_mask(mask)
        sample = self.copy(
            self.arr[mask, :],
            update={
                "scalar_fields": self.scalar_fields.sample(mask),
                "unshifted_bbox": None,
            }
        )
        return sample

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

    @classmethod
    def merge(cls, *pcds: PointCloudData) -> PointCloudData:
        # Check for same optimization shift
        scalar_fields = ScalarFieldManager.merge([pcd.scalar_fields for pcd in pcds])
        return super().merge(*pcds, scalar_fields=scalar_fields)


    @overload
    def to_o3d(self, as_tensor: Literal[False] = ...) -> o3d.geometry.PointCloud: ...
    @overload
    def to_o3d(self, as_tensor: Literal[True]) -> o3d.t.geometry.PointCloud: ...

    def to_o3d(self, as_tensor: bool = False) -> o3d.geometry.PointCloud | o3d.t.geometry.PointCloud:
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
        if self.numerical_optimization_shift is not None:
            pcd = self.copy(update={
                "numerical_optimization_shift": None,
                "scalar_fields": None
            }, link_to_same_NOS=False)
        else:
            pcd = self

        if as_tensor:
            pcd_o3d = o3d.t.geometry.PointCloud()
            pcd_o3d.point.positions = o3d.core.Tensor(pcd.xyz)

            for sf_name in set(self.scalar_fields.keys()):
                if self.scalar_fields[sf_name] is not None:
                    setattr(pcd_o3d.point, sf_name, o3d.core.Tensor(self.scalar_fields[sf_name].arr))

        else:
            pcd_o3d = o3d.geometry.PointCloud()
            pcd_o3d.points = o3d.utility.Vector3dVector(pcd.xyz)

            if 'rgb' in self.scalar_fields:
                pcd_o3d.colors = o3d.utility.Vector3dVector(self.rgb.as_normalised_float32())

            if 'normals' in self.scalar_fields:
                pcd_o3d.normals = o3d.utility.Vector3dVector(self.normals)

            for sf_name in set(self.scalar_fields.keys()).difference({'rgb', 'normals'}):
                if self.scalar_fields[sf_name] is not None:
                    logger.warning(f"Cannot add scalar field '{sf_name}' to the pcd_o3d object")

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
