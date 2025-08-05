from __future__ import annotations

from typing import Optional, Self, overload, Literal, cast, Sequence, Any
import logging

import numpy as np
import open3d as o3d
from pydantic import Field, field_validator, field_serializer

from GSEGUtils.base_types import Array_Nx3_T, IndexLike, VectorT, Array_Nx3_Float_T, Array_Nx3_Uint8_T, ArrayT
from pchandler.geometry.coordinates import CartesianCoordinates
from pchandler.scalar_fields.scalar_field_manager import ScalarFieldManager, SF_T
from pchandler.scalar_fields.scalar_fields import (
    NormalFields,
    RGBFields,
    ScalarField,
    ScalarFieldTriplet,
)

logger = logging.getLogger(__name__)

RgbInputT = Optional[RGBFields | Array_Nx3_Float_T | Array_Nx3_Uint8_T]
NormalInputT = Optional[NormalFields | Array_Nx3_Float_T]
IntensityInputT = Optional[VectorT | ArrayT]
ReflectanceInputT = Optional[VectorT | ArrayT]
SFM_T = Optional[ScalarFieldManager | dict[str, ScalarField | ScalarFieldTriplet | Array_Nx3_T | VectorT | Sequence]]


class PointCloudData(CartesianCoordinates):
    """

    Parameters
    ----------
    xyz
    rgb
    normals
    intensity
    reflectance
    scalar_fields
    """
    scalar_fields: ScalarFieldManager = Field(default_factory=ScalarFieldManager)

    def __init__(self,
                 /,
                 xyz=None,
                 *,
                 rgb: RgbInputT = None,
                 normals: NormalInputT = None,
                 intensity: IntensityInputT = None,
                 reflectance: ReflectanceInputT = None,
                 scalar_fields: SFM_T = None,
                 **kwargs: Any):

        kwargs = {} | kwargs
        kwargs['scalar_fields'] = scalar_fields if not None else None

        super().__init__(xyz=xyz, **kwargs) # type: ignore[call-overload]

        self.scalar_fields.parent = self
        self.scalar_fields.validate_lengths()

        if rgb is not None:
            self.rgb = rgb

        if normals is not None:
            self.normals = normals

        if intensity is not None:
            self.intensity = intensity

        if reflectance is not None:
            self.reflectance = reflectance

    @property
    def nbPoints(self) -> int:
        return len(self)

    @field_validator('scalar_fields', mode="before")
    @classmethod
    def convert_sfm(cls, value):
        if isinstance(value, dict):
            value = ScalarFieldManager(fields=value)
        elif value is None:
            value = ScalarFieldManager()
        return value

    @field_serializer('scalar_fields')
    def drop_parent_weakref(self, scalar_fields: ScalarFieldManager):
        scalar_fields._parent = None
        return scalar_fields

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
    def reflectance(self: Self) -> Optional[ScalarField]:
        return self.scalar_fields.reflectance

    @reflectance.setter
    def reflectance(self, value: Optional[VectorT | ScalarField]) -> None:
        self.scalar_fields.reflectance = value

    def __setattr__(self, key, value):
        super().__setattr__(key, value)

    def __setitem__(self, key: IndexLike, value: ArrayT | PointCloudData) -> None:
        raise IndexError(
            f"Setting items in PointCloudData is not supported. Consider using the copy or "
            f"dump data to a dict and reinstantiate."
        )

    def __hash__(self) -> int:
        return id(self)

    @classmethod
    def _reconstruct(cls, state: dict) -> Self:
        obj: Self = super(cls, cls)._reconstruct(state)
        obj.scalar_fields.parent = obj
        return obj

    def sample(self, mask: IndexLike) -> PointCloudData:
        mask = self.create_mask(mask)
        sample = self.copy(
            self.arr[mask, :],
            update={
                "scalar_fields": self.scalar_fields.sample(mask),
                "unshifted_bbox": None,
            }
        )
        return sample

    def reduce(self, mask: IndexLike) -> None:
        super().reduce(mask)
        self.scalar_fields.reduce(mask)
        if 'spher' in self.__dict__:
            self.__dict__['spher'] = self.__dict__['spher'][mask]

    def extract(self, mask: IndexLike) -> Self:
        extracted = super().extract(mask)
        return extracted

    @classmethod
    def merge(cls, *pcds: Self, **kwargs,) -> Self:
        scalar_fields = ScalarFieldManager.merge([pcd.scalar_fields for pcd in pcds])
        return super(cls, cls).merge(*pcds, scalar_fields=scalar_fields, **kwargs)

    @overload
    def to_o3d(self, as_tensor: Literal[False] = ...) -> o3d.geometry.PointCloud: ...
    @overload
    def to_o3d(self, as_tensor: Literal[True]) -> o3d.t.geometry.PointCloud: ...

    def to_o3d(self, as_tensor: bool = False) -> o3d.geometry.PointCloud | o3d.t.geometry.PointCloud:
        """
            Converts the point cloud to an Open3D `PointCloud` object.
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
                    setattr(pcd_o3d.point, sf_name, o3d.core.Tensor(cast(SF_T, self.scalar_fields[sf_name]).arr))

        else:
            pcd_o3d = o3d.geometry.PointCloud()
            pcd_o3d.points = o3d.utility.Vector3dVector(pcd.xyz)

            if self.rgb is not None:
                pcd_o3d.colors = o3d.utility.Vector3dVector(self.rgb.as_normalised_float32())

            if self.normals is not None:
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

        # Tensor object -> everything is a field
        if isinstance(pcd_o3d, o3d.t.geometry.PointCloud):
            pcd = PointCloudData(pcd_o3d.point.positions.numpy())

            for name, value in pcd_o3d.point.items():
                if name != 'positions':
                    setattr(pcd, name, value.numpy())

        # Original o3d object -> Other fields not supported
        elif isinstance(pcd_o3d, o3d.geometry.PointCloud):
            pcd = PointCloudData(np.asarray(pcd_o3d.points))
            if len(pcd_o3d.colors):
                pcd.rgb = np.asarray(pcd_o3d.colors)

            if len(pcd_o3d.normals):
                pcd.normals = np.asarray(pcd_o3d.normals)

        else:
            raise TypeError(f"Input point cloud is not an open 3d type but {type(pcd_o3d)=}")

        return pcd
