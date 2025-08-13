"""The base point cloud class for working with point cloud data."""

from __future__ import annotations

import logging
from typing import Any, Literal, Optional, Self, Sequence, cast, overload, TypeAlias, Union

import numpy as np
import open3d as o3d

from pydantic import Field, field_serializer, field_validator

from GSEGUtils.base_types import (
    Array_Nx3_Float_T,
    Array_Nx3_T,
    Array_Nx3_Uint8_T,
    ArrayT,
    IndexLike,
    VectorT,
    Vector_3_Float_T
)

from pchandler.geometry.coordinates import CartesianCoordinates
from pchandler.scalar_fields import (
    SF_T,
    NormalFields,
    RGBFields,
    ScalarField,
    ScalarFieldManager,
    ScalarFieldTriplet,
)

__all__ = ['PointCloudData',]

logger = logging.getLogger(__name__)


class PointCloudData(CartesianCoordinates):
    """Point Cloud Class with automatic validation and coordinate optimisation"""
    scalar_fields: ScalarFieldManager = Field(default_factory=ScalarFieldManager)

    # TODO decide on the kwargs and unpacking approach
    def __init__(
        self,
        /,
        xyz=None,
        *,
        rgb: RGBFields | Array_Nx3_Float_T | Array_Nx3_Uint8_T | None = None,
        normals: NormalFields | Array_Nx3_Float_T | None = None,
        intensity: ScalarField | VectorT | None = None,
        reflectance: ScalarField | VectorT | None = None,
        scalar_fields: (ScalarFieldManager
                                 | dict[str, ScalarField | ScalarFieldTriplet | Array_Nx3_T | VectorT | Sequence]
                                 | None) = None,
        socs_origin: Vector_3_Float_T = np.zeros(3, dtype=np.float32),
        **kwargs: Any,
    ):
        """

        Parameters
        ----------
        xyz : |Array_Nx3_Float_T|
            Input coordinates
        rgb : :class:`RGBFields` | |Array_Nx3_Float_T| | |Array_Nx3_Uint8_T| | None
        normals : |NormalFields| | |Array_Nx3_Float_T| | None
            Normal vectors corresponding for each point (normalized to unit vectors)
        intensity : |VectorT| | |ArrayT| | None
        reflectance : VectorT | ArrayT | None
        scalar_fields : :class:`ScalarFieldManager` | dict[str, :class:`ScalarField` | |ScalarFieldTriplet| | |Array_Nx3_T| | |VectorT| | Sequence] | None
            Additional custom scalar fields
        socs_origin: |Vector_3_Float_T|

            Scan original coordinate system (SOCS). Reference point for conversion to spherical coordinates.
        kwargs : dict[str, Any]
        """

        kwargs = {} | kwargs
        kwargs["scalar_fields"] = scalar_fields if not None else None
        kwargs['socs_origin'] = socs_origin

        super().__init__(xyz=xyz, **kwargs)  # type: ignore[call-overload]

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
        """Returns the number of points in the point cloud

        Returns
        -------
        int
        """
        return len(self)

    @field_validator("scalar_fields", mode="before")
    @classmethod
    def _convert_sfm(cls, value):
        """Ensure scalar_fields input is converted to a ScalarFieldManager"""
        if isinstance(value, dict):
            value = ScalarFieldManager(fields=value)
        elif value is None:
            value = ScalarFieldManager()
        return value

    @field_serializer("scalar_fields")
    def _drop_parent_weakref(self, scalar_fields: ScalarFieldManager):
        """Drop weakref to parent on serialization"""
        scalar_fields._parent = None
        return scalar_fields

    @property
    def normals(self: Self) -> NormalFields | None:
        """Returns the normal field

        Returns
        -------
        NormalFields | None
        """
        return self.scalar_fields.normals

    @normals.setter
    def normals(self, value: Optional[Array_Nx3_Float_T | NormalFields]) -> None:
        """Set the normal field (`None` will clear the field)

        Parameters
        ----------
        value : |Array_Nx3_Float_T| | NormalFields | None

        Returns
        -------

        """
        self.scalar_fields.normals = value

    @property
    def rgb(self) -> Optional[RGBFields]:
        """Returns the RGB field

        Returns
        -------
        RGBFields | None
        """
        return self.scalar_fields.rgb

    @rgb.setter
    def rgb(self, value: Optional[Array_Nx3_Float_T | Array_Nx3_Uint8_T | RGBFields]) -> None:
        """Set the RGB field (`None` will clear the field)

        Parameters
        ----------
        value : |Array_Nx3_Float_T| | |Array_Nx3_Uint8_T| | RGBFields | None

        Returns
        -------

        """
        self.scalar_fields.rgb = value

    @property
    def intensity(self) -> Optional[ScalarField]:
        """Returns the intensity field

        Returns
        -------
        ScalarField | None
        """
        return self.scalar_fields.intensity

    @intensity.setter
    def intensity(self, value: Optional[VectorT | ScalarField]) -> None:
        """Set the intensity field (`None` will clear the field)

        Parameters
        ----------
        value: |VectorT| | ScalarField | None

        Returns
        -------

        """
        self.scalar_fields.intensity = value

    @property
    def reflectance(self: Self) -> Optional[ScalarField]:
        """Returns the reflectance field

        Returns
        -------
        ScalarField | None
        """
        return self.scalar_fields.reflectance

    @reflectance.setter
    def reflectance(self, value: Optional[VectorT | ScalarField]) -> None:
        """Set the reflectance field (`None` will clear the field)

        Parameters
        ----------
        value: |VectorT| | ScalarField | None

        Returns
        -------

        """
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
        """Sample a copy of the point cloud

        Parameters
        ----------
        mask: IndexLike
            A vector like index object that corresponds to the number of points in the point cloud.

        Returns
        -------
        PointCloudData
        """
        mask = self.create_mask(mask)
        sample = self.copy(
            self.arr[mask, :],
            update={
                "scalar_fields": self.scalar_fields.sample(mask),
                "unshifted_bbox": None,
            },
        )
        return sample

    def reduce(self, mask: IndexLike) -> None:
        """Reduce the point cloud to a subset of points by a given mask

        Parameters
        ----------
        mask: IndexLike
            A vector like index object that corresponds to the number of points in the point cloud.

        Returns
        -------

        """
        super().reduce(mask)
        self.scalar_fields.reduce(mask)
        if "spher" in self.__dict__:
            self.__dict__["spher"] = self.__dict__["spher"][mask]

    def extract(self, mask: IndexLike) -> Self:
        """Extract a subset of points from the point cloud by a given mask.

        The object `self` that extract is called on will be reduced by this point set and these points will be returned
        as a new object

        Parameters
        ----------
        mask: IndexLike
            A vector like index object that corresponds to the number of points in the point cloud.

        Returns
        -------

        """
        extracted = super().extract(mask)
        return extracted

    @classmethod
    def merge(cls, *pcds: Self, **kwargs) -> Self:
        """ Merge a set of point clouds together

        If point clouds contain similar scalar fields, these will be also merged.
        Where a scalar field is missing in one point cloud, then that field will not be retained.

        The merge function will also manage any optimised shifts required by the new objects

        Parameters
        ----------
        pcds: PointCloudData
        kwargs: dict[str, Any]

        Returns
        -------
        PointCloudData
        """
        scalar_fields = ScalarFieldManager.merge([pcd.scalar_fields for pcd in pcds])
        return super(cls, cls).merge(*pcds, scalar_fields=scalar_fields, **kwargs)

    @overload
    def to_o3d(self, as_tensor: Literal[False] = ...) -> o3d.geometry.PointCloud: ...
    @overload
    def to_o3d(self, as_tensor: Literal[True]) -> o3d.t.geometry.PointCloud: ...

    def to_o3d(self, as_tensor: bool = False) -> o3d.geometry.PointCloud | o3d.t.geometry.PointCloud:
        """Converts the point cloud to an Open3D `PointCloud` object.

        Parameters
        ----------
        as_tensor: bool
            Flag as to convert to an Open3D tensor based PointCloud object.

        Returns
        -------
        |o3d.geometry.PointCloud| | |o3d.t.geometry.PointCloud|
        """

        if self.numerical_optimization_shift is not None:
            pcd = self.copy(
                update={"numerical_optimization_shift": None, "scalar_fields": None}, link_to_same_NOS=False
            )
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

            for sf_name in set(self.scalar_fields.keys()).difference({"rgb", "normals"}):
                if self.scalar_fields[sf_name] is not None:
                    logger.warning(f"Cannot add scalar field '{sf_name}' to the pcd_o3d object")

        return pcd_o3d

    @overload
    def from_o3d(self, pcd_o3d: o3d.geometry.PointCloud) -> PointCloudData: ...
    @overload
    def from_o3d(self, pcd_o3d: o3d.t.geometry.PointCloud) -> PointCloudData: ...

    @classmethod
    def from_o3d(cls, pcd_o3d: o3d.geometry.PointCloud | o3d.t.geometry.PointCloud) -> PointCloudData:
        """Convert an Open3D `PointCloud` object to a `PointCloudData` object.

        Parameters
        ----------
        pcd_o3d: |o3d.geometry.PointCloud| | |o3d.t.geometry.PointCloud|

        Returns
        -------
        PointCloudData
        """

        if isinstance(pcd_o3d, o3d.t.geometry.PointCloud):
            pcd = PointCloudData(pcd_o3d.point.positions.numpy())

            for name, value in pcd_o3d.point.items():
                if name != "positions":
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
