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

"""The base point cloud class for working with point cloud data."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, Optional, Self, Sequence, cast, overload

import numpy as np
from GSEGUtils.base_types import (
    Array_Nx3_Float_T,
    Array_Nx3_T,
    Array_Nx3_Uint8_T,
    ArrayT,
    IndexLike,
    Vector_3_Float_T,
    VectorT,
)
from pydantic import ConfigDict, Field, TypeAdapter, field_serializer, field_validator

from pchandler.geometry.coordinates import CartesianCoordinates
from pchandler.scalar_fields import (
    SF_T,
    NormalFields,
    RGBFields,
    ScalarField,
    ScalarFieldManager,
    ScalarFieldTriplet,
)

# WR-01 (Phase 3 code review) / SEC-02 extension: ``scalar_fields`` is a
# subclass-only field on :class:`PointCloudData`, so the parent's
# ``_FIELD_VALIDATORS`` dict (in ``coordinates.py``) does not cover it. A
# hostile pickle stream could otherwise ship anything in
# ``state["scalar_fields"]`` and bypass the per-field validation pass.
# :class:`ScalarFieldManager` is a :class:`pydantic.BaseModel` subclass, so
# strict-mode validation without ``arbitrary_types_allowed`` is sufficient.
_SCALAR_FIELDS_ADAPTER: TypeAdapter[ScalarFieldManager] = TypeAdapter(
    ScalarFieldManager,
    config=ConfigDict(arbitrary_types_allowed=True),
)

if TYPE_CHECKING:
    import open3d as o3d
    from py4dgeo import Epoch


__all__ = [
    "PointCloudData",
]

logger = logging.getLogger(__name__)


class PointCloudData(CartesianCoordinates):
    """Point cloud class with automatic validation and coordinate optimisation."""

    #: Contains and manages all the scalar fields associated with the point cloud coordinates
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
        scalar_fields: (
            ScalarFieldManager | dict[str, ScalarField | ScalarFieldTriplet | Array_Nx3_T | VectorT | Sequence] | None
        ) = None,
        socs_origin: Vector_3_Float_T | None = None,
        **kwargs: Any,
    ):
        """Construct a :class:`PointCloudData` from XYZ coordinates plus optional per-point scalar fields.

        Parameters
        ----------
        xyz : |Array_Nx3_Float_T|
            Input coordinates.
        rgb : :class:`RGBFields` | |Array_Nx3_Float_T| | |Array_Nx3_Uint8_T| | None
            Optional RGB colour per point.
        normals : |NormalFields| | |Array_Nx3_Float_T| | None
            Normal vectors corresponding to each point (normalised to unit vectors).
        intensity : :class:`ScalarField` | |VectorT| | None
            Optional intensity scalar field.
        reflectance : :class:`ScalarField` | |VectorT| | None
            Optional reflectance scalar field.
        scalar_fields : ScalarFieldManager | dict[str, ScalarField | Array_Nx3_T | VectorT | Sequence] | None
            Additional custom scalar fields. ``dict`` values may also be
            :class:`RGBFields`, :class:`NormalFields`, or
            :class:`ScalarFieldTriplet` instances.
        socs_origin : |Vector_3_Float_T|
            Scan original coordinate system (SOCS). Reference point for
            conversion to spherical coordinates.
        **kwargs : Any
            Additional keyword arguments forwarded to
            :class:`CartesianCoordinates`.
        """
        kwargs = {} | kwargs
        kwargs["scalar_fields"] = scalar_fields
        kwargs["socs_origin"] = socs_origin

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
        """Return the number of points in the point cloud.

        Returns
        -------
        int
            Point count (number of XYZ rows).
        """
        return len(self)

    @field_validator("scalar_fields", mode="before")
    @classmethod
    def _convert_sfm(cls, value: dict | ScalarFieldManager | None = None) -> ScalarFieldManager:
        """Coerce ``scalar_fields`` input into a :class:`ScalarFieldManager`.

        Parameters
        ----------
        value : dict | ScalarFieldManager | None
            Mapping of field name to data, an existing manager, or ``None``.

        Returns
        -------
        ScalarFieldManager
            A populated manager (empty if ``value`` is ``None``).
        """
        if isinstance(value, dict):
            value = ScalarFieldManager(fields=value)
        elif value is None:
            value = ScalarFieldManager()
        return value

    @field_serializer("scalar_fields")
    def _drop_parent_weakref(self, scalar_fields: ScalarFieldManager) -> ScalarFieldManager:
        """Drop the weakref to the parent point cloud on serialisation.

        Parameters
        ----------
        scalar_fields : ScalarFieldManager
            The manager about to be serialised.

        Returns
        -------
        ScalarFieldManager
            The same manager with ``_parent`` cleared.
        """
        scalar_fields._parent = None
        return scalar_fields

    @property
    def normals(self: Self) -> NormalFields | None:
        """Return the normal field, if set.

        Returns
        -------
        NormalFields | None
            Per-point unit normal vectors, or ``None`` if not set.
        """
        return self.scalar_fields.normals

    @normals.setter
    def normals(self, value: Optional[Array_Nx3_Float_T | NormalFields]) -> None:
        """Set the normal field (``None`` clears it).

        Parameters
        ----------
        value : Array_Nx3_Float_T | NormalFields | None
            Unit normal vectors per point, or ``None`` to clear.
        """
        self.scalar_fields.normals = value

    @property
    def rgb(self) -> Optional[RGBFields]:
        """Return the RGB field, if set.

        Returns
        -------
        RGBFields | None
            Per-point RGB colour, or ``None`` if not set.
        """
        return self.scalar_fields.rgb

    @rgb.setter
    def rgb(self, value: Optional[Array_Nx3_Float_T | Array_Nx3_Uint8_T | RGBFields]) -> None:
        """Set the RGB field (``None`` clears it).

        Parameters
        ----------
        value : Array_Nx3_Float_T | Array_Nx3_Uint8_T | RGBFields | None
            Per-point RGB colour data, or ``None`` to clear.
        """
        self.scalar_fields.rgb = value

    @property
    def intensity(self) -> Optional[ScalarField]:
        """Return the intensity field, if set.

        Returns
        -------
        ScalarField | None
            Per-point intensity scalar field, or ``None`` if not set.
        """
        return self.scalar_fields.intensity

    @intensity.setter
    def intensity(self, value: Optional[VectorT | ScalarField]) -> None:
        """Set the intensity field (``None`` clears it).

        Parameters
        ----------
        value : VectorT | ScalarField | None
            Per-point intensity data, or ``None`` to clear.
        """
        self.scalar_fields.intensity = value

    @property
    def reflectance(self: Self) -> Optional[ScalarField]:
        """Return the reflectance field, if set.

        Returns
        -------
        ScalarField | None
            Per-point reflectance scalar field, or ``None`` if not set.
        """
        return self.scalar_fields.reflectance

    @reflectance.setter
    def reflectance(self, value: Optional[VectorT | ScalarField]) -> None:
        """Set the reflectance field (``None`` clears it).

        Parameters
        ----------
        value : VectorT | ScalarField | None
            Per-point reflectance data, or ``None`` to clear.
        """
        self.scalar_fields.reflectance = value

    # def __setattr__(self, key, value) -> None:  # Todo: Is this necessary?
    #     super().__setattr__(key, value)

    def __setitem__(self, key: IndexLike, value: ArrayT | PointCloudData) -> None:
        """Raise :class:`IndexError` — index assignment is unsupported.

        Parameters
        ----------
        key : IndexLike
            Unused — present for the Python sequence protocol.
        value : ArrayT | PointCloudData
            Unused — present for the Python sequence protocol.

        Raises
        ------
        IndexError
            Always — ``PointCloudData`` is immutable via ``__setitem__``;
            use :meth:`copy` or serialise to a ``dict`` and reinstantiate.
        """
        raise IndexError(
            "Setting items in PointCloudData is not supported. Consider using the copy or "
            "dump data to a dict and reinstantiate."
        )

    def __hash__(self) -> int:
        """Return an identity-based hash so the model is usable in sets.

        Returns
        -------
        int
            The CPython ``id(self)`` of the instance.
        """
        return id(self)

    @classmethod
    def _reconstruct(cls, state: dict[str, Any]) -> Self:
        """Re-bind the scalar-field manager's parent weakref after pickle reconstruction.

        Per-field validation policy (SEC-02 / D-10 / WR-01 extension): the
        parent ``CartesianCoordinates._reconstruct`` already validates every
        field in its ``_FIELD_VALIDATORS`` dict, but ``scalar_fields`` is a
        subclass-only field and falls through unvalidated in the parent. This
        override validates ``state["scalar_fields"]`` against a dedicated
        :class:`pydantic.TypeAdapter` **before** delegating to the parent, so
        the "every pickled field validated" contract holds for
        ``PointCloudData``-shaped pickles too. A hostile pickle stream that
        smuggles a non-:class:`ScalarFieldManager` payload now raises
        :class:`pydantic.ValidationError` rather than silently producing a
        broken instance.

        Parameters
        ----------
        state : dict[str, Any]
            Pickled instance state forwarded to the parent class.

        Returns
        -------
        Self
            The reconstructed instance with its scalar-field parent re-linked.

        Raises
        ------
        pydantic.ValidationError
            If ``state["scalar_fields"]`` does not validate as a
            :class:`ScalarFieldManager` (e.g. wrong type, malformed contents),
            **or** if any field validated by the parent fails its own
            :class:`TypeAdapter`.
        """
        if "scalar_fields" in state:
            state = {
                **state,
                "scalar_fields": _SCALAR_FIELDS_ADAPTER.validate_python(state["scalar_fields"]),
            }
        obj: Self = super(cls, cls)._reconstruct(state)
        obj.scalar_fields.parent = obj
        return obj

    def sample(self, mask: IndexLike) -> PointCloudData:
        """Sample a copy of the point cloud restricted to ``mask``.

        Parameters
        ----------
        mask : IndexLike
            A vector-like index object that corresponds to the number of points in the point cloud.

        Returns
        -------
        PointCloudData
            A new point cloud containing only the masked points (and their scalar fields).
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
        """Reduce the point cloud in-place to the subset selected by ``mask``.

        Parameters
        ----------
        mask : IndexLike
            A vector-like index object that corresponds to the number of points in the point cloud.
        """
        super().reduce(mask)
        self.scalar_fields.reduce(mask)
        if "spher" in self.__dict__:
            self.__dict__["spher"] = self.__dict__["spher"][mask]

    def extract(self, mask: IndexLike) -> Self:
        """Extract a subset of points from the point cloud, removing them from ``self``.

        The object ``self`` that ``extract`` is called on is reduced by this point set, and the
        extracted points are returned as a new object.

        Parameters
        ----------
        mask : IndexLike
            A vector-like index object that corresponds to the number of points in the point cloud.

        Returns
        -------
        Self
            A new instance carrying the extracted points (and their scalar fields).
        """
        extracted = super().extract(mask)
        return extracted

    @classmethod
    def merge(
        cls,
        *pcds: Self,
        **kwargs: dict[str, Any],
    ) -> Self:
        """Merge a set of point clouds together.

        If point clouds contain similar scalar fields, these are also merged.
        Where a scalar field is missing in one point cloud, that field is not retained.

        The merge function also manages any optimised shifts required by the new objects.

        Parameters
        ----------
        *pcds : PointCloudData
            Point clouds to merge in order.
        **kwargs : dict[str, Any]
            Additional keyword arguments forwarded to the base ``merge`` implementation.

        Returns
        -------
        PointCloudData
            A new point cloud whose coordinates and scalar fields are the row-wise concatenation of ``pcds``.
        """
        scalar_fields = ScalarFieldManager.merge([pcd.scalar_fields for pcd in pcds])
        return super(cls, cls).merge(*pcds, scalar_fields=scalar_fields, **kwargs)

    @overload
    def to_o3d(self, as_tensor: Literal[False] = ...) -> o3d.geometry.PointCloud: ...
    @overload
    def to_o3d(self, as_tensor: Literal[True]) -> o3d.t.geometry.PointCloud: ...

    def to_o3d(self, as_tensor: bool = False) -> o3d.geometry.PointCloud | o3d.t.geometry.PointCloud:
        """Convert the point cloud to an Open3D ``PointCloud`` object.

        Parameters
        ----------
        as_tensor : bool
            If ``True``, build an Open3D tensor-based ``PointCloud`` (``o3d.t.geometry``).

        Returns
        -------
        |o3d.geometry.PointCloud| | |o3d.t.geometry.PointCloud|
            The Open3D point cloud, in legacy or tensor form depending on ``as_tensor``.

        Raises
        ------
        ModuleNotFoundError
            If ``open3d`` is not installed.
        """
        try:
            import open3d as _o3d
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("Open3d is not installed.") from e

        if self.numerical_optimization_shift is not None:
            pcd = self.copy(
                update={"numerical_optimization_shift": None, "scalar_fields": None}, link_to_same_NOS=False
            )
        else:
            pcd = self

        if as_tensor:
            pcd_o3d = _o3d.t.geometry.PointCloud()
            pcd_o3d.point.positions = _o3d.core.Tensor(pcd.xyz)

            for sf_name in set(self.scalar_fields.keys()):
                if self.scalar_fields[sf_name] is not None:
                    setattr(pcd_o3d.point, sf_name, _o3d.core.Tensor(cast(SF_T, self.scalar_fields[sf_name]).arr))

        else:
            pcd_o3d = _o3d.geometry.PointCloud()
            pcd_o3d.points = _o3d.utility.Vector3dVector(pcd.xyz)

            if self.rgb is not None:
                pcd_o3d.colors = _o3d.utility.Vector3dVector(self.rgb.as_normalised_float32())

            if self.normals is not None:
                pcd_o3d.normals = _o3d.utility.Vector3dVector(self.normals)

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
        """Convert an Open3D ``PointCloud`` object to a :class:`PointCloudData`.

        Parameters
        ----------
        pcd_o3d : |o3d.geometry.PointCloud| | |o3d.t.geometry.PointCloud|
            Open3D point cloud, either legacy or tensor-based.

        Returns
        -------
        PointCloudData
            The converted point cloud (colours/normals are carried over from the legacy Open3D variant; scalar
            attributes are carried over from the tensor variant).

        Raises
        ------
        ModuleNotFoundError
            If ``open3d`` is not installed.
        TypeError
            If ``pcd_o3d`` is neither an ``o3d.geometry.PointCloud`` nor an ``o3d.t.geometry.PointCloud``.
        """
        try:
            import open3d as _o3d
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("Open3d is not installed.") from e

        if isinstance(pcd_o3d, _o3d.t.geometry.PointCloud):
            pcd = PointCloudData(pcd_o3d.point.positions.numpy())

            for name, value in pcd_o3d.point.items():
                if name != "positions":
                    setattr(pcd, name, value.numpy())

        # Original o3d object -> Other fields not supported
        elif isinstance(pcd_o3d, _o3d.geometry.PointCloud):
            pcd = PointCloudData(np.asarray(pcd_o3d.points))
            if len(pcd_o3d.colors):
                pcd.rgb = np.asarray(pcd_o3d.colors)

            if len(pcd_o3d.normals):
                pcd.normals = np.asarray(pcd_o3d.normals)

        else:
            raise TypeError(f"Input point cloud is not an open 3d type but {type(pcd_o3d)=}")

        return pcd

    def to_py4dgeo(self) -> Epoch:
        """Convert the :class:`PointCloudData` to a py4dgeo :class:`Epoch`.

        The optional numerical optimization shift is reverted so py4dgeo
        receives float64 world-frame coordinates (mirrors :meth:`to_o3d`).

        Returns
        -------
        py4dgeo.Epoch
            Epoch carrying ``xyz`` in world-frame, plus optional normals and
            additional scalar-field dimensions.

        Raises
        ------
        ModuleNotFoundError
            If ``py4dgeo`` is not installed.
        """
        try:
            from py4dgeo import Epoch as _Epoch
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("py4dgeo is not installed. Install it to use PointCloudData.to_py4dgeo().") from e

        # BUG-07 (D-15): mirror to_o3d's un-shift pattern so py4dgeo receives
        # float64 world-frame coordinates rather than shifted float32.
        # The temporary copy exists *solely* to obtain world-frame XYZ -- it is
        # constructed with ``scalar_fields=None`` which causes ``_convert_sfm``
        # to substitute an empty :class:`ScalarFieldManager`. Therefore normals
        # and scalar fields must be read from ``self`` (the original), not from
        # ``pcd`` (the cleared copy); the latter would silently ship empty
        # arrays to py4dgeo. CR-01 (Phase 3 code review) -- ``to_o3d`` follows
        # the same pattern, reading sfm fields from ``self``.
        if self.numerical_optimization_shift is not None:
            pcd = self.copy(
                update={"numerical_optimization_shift": None, "scalar_fields": None},
                link_to_same_NOS=False,
            )
        else:
            pcd = self

        return _Epoch(
            cloud=pcd.xyz,
            normals=self.normals if self.normals is not None else None,
            additional_dimensions=self.scalar_fields.as_struct_array(),
        )

    @classmethod
    def from_py4dgeo(cls, epoch: Epoch) -> PointCloudData:
        """Convert a py4dgeo :class:`Epoch` to a :class:`PointCloudData`.

        Parameters
        ----------
        epoch : Epoch
            A py4dgeo ``Epoch`` to convert.

        Returns
        -------
        PointCloudData
            A new point cloud built from the epoch's cloud, normals,
            and additional dimensions.

        Raises
        ------
        ModuleNotFoundError
            If ``py4dgeo`` is not installed.
        """
        try:
            from py4dgeo import Epoch as _Epoch  # noqa: F401  # Availability probe; _Epoch is not used directly here.
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "py4dgeo is not installed. Install it to use PointCloudData.from_py4dgeo()."
            ) from e

        sfs = {}
        for name in epoch.additional_dimensions.dtype.names:
            sfs[name] = epoch.additional_dimensions[name].squeeze()

        pcd = cls(epoch.cloud, scalar_fields=sfs)
        # IN-04 (Phase 3 code review): use py4dgeo's public property
        # ``Epoch.normals`` rather than the implementation-detail private
        # slot ``epoch.__dict__["_normals"]``. The property returns ``None``
        # when normals are unset, matching the prior guard semantics without
        # the brittle dict-key reach.
        normals = epoch.normals
        if normals is not None:
            pcd.normals = normals

        return pcd
