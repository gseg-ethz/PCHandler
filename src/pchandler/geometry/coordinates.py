# pchandler - Toolbox for point-cloud handling, processing and analysis
#
# Copyright (c) 2025, Nicholas Meyer, Geosensors and Engineering Geodesy,
# Institute of Geodesy and Photogrammetry, ETH Zurich, Switzerland
# SPDX-License-Identifier: BSD-3-Clause
#
# Author: Nicholas Meyer (meyernic@ethz.ch)

"""Cartesian coordinate classes with numerical-precision shift management.

Defines :class:`AbstractCoordinates` and :class:`Abstract3dCoordinates` base
classes, :class:`CartesianCoordinates` (the main user-facing 3D coordinate
container), plus the ``rhv2xyz`` / ``xyz2rhv`` spherical/Cartesian conversion
helpers.
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from functools import cached_property
from typing import (
    Any,
    MutableMapping,
    NotRequired,
    Optional,
    Self,
    Type,
    TypedDict,
    TypeVar,
    Union,
    Unpack,
    overload,
)

import numpy as np
import numpy.typing as npt
from GSEGUtils.base_arrays import ArrayNx2, ArrayNx3, FixedLengthArray
from GSEGUtils.base_types import (
    Array_3x3_T,
    Array_4x4_T,
    Array_Nx2_Float_T,
    Array_Nx3_Float_T,
    Array_Nx3_T,
    IndexLike,
    Vector_3_T,
    Vector_Float_T,
)
from GSEGUtils.constants import DEFAULT_CONFIG
from pydantic import UUID4, AliasChoices, Field, PrivateAttr, validate_call

from pchandler.geometry import OptimizedShift, OptimizedShiftManager
from pchandler.geometry.spherical import FoV
from pchandler.geometry.transforms import (
    Transform,
    _Transform3x3,
    _Transform4x4,
)
from pchandler.geometry.util import MinMaxPoints

__all__ = ["CartesianCoordinates", "AbstractCoordinates", "Abstract2dCoordinates", "Abstract3dCoordinates"]

logger = logging.getLogger(__name__)

TransformT = Union[_Transform4x4, _Transform3x3, Transform]

CartesianT = TypeVar("CartesianT", bound="CartesianCoordinates")


class Abstract3dKw(TypedDict, total=False):
    """Optional keyword arguments accepted by :class:`Abstract3dCoordinates`."""

    project_transformation: NotRequired[Array_4x4_T]
    socs_origin: NotRequired[Optional[Vector_3_T]]


class CartesianKw(Abstract3dKw, total=False):
    """Optional keyword arguments accepted by :class:`CartesianCoordinates`."""

    numerical_optimization_shift: NotRequired[Optional[OptimizedShift]]
    unshifted_bbox: NotRequired[Optional[MinMaxPoints]]
    _shift_applied_by: NotRequired[Optional[OptimizedShift]]


class CartesianKwFull(CartesianKw, total=False):
    """Full keyword surface for :class:`CartesianCoordinates` (includes ``arr``)."""

    arr: NotRequired[Array_Nx3_T]


class AbstractCoordinates(FixedLengthArray, ABC):
    """Abstract base for all coordinate classes; carries a UUID4 identity."""

    id: UUID4 = Field(default_factory=uuid.uuid4, alias="_id")


class Abstract2dCoordinates(ArrayNx2, AbstractCoordinates, ABC):
    """Abstract base for 2D coordinate classes exposing ``row`` and ``col``."""

    @property
    @abstractmethod
    def row(self) -> npt.NDArray[Any]:
        """Return the row component of the 2D coordinates."""
        ...

    @property
    @abstractmethod
    def col(self) -> npt.NDArray[Any]:
        """Return the column component of the 2D coordinates."""
        ...


class Abstract3dCoordinates(ArrayNx3, AbstractCoordinates, ABC):
    """Abstract coordinates with support for a transformation to custom coordinate system and a scan center parameter.

    Parameters
    ----------
    project_transformation : Array_4x4_T | None
        Affine transformation array representing transform from scan coordinates to project coordinates.

    socs_origin : Vector_3_T | None
        Scan center coordinate in the current coordinate system.

    """

    project_transformation: Optional[Array_4x4_T] = None
    socs_origin: Optional[Vector_3_T] = None

    @property
    @abstractmethod
    def xyz(self) -> npt.NDArray[np.floating]:
        """Return the Cartesian coordinates as ``(N, 3)`` XYZ."""
        ...

    @property
    @abstractmethod
    def spher(self) -> npt.NDArray[np.floating]:
        """Return the spherical coordinates as ``(N, 3)`` RHV."""
        ...

    def __matmul__(self, other: Any) -> Self:
        """Reject left matrix multiplication on coordinates (use ``self.arr @ b`` instead)."""
        raise NotImplementedError(
            "Left matrix multiplication is not supported for coordinates class. "
            "If necessary, access the array data directly: y = a.arr @ b"
        )

    def __rmatmul__(self, matrix: Any) -> Self:
        """Right matrix multiplication: apply a 4x4 homogeneous or 3x3 rotation/scale matrix."""
        # 4x4 Homogeneous transform matrix
        if matrix.shape == (4, 4):
            if np.all(matrix[3, :] == [0, 0, 0, 1]):
                temp = (matrix @ self.H.T).T[:, :3]
            else:
                raise ValueError(f"4x4 matrix is not a transformation matrix with all nonzero values: {matrix}")

        # Rotation / scale matrix
        elif matrix.shape == (3, 3):
            temp = (matrix @ self.T).T

        # Otherwise, apply it
        else:
            return matrix @ self.arr

        return self.copy(temp)

    @validate_call(config=DEFAULT_CONFIG)
    def __imatmul__(self, other: Any) -> Self:
        """Reject in-place matrix multiplication; use ``x = A @ x`` instead."""
        raise NotImplementedError(
            "In place matrix multiplication not supported to avoid ambiguity between left and right multiplication.\n"
            "Use direct assignment instead. E.g. x = A @ x"
        )


class CartesianCoordinates(Abstract3dCoordinates):
    """Cartesian Coordinate class with support for numerical optimizations and helper access methods.

    Parameters
    ----------
    arr : Array_Nx3_T
        Raw coordinate data
    unshifted_bbox : MinMaxPoints | None
        Bounding box of the original input coordinates (before optimization shift)
    numerical_optimization_shift : OptimizedShift | None
        Optimization shift applied to the point cloud.
    socs_origin : Vector_3_T | None
        Scan center coordinate in the current coordinate system.

    """

    arr: Array_Nx3_T = Field(..., validation_alias=AliasChoices("arr", "xyz"))
    unshifted_bbox: Optional[MinMaxPoints] = Field(default=None)
    _shift_applied_by: Optional[OptimizedShift] = PrivateAttr(default=None)
    numerical_optimization_shift: Optional[OptimizedShift] = Field(default_factory=OptimizedShift, exclude=False)

    @overload
    def __init__(self, /, xyz: Array_Nx3_T | Self, **kwargs: Unpack[CartesianKw]): ...

    @overload
    def __init__(self, /, **kwargs: Unpack[CartesianKwFull]): ...

    def __init__(self, /, xyz: Optional[Array_Nx3_T | Self] = None, **kwargs: Unpack[CartesianKwFull]):
        """Build a :class:`CartesianCoordinates`, auto-computing a numerical-precision shift.

        On construction the object attempts to apply a numerical-precision
        shift so the coordinates fit in ``float32`` without loss. The user can
        pass ``numerical_optimization_shift=None`` to force the coordinates to
        keep the input precision and original coordinate system.

        Parameters
        ----------
        xyz : Array_Nx3_T | Self, optional
            XYZ coordinates as an ``(N, 3)`` array, or another
            :class:`CartesianCoordinates` instance to copy from. May also be
            supplied via the ``arr=`` keyword.
        **kwargs
            Optional fields: ``socs_origin``, ``project_transformation``,
            ``numerical_optimization_shift``, ``unshifted_bbox``,
            ``_shift_applied_by``.
        """
        # Accept xyz/arr as a positional argument
        if xyz is not None:
            if "arr" in kwargs:
                raise ValueError("Cannot pass both positional and keyword for xyz/arr")
            kwargs["arr"] = xyz

        prev_shift: Optional[OptimizedShift] = kwargs.pop("_shift_applied_by", None)
        super().__init__(**kwargs)  # type: ignore[misc]

        # Set the private attribute after initialization
        object.__setattr__(self, "_shift_applied_by", prev_shift)

        self.compute_unshifted_bbox()
        self._process_shift()

    def compute_unshifted_bbox(self, overwrite: bool = False):
        """Compute the bounding box of the point cloud's original (unshifted) coordinates.

        Parameters
        ----------
        overwrite : bool, default=False
            Recompute the bounding box even if one is already cached.
        """
        if self.unshifted_bbox is None or overwrite:
            applied_shift = None if self._shift_applied_by is None else self._shift_applied_by.value

            object.__setattr__(
                self, "unshifted_bbox", MinMaxPoints.from_points(self.arr, already_applied_shift_vec=applied_shift)
            )

    def _process_shift(self):
        """Resolve the four-case state machine between ``_shift_applied_by`` and ``numerical_optimization_shift``.

        Cases:

        - Case 1 -- ``prev_shift is None``, ``NOS is None``:
          basic init, register to NOS.
        - Case 2 -- ``prev_shift is None``, ``NOS`` exists:
          revert to ``prev_shift``, convert to ``float64``, unregister
          ``prev_shift``.
        - Case 3 -- ``prev_shift`` exists, ``NOS is None``:
          register to NOS.
        - Case 4 -- ``prev_shift`` exists, ``NOS`` exists:
          if same, register to NOS; else apply the difference, unregister
          ``prev``, register ``NOS``.
        """
        prev_shift = self._shift_applied_by
        if self.numerical_optimization_shift is not None:
            self._register_with_shift_at_osm()  # This could possibly set self.nos to None

        # Case 1 - Initialisation with NOS = None
        if prev_shift is None and self.numerical_optimization_shift is None:
            return

        # Case 2 - Initialization with a NOS value
        elif prev_shift is None and self.numerical_optimization_shift is not None:
            self.update_shift(-self.numerical_optimization_shift.value)

        # Case 3 - Case where a point cloud has a previous shift but should no longer be "optimized"
        elif prev_shift is not None and self.numerical_optimization_shift is None:
            self.update_shift(prev_shift.value)
            prev_shift.unregister(self)

        # Case 4 - Case when pcd has been pickled but another optimization shift is provided,
        #          or has changed since it was pickled
        elif prev_shift is not None and self.numerical_optimization_shift is not None:
            if prev_shift is not self.numerical_optimization_shift:
                delta_shift = prev_shift.value - self.numerical_optimization_shift.value
                self.update_shift(delta_shift)
                prev_shift.unregister(self)

        object.__setattr__(self, "_shift_applied_by", self.numerical_optimization_shift)

    def reduce(self, index: IndexLike) -> None:
        """Reduce the coordinates to the point set selected by ``index`` (in place).

        Supports both numpy basic and advanced indexing.

        Parameters
        ----------
        index : IndexLike
            Index or boolean mask selecting the rows to keep.
        """
        super().reduce(index)
        self.compute_unshifted_bbox(overwrite=True)  # In case limits have been reduced

    def sample(self, index: IndexLike) -> Self:
        """Sample a copy of the coordinates using ``index``.

        Parameters
        ----------
        index : IndexLike
            Index or boolean mask selecting the rows to keep.

        Returns
        -------
        CartesianCoordinates
            New :class:`CartesianCoordinates` carrying the sampled rows.
        """
        new_sample: Self = super().sample(index)
        new_sample.compute_unshifted_bbox(overwrite=True)
        return new_sample

    def update_shift(self, delta_shift: Vector_3_T) -> None:
        """Apply ``delta_shift`` to the coordinates and adjust the dtype accordingly.

        The dtype of the updated vectors is ``float32`` when a
        numerical-optimization shift is registered, and ``float64`` otherwise.

        Parameters
        ----------
        delta_shift : Vector_3_T
            The shift vector to be added.
        """
        target_dtype = np.float64 if self.numerical_optimization_shift is None else np.float32
        self.arr = (self.arr + delta_shift).astype(target_dtype, copy=False)
        if self.socs_origin is not None:
            self.socs_origin = (self.socs_origin + delta_shift).astype(target_dtype, copy=False)

    def _register_with_shift_at_osm(self) -> None:
        """Register ``self.numerical_optimization_shift`` with the :class:`OptimizedShiftManager`.

        The manager determines whether the requested shift is feasible. It may
        return a different shift, or ``None`` if no shift is feasible — in
        which case ``self.numerical_optimization_shift`` is reset to ``None``
        and the coordinates continue in ``float64`` mode.
        """
        osm = OptimizedShiftManager()
        try:
            shift = osm.register_coordinates_to_shift(self, self.numerical_optimization_shift)  # type: ignore[arg-type]
            if shift is not self.numerical_optimization_shift:
                logger.info("The input numerical_optimization_shift was not feasible and was replaced by a new one.")
                object.__setattr__(self, "numerical_optimization_shift", shift)

        except OptimizedShiftManager.ShiftNotFeasibleError:
            logger.warning("No numerical_optimization_shift was feasible. Will continue in float64 mode.")
            object.__setattr__(self, "numerical_optimization_shift", None)

    def __setattr__(self, key, value):
        """Guard ``_shift_applied_by`` and rerun ``_process_shift`` on ``numerical_optimization_shift`` assignment."""
        if key == "_shift_applied_by":
            raise AttributeError("Cannot assign to '{key}'")
        if key == "numerical_optimization_shift":
            object.__setattr__(self, key, value)
            self._process_shift()
            return
        super().__setattr__(key, value)

    def __hash__(self) -> int:
        """Return a hash based on object identity (id)."""
        return id(self)

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Dump the model to a dict, preserving the private ``_shift_applied_by`` attribute.

        Parameters
        ----------
        **kwargs
            Passed through to :meth:`pydantic.BaseModel.model_dump`.

        Returns
        -------
        dict[str, Any]
            Model state including ``_shift_applied_by``.
        """
        data = super().model_dump(**kwargs)
        data["_shift_applied_by"] = self._shift_applied_by
        return data

    def __reduce__(self) -> Any:
        """Return the (callable, state) tuple used by :mod:`pickle` to reconstruct ``self``."""
        logger.debug(f"Running `{self.__class__}.reduce()` on {self.id}")
        state = self.model_dump()
        # state["_shift_applied"] = self._shift_applied
        return self._reconstruct, (state,)

    @classmethod
    def _reconstruct(cls, state: dict) -> Self:
        obj: Self = cls.model_construct(**state)
        logger.debug(f"{cls} with id={obj.id} reconstructed")
        obj._process_shift()

        return obj

    # noinspection PyPep8Naming
    def copy(
        self: Self,  # type: ignore[override]
        array: Optional[Array_Nx3_Float_T | Self] = None,
        *,
        deep: bool = True,
        update: Optional[MutableMapping[str, Any]] = None,
        link_to_same_NOS: bool = True,
        **kwargs: dict[str, Any],
    ) -> Self:
        """Produce a deep or shallow copy of the model.

        Updates the model also if this parameter is parsed.

        Parameters
        ----------
        array: Array_Nx3_Float_T | Self
            A new object will be created with all other parameters and have it's coordinates update with this array.
        deep: bool
            Flag to indicate if a deep copy should be made.
        update: dict[str, Any] | None
            Dictionary of parameters to update.
        link_to_same_NOS: bool
            Flag if the object should be tied to the same optimal shift object. Thus attached to the same coordinate
            system.
        kwargs: dict[str, Any]

        Returns
        -------
        CartesianCoordinates
        """
        update = {} if update is None else update

        if link_to_same_NOS and "numerical_optimization_shift" not in update:
            update["numerical_optimization_shift"] = self.numerical_optimization_shift

        update["_shift_applied_by"] = self._shift_applied_by  # TODO: Rework structure!
        update["id"] = None

        return super().copy(array=array, deep=deep, update=update, **kwargs)

    @classmethod
    def merge(cls: Type[Self], *cart_coords: Self, **kwargs) -> Self:
        """Merge multiple cartesian coordinate sets into one.

        Attempts will be made to automatically resolve the numerical shift applied to each point cloud, if applied.

        Parameters
        ----------
        cart_coords: CartesianCoordinates
        kwargs: dict[str, Any]

        Returns
        -------
        CartesianCoordinates
        """
        if len(cart_coords) == 0:
            raise ValueError("Cannot merge empty list of CartesianCoordinates")

        elif len(cart_coords) == 1:
            return cart_coords[0].copy()

        elif len(set(cart_coord.numerical_optimization_shift for cart_coord in cart_coords)) > 1:
            logger.info(f"{type(cart_coords[0])} objects do not share a common numerical optimization shift object.")
            bbox_pts = np.vstack(tuple(np.array(cart_coord.unshifted_bbox) for cart_coord in cart_coords))

            if cart_coords[0].numerical_optimization_shift and cart_coords[
                0
            ].numerical_optimization_shift.check_addibility(bbox_pts):
                logger.info("Linking to numerical optimization shift of first instance.")
                common_nos = cart_coords[0].numerical_optimization_shift

            elif OptimizedShiftManager().is_shift_possible(bbox_pts):
                logger.info("Creating new numerical optimization shift instance.")
                common_nos = OptimizedShift()

            else:
                logger.info("Unable to create new numerical optimization shift instance applicable to all points.")
                common_nos = None

            cart_coord_copies = list()
            update = {"numerical_optimization_shift": common_nos}
            update.update({k: None for k in kwargs})

            for cart_coord in cart_coords:
                cart_coord_copies.append(cart_coord.copy(update=update, link_to_same_NOS=False))
            cart_coords = tuple(cart_coord_copies)

        new_arr = np.vstack(tuple(cart_coord.arr for cart_coord in cart_coords))
        return cls(
            xyz=new_arr,
            numerical_optimization_shift=cart_coords[0].numerical_optimization_shift,
            socs_origin=None,
            project_transformation=None,
            _shift_applied_by=cart_coords[0].numerical_optimization_shift,
            **kwargs,
        )

    @property
    def numerically_optimized(self) -> bool:
        """Flag if the coordinates are numerically optimized."""
        return not (
            self.numerical_optimization_shift is None or np.allclose(self.numerical_optimization_shift.value, 0)
        )

    @property
    def x(self) -> Vector_Float_T:
        """Return the X component of the coordinates."""
        return self.arr[:, 0]

    @property
    def y(self) -> Vector_Float_T:
        """Return the Y component of the coordinates."""
        return self.arr[:, 1]

    @property
    def z(self) -> Vector_Float_T:
        """Return the Z component of the coordinates."""
        return self.arr[:, 2]

    @property
    def xyz(self) -> Array_Nx3_Float_T:
        """Return the XYZ coordinates as a numpy array."""
        return self.arr

    @property
    def yxz(self) -> Array_Nx3_Float_T:
        """Return the coordinates in YXZ column order."""
        return self.xyz[:, [1, 0, 2]]

    @cached_property
    def spher(self) -> Array_Nx3_Float_T:
        """Return the coordinates in spherical RHV form relative to ``socs_origin``.

        When ``socs_origin`` is undefined, ``(0, 0, 0)`` is treated as the
        origin.

        Returns
        -------
        Array_Nx3_Float_T
            Spherical coordinates as ``(N, 3)`` ``[range, horizontal, vertical]``.
        """
        if self.socs_origin is not None:
            return xyz2rhv(self.arr, self.socs_origin)
        elif self.numerical_optimization_shift is not None:
            return xyz2rhv(self.arr, -self.numerical_optimization_shift.value)
        else:
            return xyz2rhv(self.arr, np.zeros(3, dtype=np.float32))

    @property
    def r(self) -> Vector_Float_T:
        """Returns radial component of the spherical coordinates.

        Returns
        -------
        Vector_Float_T
        """
        return self.spher[:, 0]

    @property
    def hz(self) -> Vector_Float_T:
        """Return the horizontal angles of the spherical coordinates.

        Returns
        -------
        Vector_Float_T
        """
        return self.spher[:, 1]

    @property
    def v(self) -> Vector_Float_T:
        """Return the vertical angles of the spherical coordinates.

        Returns
        -------
        Vector_Float_T
        """
        return self.spher[:, 2]

    @property
    def rhv(self) -> Array_Nx3_Float_T:
        """Returns the coordinates as spherical coordinates.

        Returns
        -------
        Array_Nx3_Float_T
        """
        return self.spher

    @property
    def _hz_v(self) -> Array_Nx2_Float_T:
        """Return the horizontal and vertical angles of the spherical coordinates.

        Returns
        -------
        Array_Nx2_Float_T
        """
        return self.rhv[:, 1:]

    @cached_property
    def fov(self) -> FoV:
        """Returns the field of view of the point cloud.

        Returns
        -------
        FoV
        """
        return FoV.from_angles(self.hz, self.v)

    def rotate(self, rotation: Array_3x3_T) -> None:
        """Rotate the current coordinates in place by a 3x3 rotation matrix.

        Parameters
        ----------
        rotation : Array_3x3_T
            3x3 rotation matrix.
        """
        self.arr = (rotation @ self.T).T

    def translate(self, translation: Vector_3_T) -> None:
        """Translate the current coordinates in place.

        Parameters
        ----------
        translation : Vector_3_T
            Translation vector added to each coordinate.
        """
        self.arr += translation

    def scale(self, scale: Vector_3_T) -> None:
        """Scale the current coordinates in place (component-wise).

        Parameters
        ----------
        scale : Vector_3_T
            Per-axis scale vector multiplied onto the coordinates.
        """
        self.arr *= scale

    def transform(self, affine: Array_4x4_T) -> None:
        """Transform the current coordinates in place using a 4x4 affine matrix.

        Parameters
        ----------
        affine : Array_4x4_T
            4x4 affine transformation matrix.
        """
        self.arr = (affine @ self.H.T).T[:, :3]

    @classmethod
    def from_spherical(cls, spher: Array_Nx3_T) -> Self:
        """Build an instance from spherical coordinates in RHV format.

        Parameters
        ----------
        spher : Array_Nx3_T
            Spherical coordinates as ``(N, 3)`` ``[range, horizontal, vertical]``.

        Returns
        -------
        Self
            New :class:`CartesianCoordinates` built from the converted XYZ.
        """
        return cls(xyz=rhv2xyz(spher))

    # @classmethod
    # def from_spherical(cls, spher: SphericalCoordinates) -> SphericalCoordinates:
    #     return cls(**spher.model_dump(exclude={"arr"}) | {"arr": self.xyz})

    # def to_spherical(self) -> SphericalCoordinates:
    #     spherical = SphericalCoordinates(**self.model_dump(exclude={"arr"}) | {"arr": self.spher})
    #     delattr(self, "spher")
    #     return spherical


# class SphericalCoordinates(Abstract3dCoordinates):
#     arr: Annotated[Array_Nx3_T, Field(validation_alias="spher"), BeforeValidator(validate_spherical_angles)]
#
#     @cached_property
#     def fov(self) -> FoV:
#         return FoV.from_angles(self.hz, self.v)
#
#     @property
#     def spher(self) -> npt.NDArray[np.floating]:
#         return self.arr
#
#     @property
#     def rhv(self) -> npt.NDArray[np.floating]:
#         return self.arr
#
#     @property
#     def r(self) -> npt.NDArray[np.floating]:
#         return self.rhv[:, 0]
#
#     @property
#     def hz(self) -> np.ndarray:
#         return self.rhv[:, 1]
#
#     @property
#     def v(self) -> npt.NDArray[np.floating]:
#         return self.rhv[:, 2]
#
#     @property
#     def _hz_v(self) -> npt.NDArray[np.floating]:
#         return self.rhv[:, 1:]
#
#     @cached_property
#     def xyz(self) -> npt.NDArray[np.floating]:
#         if self.socs_origin is None:
#             warnings.warn("Spherical origin was not defined, so coordinates assumed to be at scan origin")
#             return rhv2xyz(self.arr, np.zeros(3, dtype=np.float32))
#         return rhv2xyz(self.arr, self.socs_origin)
#
#     @property
#     def x(self) -> npt.NDArray[np.floating]:
#         return self.xyz[:, 0]
#
#     @property
#     def y(self) -> npt.NDArray[np.floating]:
#         return self.xyz[:, 1]
#
#     @property
#     def z(self) -> npt.NDArray[np.floating]:
#         return self.xyz[:, 2]
#
#     def to_cartesian(self) -> CartesianCoordinates:
#         cartesian = CartesianCoordinates(**self.model_dump(exclude={"arr"}) | {"arr": self.xyz})
#         delattr(self, "xyz")
#         return cartesian
#
#     @classmethod
#     def from_cartesian(cls, cartesian: CartesianCoordinates) -> Self:
#         spherical = cls(**cartesian.model_dump(exclude={"arr"}) | {"arr": cartesian.spher})
#         delattr(cartesian, "spher")
#         return spherical


@validate_call(config=DEFAULT_CONFIG)
def rhv2xyz(spher: Array_Nx3_T, scan_origin: Optional[Vector_3_T] = None) -> Array_Nx3_T:
    """Convert spherical coordinates (RHV) to Cartesian coordinates (XYZ).

    This function transforms 3D points given in spherical coordinates
    (radius, azimuth, elevation) to Cartesian coordinates (x, y, z).
    Optionally, a scan origin can be added to the resulting Cartesian
    coordinates.

    Parameters
    ----------
    spher : Array_Nx3_T
        Input array of shape (N, 3) representing spherical coordinates.
    scan_origin : Optional[Vector_3_T], optional
        A 3D vector representing the scan origin, by default None.

    Returns
    -------
    Array_Nx3_T
        Output array of shape (N, 3) representing Cartesian coordinates.
    """
    xyz: np.ndarray = np.zeros_like(spher)
    xyz[:, 0] = spher[:, 0] * np.sin(spher[:, 2]) * np.cos(spher[:, 1])
    xyz[:, 1] = spher[:, 0] * np.sin(spher[:, 2]) * np.sin(spher[:, 1])
    xyz[:, 2] = spher[:, 0] * np.cos(spher[:, 2])

    return xyz if scan_origin is None else xyz + scan_origin


@validate_call(config=DEFAULT_CONFIG)
def xyz2rhv(xyz: Array_Nx3_T, scan_origin: Optional[Vector_3_T] = None) -> Array_Nx3_T:
    """Convert Cartesian coordinates (XYZ) to spherical coordinates (RHV).

    The resulting spherical coordinates include the slope distance, horizontal angle,
    and zenith angle calculated from Cartesian coordinates. If a scan origin is
    provided, the Cartesian input is shifted accordingly.

    Parameters
    ----------
    xyz : Array_Nx3_T
        Array of Cartesian coordinates with shape (N, 3). Each row represents a point as [x, y, z].
    scan_origin : Optional[Vector_3_T], optional
        The origin point from which to calculate spherical coordinates. If None, it assumes the origin is at (0, 0, 0).

    Returns
    -------
    Array_Nx3_T
        Output array of shape (N, 3) representing Cartesian coordinates.
    """
    spher: np.ndarray = np.zeros_like(xyz)

    if scan_origin is not None:
        xyz = xyz - scan_origin

    xy_2: npt.ArrayLike = xyz[:, 0] ** 2 + xyz[:, 1] ** 2

    spher[:, 0] = np.sqrt(xy_2 + xyz[:, 2] ** 2)  # [  0, inf] slope distance
    spher[:, 1] = np.arctan2(xyz[:, 1], xyz[:, 0])  # [-pi, +pi] horizonal angle
    spher[:, 2] = np.arctan2(np.sqrt(xy_2), xyz[:, 2])  # [  0, +pi] zenith angle

    return spher
