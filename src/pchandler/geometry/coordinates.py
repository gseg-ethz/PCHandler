from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Annotated, Optional, Self, Any, Type, TypeVar, Unpack, TypedDict, Required, NotRequired
import uuid

import numpy as np
import numpy.typing as npt
from pydantic import BeforeValidator, Field, validate_call, PrivateAttr, field_validator, AliasChoices

from pchandler.geometry.fov import FoV
from pchandler.geometry.util import MinMaxPoints
from pchandler.base_arrays import ArrayNx2, ArrayNx3, FixedLengthArray
from pchandler.base_types import Array_4x4_T, Vector_3_T, Array_Nx3_T, Array_3x3_T, IndexLike
from pchandler.constants import DEFAULT_CONFIG
from pchandler.validators import validate_spherical_angles
from .transforms import (
    Transform,
    _Transform3x3,
    _Transform4x4,
)
from pchandler.geometry.optimal_shift import OptimizedShift, OptimizedShiftManager

logger = logging.getLogger(__name__)

TransformT = _Transform4x4 | _Transform3x3 | Transform

CartesianT  = TypeVar("CartesianT", bound="CartesianCoordinates")

class Abstract3dKw(TypedDict, total=False):
    arr: Array_Nx3_T
    project_transformation: NotRequired[Array_4x4_T]
    socs_origin: NotRequired[Vector_3_T]

class CartesianKw(Abstract3dKw, total=False):
    numerical_optimization_shift: NotRequired[Optional[OptimizedShift]]
    unshifted_bbox: NotRequired[Optional[MinMaxPoints]]
    _shift_applied_by: NotRequired[Optional[OptimizedShift]]


class AbstractCoordinates(FixedLengthArray, ABC):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, alias="_id")

    @field_validator("id", mode="before")
    @classmethod
    def populate_id(cls, v):
        if v is None:
            return uuid.uuid4()
        return v


    def __matmul__(self, transpose_matrix: TransformT | npt.NDArray[np.floating]) -> Self | np.ndarray:
        raise NotImplementedError(
            "Left matrix multiplication is not supported.\n"
            "For 3D coordinates use the formula: \n"
            "     y = Tx\n"
            "where x are coordinates and A the transformation. In python: \n"
            "     y = A @ x"
        )


class Abstract2dCoordinates(ArrayNx2, AbstractCoordinates):
    @property
    @abstractmethod
    def row(self) -> npt.NDArray[Any]: ...

    @property
    @abstractmethod
    def col(self) -> npt.NDArray[Any]: ...


class Abstract3dCoordinates(ArrayNx3, AbstractCoordinates):
    project_transformation: Optional[Array_4x4_T] = None
    socs_origin: Optional[np.ndarray] = None

    @property
    @abstractmethod
    def xyz(self) -> npt.NDArray[np.floating]: ...

    @property
    @abstractmethod
    def spher(self) -> npt.NDArray[np.floating]: ...

    @validate_call(config=DEFAULT_CONFIG)
    def __rmatmul__(self, matrix: TransformT | npt.NDArray[np.floating]) -> Self | npt.NDArray[np.floating]:
        if isinstance(matrix, TransformT):
            matrix: np.ndarray = matrix.arr

        if matrix.shape == (4, 4):
            temp = (matrix @ self.H.T).T[:, :3]
        #     TODO check my math and if this needs to be divided by the final row / column
        elif matrix.shape == (3, 3):
            temp = (matrix @ self.T).T
        else:
            return matrix @ self.arr

        temp = self.copy(temp)
        return temp

    @validate_call(config=DEFAULT_CONFIG)
    def __imatmul__(self, transpose_matrix: TransformT | npt.NDArray[np.floating]) -> Self | npt.NDArray[np.floating]:
        raise NotImplementedError(
            "In place matrix multiplication not supported due to ambiguity between left and right multiplication.\n\n"
            "For 3D coordinates follow the right matrix multiplication formula of:"
            "       y = Ax\n"
            "where x are coordinates and A the transformation. In python:\n"
            "       y = A @ x"
        )


class CartesianCoordinates(Abstract3dCoordinates):
    arr: Array_Nx3_T = Field(..., validation_alias=AliasChoices('arr', 'xyz'))

    numerical_optimization_shift: Optional[OptimizedShift] = Field(
        default_factory=OptimizedShift,
        exclude=False
    )
    # optimized: bool = Field(default=False, exclude=True)
    unshifted_bbox: Optional[MinMaxPoints] = Field(default=None)

    _shift_applied_by: Optional[OptimizedShift] = PrivateAttr(default=None)

    def __init__(self, *args, **kwargs: Unpack[CartesianKw]):
        # Accept xyz/arr as a positional argument
        if args:
            if len(args) > 1:
                raise AttributeError("expected at most 1 arguments, got %d" % len(args))
            if "xyz" in kwargs or "arr" in kwargs:
                raise TypeError("Cannot pass both positional and keyword for xyz/arr")
            kwargs["xyz"] = args[0]

        prev_shift: Optional[OptimizedShift] = kwargs.pop("_shift_applied_by", None)
        super().__init__(**kwargs)
        object.__setattr__(self, "_shift_applied_by", prev_shift)
        self.compute_unshifted_bbox()
        self._process_shift()

    def compute_unshifted_bbox(self, overwrite: bool = False):
        if self.unshifted_bbox is None or overwrite:
            applied_shift = self._shift_applied_by.value if self._shift_applied_by is not None else None
            object.__setattr__(
                self, "unshifted_bbox", MinMaxPoints.from_points(self.arr, already_applied_shift_vec=applied_shift)
            )

    def _process_shift(self):
        '''
        4 cases:
            prev_shift is None and NOS is None:
                Do nothing
            prev_shift is None and NOS not None:
                Do basic xyz, register to NOS
            prev_shift is not None and NOS is None:
                Revert prev_shift, convert to float64, unregister from prev_shift
            prev_shift is not None and NOS is not None:
                Check if they are the same
                    register to NOS
                else:
                    Apply difference in shift, unregister from prev_shift, register to NOS
        '''
        prev_shift = self._shift_applied_by
        if self.numerical_optimization_shift is not None:
            self._register_with_shift_at_osm() # This could possibly set self.nos to None

        if prev_shift is None and self.numerical_optimization_shift is None:
            return

        if prev_shift is not None and self.numerical_optimization_shift is None:
            self.update_shift(prev_shift.value)
            prev_shift.unregister(self)

        elif prev_shift is not None and self.numerical_optimization_shift is not None:
            if prev_shift is not self.numerical_optimization_shift:
                delta_shift = prev_shift.value - self.numerical_optimization_shift.value
                self.update_shift(delta_shift)
                prev_shift.unregister(self)

        elif prev_shift is None and self.numerical_optimization_shift is not None:
            self.update_shift(-self.numerical_optimization_shift.value)

        else:
            raise RuntimeError("Unknown edge case found.")

        object.__setattr__(self, "_shift_applied_by", self.numerical_optimization_shift)

    def reduce(self, index: IndexLike) -> None:
        super().reduce(index)
        self.compute_unshifted_bbox(overwrite=True)

    def sample(self, index: IndexLike) -> Self:
        new_sample = super().sample(index)
        new_sample.compute_unshifted_bbox(overwrite=True)
        return new_sample

    def update_shift(self, delta_shift: Vector_3_T) -> None:
        target_dtype = np.float64 if self.numerical_optimization_shift is None else np.float32
        self.arr = (self.arr + delta_shift).astype(target_dtype, copy=False)
        if self.socs_origin is not None:
            self.socs_origin = (self.socs_origin + delta_shift).astype(target_dtype, copy=False)

    def _register_with_shift_at_osm(self) -> None:
        """
        This function tries to register its numerical_optimization_shift with the osm. The osm determines if this is
        valid and possibly returns a different shift, or None if infeasible.

        Returns
        -------

        """
        osm = OptimizedShiftManager()
        try:
            shift = osm.register_with(self, self.numerical_optimization_shift)
            if shift is not self.numerical_optimization_shift:
                logger.info(f"The provided numerical_optimization_shift was not feasible and needed to be replaced"
                            f"by a new one.")
                object.__setattr__(self, "numerical_optimization_shift", shift)
        except OptimizedShiftManager.ShiftNotFeasibleError:
            logger.warning("No numerical_optimization_shift was feasible. Will continue in float64 mode.")
            object.__setattr__(self, "numerical_optimization_shift", None)

    def __setattr__(self, key, value):
        if key == "_shift_applied_by":
            raise AttributeError("Cannot assign to '{key}'")
        if key == "numerical_optimization_shift":
            object.__setattr__(self, key, value)
            self._process_shift()
            return
        super().__setattr__(key, value)

    def __hash__(self) -> int:
        return id(self)

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        data = super().model_dump(**kwargs)
        data["_shift_applied_by"] = self._shift_applied_by
        return data

    def __reduce__(self) -> Any:
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

    # TODO this supports merging of tiled data. Not tested over registration / transformation etc.
    @classmethod
    def merge(cls: Type[CartesianT], *cart_coords: CartesianT , **kwargs) -> CartesianT:
        if len(cart_coords) == 1:
            return cart_coords[0].copy()
        if len(set(cart_coord.numerical_optimization_shift for cart_coord in cart_coords)) > 1:
            logger.info(f"{type(cart_coords[0])} objects do not share a common numerical optimization shift object.")
            bbox_pts = np.vstack(tuple(cart_coord.unshifted_bbox for cart_coord in cart_coords))

            if cart_coords[0].numerical_optimization_shift.check_addibility(bbox_pts):
                logger.info(f"Linking to numerical optimization shift of first instance.")
                common_nos = cart_coords[0].numerical_optimization_shift
            elif OptimizedShiftManager().is_shift_possible(bbox_pts):
                logger.info(f"Creating new numerical optimization shift instance.")
                common_nos = OptimizedShift()
            else:
                logger.info(f"Unable to create new numerical optimization shift instance applicable to all points.")
                common_nos = None

            cart_coord_copies = list()
            update = {"numerical_optimization_shift": common_nos}
            update.update({k: None for k in kwargs})

            for cart_coord in cart_coords:
                cart_coord_copies.append(cart_coord.copy(update=update, link_to_same_NOS=False))
            cart_coords = tuple(cart_coord_copies)

        new_arr = np.vstack(tuple(cart_coord.arr for cart_coord in cart_coords))
        return cls(
            arr=new_arr,
            numerical_optimization_shift=cart_coords[0].numerical_optimization_shift,
            socs_origin=None,
            project_transformation=None,
            **kwargs
        )

    # @model_validator(mode="wrap")
    # @classmethod
    # def numeric_optimization(
    #         cls,
    #         data: Any,
    #         handler: ModelWrapValidatorHandler[Self],
    #         info: ValidationInfo
    # ) -> Self:
    #     # Create new Shift instance on Ellipsis
    #     # if isinstance(data, dict) and (
    #     #         not "numerical_optimization_shift" in data or data["numerical_optimization_shift"] is Ellipsis
    #     # ):
    #     #     data["numerical_optimization_shift"] = OptimizedShift()
    #     logger.debug(f"Running `numeric_optimization` validator on {data.__class__}")
    #     instance = handler(data)
    #
    #     if not instance._shift_applied and instance.numerical_optimization_shift is not None:
    #         osm = OptimizedShiftManager()
    #         try:
    #             shift = osm.register_with(instance, instance.numerical_optimization_shift)
    #             if shift is not instance.numerical_optimization_shift:
    #                 logger.info(f"The provided numerical_optimization_shift was not feasible and needed to be replaced"
    #                             f"by a new one.")
    #
    #                 object.__setattr__(instance, "numerical_optimization_shift", shift)
    #
    #             # Use `object.__setattr` to bypass each change re-calling the validator
    #             object.__setattr__(instance, "arr",
    #                                (instance.arr - instance.numerical_optimization_shift.value).astype(np.float32))
    #             if instance.socs_origin is not None:
    #                 object.__setattr__(instance, "socs_origin", instance.socs_origin - instance.numerical_optimization_shift.value)
    #         except OptimizedShiftManager.ShiftNotFeasibleError:
    #             logger.warning("No numerical_optimization_shift was feasible. Will continue in float64 mode.")
    #             object.__setattr__(instance, "numerical_optimization_shift", None)
    #
    #         object.__setattr__(instance, "_shift_applied", True)
    #
    #     return instance

    @property
    def x(self) -> npt.NDArray[np.floating]:
        return self.arr[:, 0]

    @property
    def y(self) -> npt.NDArray[np.floating]:
        return self.arr[:, 1]

    @property
    def z(self) -> npt.NDArray[np.floating]:
        return self.arr[:, 2]

    @property
    def xyz(self) -> npt.NDArray[np.floating]:
        return self.arr
    #
    # @xyz.setter
    # def xyz(self, value: npt.NDArray[np.floating]):
    #     # if self.model_config['frozen']:
    #     #     raise ValueError('Cannot edit XYZ coordinates of frozen object')
    #     self.arr = value

    @property
    def yxz(self) -> npt.NDArray[np.floating]:
        return self.xyz[:, [1, 0, 2]]

    @cached_property
    def spher(self) -> npt.NDArray[np.floating]:
        if self.socs_origin is None:
            logger.warning("Scan center of point cloud is ambiguous and results can not be guaranteed")
            return xyz2rhv(self.arr, np.zeros(3, dtype=np.float32))
        return xyz2rhv(self.arr, self.socs_origin)

    @property
    def r(self) -> npt.NDArray[np.floating]:
        return self.spher[:, 0]

    @property
    def hz(self) -> npt.NDArray[np.floating]:
        return self.spher[:, 1]

    @property
    def v(self) -> npt.NDArray[np.floating]:
        return self.spher[:, 2]

    @property
    def rhv(self) -> npt.NDArray[np.floating]:
        return self.spher

    @property
    def _hz_v(self) -> npt.NDArray[np.floating]:
        return self.rhv[:, 1:]

    @cached_property
    def fov(self) -> FoV:
        return FoV.from_angles(self.hz, self.v)

    @property
    def numerically_optimized(self) -> bool:
        return not (self.numerical_optimization_shift is None or self.numerical_optimization_shift.value) #TODO close to zero

    # def to_spherical(self) -> SphericalCoordinates:
    #     spherical = SphericalCoordinates(**self.model_dump(exclude={"arr"}) | {"arr": self.spher})
    #     delattr(self, "spher")
    #     return spherical

    @classmethod
    def from_spherical(cls, spher: SphericalCoordinates) -> Self:
        return cls(arr=rhv2xyz(spher))

    # TODO must define on the transformation handling -> Incl. support for the scipy.spatial.transform.rotation
    def transform(self,
                  affine: Array_4x4_T = None,
                  rotation: Array_3x3_T = None,
                  translation: Vector_3_T = None,
                  scale: Vector_3_T = None) -> None:
        affine = Transform.from_matrix(affine) if affine else np.eye(4)

        if rotation is not None:
            affine[:3, :3] @= rotation
        if translation is not None:
            affine[:3, 3] += translation
        if scale is not None:
            affine[[0, 1, 2], [0, 1, 2]] *= scale

        self.arr = (affine @ self.H.T).T[:, :3]

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
#
#     # DISCUSS - Add methods to apply tilt and yaw rotations easily (e.g. for spherical image projection shifts?
#     # def rotate(self, yaw=None, pitch=None):
#     #     if yaw:
#     #         self.arr[:, 1] = coerce_azimuths(self.hz + yaw)
#     #
#     #     if pitch:
#     #         self.arr[np.logical_or(temp < 0, temp > PI), 1] = coerce_azimuths(self.hz + TWO_PI)
#     #         self.arr[:, 2] = np.abs(temp := self.v - pitch)


@validate_call(config=DEFAULT_CONFIG)
def rhv2xyz(spher: npt.ArrayLike|npt.NDArray[np.floating], scan_origin: Optional[Vector_3_T] = None) -> np.ndarray:
    xyz: np.ndarray = np.zeros_like(spher)
    xyz[:, 0] = spher[:, 0] * np.sin(spher[:, 2]) * np.cos(spher[:, 1])
    xyz[:, 1] = spher[:, 0] * np.sin(spher[:, 2]) * np.sin(spher[:, 1])
    xyz[:, 2] = spher[:, 0] * np.cos(spher[:, 2])

    return xyz if scan_origin is None else xyz - scan_origin




# TODO fix this to support the optimal shifts (e.g. remove origin shift)
@validate_call(config=DEFAULT_CONFIG)
def xyz2rhv(cart: npt.ArrayLike|npt.NDArray[np.floating], scan_origin: Optional[Vector_3_T] = None) -> np.ndarray:
    spher: np.ndarray = np.zeros_like(cart)

    if scan_origin is not None:
        cart = (cart + scan_origin)

    xy_2: npt.ArrayLike = cart[:, 0]**2 + cart[:, 1]**2

    spher[:, 0] = np.sqrt(xy_2 + cart[:, 2]**2)         # [  0, inf] slope distance
    spher[:, 1] = np.arctan2(cart[:, 1], cart[:, 0])    # [-pi, +pi] horizonal angle
    spher[:, 2] = np.arctan2(np.sqrt(xy_2), cart[:, 2])     # [  0, +pi] zenith angle

    return spher
