from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Annotated, Optional, Self, Any

import numpy as np
import numpy.typing as npt
from pydantic import BeforeValidator, ConfigDict, Field, validate_call

from ..base_arrays import ArrayNx2, ArrayNx3, FixedLengthArray
from ..base_types import Array_4x4_T, Vector_3_T, Array_Nx3_T, Array_3x3_T
from ..constants import HALF_PI, PI, TWO_PI, DEFAULT_CONFIG
from ..validators import validate_spherical_angles
from .transforms import (
    GlobalShift,
    Transform,
    TransformLedger,
    TransformRecord,
    _Transform3x3,
    _Transform4x4,
)

TransformT = _Transform4x4 | _Transform3x3 | Transform


class AbstractCoordinates(FixedLengthArray, ABC):

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
    optimized: bool = Field(default=False, exclude=True)

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
    arr: Array_Nx3_T = Field(alias="xyz")

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

    @xyz.setter
    def xyz(self, value: npt.NDArray[np.floating]):
        # if self.model_config['frozen']:
        #     raise ValueError('Cannot edit XYZ coordinates of frozen object')
        self.arr = value

    @property
    def yxz(self) -> npt.NDArray[np.floating]:
        return self.xyz[:, [1, 0, 2]]

    @cached_property
    def spher(self) -> npt.NDArray[np.floating]:
        if self.socs_origin is None:
            warnings.warn("Scan center of point cloud is ambiguous and results can not be guaranteed")
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

    @property
    def fov(self):  # TODO must implement this
        raise NotImplementedError

    def to_spherical(self) -> SphericalCoordinates:
        spherical = SphericalCoordinates(**self.model_dump(exclude={"arr"}) | {"arr": self.spher})
        delattr(self, "spher")
        return spherical

    @classmethod
    def from_spherical(cls, spherical: SphericalCoordinates) -> Self:
        cartesian = cls(**spherical.model_dump(exclude={"arr"}) | {"arr": spherical.xyz})
        delattr(spherical, "xyz")
        return cartesian

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

# TODO check the validation process and performance
class SphericalCoordinates(Abstract3dCoordinates):
    arr: Annotated[Array_Nx3_T, Field(validation_alias="spher"), BeforeValidator(validate_spherical_angles)]

    @property
    def fov(self) -> Fov:
        raise NotImplementedError

    @property
    def spher(self) -> npt.NDArray[np.floating]:
        return self.arr

    @property
    def rhv(self) -> npt.NDArray[np.floating]:
        return self.arr

    @property
    def r(self) -> npt.NDArray[np.floating]:
        return self.rhv[:, 0]

    @property
    def hz(self) -> np.ndarray:
        return self.rhv[:, 1]

    @property
    def v(self) -> npt.NDArray[np.floating]:
        return self.rhv[:, 2]

    @property
    def _hz_v(self) -> npt.NDArray[np.floating]:
        return self.rhv[:, 1:]

    @cached_property
    def xyz(self) -> npt.NDArray[np.floating]:
        if self.socs_origin is None:
            warnings.warn("Spherical origin was not defined, so coordinates assumed to be at scan origin")
            return rhv2xyz(self.arr, np.zeros(3, dtype=np.float32))
        return rhv2xyz(self.arr, self.socs_origin)

    @property
    def x(self) -> npt.NDArray[np.floating]:
        return self.xyz[:, 0]

    @property
    def y(self) -> npt.NDArray[np.floating]:
        return self.xyz[:, 1]

    @property
    def z(self) -> npt.NDArray[np.floating]:
        return self.xyz[:, 2]

    def to_cartesian(self) -> CartesianCoordinates:
        cartesian = CartesianCoordinates(**self.model_dump(exclude={"arr"}) | {"arr": self.xyz})
        delattr(self, "xyz")
        return cartesian

    @classmethod
    def from_cartesian(cls, cartesian: CartesianCoordinates) -> Self:
        spherical = cls(**cartesian.model_dump(exclude={"arr"}) | {"arr": cartesian.spher})
        delattr(cartesian, "spher")
        return spherical

    # DISCUSS - Add methods to apply tilt and yaw rotations easily (e.g. for spherical image projection shifts?
    # def rotate(self, yaw=None, pitch=None):
    #     if yaw:
    #         self.arr[:, 1] = coerce_azimuths(self.hz + yaw)
    #
    #     if pitch:
    #         self.arr[np.logical_or(temp < 0, temp > PI), 1] = coerce_azimuths(self.hz + TWO_PI)
    #         self.arr[:, 2] = np.abs(temp := self.v - pitch)


@validate_call(config=DEFAULT_CONFIG)
def rhv2xyz(spher: npt.ArrayLike|npt.NDArray[np.floating], origin_shift: Vector_3_T | None = None) -> np.ndarray:
    xyz: np.ndarray = np.zeros_like(spher)
    xyz[:, 0] = spher[:, 0] * np.sin(spher[:, 2]) * np.cos(spher[:, 1])
    xyz[:, 1] = spher[:, 0] * np.sin(spher[:, 2]) * np.sin(spher[:, 1])
    xyz[:, 2] = spher[:, 0] * np.cos(spher[:, 2])

    return xyz if origin_shift is None else xyz - origin_shift


# TODO fix this to support the optimal shifts (e.g. remove origin shift)
@validate_call(config=DEFAULT_CONFIG)
def xyz2rhv(cart: npt.ArrayLike|npt.NDArray[np.floating], origin_shift: Optional[Vector_3_T] = np.zeros(3)) -> np.ndarray:
    spher: np.ndarray = np.zeros_like(cart)

    # Apply the shift in place to avoid creating additional copies
    dx, dy, dz = origin_shift

    xy_2: npt.ArrayLike = (cart[:, 0] + dx) ** 2 + (cart[:, 1] + dy) ** 2
    spher[:, 0] = np.sqrt(xy_2 + (cart[:, 2] + dz) ** 2)  # [  0, inf] slope distance
    spher[:, 1] = np.arctan2((cart[:, 1] + dy), (cart[:, 0] + dx))  # [-pi, +pi] horizonal angle
    spher[:, 2] = np.arctan2(np.sqrt(xy_2), cart[:, 2] + dz)  # [  0, +pi] zenith angle

    return spher
