from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from enum import IntEnum
from functools import cached_property, wraps
from typing import Optional, Annotated

import numpy as np
from pydantic import Field, model_validator, BeforeValidator, validate_call, ConfigDict, field_validator

from ..base_arrays import ArrayNx3, CustomArrayLikeT, Array_Nx3_T, Array_4x4_T, Vector_3_T, BaseArray
from ..validators import validate_spherical_angles, enforce_azimuths
from .transforms import TransformRecord, TransformLedger, GlobalShift, Transform

PI = np.pi
TWO_PI = 2 * PI
HALF_PI = 0.5 * PI

class TransformType(IntEnum):
    TRANSLATE = 0
    ROTATE = 1
    SCALE = 2
    AFFINE = 3


class CoordSysEnum(IntEnum):
    OPTIMAL = 0
    SOC = 1
    PROJECT = 2
    GLOBAL = 3


class Abstract3dCoordinates(ABC, ArrayNx3):
    transform_ledger: TransformLedger[str, [Array_4x4_T]] = Field(default_factory=TransformLedger)
    spherical_origin: Optional[Vector_3_T] = None

    @field_validator('transform_ledger', mode='before')
    @classmethod
    def initialise_empty_ledger(cls, value: dict|TransformLedger):
        if isinstance(value, dict):
            return TransformLedger(**value)
        return value

    @property
    @abstractmethod
    def xyz(self) -> np.ndarray:
        raise NotImplementedError

    @property
    @abstractmethod
    def spher(self) -> np.ndarray:
        raise NotImplementedError

    def __getitem__(self, key):
        return self.sample(*key)

    def __setitem__(self, key, value):
        self.arr[key] = value

    def __rmatmul__(self, other):
        temp = self.H.T if other.shape == (4, 4) else self.T
        # Todo add to the transformation ledger
        self.transform_ledger['RMATMUL'] = Transform.from_matrix(other)
        return self.get_copy(array=super(BaseArray, self).__rmatmul__(temp).T[:, :3])


def update_transform_ledger(name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(instance: Abstract3dCoordinates, *args, **kwargs):
            result = func(instance, *args, **kwargs)

            result.transform_ledger[name] = TransformRecord(forward=args[0])
        return wrapper
    return decorator



class CartesianCoordinates(Abstract3dCoordinates):
    current_system: CoordSysEnum = CoordSysEnum.GLOBAL
    @property
    def x(self) -> np.ndarray:
        return self.arr[:, 0]

    @property
    def y(self) -> np.ndarray:
        return self.arr[:, 1]

    @property
    def z(self) -> np.ndarray:
        return self.arr[:, 2]

    @property
    def xyz(self) -> np.ndarray:
        return self.arr

    @property
    def yxz(self) -> np.ndarray:
        return self.xyz[:, [1, 0, 2]]

    @cached_property
    def spher(self) -> np.ndarray:
        if self.spherical_origin is None:
            warnings.warn('Scan center of point cloud is ambiguous and results can not be guaranteed')
            return xyz2rhv(self.arr, np.zeros(3))
        return xyz2rhv(self.arr, self.spherical_origin)

    @property
    def r(self):
        return self.spher[:, 0]

    @property
    def hz(self):
        return self.spher[:, 1]

    @property
    def v(self):
        return self.spher[:, 2]

    @property
    def rhv(self):
        return self.spher

    @property
    def fov(self):
        return self.spher.fov

    def to_spherical(self) -> SphericalCoordinates:
        spherical = SphericalCoordinates(**dict(
            self.model_copy(update={'arr': self.spher.copy()}, deep=True)))
        delattr(self, 'spher')
        return spherical

    @classmethod
    def from_spherical(cls, spherical: SphericalCoordinates):
        cartesian = cls(**dict(
            spherical.model_copy( update={'arr': spherical.xyz.copy()}, deep=True)))
        delattr(spherical, 'xyz')
        return cartesian

    def transform(self, affine=None, rotation=None, translation=None, scale=None):

        affine = Transform.from_matrix(affine) if affine else np.eye(4)

        if rotation is not None:
            affine[:3, :3] @= rotation
        if translation is not None:
            affine[:3, 3] += translation
        if scale is not None:
            affine[[0, 1, 2], [0, 1, 2]] *= scale

        self.arr = (affine @ self.H.T).T[:, :3]
        self.transform_ledger['AFFINE'] = TransformRecord(forward=affine)


class SphericalCoordinates(Abstract3dCoordinates):
    arr: Annotated[Array_Nx3_T, BeforeValidator(validate_spherical_angles)]

    # DISCUSS - Add methods to apply tilt and yaw rotations easily (e.g. for spherical image projection shifts?
    def rotate(self, yaw=None, pitch=None):
        if yaw:
            self.arr[:, 1] = enforce_azimuths(self.hz + yaw)

        if pitch:
            self.arr[:, 2] = np.abs(temp := self.v - pitch)
            self.arr[np.logical_or(temp < 0, temp > PI), 1] = enforce_azimuths(self.hz + TWO_PI)

    @property
    def fov(self):
        raise NotImplementedError

    @property
    def spher(self) -> np.ndarray:
        return self.arr

    @property
    def rhv(self) -> np.ndarray:
        return self.arr

    @property
    def r(self) -> np.ndarray:
        return self.rhv[:, 0]

    @property
    def hz(self) -> np.ndarray:
        return self.rhv[:, 1]

    @property
    def v(self) -> np.ndarray:
        return self.rhv[:, 2]

    @cached_property
    def xyz(self) -> np.ndarray:
        if self.spherical_origin is None:
            warnings.warn('Spherical origin was not defined, so coordinates assumed to be at scan origin')
            rhv2xyz(self.arr, np.zeros(3))
        return rhv2xyz(self.arr, self.spherical_origin)

    @property
    def x(self) -> np.ndarray:
        return self.xyz[:, 0]

    @property
    def y(self) -> np.ndarray:
        return self.xyz[:, 1]

    @property
    def z(self) -> np.ndarray:
        return self.xyz[:, 2]

    def to_cartesian(self) -> CartesianCoordinates:
        cartesian = CartesianCoordinates(**dict(
            self.model_copy(
                update={'arr': self.xyz.copy()},
                deep=True )))
        delattr(self, 'xyz')
        return cartesian

    @classmethod
    def from_cartesian(cls, cartesian: CartesianCoordinates):
        spherical = cls(**dict(
            cartesian.model_copy(
                update={'arr': cartesian.spher.copy()},
                deep=True )))
        delattr(cartesian, 'spher')
        return spherical

# DISCUSS - Nomenclature needs to be decided
class OptimisedCartesianCoordinates(CartesianCoordinates):
    optimal_shift: GlobalShift = Field(default_factory=lambda: GlobalShift(arr=np.zeros(3)))

    def compute_shift(self, decimal_magnitude: int = 4):
        value = np.median(np.round(self.arr, decimals=-(decimal_magnitude - 1)), axis=0)
        self.optimal_shift.update_shift(value)

    def is_shift_needed(self, decimal_magnitude: int = 4) -> np.bool_:
        return np.any(np.abs(self) >= self._optimal_range(decimal_magnitude))

    @staticmethod
    def _optimal_range(decimal_magnitude: int = 4):
        return 10 ** decimal_magnitude

    def initialise_transform_ledger(self):
        if np.all(self.global_shift.arr == 0):
            self.transform_ledger['GLOB'] = self.global_shift.as_record()
        else:
            self.transform_ledger['OPT'] = self.global_shift.as_record()

    @model_validator(mode='after')
    def update_coordinate_system_info(self):
        if self.is_shift_needed():
            self.compute_shift()
            self.arr = self.arr - self.global_shift
            self.transform_ledger['OPT'] = self.global_shift.as_record()
            self.current_system = CoordSysEnum.OPTIMAL

        self.arr = self.arra

        if len(self.transform_ledger) == 0:
            self.initialize_transform_ledger()

    def as_global_coords(self):
        transform, _ = self.transform_ledger.rollback_record(0)
        return transform @ self

    @cached_property
    def spher(self) -> np.ndarray:
        return xyz2rhv(self.as_global_coords())

class TLSCoordinates(OptimisedCartesianCoordinates):
    """Assumption should be made then that the user has origin at 0,0,0"""
    current_system: CoordSysEnum = CoordSysEnum.SOC
    spherical_origin: Vector_3_T = np.zeros(3, dtype=np.float32)
    project_transformation: Optional[Array_4x4_T] = None
    is_soc_optimal: bool = False

    @model_validator(mode='after')
    def update_coordinate_system_info(self):
        if len(self.transform_ledger) == 0:
            if (self.project_transformation is None) or np.all(np.isclose(self.project_transformation, np.eye(4))):
                self.transform_ledger['SOC'] = TransformRecord(backward=np.eye(4))
            else:
                self.transform_ledger['PROJ'] = TransformRecord(backward=self.project_transformation)
                self.transform_ledger['SOC'] = TransformRecord(forward=np.eye(4))

        if self.is_shift_needed():
            self.compute_shift()
            self.arr -= self.global_shift
            self.transform_ledger['OPT'] = self.global_shift.as_record()
            self.current_system = CoordSysEnum.OPTIMAL
        else:
            self.is_soc_optimal = True

    def as_project_coords(self):
        first_record_name, _ = self.transform_ledger[0]
        transform, _ = self.transform_ledger.rollback_record(0)
        if 'PROJ' not in first_record_name and 'SOC' not in first_record_name:
            raise ValueError('Unknown starting transformation type. The Project coord system is there fore unknown.')
        if 'SOC' in first_record_name:
            warnings.warn('No transformation was provided with the point cloud. Scanner assumed to be at (0,0,0)')

        return transform @ self

    def as_soc_coords(self):
        first_record_name, _ = self.transform_ledger[0]
        if 'SOC' in first_record_name:
            transform, _ = self.transform_ledger.rollback_record(0)
        elif 'PROJ' in first_record_name:
            transform, _ = self.transform_ledger.rollback_record(1)
        else:
            raise ValueError('Unknown starting transformation type. Expected SOC in first two records')

        return transform @ self

    @cached_property
    def spher(self) -> np.ndarray:
        return xyz2rhv(self.as_soc_coords())

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def rhv2xyz(spher: CustomArrayLikeT, origin_shift: Vector_3_T|None = None) -> np.ndarray:
    xyz: np.ndarray = np.zeros_like(spher)
    xyz[:, 0] = spher[:, 0] * np.sin(spher[:, 2]) * np.cos(spher[:, 1])
    xyz[:, 1] = spher[:, 0] * np.sin(spher[:, 2]) * np.sin(spher[:, 1])
    xyz[:, 2] = spher[:, 0] * np.cos(spher[:, 2])

    return xyz if origin_shift is None else xyz - origin_shift

@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def xyz2rhv(cart: CustomArrayLikeT, origin_shift: Vector_3_T = np.zeros(3)) -> np.ndarray:
    spher: np.ndarray = np.zeros_like(cart)

    # Apply the shift in place to avoid creating additional copies
    dx, dy, dz = origin_shift

    xy_2: CustomArrayLikeT = (cart[:, 0] + dx)**2 + (cart[:, 1] + dy)**2
    spher[:, 0] = np.sqrt(xy_2 + (cart[:, 2] + dz)**2)  # [  0, inf] slope distance
    spher[:, 1] = np.arctan2((cart[:, 1] + dy), (cart[:, 0] + dx))  # [-pi, +pi] horizonal angle
    spher[:, 2] = np.arctan2(np.sqrt(xy_2), cart[:, 2] + dz)  # [  0, +pi] zenith angle

    return spher
