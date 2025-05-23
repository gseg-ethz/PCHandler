from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from enum import IntEnum
from functools import cached_property
from typing import Optional, Annotated

import numpy as np
from pydantic import Field, model_validator, BeforeValidator

from ..base_arrays import ArrayNx3, CustomArrayLikeT, Array_Nx3_T, Array_4x4_T
from ..validators import validate_spherical_angles
from .transforms import TransformRecord, TransformLedger, GlobalShift


class CoordSysEnum(IntEnum):
    OPTIMAL = 0
    SOC = 1
    PROJECT = 2
    GLOBAL = 3


class Abstract3dCoordinates(ABC, ArrayNx3):
    transform_ledger: TransformLedger[str, [Array_4x4_T]] = Field(default_factory=TransformLedger)
    @property
    @abstractmethod
    def xyz(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def spher(self) -> np.ndarray:
        pass


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
        return xyz2rhv(self.arr)

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

    def to_spherical(self) -> SphericalCoordinates:
        spherical = SphericalCoordinates(**(self.model_dump() | {'arr': self.spher.copy()}))
        delattr(self, 'spher')
        return spherical

    @classmethod
    def from_spherical(cls, spherical: SphericalCoordinates):
        cartesian = cls(**(spherical.model_dump() | {'arr': spherical.xyz.copy()}))
        delattr(spherical, 'xyz')
        return cartesian

    def transform(self, affine=None, rotation=None, translation=None, scale=None):
        """
        Takes the form x0 = (T @ R * s + t) @ x1

        Rotation  Translation     Scale:
        | R 0 |     | I t |      | s 0 |
        | 0 1 |     | 0 1 |      | 0 1 |
        """
        if affine is None:
            affine = np.eye(4)
            if rotation:
                affine[:3, :3] @= rotation

            if translation:
                affine[:3, 3] += translation

            if scale:
                affine[np.eye(3)] *= scale

        self.arr = affine @ self.arr
        self.transform_ledger['AFFINE'] = TransformRecord(forward=affine)


class SphericalCoordinates(Abstract3dCoordinates):
    arr: Annotated[Array_Nx3_T, BeforeValidator(validate_spherical_angles)]

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
        return rhv2xyz(self)

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
        out = CartesianCoordinates(**(self.model_dump() | {'arr': self.xyz.copy()}))
        delattr(self, 'xyz')
        return out

    @classmethod
    def from_cartesian(cls, cartesian: CartesianCoordinates):
        spherical = cls(**(cartesian.model_dump() | {'arr': cartesian.spher.copy()}))
        delattr(cartesian, 'spher')
        return spherical


class OptimisedCartesianCoordinates(CartesianCoordinates):
    global_shift: GlobalShift = Field(default_factory=lambda: GlobalShift(arr=np.zeros(3)))

    def compute_shift(self, decimal_magnitude: int = 4):
        value = np.median(np.round(self.arr, decimals=-(decimal_magnitude - 1)), axis=0)
        self.optimal_shift.update_shift(value)

    def is_shift_needed(self, decimal_magnitude: int = 4) -> np.bool_:
        return np.any(np.abs(self) >= self._optimal_range(decimal_magnitude))

    @staticmethod
    def _optimal_range(decimal_magnitude: int = 4):
        return 10 ** decimal_magnitude

    @model_validator(mode='after')
    def update_global_shift(self):
        if self.is_shift_needed():
            self.compute_shift()
            self.arr -= self.global_shift
            self.transform_ledger['OPT'] = self.global_shift.as_record()
            self.current_system = CoordSysEnum.OPTIMAL

    def as_global_coords(self):
        transform, _ = self.transform_ledger.rollback_record(0)
        return transform @ self

    def to_spherical(self):
        raise AttributeError('Cannot get spherical coordinates from an array with an undefined scan origin. '
                             'Use TLSScan if origin of cloud is at (0, 0, 0)')

    @cached_property
    def spher(self) -> np.ndarray:
        raise AttributeError('Cannot get spherical coordinates from an array with an undefined scan origin. '
                             'Use TLSScan if origin of cloud is at (0, 0, 0)')

class TLSCoordinates(OptimisedCartesianCoordinates, CartesianCoordinates):
    """Assumption should be made then that the user has origin at 0,0,0"""
    current_system: CoordSysEnum = CoordSysEnum.SOC
    project_transformation: Optional[Array_4x4_T] = None
    is_soc_optimal: bool = False

    @model_validator(mode='after')
    def update_global_shift(self):
        pass

    @model_validator(mode='after')
    def initialize_transforms(self):
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




def rhv2xyz(spher: CustomArrayLikeT) -> CustomArrayLikeT:
    xyz: CustomArrayLikeT = np.zeros_like(spher)
    xyz[:, 0] = spher[:, 0] * np.sin(spher[:, 2]) * np.cos(spher[:, 1])
    xyz[:, 1] = spher[:, 0] * np.sin(spher[:, 2]) * np.sin(spher[:, 1])
    xyz[:, 2] = spher[:, 0] * np.cos(spher[:, 2])
    return xyz


def xyz2rhv(cart: CustomArrayLikeT) -> CustomArrayLikeT:
    spher: CustomArrayLikeT = np.zeros_like(cart)
    xy_2: CustomArrayLikeT = cart[:, 0]**2 + cart[:, 1]**2
    spher[:, 0] = np.sqrt(xy_2 + cart[:, 2]**2)  # [  0, inf] slope distance
    spher[:, 1] = np.arctan2(cart[:, 1], cart[:, 0])  # [-pi, +pi] horizonal angle
    spher[:, 2] = np.arctan2(np.sqrt(xy_2), cart[:, 2])  # [  0, +pi] zenith angle
    return spher
