from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from enum import IntEnum
from functools import cached_property
from typing import Optional, Self, overload, Annotated
from unittest import case

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, computed_field, validate_call, model_validator, field_validator
from numpydantic import Shape, NDArray

from pchandler.v2.base_arrays import Array_Nx3, Vector_3, Array4x4


class CoordSysEnum(IntEnum):
    OPTIMAL = 0
    SOC = 1
    PROJECT = 2


class Abstract3dCoordinates(ABC, Array_Nx3):
    arr: Annotated[NDArray[Shape['*, 3'], Any]]
    @property
    @abstractmethod
    def xyz(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def spher(self) -> np.ndarray:
        pass

Float64Vector3 = NDArray[Shape['3'], np.float64]

class GlobalShift(Vector_3):
    arr: Float64Vector3
    old_shift: Float64Vector3

    @validate_call
    def update_shift(self, new_shift: np.ndarray|Vector_3) -> None:
        self.old_shift = self.arr.copy()
        self.arr = new_shift

class CartesianCoordinates(Abstract3dCoordinates):
    global_shift: GlobalShift
    @property
    def x(self) -> np.ndarray:
        return self._arr[:, 0]

    @property
    def y(self) -> np.ndarray:
        return self._arr[:, 1]

    @property
    def z(self) -> np.ndarray:
        return self._arr[:, 2]

    @property
    def xyz(self) -> np.ndarray:
        return self._arr

    @property
    def yxz(self) -> np.ndarray:
        return self.xyz[:, [1, 0, 2]]

    @cached_property
    def spher(self) -> np.ndarray:
        return xyz2rhv(self)

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
        if 'spher' in self.__dict__:
            return SphericalCoordinates(self.spher.copy())

        return SphericalCoordinates(xyz2rhv(self))

    @classmethod
    def from_spherical(cls, spherical: SphericalCoordinates):
        if 'xyz' in cls.__dict__:
            return cls(spherical.xyz.copy())
        return spherical.to_cartesian()


class SphericalCoordinates(Abstract3dCoordinates):
    # TODO add in some spherical coordinate validation based on defined coordinates system
    @property
    def spher(self) -> np.ndarray:
        return self._arr

    @property
    def rhv(self) -> np.ndarray:
        return self._arr

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
        return CartesianCoordinates(rhv2xyz(self))

    @classmethod
    def from_cartesian(cls, xyz: CartesianCoordinates):
        return xyz.to_spherical()

class GlobalShiftedCoordinates(CartesianCoordinates):
    project_transform: Array4x4|None = None
    current_system: CoordSysEnum = CoordSysEnum.SOC
    optimal_shift: Vector_3 = Field(default_factory=lambda: Vector_3())
    is_optimally_shifted: bool = False

    @model_validator(mode='after')
    def update_coord_systems(self):
        match self.current_system:
            case CoordSysEnum.OPTIMAL:
                self.update_global_shift()

            case CoordSysEnum.SOC:
                self.update_global_shift()

            case CoordSysEnum.PROJECT:
                self.convert_to_(CoordSysEnum.SOC)
                self.update_global_shift()
            case _:
                self.current_system = CoordSysEnum.SOC

    def update_global_shift(self):
        if self._is_shift_needed():
            shift = self.compute_shift()
            self.arr -= shift
            self.optimal_shift = shift if self.optimal_shift is None else self.optimal_shift + shift
            self.current_system = CoordSysEnum.OPTIMAL

        self.is_optimally_shifted = True

    def convert_to_(self, system: CoordSysEnum):
        if self.current_system == system:
            return

        match self.current_system:
            case CoordSysEnum.OPTIMAL:
                if system == CoordSysEnum.SOC:
                    self.arr += self.optimal_shift
                else:
                    if self.project_transform is None:
                        raise ValueError('Should not have object in Project system if no global transform exists')
                    self.arr = self.project_transform @ (self.arr + self.optimal_shift)

            case CoordSysEnum.SOC:
                if system == CoordSysEnum.PROJECT:
                    if self.project_transform is None:
                        raise ValueError('Should not have object in Project system if no global transform exists')
                    self.arr = self.project_transform @ self.arr
                else:
                    self.arr -= self.optimal_shift

            case CoordSysEnum.PROJECT:
                if system == CoordSysEnum.OPTIMAL:
                    self.arr = (np.linalg.inv(self.project_transform) @ self.arr) - self.optimal_shift
                else:
                    self.arr = (np.linalg.inv(self.project_transform) @ self.arr)

        self.current_system = system

    def _compute_shift(self, decimal_magnitude: int = 4):
        return Vector_3(np.median(np.round(self.arr, decimals=-(decimal_magnitude - 1)), axis=0))

    def _is_shift_needed(self, decimal_magnitude: int = 4) -> np.bool_:
        return np.any(np.abs(self.arr) >= 10 ** decimal_magnitude)

    def __setitem__(self, key: str, value: np.ndarray):
        if key == "_arr":
            del self.__dict__["spher"]
        super().__setitem__(key, value)

    def to_spherical(self) -> SphericalCoordinates:
        return SphericalCoordinates(xyz2rhv(self))

    @classmethod
    def from_spherical(cls, spherical: SphericalCoordinates) -> CartesianCoordinates:
        return rhv2xyz(spherical) - spherical.global_offset


def rhv2xyz(spher: np.ndarray | NpMixinT) -> np.ndarray | NpMixinT:
    xyz = np.zeros_like(spher)
    xyz[:, 0] = spher[:, 0] * np.sin(spher[:, 2]) * np.cos(spher[:, 1])
    xyz[:, 1] = spher[:, 0] * np.sin(spher[:, 2]) * np.sin(spher[:, 1])
    xyz[:, 2] = spher[:, 0] * np.cos(spher[:, 2])
    return xyz


def xyz2rhv(cart: np.ndarray | NpMixinT) -> np.ndarray:
    spher: np.ndarray = np.zeros_like(cart)
    xy_2: np.ndarray = cart[:, 0]**2 + cart[:, 1]**2
    spher[:, 0] = np.sqrt(xy_2 + cart[:, 2]**2)  # [  0, inf] slope distance
    spher[:, 1] = np.arctan2(cart[:, 1], cart[:, 0])  # [-pi, +pi] horizonal angle
    spher[:, 2] = np.arctan2(np.sqrt(xy_2), cart[:, 2])  # [  0, +pi] zenith angle
    return spher
