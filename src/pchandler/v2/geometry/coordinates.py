from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from enum import IntEnum
from functools import cached_property
from typing import Optional, Self, overload

import numpy as np

from pchandler.v2.base_arrays import


class CoordSysEnum(IntEnum):
    OPTIMAL = 0
    SOC = 1
    PROJECT = 2


class Abstract3dCoordinates(ABC, ArrayNx3):
    @property
    @abstractmethod
    def xyz(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def spher(self) -> np.ndarray:
        pass


class CartesianCoordinates(Abstract3dCoordinates):
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


class CartesianKdTree(CartesianCoordinates):
    _arr: np.ndarray = Descriptor(KdTree)

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
    global_offset: np.ndarray | Vector3 = Descriptor(Vector3, optional=False)
    transform: np.ndarray | TransformArray4x4 = Descriptor(TransformArray4x4, optional=False)
    coordinate_system: CoordSysEnum = Descriptor(CoordSysEnum, optional=False, default=CoordSysEnum.SOC)

    def __init__(
        self,
        coordinates: np.ndarray | Self,
        shift: np.ndarray = None,
        global_offset: Optional[np.ndarray | Vector3] = None,
        transform_matrix: Optional[np.ndarray | TransformArray4x4] = None,
        coord_system: Optional[CoordSysEnum] = CoordSysEnum.SOC,
    ) -> None:
        if isinstance(coordinates, GlobalShiftedCoordinates):
            self.__dict__ |= copy.deepcopy(coordinates.__dict__)
        else:
            super().__init__(coordinates)

        if global_offset is not None:
            self.global_offset = Vector3(global_offset)
        if transform_matrix is not None:
            self.transform = TransformArray4x4(transform_matrix)
        if coord_system is not None:
            self.coordinate_system = CoordSysEnum.SCAN

    # DISCUSS: the idea here is that whenever initialised, the scan center coordinates are always provided
    #  A transformation can be provided with it

    def as_proj(self) -> CartesianCoordinates:
        return self.transform @ self.as_soc()

    def as_soc(self) -> CartesianCoordinates:
        return self + self.global_offset

    def as_local(self) -> CartesianCoordinates:
        return self

    def compute_global_shift(self, xyz: ArrayNx3, decimal_magnitude: int = 4) -> tuple[ArrayNx3, Vector3]:

        delta_t: Vector3 = Vector3()
        if self._is_shift_needed(xyz):
            delta_t = self._compute_shift(xyz, decimal_magnitude)
        t_new = Vector3(self.global_offset - delta_t)
        xyz = ArrayNx3((self._arr + delta_t).astype(self.__dtype__))
        return xyz, t_new

    def __setitem__(self, key: str, value: np.ndarray):
        if key == "_arr":
            del self.__dict__["spher"]
        super().__setitem__(key, value)

    @staticmethod
    def _compute_shift(xyz, decimal_magnitude: int = 4):
        return Vector3(np.median(np.round(xyz, decimals=-(decimal_magnitude - 1)), axis=0))

    @staticmethod
    def _is_shift_needed(xyz: ArrayNx3 | np.ndarray, decimal_magnitude: int = 4) -> np.bool_:
        return np.any(np.abs(xyz) >= 10**decimal_magnitude)

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
