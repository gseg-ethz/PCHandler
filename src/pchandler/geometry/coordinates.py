from __future__ import annotations

from abc import ABC
from functools import cached_property
from typing import Self

import numpy as np

from src.pchandler.base_descriptors import ArrayDescriptor, BaseDescriptor
from src.pchandler.base_arrays import Point3d, ArrayNx3, TransformMatrix, NpMixinT

class Abstract3DCoordinates(ABC, ArrayNx3):
    _arr: ArrayNx3 = ArrayDescriptor(ArrayNx3, coerce=True)
    _shift: Point3d = ArrayDescriptor(Point3d, optional=False)
    _project_transform: TransformMatrix = ArrayDescriptor(TransformMatrix, optional=False)
    _global_shifted: bool = BaseDescriptor(bool, default=False, optional=False)

    def __init__(self,
                 coordinates: np.ndarray|Self,
                 shift: np.ndarray = np.zeros(3, dtype=np.float32),
                 transform_matrix: np.ndarray | TransformMatrix = np.eye(4, dtype=np.float32),
                 global_shifted: bool = False) -> None:
        self._arr: Self = type(self)(coordinates)
        self._shift = Point3d(shift)
        self._project_transform = TransformMatrix(transform_matrix)
        self._global_shifted = global_shifted
        super().__init__(coordinates)

# TODO the idea here is that whenever initialised, the scan center coordinates are always provided
#  A transformation can be provided with it
class CartesianCoordinates(Abstract3DCoordinates):
    # TODO implement a function to gracefully handle transformation matrices

    # TODO discuss - my idea is that much like in cloud compare, the shift is applied on load and then never updated.
    #  well... I assume it works that way
    def as_global(self) -> CartesianCoordinates:
        return self._project_transform @ self.as_local()

    def as_local(self) -> CartesianCoordinates:
        if not np.all(self._shift == 0):
            return self + self._shift
        else:
            return self

    def as_optimal(self) -> CartesianCoordinates:
        return self._arr

    def compute_global_shift(self, xyz: ArrayNx3, decimal_magnitude: int = 4) -> tuple[ArrayNx3, Point3d]:
        delta_t: Point3d = Point3d([0, 0, 0])
        if self._is_shift_needed(xyz):
            delta_t = Point3d(np.median( np.round( xyz, decimals = -( decimal_magnitude - 1 ) ), axis = 0 ))

        t_new = Point3d(self._shift - delta_t)
        xyz = ArrayNx3((self._arr + delta_t).astype(self.__dtype__))
        return xyz, t_new

    def __setitem__(self, key: str, value: np.ndarray):
        if key == '_arr':
            del self.__dict__['spher']
        super().__setitem__(key, value)

    @property
    def x(self): return self._arr[:, 0]

    @x.setter
    def x(self, value): self._arr[:, 0] = value

    @property
    def y(self): return self._arr[:, 1]

    @y.setter
    def y(self, value): self._arr[:, 1] = value

    @property
    def z(self): return self._arr[:, 2]

    @z.setter
    def z(self, value): self._arr[:, 2] = value

    @property
    def xyz(self): return self._arr

    @property
    def yxz(self): return np.vstack((self.y, self.x, self.z)).T

    @staticmethod
    def _is_shift_needed(xyz: ArrayNx3|np.ndarray, decimal_magnitude: int = 4) -> np.bool_:
        return np.any(np.abs(xyz) >= 10 ** decimal_magnitude)

    def to_spherical(self) -> SphericalCoordinates:
        return cartesian2spherical(self.as_local())

    @classmethod
    def from_spherical(cls, spherical: SphericalCoordinates) -> CartesianCoordinates:
        return spherical2cartesian(spherical)

    @cached_property
    def spher(self) -> SphericalCoordinates:
        return self.to_spherical()

    @cached_property
    def r(self):
        return self.spher.r

    @property
    def hz(self):
        return self.spher.hz

    @property
    def v(self):
        return self.spher.v


class SphericalCoordinates(Abstract3DCoordinates):
    # TODO add in some spherical coordinate validation based on defined coordinates system
    @property
    def rhv(self):
        return self._arr

    @property
    def r(self) -> np.ndarray:
        return self.rhv[:, 0]

    @r.setter
    def r(self, value: np.ndarray):
        self.rhv[:, 0] = value

    @property
    def hz(self) -> np.ndarray:
        return self.rhv[:, 1]

    @hz.setter
    def hz(self, value: np.ndarray):
        self.rhv[:, 1] = value

    @property
    def v(self):
        return self.rhv[:, 2]

    @v.setter
    def v(self, value: np.ndarray):
        self.rhv[:, 2] = value

    def to_cartesian(self):
        return spherical2cartesian(self.rhv)

    @classmethod
    def from_cartesian(cls, cartesian: CartesianCoordinates):
        return cartesian2spherical(cartesian)


def spherical2cartesian(coordinates: SphericalCoordinates) -> CartesianCoordinates:
    xyz = CartesianCoordinates(np.zeros_like(coordinates))
    xyz.x = coordinates.r * np.sin(coordinates.v) * np.cos(coordinates.hz)
    xyz.y = coordinates.r * np.sin(coordinates.v) * np.sin(coordinates.hz)
    xyz.z = coordinates.r * np.cos(coordinates.v)
    return CartesianCoordinates(xyz)

def cartesian2spherical(coordinates: CartesianCoordinates) -> SphericalCoordinates:
    spher: SphericalCoordinates = SphericalCoordinates(np.zeros_like(coordinates))
    xy_2: np.ndarray = coordinates.x ** 2 + coordinates.y ** 2
    spher.r = np.sqrt(xy_2 + coordinates.z ** 2)              # [  0, inf] slope distance
    spher.hz = np.arctan2(coordinates.y, coordinates.x)        # [-pi, +pi] horizonal angle
    spher.v = np.arctan2(np.sqrt(xy_2), coordinates.z)        # [  0, +pi] zenith angle
    return SphericalCoordinates(spher)

