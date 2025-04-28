from __future__ import annotations

from enum import IntEnum
from functools import cached_property
from typing import Literal

import numpy as np

from pchandler.geometry.util import bypass_immutable
from pchandler.base_classes import DataArrayNx3
from pchandler.geometry.validation import check_spherical_coordinates, NumOrArray


class CoordSysEnum(IntEnum):
    CART = 0
    SPHER = 1

CoordSystemNamesT = Literal['cartesian']|Literal['spherical']|CoordSysEnum



def spherical2cartesian(spherical: np.ndarray) -> np.ndarray:
    xyz: np.ndarray = np.zeros_like(spherical)
    xyz[:, 0], xyz[:, 1], xyz[:, 2] = spher2cart_vec(spherical[:, 0], spherical[:, 1], spherical[:, 2])
    return xyz

def spher2cart_vec(rho: NumOrArray, theta: NumOrArray, phi: NumOrArray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x: NumOrArray = rho * np.sin(theta) * np.cos(phi)
    y: NumOrArray = rho * np.sin(theta) * np.sin(phi)
    z: NumOrArray = rho * np.cos(theta)
    return x, y, z


def cartesian2spherical(xyz: np.ndarray) -> np.ndarray:
    spherical: np.ndarray = np.zeros_like(xyz)
    spherical[:, 0], spherical[:, 1], spherical[:, 2] = cart2spher_vec(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    return spherical

def cart2spher_vec(x: NumOrArray, y: NumOrArray, z: NumOrArray) -> tuple[NumOrArray, NumOrArray, NumOrArray]:
    xy_2: NumOrArray = x ** 2 + y ** 2
    rho: NumOrArray = np.sqrt(xy_2 + z ** 2)            # [  0, inf] slope distance
    theta: NumOrArray = np.arctan2(np.sqrt(xy_2), z)    # [  0, +pi] zenith angle
    phi: NumOrArray = np.arctan2(y, x)                  # [-pi, +pi] horizonal angle
    return rho, theta, phi



# TODO Implement a Storage and View Structure design
#  - separation of concerns
#  - shallow call paths
#  - coordinate system control
#  - caching of the complimentary coordinates
#  - cleanm descriptive code
#  - no need to recompute coordinates if not required
#  - mutability in place will depend on if the overarching array is provided as mutable
#  - invalidate cache on mutation
class CoordinateSet3D(DataArrayNx3):
    coord_system: CoordSysEnum = CoordSysEnum.CART

    def __init__(self, array, coord_system: CoordSystemNamesT = CoordSysEnum.CART, **kwargs):

        super().__init__(array, **kwargs)
        if coord_system not in CoordSysEnum:
            raise ValueError("coord_system must be from 'CoordSysEnum'")

        self.coord_system = CoordSysEnum[coord_system] if isinstance(coord_system, str) else coord_system

    @property
    def num_points(self) -> int:
        return len(self)

    @property
    def _prop_names(self) -> frozenset[str]:
        return frozenset([])

    def invalidate_cache(self):
        for name in self._prop_names:
            if name in self.__dict__:
                del self.__dict__[name]

    def _convert_to_system(self, target_system: CoordSysEnum):
        if self.coord_system != target_system:

            self.invalidate_cache()
            self.coord_system = target_system

            if self.coord_system == CoordSysEnum.CART:
                self.arr = spherical2cartesian(self.arr)
            else:
                self.arr = cartesian2spherical(self.arr)

    def validate(self, array: np.ndarray) -> np.ndarray:
        array = super().validate(array)

        if not np.issubdtype(array.dtype, np.floating):
            raise TypeError(f"Expected floating point array. Received {array.dtype}.")

        if self.coord_system == CoordSysEnum.SPHER:
            check_spherical_coordinates(array)
        return array


class CartesianCoordinates(CoordinateSet3D):
    @property
    def _prop_names(self) -> frozenset[str]:
        return super()._prop_names | frozenset({'_xyz'})

    @cached_property
    def _xyz(self) -> np.ndarray:
        return spherical2cartesian(self.arr)

    @property
    def xyz(self) -> np.ndarray:
        return self._get_cartesian_data()

    @xyz.setter
    def xyz(self, xyz: np.ndarray) -> None:
        if self.coord_system != CoordSysEnum.CART:
            self._convert_to_system(CoordSysEnum.CART)
        self.arr = xyz

    @property
    def x(self) -> NumOrArray:
        return self.xyz[:, 0].view()

    @x.setter
    def x(self, x: NumOrArray) -> None:
        if self.coord_system != CoordSysEnum.CART:
            raise ValueError("Cannot set 'x' whilst coord system is SPHERICAL")
        self.xyz[:, 0] = x

    @property
    def y(self) -> NumOrArray:
        return self.xyz[:, 1].view()

    @y.setter
    def y(self, y: NumOrArray) -> None:
        if self.coord_system != CoordSysEnum.CART:
            raise ValueError("Cannot set 'y' whilst coord system is SPHERICAL")
        self.xyz[:, 1] = y

    @property
    def z(self) -> NumOrArray:
        return self.xyz[:, 2].view()

    @z.setter
    def z(self, z: NumOrArray) -> None:
        if self.coord_system != CoordSysEnum.CART:
            raise ValueError("Cannot set 'z' whilst coord system is SPHERICAL")
        self.xyz[:, 2] = z

    def _get_cartesian_data(self):
        if self.coord_system == CoordSysEnum.SPHER:
            return self._xyz
        return self.arr.view()

    @bypass_immutable
    def to_spherical(self):
        self._convert_to_system(CoordSysEnum.SPHER)


class SphereCoordinates(CoordinateSet3D):
    @property
    def _prop_names(self) -> frozenset[str]:
        return super()._prop_names | frozenset(['_spher'])

    @property
    def spher(self) -> np.ndarray:
        return self._get_spherical_data()

    @spher.setter
    def spher(self, spher: np.ndarray) -> None:
        if self.coord_system != CoordSysEnum.SPHER:
            self._convert_to_system(CoordSysEnum.SPHER)
        self.arr = spher


    @cached_property
    def _spher(self) -> np.ndarray:
        return cartesian2spherical(self.arr)

    @property
    def r(self):
        return self.spher[:, 0].view()

    @r.setter
    def r(self, r: NumOrArray) -> None:
        if self.coord_system != CoordSysEnum.SPHER:
            raise ValueError("Cannot set 'r' whilst coord system is CARTESIAN")
        self.spher[:, 0] = r

    @property
    def v(self):
        return self.spher[:, 1].view()

    @v.setter
    def v(self, v: NumOrArray) -> None:
        if self.coord_system != CoordSysEnum.SPHER:
            raise ValueError("Cannot set 'v' whilst coord system is SPHERICAL")
        self.spher[:, 1] = v

    @property
    def hz(self):
        return self.spher[:, 2].view()

    @hz.setter
    def hz(self, hz: NumOrArray) -> None:
        if self.coord_system != CoordSysEnum.SPHER:
            raise ValueError("Cannot set 'hz' whilst coord system is SPHERICAL")
        self.spher[:, 2] = hz

    @property
    def rho(self):
        return self.r

    @rho.setter
    def rho(self, rho: NumOrArray) -> None:
        if self.coord_system != CoordSysEnum.SPHER:
            raise ValueError("Cannot set 'rho' whilst coord system is SPHERICAL")
        self.spher[:, 0] = rho

    @property
    def theta(self):
        return self.v

    @theta.setter
    def theta(self, theta: NumOrArray) -> None:
        if self.coord_system != CoordSysEnum.SPHER:
            raise ValueError("Cannot set 'theta' whilst coord system is SPHERICAL")
        self.spher[:, 1] = theta

    @property
    def phi(self):
        return self.hz

    @phi.setter
    def phi(self, phi: NumOrArray) -> None:
        if self.coord_system != CoordSysEnum.SPHER:
            raise ValueError("Cannot set 'phi' whilst coord system is SPHERICAL")
        self.spher[:, 2] = phi

    def _get_spherical_data(self):
        if self.coord_system == CoordSysEnum.CART:
            return self._spher
        return self.arr.view()

    @bypass_immutable
    def to_cartesian(self):
        self._convert_to_system(CoordSysEnum.CART)


class GeneralCoordinates(CartesianCoordinates, SphereCoordinates):
    @property
    def _prop_names(self):
        return super()._prop_names