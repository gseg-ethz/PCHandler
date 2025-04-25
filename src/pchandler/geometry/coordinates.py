from __future__ import annotations

from enum import IntEnum
from functools import cached_property
from typing import Literal

import numpy as np

from pchandler.geometry.base_classes import DataArrayNx3
from pchandler.geometry.validation import check_spherical_coordinates, NumOrArray


class CoordSysEnum(IntEnum):
    CART = 0
    SPHER = 1


def spherical2cartesian(coords: np.ndarray) -> np.ndarray:
    def _spher2cart_arr(spherical: np.ndarray) -> np.ndarray:
        xyz: np.ndarray = np.zeros_like(spherical)
        xyz[:, 0], xyz[:, 1], xyz[:, 2] = _spher2cart_vec(spherical[:, 0], spherical[:, 1], spherical[:, 2])
        return xyz

    def _spher2cart_vec(rho: NumOrArray, theta: NumOrArray, phi: NumOrArray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x: NumOrArray = rho * np.sin(theta) * np.cos(phi)
        y: NumOrArray = rho * np.sin(theta) * np.sin(phi)
        z: NumOrArray = rho * np.cos(theta)
        return x, y, z

    return _spher2cart_arr(coords)

def cartesian2spherical(coords: np.ndarray) -> np.ndarray:
    def _cart2spher_arr(xyz: np.ndarray) -> np.ndarray:
        spherical: np.ndarray = np.zeros_like(xyz)
        spherical[:, 0], spherical[:, 1], spherical[:, 2] = _cart2spher_vec(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2])
        return spherical

    def _cart2spher_vec(x: NumOrArray, y: NumOrArray, z: NumOrArray) -> tuple[NumOrArray, NumOrArray, NumOrArray]:
        xy_2: NumOrArray = x ** 2 + y ** 2
        rho: NumOrArray = np.sqrt(xy_2 + z ** 2)            # [  0, inf] slope distance
        theta: NumOrArray = np.arctan2(np.sqrt(xy_2), z)    # [  0, +pi] zenith angle
        phi: NumOrArray = np.arctan2(y, x)                  # [-pi, +pi] horizonal angle
        return rho, theta, phi

    return _cart2spher_arr(coords)


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
    _coord_system: CoordSysEnum = CoordSysEnum.CART

    def __init__(self, array,
                 coord_system: Literal['cartesian']|Literal['spherical']|CoordSysEnum = CoordSysEnum.CART, **kwargs):

        super().__init__(array, **kwargs)
        if coord_system not in CoordSysEnum:
            raise ValueError("coord_system must be from 'CoordSysEnum'")

        self._coord_system = CoordSysEnum[coord_system] if isinstance(coord_system, str) else coord_system

    @property
    def num_points(self) -> int:
        return len(self)

    @property
    def _cartesian_view(self):
        return CartesianView(self)

    @property
    def _spherical_view(self):
        return SphericalView(self)

    def clear_cache(self):
        self._cartesian_view._invalidate_cache()
        self._spherical_view._invalidate_cache()

    def _convert_to_system(self, system: CoordSysEnum):
        if self._coord_system != system:

            self.clear_cache()
            self._coord_system = system

            if self._coord_system == CoordSysEnum.SPHER:
                self.arr = spherical2cartesian(self.arr)
            else:
                self.arr = cartesian2spherical(self.arr)

    def to_cartesian(self):
        self._convert_to_system(CoordSysEnum.CART)

    def to_spherical(self):
        self._convert_to_system(CoordSysEnum.SPHER)

    def validate(self, array: np.ndarray) -> np.ndarray:
        if not np.issubdtype(array.dtype, np.floating):
            raise TypeError(f"Expected floating point array. Received {array.dtype}.")

        if self._coord_system == CoordSysEnum.SPHER:
            check_spherical_coordinates(array)
        return array


class CartesianView:
    _prop_names: tuple[str, ...] = ("x", "y", "z", "xyz")

    def __init__(self, parent: CoordinateSet3D):
        self.parent: CoordinateSet3D = parent
        self.parent.xyz = self.xyz
        self.parent.x = self.xyz[:, 0].view()
        self.parent.y = self.xyz[:, 1].view()
        self.parent.z = self.xyz[:, 2].view()

    def _get_data(self):
        if self.parent._coord_system == CoordSysEnum.SPHER:
            return spherical2cartesian(self.parent.arr)
        return self.parent.arr.view()

    @cached_property
    def xyz(self) -> np.ndarray:
        return self._get_data()

    def _invalidate_cache(self):
        for name in self._prop_names:
            if name in self.__dict__:
                del self.__dict__[name]


class SphericalView:
    _prop_names: tuple[str, ...] = ("rho", "theta", "phi", 'hz', 'v', 'r', 'spher')

    def __init__(self, parent: CoordinateSet3D):
        self.parent: CoordinateSet3D = parent
        self.parent.spher = self.spher
        self.parent.r = self.rho = self.spher[:, 0].view()
        self.parent.v = self.theta = self.spher[:, 1].view()
        self.parent.hz = self.phi = self.spher[:, 2].view()

    def _get_data(self):
        if self.parent._coord_system == CoordSysEnum.CART:
            return cartesian2spherical(self.parent.arr)
        return self.parent.arr.view()

    @cached_property
    def spher(self) -> np.ndarray:
        return self._get_data()

    def _invalidate_cache(self):
        for name in self._prop_names:
            if name in self.__dict__:
                del self.__dict__[name]
