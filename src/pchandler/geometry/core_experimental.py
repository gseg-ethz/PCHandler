from __future__ import annotations

from abc import ABC
from typing import Optional, Any, Self

import numpy as np

from pchandler.geometry.base_classes import ImmutableField
from pchandler.geometry.coordinates import CoordinateSet3D, CartesianCoordinates, SphericalCoordinates


class PointCloud(CoordinateSet3D, ABC):
    arr: CoordinateSet3D = ImmutableField[CoordinateSet3D]("arr", type_=CoordinateSet3D)
    scalar_fields: dict = ImmutableField[dict]("scalar_fields", type_=dict)
    _optimized: bool = ImmutableField[bool]("_optimized", type_=bool)

    def __init__(self, array: np.ndarray|Self, *args, immutable:bool = False, optimize: bool = True, **kwargs):
        self.reduction_pt: Optional[np.ndarray] = kwargs.pop("reduction_pt", None)
        self.scalar_fields = kwargs
        
        if optimize:
            self.optimize(array)

        self._optimized = optimize
        super().__init__(array, *args, immutable=immutable, **kwargs)

    def set_immutability(self, value: bool) -> None:
        self.reduction_pt.setflags(write=not value)
        super().set_immutability(value)

    # TODO not working yet
    def __getitem__(self, index: Any) -> Self:
        array: np.ndarray = self.arr[index]

        for k, v in self.scalar_fields.items():
            self.scalar_fields[k] = v[index].copy() if self.immutable else v[index]

        return type(self)(array, immutable=self.immutable, **self.scalar_fields)

    def optimize(self, array: np.ndarray|Self) -> Self:
        self.reduction_pt: Optional[np.ndarray] = self._compute_reduction_point(array)
        array -= self.reduction_pt
        self._optimized = True

    @property
    def reduced_(self) -> Self:
        return self

    @property
    def local_(self) -> Self:
        if self._optimized:
            return self + self.reduction_pt
        return self

    @property
    def global_(self) -> Self:
        return self.local_

    def _compute_reduction_point(self, xyz: np.ndarray) -> Optional[np.ndarray]:
        if self._check_if_required(xyz):
            raise NotImplementedError
        return None

    @staticmethod
    def _check_if_required(array: np.ndarray) -> bool:
        raise NotImplementedError

    @property
    def fov(self) -> FoV:
        raise NotImplementedError


class BasePointCloud(AbstractPointCloud, CartesianCoordinates):
    arr: CartesianCoordinates = ImmutableField[CartesianCoordinates](
        "arr", type_=CartesianCoordinates)


class SphericalPointCloud(AbstractPointCloud, SphericalCoordinates):
    arr: SphericalCoordinates = ImmutableField[SphericalCoordinates](
        "arr", type_=SphericalCoordinates)


class TlsPointCloud(BasePointCloud):
    global_transform: np.ndarray = ImmutableField[np.ndarray]('global_transform', type_=np.ndarray)

    @property
    def intensity(self) -> np.ndarray:
        if 'intensity' in self.scalar_fields.keys():
            return self.scalar_fields['intensity']
        else:
            raise ValueError("No intensity scalar field existing in point cloud")

    @property
    def global_(self):
        array: Self = super().global_
        return self.global_transform @ array


class MultiScanPointCloud(BasePointCloud):
    pass


class FoV:
    raise NotImplementedError

