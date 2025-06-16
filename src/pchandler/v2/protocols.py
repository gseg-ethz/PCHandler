from __future__ import annotations

from typing import Any, Protocol, Self

import numpy as np


class CoordinateSet(Protocol):
    __num_cols: int

    def __array__(self) -> np.ndarray: ...


class CartesianCoordProtocol(Protocol):
    @property
    def xyz(self) -> np.ndarray: ...

    @property
    def x(self): ...

    @property
    def y(self): ...

    @property
    def z(self): ...

    @property
    def spher(self): ...

    def to_spherical(self) -> SphericalCoordProtocol: ...

    @classmethod
    def from_spherical(cls, xyz: np.ndarray) -> Self: ...


class SphericalCoordProtocol(Protocol):
    @property
    def spher(self) -> np.ndarray: ...

    @property
    def hz(self) -> np.ndarray: ...

    @property
    def v(self): ...

    @property
    def r(self): ...

    def to_cartesian(self) -> CartesianCoordProtocol: ...

    def from_cartesian(self, hz: np.ndarray | CartesianCoordProtocol) -> CartesianCoordProtocol: ...


class PointCloudProtocol(CartesianCoordProtocol, SphericalCoordProtocol, Protocol):
    @property
    def coords(self) -> CartesianCoordProtocol | SphericalCoordProtocol: ...

    @property
    def scalar_fields(self) -> np.ndarray: ...

    @property
    def intensity(self) -> np.ndarray: ...

    @property
    def normals(self) -> np.ndarray: ...

    @property
    def rgb(self) -> np.ndarray: ...

    @property
    def reflectance(self) -> np.ndarray: ...

    @property
    def transformations(self) -> Any: ...

    @property
    def metadata(self) -> dict[str, Any]: ...


class FilterResult(Protocol):
    @property
    def filter_mask(self) -> np.ndarray: ...

    def inside(self) -> np.ndarray: ...

    def outside(self) -> np.ndarray: ...
