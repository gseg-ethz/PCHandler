from __future__ import annotations

import copy
from enum import StrEnum
from abc import abstractmethod, ABC
from functools import wraps, cached_property
from typing import Generic, TypeVar, Optional, Union, Any, Self, overload, TypeAlias, Type, Callable, Tuple

import numpy as np

T = TypeVar("T")
FoV = TypeVar("FoV")
NumOrArray: TypeAlias  = np.ndarray|float|int|list|tuple


class CoordSystemEnum(StrEnum):
    Cartesian = 'cart'
    Spherical = 'spher'


def check_in_range(value: NumOrArray, target_min: float, target_max: float):
    value: np.ndarray = np.asarray(value)
    val_min: float|int = value.min()
    val_max: float|int = value.max()

    if (val_min < target_min) and (val_max > target_max):
        raise ValueError(f'Min and max values [{val_min},{val_max}] exceeds bounds [{target_min},{target_max}].')

    elif val_min < target_min:
        raise ValueError(f'Min value {val_min} exceeds lower limit {target_min}.')

    elif val_max > target_max:
        raise ValueError(f'Max value {val_max} exceeds upper limit {target_max}.')

def check_hz_angles(array: NumOrArray):
    check_in_range(array, -np.pi, np.pi)

def check_zenith_angles(array: NumOrArray):
    check_in_range(array, 0, np.pi)

def check_azimuth_angles(array: NumOrArray):
    check_in_range(array, 0, 2*np.pi)

def check_radial_distances(array: NumOrArray):
    check_in_range(array, 0, np.inf)

def check_inclination_angles(array: NumOrArray):
    check_in_range(array, -np.pi/2, np.pi/2)

def check_spherical_coordinates(array: np.ndarray):
    check_radial_distances(array[:, 0])
    check_zenith_angles(array[:, 1])
    check_hz_angles(array[:, 2])


def bypass_immutable(method):

    @wraps(method)
    def wrapper(self: DataArray, *args, **kwargs):
        original_state: bool = getattr(self, '_immutable', False)
        self.set_immutability(not original_state)

        try:
            return method(self, *args, **kwargs)

        finally:
            self.set_immutability(original_state)

    return wrapper


def return_copy(deep=True):

    def decorator(method):

        @wraps(method)
        def wrapper(self: DataArray, *args, **kwargs):
            result = method(self, *args, **kwargs)

            if not self._immutable:
                return result

            return copy.deepcopy(result) if deep else copy.copy(result)
        return wrapper
    return decorator


class ImmutableField(Generic[T]):
    def __init__(self, name: str, type_: Optional[Union[Type[T], tuple[Type[Any], ...]]] = None):
        self.name = "_" + name
        self.type_ = type_

    def __get__(self, obj: Any, objtype=None) -> T:
        return getattr(obj, self.name)

    def __set__(self, obj: Any, value: T) -> None:
        if getattr(obj, '_immutable', False):
            raise AttributeError(f"Cannot modify '{self.name}'; object is immutable.")

        if self.type_ is not None and not isinstance(value, self.type_):
            raise TypeError(f"Expected value of type {self.type_}, got {type(value)}.")

        setattr(obj, self.name, value)


class DataArray(np.lib.mixins.NDArrayOperatorsMixin):
    _num_rows: Optional[int] = None
    _num_cols: Optional[int] = None
    arr: np.ndarray = ImmutableField[np.ndarray]("arr", type_=np.ndarray)

    def __init__(self, array: np.ndarray|Self, *, immutable: bool = False):
        self._immutable = False

        if isinstance(array, DataArray):
            self.__dict__ = copy.deepcopy(self.__dict__)

        else:
            self._arr: np.ndarray = self.validate(np.asarray(array))

        self.set_immutability(immutable)

    @return_copy(deep=True)
    def __getitem__(self, index: Any) -> np.ndarray:
        return self.arr[index]             # Returns a view of the array

    def __setitem__(self, index, value: np.ndarray|float|int|bool):
        if self.immutable:
            raise AttributeError("Cannot modify coordinates; object is immutable.")

        if isinstance(value, DataArray):
            self.arr[index] = value[index]

        else:
            self.arr[index] = value

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._arr.shape

    @property
    def ndim(self) -> int:
        return self._arr.ndim

    @property
    def size(self) -> int:
        return self._arr.size

    @property
    def base(self) -> np.ndarray|None:
        return self._arr.base

    def __array__(self) -> np.ndarray:
        return self._arr

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs) -> np.ndarray|tuple[np.ndarray,...]|tuple[DataArray,...]:
        arrays = [x.arr if isinstance(x, DataArray) else x for x in inputs]
        result = getattr(ufunc, method)(*arrays, **kwargs)

        if isinstance(result, tuple):
            return tuple(x if np.issubdtype(x.dtype, np.bool_) else DataArray(x) for x in result)

        elif isinstance(result, np.ndarray):
            return result if np.issubdtype(result.dtype, np.bool_) else DataArray(result)

        else:
            return result

    @property
    def immutable(self) -> bool:
        return self._immutable

    def set_immutability(self, value: bool) -> None:
        self._arr.setflags(write=not value)   # Blocks assignment to array as well
        self._immutable: bool = value         # Blocks attribute setting

    def copy(self, index: Optional[Any] = None, deep=False) -> Self:
        if deep or self.immutable:
            func: Callable = copy.deepcopy

        else:
            func: Callable = copy.copy

        if index is None:
            return func(self)

        return func(self[index])

    def validate(self, array: np.ndarray) -> np.ndarray:
        if not isinstance(array, np.ndarray):
            raise TypeError(f"Expected np.ndarray object. Received {type(array)}.")

        if not (np.issubdtype(array.dtype, np.floating) or np.issubdtype(array.dtype, np.integer)):
            raise TypeError(f"Expected floating point array. Received {array.dtype}.")

        if array.size == 0:
            raise ValueError('Array is empty.')

        if array.ndim == 0:
            array = array.reshape(1,)

        if array.ndim not in (1, 2, 3):
            # Assumption is that this will support coordinates or image arrays. Time series data (e.g. image frames or
            # multiple epochs should be treated as a separate class or extend the class to fit
            raise ValueError( f"Expected 2D or 3D array containing coordinates. Received {array.shape=}.")

        if array.shape[0] != self._num_rows and self._num_rows is not None:
            raise ValueError( f"Expected array with {self._num_rows} rows. Received array shape {array.shape=}.")

        if array.ndim > 1:
            if array.shape[1] != self._num_cols and self._num_cols is not None:
                raise ValueError( f"Expected array with {self._num_cols} columns. Received array shape {array.shape=}.")

        return array


class Coordinates3D(DataArray):
    _num_cols = 3
    _NOT_IMPLEMENTED = NotImplementedError('Numpy array input but unknown Coordinate Type')

    def __len__(self) -> int:
        return self.arr.shape[0]

    @property
    def num_points(self) -> int:
        return len(self)

    @property
    def xyz(self) -> np.ndarray:
        raise self._NOT_IMPLEMENTED

    @property
    def spher(self) -> np.ndarray:
        raise self._NOT_IMPLEMENTED

    def validate(self, array: np.ndarray) -> np.ndarray:
        if not np.issubdtype(array.dtype, np.floating):
            raise TypeError(f"Expected floating point array. Received {array.dtype}.")

        return super().validate(array)

    @property
    def x(self) -> np.ndarray:
        return self.xyz[:, 0]

    @property
    def y(self) -> np.ndarray:
        return self.xyz[:, 1]

    @property
    def z(self) -> np.ndarray:
        return self.xyz[:, 2]

    @property
    def r(self) -> np.ndarray:
        return self.spher[:, 0]

    @property
    def v(self) -> np.ndarray:
        return self.spher[:, 1]

    @property
    def hz(self) -> np.ndarray:
        return self.spher[:, 2]

    @property
    def rho(self) -> np.ndarray:
        return self.r

    @property
    def theta(self) -> np.ndarray:
        return self.v

    @property
    def phi(self) -> np.ndarray:
        return self.hz

    @abstractmethod
    def to_spherical(self):
        raise self._NOT_IMPLEMENTED

    @abstractmethod
    def to_cartesian(self):
        raise self._NOT_IMPLEMENTED

    @classmethod
    @abstractmethod
    def from_spherical(cls, coords: "SphericalCoordinates") -> CartesianCoordinates:
        raise cls._NOT_IMPLEMENTED

    @classmethod
    @abstractmethod
    def from_cartesian(cls, coords: "CartesianCoordinates") -> SphericalCoordinates:
        raise cls._NOT_IMPLEMENTED


class CartesianCoordinates(Coordinates3D):
    @property
    def xyz(self) -> np.ndarray:
        return self.arr

    @xyz.setter
    def xyz(self, value: np.ndarray):
        self.arr = self.validate(value)

    def unpack(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.x, self.y, self.z

    @cached_property
    def spher(self) -> np.ndarray:
        return self.to_spherical().arr

    def to_spherical(self) -> SphericalCoordinates:
        return cartesian2spherical(self)

    def to_cartesian(self) -> CartesianCoordinates:
        return self

    @classmethod
    def from_spherical(cls, coords: SphericalCoordinates) -> CartesianCoordinates:
        return spherical2cartesian(coords)

    @classmethod
    def from_cartesian(cls, coords: CartesianCoordinates) -> CartesianCoordinates:
        return cls(coords)


class SphericalCoordinates(Coordinates3D):
    @property
    def spher(self):
        return self.arr

    @spher.setter
    def spher(self, value: np.ndarray):
        self.arr = self.validate(value)

    def unpack(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.r, self.v, self.hz

    @cached_property
    def xyz(self) -> np.ndarray:
        return self.to_cartesian().arr

    def to_cartesian(self) -> CartesianCoordinates:
        return spherical2cartesian(self)

    def to_spherical(self) -> SphericalCoordinates:
        return self

    @classmethod
    def from_cartesian(cls, coords: CartesianCoordinates) -> SphericalCoordinates:
        return cartesian2spherical(coords)

    @classmethod
    def from_spherical(cls, coords: SphericalCoordinates) -> SphericalCoordinates:
        return cls(coords)

    def validate(self, array: np.ndarray) -> np.ndarray:
        check_spherical_coordinates(array)
        return super().validate(array)


class AbstractPointCloud(Coordinates3D, ABC):
    arr: Coordinates3D = ImmutableField[Coordinates3D]("arr", type_=Coordinates3D)
    scalar_fields: dict = ImmutableField[dict]("scalar_fields", type_=dict)
    _optimized: bool = ImmutableField[bool]("_optimized", type_=bool)

    def __init__(self, array: np.ndarray|Self, /, immutable:bool = False, optimize: bool = True, **kwargs):
        self.reduction_pt: Optional[np.ndarray] = kwargs.pop("reduction_pt", None)
        self.scalar_fields = kwargs
        
        if optimize:
            self.optimize(array)

        self._optimized = optimize
        super().__init__(array, immutable=immutable)

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
        if self._check_if_reduction_required(xyz):
            raise NotImplementedError
        return None

    @staticmethod
    def _check_if_reduction_required(array: np.ndarray) -> bool:
        raise NotImplementedError

    @property
    def fov(self) -> FoV:
        raise NotImplementedError


class BasePointCloud(AbstractPointCloud, CartesianCoordinates):
    arr: CartesianCoordinates = ImmutableField[CartesianCoordinates]("arr", type_=CartesianCoordinates)


class SphericalPointCloud(AbstractPointCloud, SphericalCoordinates):
    arr: SphericalCoordinates = ImmutableField[SphericalCoordinates]("arr", type_=SphericalCoordinates)


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


@overload
def spherical2cartesian(coords: np.ndarray) -> np.ndarray: ... # pragma: no cover

@overload
def spherical2cartesian(coords: SphericalCoordinates) -> CartesianCoordinates: ... # pragma: no cover

def spherical2cartesian(coords: np.ndarray|SphericalCoordinates) -> np.ndarray|CartesianCoordinates:
    def _spher2cart_arr(spherical: np.ndarray) -> np.ndarray:
        xyz = np.zeros_like(spherical)
        xyz[:, 0], xyz[:, 1], xyz[:, 2] = _spher2cart_vec(rho=spherical[:, 0], theta=spherical[:, 1], phi=spherical[:, 2])
        return xyz

    def _spher2cart_vec(rho: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = rho * np.sin(theta) * np.cos(phi)
        y = rho * np.sin(theta) * np.sin(phi)
        z = rho * np.cos(theta)
        return x, y, z

    if not isinstance(coords, SphericalCoordinates|np.ndarray):
        raise TypeError("Input coordinates must either be SphericalCoordinates or np.ndarray")

    return CartesianCoordinates(_spher2cart_arr(coords.__array__()))

@overload
def cartesian2spherical(coords: CartesianCoordinates) -> SphericalCoordinates: ... # pragma: no cover

@overload
def cartesian2spherical(coords: np.ndarray) -> np.ndarray: ... # pragma: no cover

def cartesian2spherical(coords: np.ndarray|CartesianCoordinates) -> np.ndarray|SphericalCoordinates:
    def _cart2spher_arr(xyz: np.ndarray) -> np.ndarray:
        spherical = np.zeros_like(xyz)
        spherical[:, 0], spherical[:, 1], spherical[:, 2] = _cart2spher_vec(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2])
        return spherical

    def _cart2spher_vec(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xy_2 = x ** 2 + y ** 2
        rho = np.sqrt(xy_2 + z ** 2)            # [  0, inf] slope distance
        theta = np.arctan2(np.sqrt(xy_2), z)    # [  0, +pi] zenith angle
        phi = np.arctan2(y, x)                  # [-pi, +pi] horizonal angle
        return rho, theta, phi

    if not isinstance(coords, CartesianCoordinates|np.ndarray):
        raise TypeError("Input coordinates must either be CartesianCoordinates or np.ndarray")

    return SphericalCoordinates(_cart2spher_arr(coords.__array__()))

