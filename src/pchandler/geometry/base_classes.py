from __future__ import annotations

import copy
from typing import Generic, TypeVar, Optional, Any, Self, Callable

import numpy as np

from pchandler.geometry.util import return_copy

T = TypeVar("T")
FoV = TypeVar("FoV")


class ImmutableField(Generic[T]):
    def __init__(self, name: str, type_: Optional[T | tuple[Any, ...]] = None):
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
    _array_ndims: Optional[int] = None
    arr: np.ndarray = ImmutableField[np.ndarray]("arr", type_=np.ndarray)

    def __init__(self, array: np.ndarray|Self, *args, immutable: bool = False, **kwargs):
        self._immutable: bool = False   # Start as false and then update once an object has been initialized

        if isinstance(array, DataArray):
            self.__dict__ = copy.deepcopy(self.__dict__)

        else:
            self._arr: np.ndarray = self.validate(array)
            self.set_immutability(immutable)

    # TODO discuss and decide on a copy / deepcopy / view approach
    @return_copy(deep=True)
    def __getitem__(self, index: Any) -> np.ndarray:
        return self.arr[index]             # Returns a view of the array

    # TODO resolve an index type
    def __setitem__(self, index, value: np.ndarray|float|int|bool):
        if self.immutable:
            raise AttributeError("Cannot modify coordinates; object is immutable.")

        self.arr[index] = value[index] if isinstance(value, DataArray) else value

    @property
    def ndim(self) -> int:
        return self._arr.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        return self._arr.shape

    @property
    def size(self) -> int:
        return self._arr.size

    @property
    def dtype(self) -> np.dtype:
        return self._arr.dtype

    @property
    def base(self) -> np.ndarray|None:
        return self._arr.base

    def __array__(self) -> np.ndarray:
        return self._arr

    @property
    def __array_interface__(self) -> dict:
        return self._arr.__array_interface__

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
        return func(self if index is None else self[index])

    def validate(self, array: np.ndarray) -> np.ndarray:
        if not isinstance(array, np.ndarray):
            raise TypeError(f"Expected np.ndarray object. Received {type(array)}.")

        if not (np.issubdtype(array.dtype, np.floating) or np.issubdtype(array.dtype, np.integer)):
            raise TypeError(f"Expected floating point array. Received {array.dtype}.")

        if array.size == 0:
            raise ValueError('Array is empty.')

        if self._array_ndims is not None:
            if array.ndim == 0:
                array = array.reshape(1,)
            elif array.ndim != self._array_ndims:
                raise ValueError(f"Expected array with {self._array_ndims} dimensions. Received [{array.shape=}].")

        return array

class DataArray1D(DataArray):
    _array_ndims: int = 1
    _num_rows: Optional[int] = None

    def validate(self, array: np.ndarray) -> np.ndarray:
        if self._array_ndims == 1:
            if array.ndim == (self._array_ndims+1) and 1 in array.shape:
                array = array.reshape(-1)

        array = super().validate(array)

        if array.shape[0] != self._num_rows and self._num_rows is not None:
            raise ValueError(f"Expected array with {self._num_rows} rows. Received array shape {array.shape=}.")

        return array

class DataArray2D(DataArray):
    _array_ndims: int = 2
    _num_cols: Optional[int] = None

    def validate(self, array: np.ndarray) -> np.ndarray:
        array = super().validate(array)

        if array.shape[1] != self._num_cols and self._num_cols is not None:
            raise ValueError(f"Expected array with {self._num_cols} columns. Received array shape {array.shape=}.")

        return array


class DataArrayNx2(DataArray2D):
    _num_cols: int = 2

    def __len__(self) -> int:
        return self.arr.shape[0]


class DataArrayNx3(DataArrayNx2):
    _num_cols: int = 3


class DataArray3D(DataArray2D):
    _array_ndims: int = 3
    _num_channels: Optional[int] = None

    def validate(self, array: np.ndarray) -> np.ndarray:
        array: np.ndarray = super().validate(array)
        if array.shape[2] != self._num_channels and self._num_channels is not None:
            raise ValueError(f"Expected array with {self._num_channels} channels/epochs. "
                             f"Received array shape {array.shape=}.")

        return array


class DataArrayMxNx3(DataArray3D):
    _num_channels: int = 3

class DataArray4D(DataArray3D):
    _array_ndims = 4