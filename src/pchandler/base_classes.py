from __future__ import annotations

from enum import StrEnum
import copy
from typing import Generic, TypeVar, Optional, Any, Self, Callable

import numpy as np
import numpy.typing as npt

from pchandler.geometry.util import return_copy, enforce_immutability

T = TypeVar("T")
FoV = TypeVar("FoV")


class MutabilityOpts(StrEnum):
    MUTABLE = 'mutable'
    IMMUTABLE = 'immutable'
    READONLY = 'readonly'

class CopyOpts(StrEnum):
    BY_ID = 'by_id'
    VIEW = 'view'
    COPY = 'copy'
    DEEP = 'deep'


class BaseField(Generic[T]):
    """ The base field only performs type validation"""
    def __init__(self,
                 name: str,
                 type_: Optional[T | tuple[Any, ...]] = None,
                 ):
        self.name = "_" + name
        self.type_ = type_

    def __get__(self, instance: Any, objtype=None) -> T:
        return getattr(instance, self.name)

    def __set__(self, instance: Any, value: T) -> None:
        self._validate(instance, value)
        setattr(instance, self.name, value)

    def _validate(self, instance: Any, value: T) -> None:
        if (getattr(instance, '_immutable', MutabilityOpts.MUTABLE) in
                (MutabilityOpts.READONLY, MutabilityOpts.IMMUTABLE)):
            raise AttributeError(f"Cannot modify '{self.name}'; object is immutable.")

        if self.type_ is not None and not isinstance(value, self.type_):
            raise TypeError(f"Expected value of type {self.type_}, got {type(value)}.")


class ValidatedField(BaseField[T]):
    """ Extends the BaseField by allowing an additional validation function to be passed"""
    def __init__(self,
                 name: str,
                 type_: Optional[T | tuple[Any, ...]] = None,
                 validate_func: Optional[Callable] = None,
                 ):
        super().__init__(name, type_)
        self.validate_func = validate_func

    def _validate(self, obj: Any, value: T) -> T:
        value = self.validate_func(value)
        super().__set__(obj, value)

class ValidateNumpyField(ValidatedField[T]):
    def __init__(self,
                 name: str,
                 validate_func: Optional[Callable] = None,
                 dtype_: Optional[npt.DTypeLike] = None,
                 ndims_: Optional[int] = None,
                 shape_constraints: dict[int, int] = None
                 ):
        super().__init__(name, np.ndarray, validate_func)
        self.dtype_ = dtype_
        self.ndims_ = ndims_
        self._shape_constraints = shape_constraints if shape_constraints is not None else {}

    def _validate(self, obj: Any, value: T) -> T:
        value = self.validate_func(value)

        if not isinstance(value, np.ndarray):
            raise TypeError(f"Input value is not numpy array. Input: '{type(value)}'")

        if value.dtype != self.dtype_:
            raise TypeError(f"Expected array dtype of {self.dtype_}, got {value.dtype}.")

        for k, v in self._shape_constraints.items():
            if value.shape[k] != v:
                raise ValueError(f'Size along ndim {k} of {value.shape[k]} must be equal to the constraint {v}.')

        super().__set__(obj, value)



class DataArray(np.lib.mixins.NDArrayOperatorsMixin):
    _array_ndims: Optional[int] = None
    _shape_constraints: dict[int, int] = {}
    arr: np.ndarray = ValidateNumpyField(name="arr")

    def __init__(self,
                 array: np.ndarray|Self,
                 *args,
                 copy_option: CopyOpts = 'view',
                 immutable: MutabilityOpts = 'mutable',
                 **kwargs):
        self._mutability: MutabilityOpts = immutable

        if isinstance(array, DataArray):
            self.__dict__ = copy.deepcopy(self.__dict__)

        else:
            self._arr: np.ndarray = self.validate(array)
            self.set_mutability(immutable)

    # TODO fix the copy and view methodolgy.
    #  E.g. buffer w/ read only. View muted / frozen access full object, copy everytime
    @return_copy(deep=True)
    def __getitem__(self, index: Any) -> np.ndarray:
        return self.arr[index]             # Returns a view of the array

    # TODO resolve an index type
    @enforce_immutability
    def __setitem__(self, index, value: np.ndarray|float|int|bool) -> DataArray|np.ndarray|float|int|bool:
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

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs) \
            -> np.ndarray|tuple[np.ndarray,...]|tuple[DataArray,...]:

        arrays = [x.arr if isinstance(x, DataArray) else x for x in inputs]
        result = getattr(ufunc, method)(*arrays, **kwargs)

        if isinstance(result, tuple):
            return tuple(x if np.issubdtype(x.dtype, np.bool_) else DataArray(x) for x in result)

        elif isinstance(result, np.ndarray):
            return result if np.issubdtype(result.dtype, np.bool_) else DataArray(result)

        else:
            return result

    @property
    def mutable(self) -> MutabilityOpts:
        return self._mutability

    def set_mutability(self, value: MutabilityOpts) -> None:
        if value == MutabilityOpts.READONLY:
            self._arr.setflags(write=False)
        else:
            self._arr.setflags(write=True)

        self._mutability: MutabilityOpts = value


    def copy(self, index: Optional[Any] = None, opt: CopyOpts = CopyOpts.VIEW) -> Self:
        match opt:
            case CopyOpts.BY_ID:
                return self     # Creates a pointer to the original object as they're represented by the same ID

            case CopyOpts.COPY:
                method = copy.copy

            case CopyOpts.DEEP:
                method = copy.deepcopy

            case CopyOpts.VIEW:
                temp: DataArray = copy.deepcopy(self)   # Create a deep copy of all other fields, but overwrite the array with a view to the original
                temp._arr = self.arr.view()
                return temp

            case _:
                raise ValueError(f"Invalid copy option entered {opt}")

        return method(self if index is None else self[index])

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


class DataArray2D(DataArray):
    _array_ndims: int = 2


class DataArrayNx2(DataArray2D):
    _shape_constraints: dict[int, int] = {1: 2}

    def __len__(self) -> int:
        return self.arr.shape[0]


class DataArrayNx3(DataArrayNx2):
    _shape_constraints: dict[int, int] = {1: 3}

    def __len__(self) -> int:
        return self.arr.shape[0]

class DataArray3D(DataArray2D):
    _array_ndims: int = 3


class DataArrayMxNx3(DataArray3D):
    _shape_constraints = {2: 3}

class DataArray4D(DataArray3D):
    _array_ndims = 4