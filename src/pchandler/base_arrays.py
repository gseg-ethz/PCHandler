from __future__ import annotations
import copy
from typing import Any, Optional

import numpy as np
import numpy.typing as npt


from src.pchandler.base_descriptors import ArrayDescriptor

NpMixinT = np.lib.mixins.NDArrayOperatorsMixin

class ValidatedArray(NpMixinT):
    __ndim__: int | None = None
    __shape__: tuple[Optional[int], ...] = (None,)
    __dtype__: np.dtype = None
    _arr: ArrayDescriptor = ArrayDescriptor(np.ndarray)

    def __init__(self, array: np.ndarray | ValidatedArray):

        if isinstance(array, ValidatedArray):
            self.validate(array._arr)
            self.__dict__ = copy.deepcopy(self.__dict__)
        else:
            array = self.coerce_array(array)
            self.validate(array)
            self._arr: np.ndarray = array

    def __getitem__(self, index: Any) -> np.ndarray:
        return self._arr[index]

    def __setitem__(self, index, value: np.ndarray | NpMixinT |  float | int | bool) \
            -> ValidatedArray | NpMixinT | np.ndarray | float | int | bool:
        # TODO need to re-implement the logic here to validate the item being set.
        self._arr[index] = value[index] if isinstance(value, (np.ndarray, NpMixinT)) else value

    def __array__(self) -> np.ndarray:
        return self._arr

    @property
    def __array_interface__(self) -> dict:
        return self._arr.__array_interface__

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs) \
            -> np.ndarray|tuple[np.ndarray,...]|tuple[ValidatedArray,...]:

        arrays = [x._arr if isinstance(x, ValidatedArray) else x for x in inputs]
        result = getattr(ufunc, method)(*arrays, **kwargs)

        if isinstance(result, tuple):
            return tuple(x if np.issubdtype(x.dtype, np.bool_) else type(self)(x) for x in result)

        elif isinstance(result, np.ndarray):
            return result if np.issubdtype(result.dtype, np.bool_) else type(self)(result)

        else:
            return result

    def validate(self, value: np.ndarray) -> None:
        self._check_ndarray(value)

    def _check_ndarray(self, value: np.ndarray):
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Input of '{value}' is not actually a numpy array: {type(value)}")

        if self.__dtype__ is not None and self.__dtype__ != value.dtype:
            raise TypeError(f"Input array does not have the expected type of {self.__dtype__}")

        if self.__ndim__ is not None and self.__ndim__ != value.ndim:
            raise ValueError(f"Expected array with {self.__ndim__} dimensions. Received [{value.shape=}].")

        if self.__shape__ is not None:
            for i, size_dim in enumerate(self.__shape__):
                if size_dim is not None and size_dim != value.shape[i]:
                    raise ValueError(f"Dimension {i} shape does not match, expected {size_dim} != {value.shape[i]}")

    def coerce_array(self, array: np.ndarray) -> np.ndarray:
        return array

    @property
    def ndim(self) -> int:
        return self._arr.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        return self._arr.shape

    @property
    def dtype(self) -> np.dtype:
        return self._arr.dtype

    @property
    def size(self) -> int:
        return self._arr.size

    @property
    def base(self):
        return self._arr.base

    def copy(self, deep: bool = False) -> ValidatedArray:
        return copy.deepcopy(self) if deep else copy.copy(self)

    @property
    def view(self):
        return self._arr.view(type(self))

class OptionalArray(ValidatedArray):
    _arr: ArrayDescriptor = ArrayDescriptor(ValidatedArray, optional=True)

class ReadOnlyArray(ValidatedArray):
    _arr: ArrayDescriptor = ArrayDescriptor(ValidatedArray, freezable=True, coerce=True)

class _LengthArray(ValidatedArray):
    def __len__(self) -> int:
        return self._arr.shape[0]

class Vector(_LengthArray):
    __ndim__: int = 1
    __shape__: tuple[Optional[int]] = (None,)

    def coerce_array(self, value: np.ndarray) -> np.ndarray:
        value = value.squeeze()
        return super().coerce_array(value)


class Point2d(Vector):
    _arr: ArrayDescriptor = ArrayDescriptor(Vector, default=np.zeros(2).astype(np.float32) )
    __shape__: tuple[int] = (2,)


class Point3d(Vector):
    # TODO should we make the dtype default to float32?
    _arr: ArrayDescriptor = ArrayDescriptor(Vector, default=np.zeros(3).astype(np.float32) )
    __shape__: tuple[int] = (3,)


class Array2d(ValidatedArray):
    __ndim__: int = 2
    __shape__: tuple[Optional[int], Optional[int]] = (None, None)


class ArrayNx2(Array2d, _LengthArray):
    __shape__: tuple[Optional[int], int] = (None, 2)


class ArrayNx3(Array2d, _LengthArray):
    __shape__: tuple[Optional[int], int] = (None, 3)


class TransformMatrix(Array2d):
    _arr: ArrayDescriptor = ArrayDescriptor(Array2d, default=np.eye(4).astype(np.float32) )
    __shape__: tuple[int, int] = (4, 4)
    __dtype__: np.dtype = np.float32



class Array3d(ValidatedArray):
    __ndim__: int = 3
    __shape__: tuple[Optional[int], Optional[int], Optional[int]] = (None, None, None)

