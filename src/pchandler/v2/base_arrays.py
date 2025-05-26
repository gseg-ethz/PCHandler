from __future__ import annotations

from pathlib import Path
import uuid
import functools
from typing import Any, Optional, Literal
from copy import deepcopy
from functools import wraps
from unittest import case

# noinspection PyPackageRequirements
import numpy as np
# noinspection PyPackageRequirements
from numpydantic import NDArray, Shape
# noinspection PyPackageRequirements
from pydantic import BaseModel, ConfigDict, model_validator, field_validator, UUID4, Field

NpMixinT = np.lib.mixins.NDArrayOperatorsMixin

def make_ndarray_type(*args: Optional[int|str], dtype = None):
    """
    Helper function to generate the numpydantic type for a ndarray.

    Calling 'make_ndarray_type(None, 3, dtype=np.float32)' would return a numpydantic dtype corresponding to an array
    of shape (N, 3) with dtype = np.float32 and would provide pydantic validation on this
    """
    if len(args) == 0:
        shape_list = ['*', '...']
    else:
        shape_list: list = [str(x) if x is not None else "*" for x in args]

    return NDArray[Shape[', '.join(shape_list)], dtype if dtype is not None else Any]


Array_T = NDArray[Shape['*, ...'], Any]
Array_NxM_T = NDArray[Shape['*, *'], Any]
Array_Nx2_T = NDArray[Shape['*, 2'], Any]
Array_Nx3_T = NDArray[Shape['*, 3'], Any]
Array_3x3_T = NDArray[Shape['4, 4'], Any]
Array_4x4_T = NDArray[Shape['4, 4'], Any]
Vector_N_T = NDArray[Shape['*'], Any]
Vector_2_T = NDArray[Shape['2'], Any]
Vector_3_T = NDArray[Shape['3'], Any]


# TODO need to find a nice way to get all the relevant right mixins
def output_array_like(enabled=False, as_update=True):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(instance: BaseArray, *args, **kwargs):
            arraylike: bool = isinstance(instance, BaseArray)
            if not arraylike:
                raise TypeError('This decorated function must be attached to a method of BaseArray subclass')

            result = func(instance, *args, **kwargs)

            if not enabled:
                return result

            if not as_update:
                return type(instance)(arr=result)
            return instance.get_copy(update={'arr': result})

        return wrapper
    return decorator


def __get_args_as_arrays__(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        array_args = list(args)
        for i, x in enumerate(args[3:], 3):
            array_args[i] = x.arr if isinstance(x, type(array_args[0])) else np.asarray(x)
        return func(*array_args, **kwargs)
    return wrapper

# TODO find a way to enable positional args without error and type highlighting
class BaseArray(BaseModel, NpMixinT):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        revalidate_instances="always",
        validate_default=True,
        frozen=False,
        extra='ignore')
    arr: Array_T = Field(kw_only=False)
    cache_uuid: UUID4 = Field(default_factory=lambda: uuid.uuid4(), exclude=True)
    offloaded: bool = Field(default=False, exclude=True)


    # noinspection PyNestedDecorators
    @field_validator('arr', mode='before')
    @classmethod
    def coerce(cls, array: list | tuple | np.ndarray) -> Any:
        if isinstance(array, (list, tuple)) or issubclass(type(array), NpMixinT):
            array = np.asarray(array)
        elif not isinstance(array, np.ndarray) and not issubclass(type(array), cls):
            raise TypeError(f'Cannot convert {type(array)} to numpy array')

        if isinstance(cls.model_fields['arr'].annotation, NDArray):

            base_type = cls.model_fields['arr'].annotation.__dict__['__args__'][1]
            if base_type is not Any:
                if np.can_cast(array.dtype, base_type):
                    return array.astype(base_type)
                raise TypeError(f'Cannot convert {array.dtype} to {cls.arr.dtype}')

        return array

    def get_copy(self, update=None, include=None, exclude=None):
        data = self.model_dump(
            include=include,
            exclude=set(set(exclude or {}) | set((update or {}).keys())))
        return self.model_validate({**data, **(deepcopy(update) or {})})

    @model_validator(mode='after')
    def freeze(self):
        if self.model_config['frozen']:
            self.arr.setflags(write=False)
        return self


    @__get_args_as_arrays__
    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        return getattr(ufunc, method)(*args, **kwargs)

    def __array__(self):
        return self.arr

    @property
    def T(self):
        return self.arr.T

    @property
    def __array_interface__(self):
        return self.arr.__array_interface__

    # def __array_function__(self, *args, **kwargs):
    #     pass

    @property
    def shape(self):
        return self.arr.shape

    @property
    def dtype(self):
        return self.arr.dtype

    @property
    def ndim(self):
        return self.arr.ndim

    @property
    def base(self):
        return self.arr.base

    @property
    def size(self):
        return self.arr.size

    def min(self):
        return self.arr.min()

    def max(self):
        return self.arr.max()

    def __len__(self):
        raise NotImplementedError('Length of an undefined array shape is not clear')

    def __getitem__(self, item):
        return self.sample(item)

    def __setitem__(self, key, value):
        self.arr[key] = value

    def __delitem__(self, key):
        np.delete(self.arr, self.create_mask(key))

    def create_mask(self, indices):
        mask = np.zeros_like(self.arr, dtype=np.bool_)
        mask[indices] = True
        return mask

    def sample(self, index):
        return self.get_copy(update={'arr': self.arr[self.create_mask(index)]})

    def reduce(self, index):
        self.arr = self.arr[self.create_mask(index)]

    def extract(self, index):
        extracted = self.sample(index)
        self.reduce(~index)
        return extracted

CustomArrayLikeT = np.ndarray | NpMixinT


class _FixedLengthArray(BaseArray):
    def __len__(self):
        return self.arr.shape[0]

    def create_mask(self, indices):
        mask = np.zeros_like(self, dtype=np.bool_)
        if mask.ndim == 1:
            mask[indices] = True
        else:
            mask[indices, :] = True
        return mask

    @property
    def H(self):
        return np.column_stack((self.arr, np.ones(len(self), dtype=self.dtype)))


class BaseVector(_FixedLengthArray):
    arr: Vector_N_T


class Array2D(BaseArray):
    arr: Array_NxM_T


class ArrayNx2(_FixedLengthArray):
    arr: Array_Nx2_T


class ArrayNx3(_FixedLengthArray):
    arr: Array_Nx3_T


class _TransformMatrixType(BaseArray):


class Array3x3(_TransformMatrixType):
    arr: Array_3x3_T = Field(default_factory=lambda: Array4x4(arr=np.eye(3)))


class Array4x4(_TransformMatrixType):
    arr: Array_4x4_T = Field(default_factory=lambda: Array4x4(arr=np.eye(4)))


class _ReadOnly:
    # noinspection PyNestedDecorators
    @field_validator('arr', mode='before')
    @classmethod
    def coerce(cls, array: np.ndarray) -> np.ndarray:
        return array

    # noinspection PyNestedDecorators
    @field_validator('arr', mode='before')
    @classmethod
    def copy(cls, array: np.ndarray) -> np.ndarray:
        return array


class ReadOnlyArray(_ReadOnly, BaseArray):
    model_config = ConfigDict(strict=True, frozen=True)


class ReadOnlyVector(_ReadOnly, BaseVector):
    model_config = ConfigDict(strict=True, frozen=True)

