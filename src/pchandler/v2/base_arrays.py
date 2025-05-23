from __future__ import annotations

from pathlib import Path
import uuid
import functools
from typing import Any, Optional
from copy import deepcopy
from functools import wraps

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


def force_output_type(enabled=False):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            base_t = type(args[0])

            if not enabled:
                return result

            if isinstance(result, (list, tuple, set)):
                return type(result)(base_t(arr=item)
                                    if issubclass(type(item), np.ndarray|base_t)
                                    else item
                                    for item in result)

            return base_t(arr=result) if issubclass(type(result), np.ndarray|base_t) else result
        return wrapper
    return decorator


def __get_args_as_arrays__(instance, *args, **kwargs):
    arrays = []
    for x in args:
        if isinstance(x, type(instance)):
            arrays.append(x.arr)
        else:
            arrays.append(np.asarray(x))
    return arrays


class BaseArray(BaseModel, NpMixinT):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        revalidate_instances="always",
        validate_default=True,
        frozen=False,
        extra='allow')
    arr: Array_T
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

    # noinspection PyNestedDecorators
    @field_validator('arr', mode='before')
    @classmethod
    def copy(cls, array: Any) -> Any:
        return deepcopy(array)

    @model_validator(mode='after')
    def freeze(self):
        if self.model_config['frozen']:
            self.arr.setflags(write=False)
        return self

    # def __array_function__(self, func, types, *args, **kwargs):
    #     arrays = __get_args_as_arrays__(self, args, **kwargs)
    #     return func(arrays, **kwargs)

    def __array__(self):
        return self.arr

    @property
    def T(self):
        return self.arr.T

    @property
    def __array_interface__(self):
        return self.arr.__array_interface__

    @force_output_type(enabled=False)
    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        arrays = __get_args_as_arrays__(self, *args, **kwargs)
        return getattr(ufunc, method)(*arrays, **kwargs)

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

    def copy_array(self):
        return self.arr.copy()

    def get_view(self):
        return self.arr.view()

    def __len__(self):
        raise NotImplementedError('Length of an undefined array shape is not clear')

    def __getitem__(self, item):
        return self.arr[item]

    def __setitem__(self, key, value):
        self.arr[key] = value

CustomArrayLikeT = np.ndarray | NpMixinT


class LimitedColumnArray(BaseArray):
    def __len__(self):
        return self.arr.shape[0]


class BaseVector(LimitedColumnArray):
    arr: Vector_N_T


class Array2D(BaseArray):
    arr: Array_NxM_T


class ArrayNx2(LimitedColumnArray):
    arr: Array_Nx2_T


class ArrayNx3(LimitedColumnArray):
    arr: Array_Nx3_T


class Array3x3(BaseArray):
    arr: Array_3x3_T = Field(default_factory=lambda: Array4x4(arr=np.eye(3)))


class Array4x4(BaseArray):
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

