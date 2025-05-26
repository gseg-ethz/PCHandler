from __future__ import annotations

import uuid
import functools
from typing import Any, Optional, Callable
from copy import deepcopy


import numpy as np
from numpydantic import NDArray, Shape
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

# Validation types for Pydantic
Array_T = NDArray[Shape['*, ...'], Any]
Array_NxM_T = NDArray[Shape['*, *'], Any]
Array_Nx2_T = NDArray[Shape['*, 2'], Any]
Array_Nx3_T = NDArray[Shape['*, 3'], Any]
Array_3x3_T = NDArray[Shape['4, 4'], Any]
Array_4x4_T = NDArray[Shape['4, 4'], Any]
Vector_N_T = NDArray[Shape['*'], Any]
Vector_2_T = NDArray[Shape['2'], Any]
Vector_3_T = NDArray[Shape['3'], Any]

CustomArrayLikeT = np.ndarray | NpMixinT

def override_ufuncs(method_map: dict[str, Callable]):
    def class_decorator(cls):
        for method_name, func in method_map.items():
            setattr(cls, method_name, func)
        return cls
    return class_decorator


def __get_args_as_arrays__(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        array_args = list(args)

        for i, x in enumerate(args[3:], 3):
            array_args[i] = x.arr if isinstance(x, type(array_args[0])) else np.asarray(x)

        return func(*array_args, **kwargs)
    return wrapper


class BaseArray(BaseModel, NpMixinT):
    """
    BaseArray is designed to be a subclassable, automatic validator for array based classes.

    In line with the PCHandler project, the main idea is that it can be extended to support the following:
        -> Coordinate Classes
        -> Scalar Fields
        -> Transformation Matrices (4x4 Affine and 3x3 Rotation (for example))
        -> Scalar Fields

    The shape of each should be clearly validated.
    The dtype should also be able to be validated.
    The objects should support additional attribute information.
    Arrays can also be lazy loaded.

    These should also be easily sampled from

    """
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

    @model_validator(mode='after')
    def freeze(self):
        if self.model_config['frozen']:
            self.arr.setflags(write=False)
        return self

    @property
    def __array_interface__(self):
        return self.arr.__array_interface__

    @__get_args_as_arrays__
    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        # TODO best way to apply functions in place for 'out'
        if 'out' in kwargs:
            output_target = kwargs.pop('out')[0]

            if isinstance(output_target, type(self)):
                output_target.arr[:] = getattr(ufunc, method)(*args, **kwargs)
            else:
                output_target[:] = getattr(ufunc, method)(*args, **kwargs)

            return output_target
        return getattr(ufunc, method)(*args, **kwargs)

    def get_copy(self, array=None, *, update=None, include=None, exclude=None):
        """If no parameters, copy the whole array as is"""
        update = update or {}

        if array is not None:
            # This will override any 'update' parameter
            array = array.reshape(-1, 3) if array.size == 3 else array
            update |= {'arr': array}

        data = self.model_dump(
            include=include,
            exclude=set(set(exclude or {}) | set((update or {}).keys())))

        return self.model_validate(deepcopy({**data, **update}))

    @property
    def T(self):
        return self.arr.T

    # TODO decide on helper name
    @property
    def H(self):
        return self.get_copy(
            np.column_stack((self.arr, np.ones(len(self), dtype=self.dtype)))
        )

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

    # TODO overwrite these for coordinate_sets
    def __getitem__(self, item):
        return self.arr[*item]

    def __setitem__(self, key, value):
        self.arr[key] = value

    def __delitem__(self, key):
        np.delete(self.arr, self.create_mask(key))

    def create_mask(self, *indices):
        mask = np.zeros_like(self.arr, dtype=np.bool_)
        mask[*indices] = True
        return mask

    def sample(self, *index, sub_ok=True):
        mask = self.create_mask(*index)

        if sub_ok:
            return self.get_copy(self.arr[mask])

        return self.arr[mask].copy()

    def reduce(self, *index):
        self.arr = self.arr[self.create_mask(*index)]

    def extract(self, index):
        mask = self.create_mask(*index)
        extracted = self.sample(mask)
        self.reduce(~mask)
        return extracted


class _FixedLengthArray(BaseArray):
    def __len__(self):
        return self.arr.shape[0]

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


class _TransformArray(BaseArray):
    def __matmul__(self, other):
        if isinstance(other, ArrayNx3):
            return other.__rmatmul__(self)
        if isinstance(other, type(self)):
            return self.get_copy(array=self.__matmul__(other))
        else:
            return self.__matmul__(other)



class RotationArray(_TransformArray):
    arr: Array_3x3_T = Field(default_factory=lambda: RotationArray(arr=np.eye(3)))


class AffineArray(_TransformArray):
    arr: Array_4x4_T = Field(default_factory=lambda: AffineArray(arr=np.eye(4)))


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

