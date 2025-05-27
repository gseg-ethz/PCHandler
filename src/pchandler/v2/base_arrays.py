from __future__ import annotations

from functools import cached_property
from typing import Any, Optional, TypedDict, Unpack

import numpy as np
from numpydantic import NDArray, Shape
from pydantic import BaseModel, ConfigDict, model_validator, field_validator, UUID4, Field

# Base types for Pydantic validation
Array_T = NDArray[Shape['*, ...'], Any]         # Any array like object. E.g. image stacks (
Array_NxM_T = NDArray[Shape['*, *'], Any]       # Intensity image, depth image
Array_NxM_3_T = NDArray[Shape['*, *, 3'], Any]    # RGB or normal image
Array_Nx2_T = NDArray[Shape['*, 2'], Any]       # Image coordinates or plane coordinates
Array_Nx3_T = NDArray[Shape['*, 3'], Any]       # 3D Coordinates / Scalar Field triplets (RGB, normals)
Array_3x3_T = NDArray[Shape['4, 4'], Any]       # Rotation Matrix
Array_4x4_T = NDArray[Shape['4, 4'], Any]       # Affine Transformation
Vector_N_T = NDArray[Shape['*'], Any]           # Scalar Fields
Vector_2_T = NDArray[Shape['2'], Any]           # Image coordinate / translation vector
Vector_3_T = NDArray[Shape['3'], Any]           # 3D coordinate / translation vector


# TODO define the scope -> Do we want an all functioning array class or just define the critical ones
#  Scalar Fields
#  Coordinates
#  Transformations
#  Images (depth / RGB / stacks)
#  This will then define if Mixins to be included at base or not
#  Gut feeling is to include default mixins but always return numpy array
#   -> This is based on the idea that other 'subclassed' or container instances will be included in these funcs.
class BaseArray(BaseModel):
    """
    BaseArray is designed to be a subclassable, automatic validator for array based classes.

    In line with the PCHandler project, the main idea is that it can be extended to support the following:
        -> Coordinate Classes
        -> Scalar Fields
        -> Transformation Matrices (4x4 Affine and 3x3 matrices -> intrinsic / extrinsic)

    The shape of each should be validated any time an object or attribute is changed.
    The dtype should also be able to be validated (e.g. RGB or Optimised Coordinates).
    Objects can also be easily extended to support other attribute information (e.g. lazy loading, image metadata).
    Objects can also be frozen absolutely -> e.g. a loaded read-only array's coordinates "never" change"
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        revalidate_instances="always",
        validate_default=True,
        frozen=False,
        extra='ignore')
    arr: Array_T

    @model_validator(mode='after')
    def freeze(self):
        if self.model_config['frozen']:
            self.arr.setflags(write=False)
        return self

    # TODO THIS MUST STAY - __array__ is deprecated
    @property
    def __array_interface__(self) -> dict:
        """
        Gives access for all numpy functions to the root array object
        """
        return self.arr.__array_interface__

    def __array_wrap__(self, out_arr, context=None, return_scalar=False):
        """
        This function provides access to the returned object from numpy class methods / functions

        for example np.add(a, b) returns a numpy array but can be caught here to return always as type(self).
        """
        return super().__array_wrap__(self, out_arr, context, return_scalar)

    def __array_finalize__(self, obj):
        """
        Finalise the createion process of a new instance (after __new__)
        """
        return super().__array_finalize__()

    def update_copy(self, array: np.array|BaseArray|None = None, *args, deep: bool = False, **kwargs) -> BaseArray:
        """
        This function is designed to be more efficient by not dumping the memory heavy array if it's to be updated in
        the new instance.
        E.g. if 'arr' is in the update dict {'arr': np.random.rand(10000, 3)}, don't dump the existing, just add this
        new value.
        """
        update = kwargs.get('update', {})
        include = set(kwargs.get('include', []))
        exclude = set(kwargs.get('exclude', [])) | {'T'}

        update = update or {}
        # NOTE array positional arg takes precedence over update dict
        # TODO decide if extra logic required

        if array is not None:
            update['arr'] = array.arr if isinstance(array, BaseArray) else array

        return self.copy(deep=deep, include=include, exclude=exclude, update=update)

    # TODO discuss -> default is ALWAYS a copy to avoid references and views -> Particularly with global shift.
    #  Views for point cloud data are of little benefit unless 'slicing'
    #  There may be more benefit on image operations
    def copy(self, *args, deep=True, **kwargs):
        """
        Produce a deep or shallow copy of the model. If no 'update' parameters passed then copy the whole model.
        """
        update = kwargs.get('update', None)
        if update is not None:
            new_model = self.model_copy(update=update, deep=deep)
            new_model.model_validate(new_model, strict=True)
            return new_model
        return self.model_copy(deep=deep, *args, **kwargs)

    def view(self, cls: Optional[type] = None):
        raise NotImplementedError

    @cached_property
    def T(self): return self.arr.T

    @property
    def shape(self): return self.arr.shape

    @property
    def dtype(self): return self.arr.dtype

    @property
    def ndim(self): return self.arr.ndim

    @property
    def base(self): return self.arr.base

    @property
    def size(self): return self.arr.size

    def min(self): return self.arr.min()

    def max(self): return self.arr.max()

    def __len__(self):
        raise NotImplementedError('Length of an undefined array shape is not clear')

    # TODO overwrite these for coordinate_sets -> This could become abstract as well as others to force this object not
    #  to be usable
    def __getitem__(self, item):
        return self.arr[*item]

    def __setitem__(self, key, value):
        self.arr[key] = value

    def __delitem__(self, key):
        np.delete(self.arr, self.create_mask(key))

    def create_mask(self, *indices):
        if isinstance(indices, slice):
            mask = indices
        else:
            mask = np.zeros_like(self.arr, dtype=np.bool_)
            mask[*indices] = True
        return mask

    def sample(self, *index, as_ndarray=False):
        """Sample produces a copy of the sampled points"""
        mask = self.create_mask(*index)
        if as_ndarray:
            return self.arr[mask].copy()
        return self.updated_copy(array=self.arr[mask])

    def reduce(self, *index):
        """Reduces the array to the points indexed"""
        self.arr = self.arr[self.create_mask(*index)]

    def extract(self, index):
        """Returns the points indexed but also reduces the indexed array by these points"""
        mask = self.create_mask(*index)
        extracted = self.sample(mask)
        self.reduce(~mask)
        return extracted


class _FixedLengthArray(BaseArray):
    def __len__(self):
        return self.arr.shape[0]


class _HomogeneousArray(BaseArray):
    @property
    def H(self):
        return np.column_stack((self.arr, np.ones(len(self), dtype=self.dtype)))


class BaseVector(_FixedLengthArray):
    arr: Vector_N_T


class Array2D(BaseArray):
    arr: Array_NxM_T


class ArrayNx2(_FixedLengthArray, _HomogeneousArray):
    arr: Array_Nx2_T


class ArrayNx3(_FixedLengthArray, _HomogeneousArray):
    arr: Array_Nx3_T


class _TransformArray(BaseArray):
    def __matmul__(self, other):
        # This is transforming the other object. Therefore use it's __rmatmul__ to enable adding the transform to ledger
        if isinstance(other, (ArrayNx3, ArrayNx2)):
            return other.__rmatmul__(self)

        if isinstance(other, type(self)):
            # DISCUSS do transforms need
            return self.get_copy(array=self.__matmul__(other))

        return self.__matmul__(other)


class Transform3x3(_TransformArray):
    arr: Array_3x3_T = Field(default_factory=lambda: Transform3x3(arr=np.eye(3)))


class Transform4x4(_TransformArray):
    arr: Array_4x4_T = Field(default_factory=lambda: Transform4x4(arr=np.eye(4)))


# TODO decide if to keep these or add more
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
