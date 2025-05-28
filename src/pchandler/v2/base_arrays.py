from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, Optional, Generator, Mapping, Self, Union

import numpy as np
import numpy.typing as npt
from numpydantic import NDArray, Shape
from numpydantic.dtype import Integer, Float, Bool
from pydantic import BaseModel, ConfigDict, model_validator, field_validator, Field

from .custom_types import IndexLike

ArrayDtypes = (Integer, Float, Bool)

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

# DECISION -> Should we shift this to custom types? For now this is a rather independent module
Array_T = NDArray[Shape['*, ...'], ArrayDtypes]         # Any array like object. E.g. image stacks (
Array_NxM_T = NDArray[Shape['*, *'], ArrayDtypes]       # Intensity image, depth image
Array_NxM_3_T = NDArray[Shape['*, *, 3'], ArrayDtypes]  # RGB or normal image
Array_Nx2_T = NDArray[Shape['*, 2'], ArrayDtypes]       # Image coordinates or plane coordinates
Array_Nx3_T = NDArray[Shape['*, 3'], ArrayDtypes]       # 3D Coordinates / Scalar Field triplets (RGB, normals)
Array_3x3_T = NDArray[Shape['4, 4'], ArrayDtypes]       # Rotation Matrix
Array_4x4_T = NDArray[Shape['4, 4'], ArrayDtypes]       # Affine Transformation
Vector_N_T = NDArray[Shape['*'], ArrayDtypes]           # Scalar Fields
Vector_2_T = NDArray[Shape['2'], ArrayDtypes]           # Image coordinate / translation vector
Vector_3_T = NDArray[Shape['3'], ArrayDtypes]           # 3D coordinate / translation vector


# TODO define the scope -> Do we want an all functioning array class or just define the critical ones
#  Scalar Fields
#  Coordinates
#  Transformations
#  Images (depth / RGB / stacks)
#  This will then define if Mixins to be included at base or not
#  Gut feeling is to include default mixins but always return numpy array
#   -> This is based on the idea that other 'subclassed' or container instances will be included in these funcs.
class BaseArray(ABC, BaseModel):
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
        strict=True,
        frozen=False,
        extra='ignore')
    arr: Array_T

    # TODO performance test the difference -> if any
    @property
    def __array_interface__(self) -> dict:
        """ Gives access for all numpy functions to the root array object

        All objects will be converted to numpy arrays when processed with numpy functions.
        - __array__ will be deprecated in future -> more reason to use this
        E.g. any function will use np.asarray(base_arraylike.arr.__array_interface__)
        """
        return self.arr.__array_interface__

    def __array__(self) -> Array_T:
        """This is a backup for __array_interface__"""
        return self.arr

    @cached_property
    def T(self) -> npt.NDArray:
        return self.arr.T

    @property
    def shape(self) -> tuple[int, ...]:
        return self.arr.shape

    @property
    def dtype(self) -> npt.DTypeLike:
        return self.arr.dtype

    @property
    def ndim(self) -> int:
        return self.arr.ndim

    @property
    def base(self) -> npt.NDArray|None:
        return self.arr.base

    @property
    def size(self) -> int:
        return self.arr.size

    def min(self, **kwargs) -> np.number | npt.NDArray:
        return self.arr.min(**kwargs)

    def max(self, **kwargs) -> np.number | npt.NDArray:
        return self.arr.max(**kwargs)

    # To better catch any attempts to coerce as strict seems to fail
    @field_validator('arr', mode='before')
    @classmethod
    def validate_ndarray(cls, arr):
        if not isinstance(arr, np.ndarray|type(cls)):
            raise TypeError(f'Invalid array type: {type(arr)}')

        return arr

    @model_validator(mode='after')
    def freeze(self) -> Self:
        if self.model_config['frozen']:
            self.arr.setflags(write=False)
        return self

    def update_copy(self,
                    array: npt.NDArray|BaseArray|None = None, *,
                    deep: bool = True,
                    update: Mapping[str, Any] = None) -> Self:
        """
        This function is designed to be more efficient by not dumping the memory heavy array if it's to be updated in
        the new instance.
        E.g. if 'arr' is in the update dict {'arr': np.random.rand(10000, 3)}, don't dump the existing, just add this
        new value.
        """
        update = update or {}

        if array is not None:
            update['arr'] = array.arr if isinstance(array, BaseArray) else array

        return self.copy(deep=deep, update=update)

    def copy(self, *, deep: bool = True, **kwargs) -> Self:
        """
        Produce a deep or shallow copy of the model. Updates the model also if parameter is parsed.
        """
        if not deep:
            raise NotImplementedError(f'Shallow copy is not implemented on this class: {type(self)}')

        update = kwargs.get('update', {})

        result = self.model_copy(deep=deep, update=update)

        # Delete excluded fields on the copy
        for name, field_info in result.model_fields.items():
            if field_info.exclude:
                delattr(result, name)

        return result.model_validate(result, strict=True)

    def view(self, cls: Optional[type] = None) -> Self:
        # This is a placeholder for the ability to subclass an array to act like a view
        raise NotImplementedError


    def __len__(self) -> int:
        raise NotImplementedError('Length of an undefined array shape is not clear')

    def __getitem__(self, key: IndexLike) -> np.ndarray|Self:
        if isinstance(key, slice):
            key = [key]

        if isinstance(key, int):
            result = self.arr[key]
        else:
            result = self.arr[*key]
        if isinstance(result, np.ndarray):
            return self.update_copy(result)
        return result

    def __setitem__(self, key: IndexLike, value: npt.NDArray|BaseArray) -> None:
        if isinstance(key, slice):
            key = [key]
        if isinstance(value, BaseArray):
            self.arr[*key] = value.arr
        else:
            self.arr[*key] = value


class _SampleArray(BaseArray):
    def create_mask(self, selection: IndexLike) -> Vector_N_T:
        """Creates a boolean mask for the whole array

        This ensures all new objects are a copy of an array and no views/references
        """
        if isinstance(selection, np.ndarray) and selection.dtype == np.bool_:
            return selection

        mask = np.zeros_like(self.arr, dtype=np.bool_)
        if isinstance(selection, list):
            mask[np.array(selection)] = True

        elif isinstance(selection, np.ndarray):
            mask[selection] = True

        elif not isinstance(selection, (tuple, slice, int)):
            raise TypeError(f"Unsupported selection type: {type(selection)}. Must be slice, list, or np.ndarray.")
        else:
            mask[selection] = True
        return mask

    def sample(self, index: IndexLike) -> Self:
        """Return a sample of the array"""
        mask = self.create_mask(index)
        return self.update_copy( array=self.arr[mask] )

    def reduce(self, index: IndexLike) -> None:
        """Reduces the array to the points indexed"""
        mask = self.create_mask(index)
        self.arr = self.arr[mask]

    def extract(self, index: IndexLike) -> Self:
        """Returns the points indexed but also reduces the indexed array by these points"""
        mask = self.create_mask(index)
        extracted = self.sample(mask)
        self.reduce(~mask)
        return extracted


class _FixedLengthArray(_SampleArray):
    """
    Array to support objects like Coordinate sets or vectors which have "len()" or number of items == rows
    """
    def __len__(self) -> int:
        return self.arr.shape[0]

    def __iter__(self) -> Generator[np.ndarray]:
        """Function to allow iteration through vector items or rows of array"""
        for i in self.arr:
            yield i

    def create_mask(self, selection: IndexLike) -> Vector_N_T:

        if isinstance(selection, np.ndarray) and selection.dtype == np.bool_:
            if selection.shape != (self.nbPoints,):
                raise ValueError(f"Boolean mask must have shape ({self.nbPoints},), but got {selection.shape}")
            return selection

        mask = np.zeros(len(self), dtype=np.bool_)

        if isinstance(selection, list):
            mask[np.array(selection)] = True

        elif isinstance(selection, np.ndarray):
            mask[selection] = True

        elif not isinstance(selection, (tuple, slice, int)):
            raise TypeError(f"Unsupported selection type: {type(selection)}. Must be slice, list, or np.ndarray.")
        else:
            mask[selection] = True

        if mask.ndim != 1:
            raise ValueError(f'Input selection must correspond to a selection of rows in the array (1D).')

        return mask

    def sample(self, index: IndexLike) -> _FixedLengthArray:
        mask = self.create_mask(index)
        return self.update_copy( array=self.arr[mask] if self.ndim == 1 else self.arr[mask, :] )

    def reduce(self, index: IndexLike) -> None:
        """Reduces the array to the points indexed"""
        mask = self.create_mask(index)
        self.arr = self.arr[mask] if self.ndim == 1 else self.arr[mask, :]

    def extract(self, index: IndexLike) -> Self:
        """Returns the points indexed but also reduces the indexed array by these points"""
        mask = self.create_mask(index)
        extracted = self.sample(mask)
        self.reduce(~mask)
        return extracted

    def __add__(self, other):
        return self.update_copy(self.arr + other)

    def __radd__(self, other):
        return self.update_copy(other + self.arr)

    def __iadd__(self, other):
        self.arr += other
        return self

    def __sub__(self, other):
        return self.update_copy(self.arr - other)

    def __rsub__(self, other):
        return self.update_copy(other - self.arr)

    def __isub__(self, other):
        self.arr -= other
        return self

    def __mul__(self, other):
        return self.update_copy(self.arr * other)

    def __rmul__(self, other):
        return self.update_copy(other * self.arr)

    def __imul__(self, other):
        self.arr *= other
        return self

    def __truediv__(self, other):
        return self.update_copy(self.arr / other)

    def __rtruediv__(self, other):
        return self.update_copy(other / self.arr )

    def __itruediv__(self, other):
        self.arr /= other
        return self


class BaseVector(_FixedLengthArray):
    arr: Vector_N_T


class _HomogeneousArray(_FixedLengthArray):
    @property
    def H(self) -> np.ndarray:
        return np.column_stack((self.arr, np.ones(len(self), dtype=self.dtype)))


class ArrayNx2(_HomogeneousArray):
    arr: Array_Nx2_T


class ArrayNx3(_HomogeneousArray):
    arr: Array_Nx3_T


class ReadOnlyArray(BaseArray):
    model_config = ConfigDict(strict=True, frozen=True)


class ReadOnlyVector(BaseVector):
    model_config = ConfigDict(strict=True, frozen=True)



class _ImageLike(_SampleArray, ABC):
    arr: Array_NxM_T|Array_NxM_3_T
    # Update implementation based on if you want to support slicing / views or not
    def __getitem__(self, key: IndexLike) -> Any:
        return self.arr[*key]

    def create_mask(self, *indices):
        if isinstance(indices, slice):
            mask = indices
        else:
            mask = np.zeros_like(self.arr, dtype=np.bool_)
            mask[*indices] = True
        return mask

    def view(self, cls: Optional[type] = None) -> Self:
        return self.updated_copy(self.arr.view(cls=cls), deep=False)