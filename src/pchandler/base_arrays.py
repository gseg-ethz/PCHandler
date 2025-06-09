from __future__ import annotations

from abc import ABC
from functools import cached_property
from typing import Any, Optional, Generator, Mapping, Self

import numpy as np
import numpy.typing as npt
from numpydantic import NDArray, Shape
from pydantic import BaseModel, ConfigDict, model_validator, field_validator


from .base_types import ArrayT, VectorT, Array_NxM_T, Array_Nx3_T, Array_Nx2_T, Array_NxM_3_T
from .base_types import IndexLike


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
        extra='ignore',
        populate_by_name=True)
    arr: ArrayT

    @property
    def __array_interface__(self) -> dict:
        """ Gives access for all numpy functions to the root array object

        All objects will be converted to numpy arrays when processed with numpy functions.
        - __array__ will be deprecated in future -> more reason to use this
        E.g. any function will use np.asarray(base_arraylike.arr.__array_interface__)
        """
        return self.arr.__array_interface__

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

    @field_validator('arr', mode='before')
    @classmethod
    def validate_ndarray(cls, arr: np.ndarray):

        if not isinstance(arr, np.ndarray|type(cls)):
            raise TypeError(f'Invalid array type: {type(arr)}')

        return arr

    @model_validator(mode='after')
    def freeze(self) -> Self:
        if self.model_config['frozen']:
            self.arr.setflags(write=False)
        return self

    def model_dump(self, *args, **kwargs) -> dict:
        kwargs = kwargs | {'exclude': {'spher'}}
        return super().model_dump(*args, **kwargs)

    # TODO need to change this process to dump and then build new. This ensures cached properties are not copied
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

    def __lt__(self, other: Any) -> npt.NDArray[np.bool_]|bool: return self.arr < other
    def __le__(self, other: Any) -> npt.NDArray[np.bool_]|bool: return self.arr <= other
    def __eq__(self, other: Any) -> npt.NDArray[np.bool_]|bool: return self.arr == other
    def __ne__(self, other: Any) -> npt.NDArray[np.bool_]|bool: return self.arr != other
    def __ge__(self, other: Any) -> npt.NDArray[np.bool_]|bool: return self.arr >= other
    def __gt__(self, other: Any) -> npt.NDArray[np.bool_]|bool: return self.arr > other


class SampleArray(BaseArray):
    def create_mask(self, selection: IndexLike, as_vector=False) -> NDArray[np.bool_]|NDArray[np.int_]:
        """Creates a boolean mask for the whole array

        This ensures all new objects are a copy of an array and no views/references
        """
        if isinstance(selection, np.ndarray) and selection.dtype == np.bool_:
            selection = selection.squeeze()
            if as_vector:
                if selection.ndim > 1:
                    raise ValueError(f'Selection mask must be a vector like')
                return selection

            if selection.shape == self.shape:
                return selection
            else:
                raise ValueError(f'Invalid selection mask shape: {selection.shape}')

        if as_vector:
            mask = np.zeros(len(self), dtype=np.bool_)
        else:
            mask = np.zeros_like(self.arr, dtype=np.bool_)

        if isinstance(selection, list):
            selection = np.array(selection)

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


class _NumericMixins(BaseArray):
    def __add__(self, other: Any) -> Self: return self.update_copy(self.arr + other)
    def __sub__(self, other: Any) -> Self: return self.update_copy(self.arr - other)
    def __mul__(self, other: Any) -> Self: return self.update_copy(self.arr * other)
    def __truediv__(self, other: Any) -> Self: return self.update_copy(self.arr / other)
    def __floordiv__(self, other: Any) -> Self: return self.update_copy(self.arr // other)
    def __mod__(self, other: Any) -> Self: return self.update_copy(self.arr % other)
    def __pow__(self, other: Any) -> Self: return self.update_copy(self.arr ** other)
    def __radd__(self, other: Any) -> Self: return self.update_copy(other + self.arr)
    def __rsub__(self, other: Any) -> Self: return self.update_copy(other - self.arr)
    def __rmul__(self, other: Any) -> Self: return self.update_copy(other * self.arr)
    def __rpow__(self, other: Any) -> Self: return self.update_copy(other ** self.arr)
    def __rtruediv__(self, other: Any) -> Self: return self.update_copy(other / self.arr )
    def __rfloordiv__(self, other: Any) -> Self: return self.update_copy(other // self.arr)
    def __rmod__(self, other: Any) -> Self: return self.update_copy(other % self.arr)
    def __divmod__(self, other: Any) -> Self: return self.update_copy(divmod(self.arr, other))
    def __neg__(self, other: Any) -> Self: return self.update_copy(-self.arr)

    def __iadd__(self, other: Any) -> Self:
        self.arr = self.arr + other
        return self

    def __isub__(self, other: Any) -> Self:
        self.arr = self.arr - other
        return self

    def __imul__(self, other: Any) -> Self:
        self.arr = self.arr * other
        return self

    def __itruediv__(self, other: Any) -> Self:
        self.arr = self.arr / other
        return self

    def __ifloordiv__(self, other: Any) -> Self:
        self.arr = self.arr // other
        return self

    def __imod__(self, other: Any) -> Self:
        self.arr = self.arr % other
        return self

    def __ipow__(self, other: Any) -> Self:
        self.arr = self.arr ** other
        return self


class FixedLengthArray(SampleArray, _NumericMixins):
    """
    Array to support objects like Coordinate sets or vectors which have "len()" or number of items == rows
    """
    def __len__(self) -> int:
        return self.arr.shape[0]

    def __iter__(self) -> Generator[np.ndarray]:
        for i in self.arr:
            yield i

    def create_mask(self, selection: IndexLike, **kwargs) -> NDArray[np.bool_]|NDArray[np.int_]:
        return super().create_mask(selection, as_vector=True)

    def sample(self, index: IndexLike) -> FixedLengthArray:
        mask = self.create_mask(index)
        return self.update_copy(
            array = self.arr[mask] if self.shape == mask.shape else self.arr[mask, :]
        )

    def reduce(self, index: IndexLike) -> None:
        """Reduces the array to the points indexed"""
        mask = self.create_mask(index)
        self.arr = self.arr[mask] if self.shape == mask.shape else self.arr[mask, :]

    def extract(self, index: IndexLike) -> Self:
        """Returns the points indexed but also reduces the indexed array by these points"""
        mask: np.ndarray[np.bool_]| np.ndarray[np.integer] = self.create_mask(index)
        extracted = self.sample(mask)
        self.reduce(~mask)
        return extracted


class BaseVector(FixedLengthArray):
    arr: VectorT


class HomoegeneousArray(FixedLengthArray):
    @property
    def H(self) -> np.ndarray:
        return np.column_stack((self.arr, np.ones(len(self), dtype=self.dtype)))


class ArrayNx2(HomoegeneousArray):
    arr: Array_Nx2_T


class ArrayNx3(HomoegeneousArray):
    arr: Array_Nx3_T


class ReadOnlyArray(BaseArray):
    model_config = ConfigDict(strict=True, frozen=True)


class ReadOnlyVector(BaseVector):
    model_config = ConfigDict(strict=True, frozen=True)


class _ImageLike(SampleArray, _NumericMixins, ABC):
    arr: Array_NxM_T|Array_NxM_3_T
    # Update implementation based on if you want to support slicing / views or not
    def __getitem__(self, *key: IndexLike) -> Any:
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


def unpack_npydantic_dtype(cls: type[BaseArray]) -> tuple[npt.DTypeLike, ...]:
    a = cls.model_fields['arr'].annotation.__dict__['__args__'][1]
    all_types = []

    try:
        for dt in a:
            if isinstance(dt, tuple):
                for dt_ in dt:
                    all_types.append(dt_)
            else:
                all_types.append(dt)

    # TODO write tests for this and get the appropriate Exception to catch
    except Exception as e:
        all_types.append(a)
    return tuple(all_types)