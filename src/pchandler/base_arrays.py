from __future__ import annotations

import copy
from abc import ABC
from typing import Any, Generator, MutableMapping, Optional, Self, cast

import numpy as np
import numpy.typing as npt
from numpydantic import NDArray
from pydantic import BaseModel, ConfigDict, field_validator, ValidationError

from .validators import validate_transposed_2d_array, convert_slice_to_integer_range
from .base_types import ArrayT, Array_Nx2_T, Array_Nx3_T,  IndexLike, VectorIndexLike, Vector_IndexT, VectorT


class BaseArray(ABC, BaseModel):
    """
    BaseArray is designed to be a subclassable, automatic validator for array-based classes.
    It is built around a combination of the Pydantic and Numpydantic libraries.

    In line with the PCHandler project, the main idea is that it can be extended to support the following:
        -> Coordinate Classes
        -> Scalar Fields
        -> Transformation Matrices (4x4 Affine and 3x3 matrices -> intrinsic / extrinsic)
        -> Image Arrays
        -> ...

    For now, it supports the following:
        - Boolean, Floating, SignedInteger and UnsignedInteger types
        - Scalar values (although converted to 1D arrays)
        - 1D or greater arrays (all 0D / scalars will be converted to 1D)
    """

    # Config
    model_config = ConfigDict(
        arbitrary_types_allowed=True,   # Required for the numpy types
        validate_assignment=True,       # Should validate anytime an attribute is set
        revalidate_instances="never",   # Don't keep validating instances (avoids infinite validation loops)
        validate_default=True,          # Ensure default values get validated as well
        strict=True,                    # Ensure that no coercion of types occurs - strict typechecking
        frozen=False,                   # Object can be manipulated
        extra="ignore",                 # Extra fields passed are not stored in the object (e.g. kwargs)
        serialize_by_alias=False,       # Serialisation takes the original field names (e.g. 'arr')
        populate_by_name=False,         # Field is not expected to be populated by attribute name if an alias exists
    )

    # Validated Attributes
    arr: ArrayT

    # noinspection PyNestedDecorators
    @field_validator('arr', mode='before')
    @classmethod
    def coerce_array(cls, value: Any) -> ArrayT:
        # This ensures that a copy is not made
        if isinstance(value, BaseArray):
            value = value.arr

        return np.atleast_1d(np.asarray(value))

    # @model_validator(mode="after")
    # def freeze(self) -> Self:
    #     if self.model_config["frozen"]:
    #         self.arr.setflags(write=False)
    #     return self

    @property
    def __array_interface__(self) -> dict[str, Any]:
        """Gives access for all numpy functions to the root array object

        All objects will be converted to numpy arrays when processed with numpy functions.
        - __array__ will be deprecated in the future, making more reason to use this
        E.g. any function will use np.as array(base_arraylike.arr.__array_interface__)
        """
        return cast(dict[str, Any], self.arr.__array_interface__)

    # Numpy like methods/properties
    # noinspection PyPep8Naming
    @property
    def T(self) -> ArrayT:
        return self.arr.T

    @property
    def shape(self) -> tuple[int, ...]:
        return cast(tuple[int, ...], self.arr.shape)

    @property
    def dtype(self) -> npt.DTypeLike:
        return cast(np.dtype, self.arr.dtype)

    @property
    def ndim(self) -> int:
        return cast(int, self.arr.ndim)

    @property
    def base(self) -> ArrayT|None:
        return self.arr.base

    @property
    def size(self) -> int:
        return cast(int, self.arr.size)

    def view(self, dtype: npt.DTypeLike = None, _type: type|None = None) -> ArrayT:
        dtype = self.dtype if dtype is None else dtype
        _type = type(self.arr) if _type is None else _type
        return self.arr.view(dtype=dtype, type=_type)

    def min(self, **kwargs: dict[str, Any]) -> Any:
        return self.arr.min(**kwargs)

    def max(self, **kwargs: dict[str, Any]) -> Any:
        return self.arr.max(**kwargs)

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key: IndexLike) -> ArrayT | Self:
        try:
            return self.copy(array=self.arr[key], deep=False)
        except ValidationError:
            return self.arr[key]

    def __setitem__(self, key: IndexLike, value: ArrayT | BaseArray) -> None:
        self.arr[key] = value.arr if isinstance(value, BaseArray) else value

    # Logical Mixins
    def __lt__(self, other: Any) -> npt.NDArray[np.bool_]:
        return cast(npt.NDArray[np.bool_], self.arr < other)

    def __le__(self, other: Any) -> npt.NDArray[np.bool_]:
        return cast(npt.NDArray[np.bool_], self.arr <= other)

    def __ge__(self, other: Any) -> npt.NDArray[np.bool_]:
        return cast(npt.NDArray[np.bool_], self.arr >= other)

    def __gt__(self, other: Any) -> npt.NDArray[np.bool_]:
        return cast(npt.NDArray[np.bool_], self.arr > other)

    def __eq__(self, other: Any) -> npt.NDArray[np.bool_]: # type: ignore[override]
        return cast(npt.NDArray[np.bool_], self.arr == other)

    def __ne__(self, other: Any) -> npt.NDArray[np.bool_]: # type: ignore[override]
        return cast(npt.NDArray[np.bool_], self.arr != other)

    def copy(self,  # type: ignore
             array: npt.NDArray[Any] | BaseArray | None = None,
             *,
             deep: bool = True,
             update: Optional[MutableMapping[str, Any]] = None,
             **kwargs: dict[str, Any]) -> Self:
        """

        :param array:
            Bypasses any deepcopy
        :param deep:
        :param update:
            Bypasses any deepcopy
        :param kwargs:
        :return:
        """

        update = update or {}

        if array is not None:
            update["arr"] = array

        if "arr" in update:
            if isinstance(update["arr"], BaseArray):
                update["arr"] = update["arr"].arr

        data = self.model_dump(exclude=set(update.keys()), by_alias=False)
        data = copy.deepcopy(data) if deep else data

        data.update(update)

        return type(self)(**data)


class NumericMixins(BaseArray):
    def __add__(self, other: Any) -> Self:
        return self.copy(self.arr + other)

    def __sub__(self, other: Any) -> Self:
        return self.copy(self.arr - other)

    def __mul__(self, other: Any) -> Self:
        return self.copy(self.arr * other)

    def __truediv__(self, other: Any) -> Self:
        return self.copy(self.arr / other)

    def __floordiv__(self, other: Any) -> Self:
        return self.copy(self.arr // other)

    def __mod__(self, other: Any) -> Self:
        return self.copy(self.arr % other)

    def __pow__(self, other: Any) -> Self:
        return self.copy(self.arr**other)

    def __radd__(self, other: Any) -> Self:
        return self.copy(other + self.arr)

    def __rsub__(self, other: Any) -> Self:
        return self.copy(other - self.arr)

    def __rmul__(self, other: Any) -> Self:
        return self.copy(other * self.arr)

    def __rpow__(self, other: Any) -> Self:
        return self.copy(other**self.arr)

    def __rtruediv__(self, other: Any) -> Self:
        return self.copy(other / self.arr)

    def __rfloordiv__(self, other: Any) -> Self:
        return self.copy(other // self.arr)

    def __rmod__(self, other: Any) -> Self:
        return self.copy(other % self.arr)

    def __divmod__(self, other: Any) -> Self:
        return self.copy(divmod(self.arr, other))

    def __neg__(self, other: Any) -> Self:
        return self.copy(-self.arr)

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
        self.arr = self.arr**other
        return self


class FixedLengthArray(NumericMixins):
    """
    Array to support objects like Coordinate sets or vectors which have "len()" which maps to their number of items
    """

    def __iter__(self) -> Generator[tuple[str, Any], None, None]:
        for i in self.arr:
            yield i

    def create_mask(self, selection: VectorIndexLike) -> NDArray[np.bool_] | NDArray[np.int_]:
        """Creates a boolean vector mask that corresponds to row indices"""

        if isinstance(selection, slice):    # Case 1: slice object
            vector_mask = convert_slice_to_integer_range(selection=selection, length=len(self))

        elif isinstance(selection, int):    # Case 2: single integer
            vector_mask = np.array([selection])

        else:                               # Case 3: numpy arrays and sequences
            if isinstance(selection, np.ndarray):
                selection = np.atleast_1d(selection.squeeze())
            vector_mask = Vector_IndexT(selection)

        if vector_mask.dtype == np.bool_:     # Case 3a: Boolean
            if vector_mask.shape[0] != len(self):
                raise ValueError(f"Mask has wrong number of points. Mask:{vector_mask.size}  != array:{len(self)}")
            return vector_mask

        else:                               # Case 3b: Integer
            # TODO throw a warning for attempts to oversample (multiple integers the same)
            mask = np.zeros(len(self), dtype=np.bool_)
            mask[vector_mask] = True
            return mask

    def sample(self, index: IndexLike) -> Self:
        """Return a sample of the array"""
        mask = self.create_mask(index)
        return self.copy(self.arr[mask], deep=True)

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


class BaseVector(FixedLengthArray):
    arr: VectorT

    # noinspection PyNestedDecorators
    @field_validator('arr', mode='before')
    @classmethod
    def coerce_array(cls, value: ArrayT | Self) -> VectorT:
        value = super(BaseVector, cls).coerce_array(value)
        return np.atleast_1d(value.squeeze())


class HomogeneousArray(FixedLengthArray):
    # noinspection PyPep8Naming
    @property
    def H(self) -> ArrayT:
        return np.column_stack((self.arr, np.ones(len(self), dtype=self.dtype)))


class ArrayNx2(HomogeneousArray):
    arr: Array_Nx2_T

    # noinspection PyNestedDecorators
    @field_validator('arr', mode='plain')
    @classmethod
    def coerce_array(cls, value: ArrayT) -> Array_Nx2_T:
        value = super(ArrayNx2, cls).coerce_array(value)
        return validate_transposed_2d_array(value, 2)


class ArrayNx3(HomogeneousArray):
    arr: Array_Nx3_T

    # noinspection PyNestedDecorators
    @field_validator('arr', mode='plain')
    @classmethod
    def coerce_array(cls, value: ArrayT) -> Array_Nx3_T:
        value = super(ArrayNx3, cls).coerce_array(value)
        return validate_transposed_2d_array(value, 3)

# TODO support for frozen classes
# class ReadOnlyArray(BaseArray):
#     model_config = ConfigDict(strict=True, frozen=True)
#
#
# class ReadOnlyVector(BaseVector):
#     model_config = ConfigDict(strict=True, frozen=True)
#
#
# TODO support for image like arrays? Should be done elsewhere maybe
# class _ImageLike(SampleArray, _NumericMixins, ABC):
#     arr: Array_NxM_T | Array_NxM_3_T
#
#     # Update implementation based on if you want to support slicing / views or not
#     def __getitem__(self, *key: IndexLike) -> Any:
#         return self.arr[*key]
#
#     def create_mask(self, *indices: int|slice):
#         if isinstance(indices, slice):
#             mask = indices
#         else:
#             mask = np.zeros_like(self.arr, dtype=np.bool_)
#             mask[*indices] = True
#         return mask
#
#     def view(self, cls: Optional[type] = None) -> Self:
#         return self.copy(self.arr.view(cls=cls), deep=False)

