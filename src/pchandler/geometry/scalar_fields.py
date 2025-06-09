from __future__ import annotations

import logging
from typing import Annotated, NamedTuple, Self, Any, TypeVar

import numpy as np
from numpy.typing import DTypeLike, NDArray
from pydantic import StringConstraints, model_validator, Field

from ..base_arrays import BaseVector
from ..base_types import (Uint8VectorT, Float32VectorT, Array_Nx3_uint8_T, Array_Nx3_float32_T, VectorT,
                          Uint16VectorT, Int16VectorT, Int32VectorT, Int8VectorT)
from ..validators import linear_map_dtype, extract_array, normalize_array
from ..constants import RGB_FIELD, NORMALS_FIELD


logger = logging.getLogger(__name__.split(".")[0])


LowerStr = Annotated[str, StringConstraints(strip_whitespace=True, to_lower=True),]



class DtypeState(NamedTuple):
    dtype: DTypeLike
    lower: NDArray[np.number]|float|int|None
    upper: NDArray[np.number]|float|int|None

    @classmethod
    def generate(cls, array: np.ndarray):
        return DtypeState(
            dtype=array.dtype,
            lower=array.min(),
            upper=array.max()
        )

class ScalarField(BaseVector):
    name: LowerStr
    origin_dtype: DtypeState | None = None

    @model_validator(mode='before')
    @classmethod
    def initialise_dtype(cls, kwargs: Any) -> dict[str, Any]:
        arr = kwargs.get('arr')
        arr = extract_array(arr)

        original = kwargs.get('original_dtype', None)

        if original is None:
            original = DtypeState.generate(arr)

        kwargs['original_dtype'] = original
        return kwargs

    def get_original_data(self):
        current_dtype_state = DtypeState.generate(self.arr)
        if current_dtype_state == self.origin_dtype:
            logger.info('No changes to the data as no prior conversions made')
            return self.arr

        data = self.arr.copy().astype(np.float64)
        data = normalize_array(data, 0.0, 1.0)
        data = normalize_array(data, self.origin_dtype.lower, self.origin_dtype.upper).astype(self.origin_dtype.dtype)
        return data


class ScalarFieldUInt8(ScalarField):
    arr: Uint8VectorT

class ScalarFieldUInt16(ScalarField):
    arr: Uint16VectorT

class ScalarFieldInt8(ScalarField):
    arr: Int8VectorT

class ScalarFieldInt16(ScalarField):
    arr: Int16VectorT

class ScalarFieldInt32(ScalarField):
    arr: Int32VectorT

class ScalarFieldFloat32(ScalarField):
    arr: Float32VectorT

class ScalarFieldBool(ScalarField):
    arr: Float32VectorT


class RGBFields(ScalarField):
    arr: Array_Nx3_uint8_T
    name: LowerStr = RGB_FIELD
    @property
    def rgb(self) -> Array_Nx3_uint8_T: return self.arr
    @property
    def r(self) -> Uint8VectorT: return self.arr[:, 0]
    @property
    def g(self) -> Uint8VectorT: return self.arr[:, 1]
    @property
    def b(self) -> Uint8VectorT: return self.arr[:, 2]

    @classmethod
    def initialize(cls, size: int, value: Array_Nx3_uint8_T|Array_Nx3_float32_T | None = None) -> RGBFields:
        if value is None:
            value = np.zeros((size, 3), dtype=np.uint8)
        return RGBFields(arr=value)

    def values_as_float(self, lower: float = 0.0, upper: float = 1.0):
        return normalize_array(self.arr.astype(np.float32, copy=True), lower, upper)



class NormalFields(ScalarField):
    arr: Array_Nx3_float32_T
    name: str = NORMALS_FIELD
    @property
    def normals(self) -> Array_Nx3_float32_T: return self.arr
    @property
    def nx(self) -> Float32VectorT: return self.arr[:, 0]
    @property
    def ny(self) -> Float32VectorT: return self.arr[:, 1]
    @property
    def nz(self) -> Float32VectorT: return self.arr[:, 2]

    @classmethod
    def initialize(cls, size: int, value: Array_Nx3_float32_T | None=None) -> NormalFields:
        if value is None:
            value = np.zeros((size, 3), dtype=np.float32)
        return NormalFields(arr=value)


SF_T = TypeVar('SF_T', bound=ScalarField)


class SegmentationMap(ScalarField):
    arr: Uint8VectorT | Uint16VectorT

    @classmethod
    def initialize(cls, name: LowerStr, pt_cloud_sizes: list[int]) -> Self:
        vector_length = sum(pt_cloud_sizes)
        if len(pt_cloud_sizes) >= 2**8:
            arr = np.zeros(vector_length, dtype=np.uint8)
        elif len(pt_cloud_sizes) >= 2**16:
            arr = np.zeros(vector_length, dtype=np.uint16)
        else:
            raise ValueError(f"Creating segmentation map for more than {2**16} point {len(pt_cloud_sizes)} not supported.")

        return SegmentationMap(arr=arr, name=name)

