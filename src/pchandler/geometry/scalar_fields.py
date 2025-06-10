from __future__ import annotations

import logging
from typing import Annotated, NamedTuple, Self, Any, TypeVar, Union, Optional

import numpy as np
from numpy.typing import DTypeLike, NDArray
from pydantic import StringConstraints, model_validator, Field, AfterValidator, BeforeValidator

from ..base_arrays import BaseVector
from ..validators import linear_map_dtype, extract_array, normalize_array
from ..constants import RGB_FIELD, NORMALS_FIELD
from ..base_types import (VectorT_Uint8, VectorT_Float32, Array_Nx3_uint8_T, Array_Nx3_float32_T, VectorT,
                          VectorT_Uint16, VectorT_Int16, VectorT_Int32, VectorT_Int8)


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

    def __init__(self, arr: np.ndarray|ScalarField, name: LowerStr, *args, **kwargs):
        if arr is not None:
            kwargs['arr'] = extract_array(arr)
        kwargs['name'] = name
        super().__init__(**kwargs)
        super().__init__(**kwargs)

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
    arr: VectorT_Uint8


class ScalarFieldUInt16(ScalarField):
    arr: VectorT_Uint16


class ScalarFieldInt8(ScalarField):
    arr: VectorT_Int8


class ScalarFieldInt16(ScalarField):
    arr: VectorT_Int16


class ScalarFieldInt32(ScalarField):
    arr: VectorT_Int32


class ScalarFieldFloat32(ScalarField):
    arr: VectorT_Float32


class ScalarFieldBool(ScalarField):
    arr: VectorT_Float32


class RGBFields(ScalarField):
    arr: Array_Nx3_uint8_T
    name: Annotated[LowerStr, AfterValidator(lambda _: RGB_FIELD)] = RGB_FIELD

    def __init__(self, arr: np.ndarray | RGBFields = None, name: LowerStr = RGB_FIELD, **kwargs):
        if arr is not None:
            kwargs['arr'] = extract_array(arr)
        kwargs['name'] = name
        super().__init__(**kwargs)

    @property
    def r(self) -> VectorT_Uint8: return self.arr[:, 0]
    @property
    def g(self) -> VectorT_Uint8: return self.arr[:, 1]
    @property
    def b(self) -> VectorT_Uint8: return self.arr[:, 2]

    @classmethod
    def initialize(cls, size: int, value: Array_Nx3_uint8_T | Array_Nx3_float32_T | None = None) -> RGBFields:
        if value is None:
            value = np.zeros((size, 3), dtype=np.uint8)
        return RGBFields(value)

    def values_as_float(self, lower: float = 0.0, upper: float = 1.0):
        return normalize_array(self.arr.astype(np.float32, copy=True), lower, upper)


class NormalFields(ScalarField):
    arr: Annotated[Array_Nx3_float32_T, BeforeValidator(lambda value: value.astype(np.float32))]
    name: Annotated[LowerStr, AfterValidator(lambda _: NORMALS_FIELD)] = NORMALS_FIELD

    def __init__(self, arr: np.ndarray, name: LowerStr=NORMALS_FIELD, **kwargs):
        if arr is not None:
            kwargs['arr'] = arr
        kwargs['name'] = name
        super().__init__(**kwargs)

    @property
    def x(self) -> VectorT_Float32: return self.arr[:, 0]
    @property
    def y(self) -> VectorT_Float32: return self.arr[:, 1]
    @property
    def z(self) -> VectorT_Float32: return self.arr[:, 2]

    @classmethod
    def initialize(cls, size: int, value: Array_Nx3_float32_T | None=None) -> NormalFields:
        if value is None:
            value = np.zeros((size, 3), dtype=np.float32)
        return NormalFields(value)


SF_T = TypeVar('SF_T', bound=Union[ScalarField, RGBFields, NormalFields])


class SegmentationMap(ScalarField):
    arr: VectorT_Uint8 | VectorT_Uint16

    def __init__(self, arr: np.ndarray, name: LowerStr, **kwargs):
        kwargs['arr'] = arr
        kwargs['name'] = name
        super().__init__(**kwargs)

    @classmethod
    def initialize(cls, name: LowerStr, pt_cloud_sizes: list[int]) -> Self:
        vector_length = sum(pt_cloud_sizes)
        if len(pt_cloud_sizes) >= 2**8:
            arr = np.zeros(vector_length, dtype=np.uint8)
        elif len(pt_cloud_sizes) >= 2**16:
            arr = np.zeros(vector_length, dtype=np.uint16)
        else:
            raise ValueError(f"Creating segmentation map for more than {2**16} point {len(pt_cloud_sizes)} not supported.")

        return cls(arr, name=name)

