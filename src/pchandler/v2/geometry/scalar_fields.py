from __future__ import annotations

import copy
import logging
from typing import Annotated, NamedTuple, Self, TypeVar, Union, Optional, Any

import numpy as np
from numpy.typing import DTypeLike, NDArray
from pydantic import AfterValidator, StringConstraints, model_validator

from .util import normalize_min_max
from ..base_arrays import BaseVector, ArrayNx3, BaseArray
from ..base_types import (
    Array_Nx3_float32_T,
    Array_Nx3_uint8_T,
    VectorT_Bool,
    VectorT_Float32,
    VectorT_Int8,
    VectorT_Int16,
    VectorT_Int32,
    VectorT_Uint8,
    VectorT_Uint16,
)
from ..constants import NORMALS_FIELD, RGB_FIELD


logger = logging.getLogger(__name__.split(".")[0])

LowerStr = Annotated[
    str,
    StringConstraints(strip_whitespace=True, to_lower=True),
]


class DtypeState(NamedTuple):
    dtype: DTypeLike
    lower: NDArray[np.number] | float | int | None
    upper: NDArray[np.number] | float | int | None

    @classmethod
    def generate(cls, array: np.ndarray):
        return DtypeState(dtype=array.dtype, lower=array.min(), upper=array.max())

    @staticmethod
    def validate(obj: DtypeState):
        if obj is not None and obj.lower >= obj.upper:
            raise ValueError(f"lower must be less than upper. {obj=}")



class AbstractScalarField(BaseArray):
    name: LowerStr
    origin_dtype: DtypeState | None = None

    def __init__(self,
                 arr: np.ndarray | ScalarField,
                 name: LowerStr = None,
                 *,
                 origin_dtype: Optional[DtypeState] = None):

        if isinstance(arr, ScalarField):
            arr = copy.deepcopy(arr)
            if name is None:
                name = arr.name

            super().__init__(arr=arr.arr, name=name, origin_dtype=arr.origin_dtype)
            return

        super().__init__(arr=arr, name=name, origin_dtype=origin_dtype)
        return

    @model_validator(mode='before')
    @classmethod
    def validate_model_before(cls, data: dict):
        if data['name'] is None:
            data['name'] = cls.model_fields['name'].default

        if data['origin_dtype'] is None:
            data['origin_dtype'] = DtypeState.generate(data['arr'])
        return data


    def __getitem__(self, key):
        return self.sample(key)

    def get_original_data(self):
        current_dtype_state = DtypeState.generate(self.arr)
        if current_dtype_state == self.origin_dtype:
            logger.info("No changes to the data as no prior conversions made")
            return self.arr.copy()

        return normalize_min_max(self.arr.copy(), self.origin_dtype.lower, self.origin_dtype.upper, self.origin_dtype.dtype)


    def get_normalized(self, lower=0, upper=1, target_dtype=np.float32):
        return normalize_min_max(self.arr, lower=lower, upper=upper, target_dtype=target_dtype)


class ScalarField(BaseVector, AbstractScalarField):
    def __init__(self, arr: np.ndarray, name: LowerStr = None, *, origin_dtype: Optional[DtypeState] = None):
        super().__init__(arr, name=name, origin_dtype=origin_dtype)


class _ScalarFieldTriplet(ArrayNx3, AbstractScalarField):
    def __init__(self, arr: Self|np.ndarray, name: LowerStr = None, *, origin_dtype: Optional[DtypeState] = None):
        super().__init__(arr, name=name, origin_dtype=origin_dtype)

    @classmethod
    def initialize(cls, size: int, value: Array_Nx3_float32_T | Array_Nx3_uint8_T | None = None) -> Self:
        dtype = cls.model_fields['arr'].annotation.__dict__['__args__'][1]
        if value is None:
            value = np.zeros((size, 3), dtype=dtype)
        return cls(value)


class RGBFields(_ScalarFieldTriplet):
    arr: Array_Nx3_uint8_T
    name: Annotated[LowerStr, AfterValidator(lambda _: RGB_FIELD)] = RGB_FIELD

    def __init__(self,
                 arr: Self|np.ndarray,
                 name: LowerStr = RGB_FIELD,
                 *,
                 origin_dtype: Optional[DtypeState] = None):
        super().__init__(arr, name=name, origin_dtype=origin_dtype)

    @property
    def r(self) -> VectorT_Uint8:
        return self.arr[:, 0]

    @property
    def g(self) -> VectorT_Uint8:
        return self.arr[:, 1]

    @property
    def b(self) -> VectorT_Uint8:
        return self.arr[:, 2]

    @model_validator(mode='before')
    @classmethod
    def validate_model_before(cls, data: Any):

        if data['origin_dtype'] is None:
            data['origin_dtype'] = DtypeState.generate(data['arr'])

        if isinstance(data['arr'], RGBFields):
            return data

        if not isinstance(data['arr'], np.ndarray):
            raise TypeError(f'Input value is not of type RGBFields or numpy array but of {type(data['arr'])}')


        if (not np.issubdtype(data['arr'].dtype, np.floating) and
                not np.issubdtype(data['arr'].dtype, np.integer) and
                not np.issubdtype(data['arr'].dtype, np.bool)):
            raise TypeError(f"Cannot convert numpy array of type {data['arr'].dtype}")

        if data['arr'].dtype != np.uint8:
            data['arr'] = normalize_min_max(data['arr'], 0, 255, np.uint8)

        return data


class NormalFields(_ScalarFieldTriplet):
    arr: Array_Nx3_float32_T
    name: Annotated[LowerStr, AfterValidator(lambda _: NORMALS_FIELD)] = NORMALS_FIELD

    def __init__(self,
                 arr: Self|np.ndarray,
                 name: LowerStr = NORMALS_FIELD,
                 *,
                 origin_dtype: Optional[DtypeState] = None):
        super().__init__(arr, name=name, origin_dtype=origin_dtype)


    @property
    def nx(self) -> VectorT_Float32:
        return self.arr[:, 0]

    @property
    def ny(self) -> VectorT_Float32:
        return self.arr[:, 1]

    @property
    def nz(self) -> VectorT_Float32:
        return self.arr[:, 2]





class SegmentationMap(ScalarField):
    arr: VectorT_Uint8 | VectorT_Uint16

    def __init__(self,
                 arr: Self|np.ndarray,
                 name: LowerStr = None,
                 *,
                 origin_dtype: Optional[DtypeState] = None):
        super().__init__(arr, name=name, origin_dtype=origin_dtype)

    @classmethod
    def initialize(cls, name: LowerStr, pt_cloud_sizes: list[int]) -> Self:
        vector_length = sum(pt_cloud_sizes)
        if len(pt_cloud_sizes) <= 2**8 - 1:
            arr = np.zeros(vector_length, dtype=np.uint8)
        elif len(pt_cloud_sizes) <= 2**16 - 1:
            arr = np.zeros(vector_length, dtype=np.uint16)
        else:
            raise ValueError(
                f"Creating segmentation map for more than {2 ** 16} point {len(pt_cloud_sizes)} not supported."
            )

        return cls(arr, name=name)


class ScalarFieldUInt8(ScalarField):
    arr: VectorT_Uint8

    def __init__(self, arr: np.ndarray, name: LowerStr = None, *, origin_dtype: Optional[DtypeState] = None):
        super().__init__(arr, name=name, origin_dtype=origin_dtype)


class ScalarFieldUInt16(ScalarField):
    arr: VectorT_Uint16

    def __init__(self, arr: Self|np.ndarray, name: LowerStr = None, *, origin_dtype: Optional[DtypeState] = None):
        super().__init__(arr, name=name, origin_dtype=origin_dtype)


class ScalarFieldInt8(ScalarField):
    arr: VectorT_Int8

    def __init__(self, arr: Self|np.ndarray, name: LowerStr = None, *, origin_dtype: Optional[DtypeState] = None):
        super().__init__(arr, name=name, origin_dtype=origin_dtype)


class ScalarFieldInt16(ScalarField):
    arr: VectorT_Int16

    def __init__(self, arr: Self|np.ndarray, name: LowerStr = None, *, origin_dtype: Optional[DtypeState] = None):
        super().__init__(arr, name=name, origin_dtype=origin_dtype)


class ScalarFieldInt32(ScalarField):
    arr: VectorT_Int32

    def __init__(self, arr: Self|np.ndarray, name: LowerStr = None, *, origin_dtype: Optional[DtypeState] = None):
        super().__init__(arr, name=name, origin_dtype=origin_dtype)


class ScalarFieldFloat32(ScalarField):
    arr: VectorT_Float32

    def __init__(self, arr: Self|np.ndarray, name: LowerStr = None, *, origin_dtype: Optional[DtypeState] = None):
        super().__init__(arr, name=name, origin_dtype=origin_dtype)


class ScalarFieldBool(ScalarField):
    arr: VectorT_Bool

    def __init__(self, arr: Self|np.ndarray, name: LowerStr = None, *, origin_dtype: Optional[DtypeState] = None):
        super().__init__(arr, name=name, origin_dtype=origin_dtype)


SF_T = TypeVar("SF_T", bound=Union[ScalarField, RGBFields, NormalFields])
