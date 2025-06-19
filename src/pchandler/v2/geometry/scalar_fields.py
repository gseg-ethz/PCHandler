from __future__ import annotations

import logging
from typing import Annotated, NamedTuple, Self, TypeVar, Union, Optional, Any, TypedDict, NotRequired, Unpack

import numpy as np
from numpy.typing import DTypeLike, NDArray
from pydantic import StringConstraints, model_validator

from .util import normalize_min_max
from ..base_arrays import BaseVector, ArrayNx3, FixedLengthArray
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

class _SW_KW_T(TypedDict):
    name: NotRequired[LowerStr]
    origin_dtype: NotRequired[DtypeState]


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


class AbstractScalarField(FixedLengthArray):
    name: LowerStr
    origin_dtype: DtypeState | None = None

    def __init__(self,
                 arr: np.ndarray | Self,
                 *,
                 name: LowerStr = None,
                 origin_dtype: Optional[DtypeState] = None):

        if hasattr(arr, 'name') and name is None:
            name = arr.name

        if hasattr(arr, 'origin_dtype') and origin_dtype is None:
            origin_dtype = arr.origin_dtype

        super().__init__(arr=arr, name=name, origin_dtype=origin_dtype)
        return


    @model_validator(mode='before')
    @classmethod
    def validate_model_before(cls, data: Any):
        if data['name'] is None:
            data['name'] = cls.model_fields['name'].default

        if data['origin_dtype'] is None:
            data['origin_dtype'] = DtypeState.generate(data['arr'])

        if isinstance(data['arr'], AbstractScalarField):
            return data

        if hasattr(cls, '_normalize'):
            data['arr'] = cls._normalize(data['arr'])

        return data

    def get_original_data(self):
        current_dtype_state = DtypeState.generate(self.arr)
        if current_dtype_state == self.origin_dtype:
            logger.info("No changes to the data as no prior conversions made")
            return self.arr.copy()

        return normalize_min_max(self.arr.copy(), self.origin_dtype.lower, self.origin_dtype.upper, self.origin_dtype.dtype)


class ScalarField(BaseVector, AbstractScalarField):
    def __init__(self, arr: Self|np.ndarray, **kwargs: Unpack[_SW_KW_T]):
        super().__init__(arr=arr, **kwargs)


class _ScalarFieldTriplet(ArrayNx3, AbstractScalarField):
    def __init__(self, arr: Self|np.ndarray, **kwargs: Unpack[_SW_KW_T]):
        super().__init__(arr=arr, **kwargs)

    @classmethod
    def initialize(cls, size: int, value: Array_Nx3_float32_T | Array_Nx3_uint8_T | None = None) -> Self:
        dtype = cls.model_fields['arr'].annotation.__dict__['__args__'][1]
        if value is None:
            value = np.zeros((size, 3), dtype=dtype)
        return cls(value)


class RGBFields(_ScalarFieldTriplet):
    arr: Array_Nx3_uint8_T
    name: LowerStr = RGB_FIELD

    def __init__(self, arr: Self|np.ndarray, **kwargs: Unpack[_SW_KW_T]):
        kwargs['name'] = RGB_FIELD
        super().__init__(arr, **kwargs)

    @property
    def r(self) -> VectorT_Uint8:
        return self.arr[:, 0]

    @property
    def g(self) -> VectorT_Uint8:
        return self.arr[:, 1]

    @property
    def b(self) -> VectorT_Uint8:
        return self.arr[:, 2]

    @staticmethod
    def _normalize(array: np.ndarray) -> np.ndarray:
        if array.dtype != np.uint8:
            array = normalize_min_max(array, 0, 255, np.uint8)
        return array


class NormalFields(_ScalarFieldTriplet):
    arr: Array_Nx3_float32_T
    name: LowerStr = NORMALS_FIELD

    def __init__(self, arr: Self|np.ndarray, **kwargs: Unpack[_SW_KW_T]):
        kwargs['name'] = NORMALS_FIELD
        super().__init__(arr=arr, **kwargs)

    @property
    def nx(self) -> VectorT_Float32:
        return self.arr[:, 0]

    @property
    def ny(self) -> VectorT_Float32:
        return self.arr[:, 1]

    @property
    def nz(self) -> VectorT_Float32:
        return self.arr[:, 2]

    @staticmethod
    def _normalize(array: np.ndarray) -> np.ndarray:
        if not (np.issubdtype(array.dtype, np.floating) or np.issubdtype(array.dtype, np.signedinteger)):
            raise TypeError("Dtype of normals array must be of type floating or signed integer}")

        if array.dtype == np.float32:
            return array
        array = array.astype(np.float32)
        return array / np.linalg.norm(array, axis=1).reshape(-1, 1)

class SegmentationMap(ScalarField):
    arr: VectorT_Uint8 | VectorT_Uint16

    def __init__(self, arr: Self|np.ndarray, **kwargs: Unpack[_SW_KW_T]):
        super().__init__(arr=arr, **kwargs)

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

    def __init__(self, arr: Self|np.ndarray, **kwargs: Unpack[_SW_KW_T]):
        super().__init__(arr=arr, **kwargs)


class ScalarFieldUInt16(ScalarField):
    arr: VectorT_Uint16

    def __init__(self, arr: Self|np.ndarray, **kwargs: Unpack[_SW_KW_T]):
        super().__init__(arr=arr, **kwargs)


class ScalarFieldInt8(ScalarField):
    arr: VectorT_Int8

    def __init__(self, arr: Self|np.ndarray, **kwargs: Unpack[_SW_KW_T]):
        super().__init__(arr=arr, **kwargs)


class ScalarFieldInt16(ScalarField):
    arr: VectorT_Int16

    def __init__(self, arr: Self|np.ndarray, **kwargs: Unpack[_SW_KW_T]):
        super().__init__(arr=arr, **kwargs)


class ScalarFieldInt32(ScalarField):
    arr: VectorT_Int32

    def __init__(self, arr: Self|np.ndarray, **kwargs: Unpack[_SW_KW_T]):
        super().__init__(arr=arr, **kwargs)


class ScalarFieldFloat32(ScalarField):
    arr: VectorT_Float32

    def __init__(self, arr: Self|np.ndarray, **kwargs: Unpack[_SW_KW_T]):
        super().__init__(arr=arr, **kwargs)


class ScalarFieldBool(ScalarField):
    arr: VectorT_Bool

    def __init__(self, arr: Self|np.ndarray, **kwargs: Unpack[_SW_KW_T]):
        super().__init__(arr=arr, **kwargs)


SF_T = TypeVar("SF_T", bound=Union[ScalarField, RGBFields, NormalFields])
