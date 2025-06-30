from __future__ import annotations

import logging
from typing import Annotated, NamedTuple, Self, TypeVar, Optional, Any, TypedDict, NotRequired, Unpack

import numpy as np
import numpy.typing as npt
from pydantic import StringConstraints, model_validator, BeforeValidator

from ..validators import normalize_min_max, normalize_uint8, ensure_unit_vector, normalize_int16
from ..base_arrays import BaseVector, ArrayNx3, FixedLengthArray
from ..base_types import (
    Array_Nx3_float32_T,
    Array_Nx3_uint8_T,
    VectorT_Bool,
    VectorT_Float32,
    VectorT_Int16,
    VectorT_Uint8,
    VectorT_Uint16,
)
from ..constants import NORMALS_FIELD, RGB_FIELD


logger = logging.getLogger(__name__.split(".")[0])

LowerStr = Annotated[
    str,
    StringConstraints(strip_whitespace=True, to_lower=True),
]

class ScalarKwargT(TypedDict):
    name: NotRequired[LowerStr]
    origin_dtype: NotRequired[DtypeState]


class DtypeState(NamedTuple):
    dtype: npt.DTypeLike
    lower: npt.NDArray[np.number] | float | int
    upper: npt.NDArray[np.number] | float | int

    @classmethod
    def generate(cls, array: npt.NDArray[Any]) -> DtypeState:
        return DtypeState(dtype=array.dtype, lower=array.min(), upper=array.max())

    @staticmethod
    def validate(obj: DtypeState) -> None:
        if (obj is not None) and (obj.lower >= obj.upper):
            raise ValueError(f"lower must be less than upper. {obj=}")


# TODO look at model_dict lowerstr parameter
class AbstractScalarField(FixedLengthArray):
    name: LowerStr
    origin_dtype: DtypeState

    def __init__(self,
                 arr: npt.NDArray[Any] | Self,
                 *,
                 name: Optional[LowerStr] = None,
                 origin_dtype: Optional[DtypeState] = None):

        super().__init__(arr=arr, **{'name': name, 'origin_dtype': origin_dtype})
        return


    @model_validator(mode='before')
    @classmethod
    def validate_model_before(cls, data: Any) -> Any:

        if data['name'] is None:
            if hasattr(data['arr'], 'name'):
                data['name'] = data['arr'].name
            else:
                data['name'] = cls.model_fields['name'].default


        if data['origin_dtype'] is None:
            if hasattr(data['arr'], 'origin_dtype'):
                data['origin_dtype'] = data['arr'].origin_dtype
            else:
                data['origin_dtype'] = DtypeState.generate(data['arr'])

        return data

    def get_original_data(self) -> npt.NDArray[Any]:
        current_dtype_state = DtypeState.generate(self.arr)
        if current_dtype_state == self.origin_dtype:
            logger.info("No changes to the data as no prior conversions made")
            return self.arr.copy()

        return normalize_min_max(self.arr.copy(), self.origin_dtype.lower, self.origin_dtype.upper, self.origin_dtype.dtype)


class ScalarField(BaseVector, AbstractScalarField):
    def __init__(self, arr: Self|npt.NDArray[np.generic], **kwargs: Unpack[ScalarKwargT]):
        kwargs['name'] = kwargs.get('name', None)
        super().__init__(arr=arr, **kwargs)


class ScalarFieldTriplet(ArrayNx3, AbstractScalarField):
    def __init__(self, arr: Self|npt.NDArray[Any], name: str, **kwargs: Unpack[ScalarKwargT]):
        kwargs['name'] = name
        super().__init__(arr=arr, **kwargs)

    @classmethod
    def initialize(cls, size: int, value: Array_Nx3_float32_T | Array_Nx3_uint8_T | None = None, name: str = "") -> Self:
        dtype = cls.model_fields['arr'].annotation.__dict__['__args__'][1]
        if value is None:
            value = np.zeros((size, 3), dtype=dtype)
        return cls(value, name=name)


class RGBFields(ScalarFieldTriplet):
    arr: Annotated[Array_Nx3_uint8_T, BeforeValidator(normalize_uint8)]
    name: LowerStr = RGB_FIELD

    def __init__(self, arr: Self|npt.NDArray[np.uint8|np.float32], **kwargs: Unpack[ScalarKwargT]):
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

    def as_normalised_float32(self) -> npt.NDArray[np.float32]:
        return normalize_min_max(self, 0, 1, np.float32)


class NormalFields(ScalarFieldTriplet):
    arr: Annotated[Array_Nx3_float32_T, BeforeValidator(ensure_unit_vector)]
    name: LowerStr = NORMALS_FIELD

    def __init__(self, arr: Self|npt.NDArray[np.float32], **kwargs: Unpack[ScalarKwargT]):
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


class SegmentationMap(ScalarField):
    arr: VectorT_Uint8 | VectorT_Uint16

    def __init__(self, arr: Self|npt.NDArray[Any], **kwargs: Unpack[ScalarKwargT]):
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

class ScalarFieldUint8(ScalarField):
    arr: VectorT_Uint8

    def __init__(self, arr: Self|npt.NDArray[np.uint8], **kwargs: Unpack[ScalarKwargT]):
        super().__init__(arr=arr, **kwargs)

class NormalisedInt16ScalarField(ScalarField):
    """
    Class to support importing reflectance or intensity values as they are often in a range larger than Uint8
    """
    arr: Annotated[VectorT_Int16, BeforeValidator(normalize_int16)]

    def __init__(self, arr: Self|npt.NDArray[np.int16], **kwargs: Unpack[ScalarKwargT]):
        super().__init__(arr=arr, **kwargs)

    def to_uint8(self) -> VectorT_Uint8:
        return ScalarFieldUint8(normalize_uint8(self.arr), name=self.name, origin_dtype=self.origin_dtype)


class BooleanScalarField(ScalarField):
    arr: VectorT_Bool

    def __init__(self, arr: Self|npt.NDArray[np.bool_], **kwargs: Unpack[ScalarKwargT]):
        super().__init__(arr=arr, **kwargs)


SF_T = TypeVar("SF_T", bound=AbstractScalarField)
