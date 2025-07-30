from __future__ import annotations

import logging
from typing import Annotated, NamedTuple, Self, TypeVar, Optional, Any, TypedDict, NotRequired, Unpack, TypeAlias, cast

import numpy as np
import numpy.typing as npt
from pydantic import model_validator, BeforeValidator, field_validator

from pchandler.constants import NORMAL_NAMES, RGB_NAMES
from pchandler.validators import normalize_min_max, normalize_uint8, normalize_int16
from pchandler.base_arrays import BaseVector, ArrayNx3, FixedLengthArray
from pchandler.base_types import (
    SfNameT,
    Array_Float32_T,
    Array_Nx3_T,
    Array_Nx3_Float_T,
    Array_Nx3_Float32_T,
    Array_Nx3_Uint8_T,
    VectorT,
    Vector_Bool_T,
    Vector_Float32_T,
    Vector_Int16_T,
    Vector_Uint8_T,
    Vector_Uint16_T,
    LowerStr
)


logger = logging.getLogger(__name__.split(".")[0])


#  TODO add name attribute to be able to track original names from original data files
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


SfOrigDtT: TypeAlias = Optional[DtypeState]


class ScalarKwargT(TypedDict):
    name: NotRequired[SfNameT]
    origin_dtype: NotRequired[DtypeState]


# TODO add an astype method for converting data types


class AbstractScalarField(FixedLengthArray):
    name: SfNameT
    origin_dtype: DtypeState

    def __init__(self, arr: VectorT | Array_Nx3_T | Self, name: SfNameT = None, origin_dtype: SfOrigDtT = None):
        kwargs: dict[str, Any] = {'name': name, 'origin_dtype': origin_dtype}
        super().__init__(arr, **kwargs)

    # noinspection PyNestedDecorators
    @model_validator(mode='before')
    @classmethod
    def validate_model_before(cls, data: Any) -> Any:
        if data['name'] is None:
            # Extract name from field if it exists, otherwise get default if available
            if hasattr(data['arr'], 'name'):
                data['name'] = data['arr'].name
            else:
                data['name'] = cls.model_fields['name'].default

        # Get the origin_dtype if it exists
        if data['origin_dtype'] is None:
            if hasattr(data['arr'], 'origin_dtype'):
                data['origin_dtype'] = data['arr'].origin_dtype
            else:
                data['origin_dtype'] = DtypeState.generate(data['arr'])

        return data

    def get_original_data(self) -> npt.NDArray[Any]:
        current_dtype_state = DtypeState.generate(self.arr)

        if current_dtype_state == self.origin_dtype:
            return self.arr.copy()

        return normalize_min_max(array=self.arr.copy(),
                                 lower=float(self.origin_dtype.lower),
                                 upper=float(self.origin_dtype.upper),
                                 target_dtype=self.origin_dtype.dtype)


class ScalarField(BaseVector, AbstractScalarField):
    def __init__(self, arr: VectorT|Self, name: SfNameT = None, **kwargs: Unpack[ScalarKwargT]):
        if name is not None:
            kwargs['name'] = name
        super().__init__(arr, **cast(dict[str, Any], kwargs))


class ScalarFieldTriplet(ArrayNx3, AbstractScalarField):
    def __init__(self, arr: Array_Nx3_T|Self, **kwargs: Unpack[ScalarKwargT]):
        super().__init__(arr, **cast(dict[str, Any], kwargs))

    @classmethod
    def initialize(cls, size: int, value: Array_Nx3_Uint8_T | None = None, name: str = "") -> Self:
        dtype = cls.model_fields['arr'].annotation.__dict__['__args__'][1]
        if value is None:
            value = np.zeros((size, 3), dtype=dtype)
        return cls(value, name=name)


class RGBFields(ScalarFieldTriplet):
    arr: Array_Nx3_Uint8_T
    name: SfNameT = RGB_NAMES.base

    def __init__(self, arr: Array_Nx3_Uint8_T|Array_Nx3_Float_T|Self, **kwargs: Unpack[ScalarKwargT]):
        kwargs['name'] = RGB_NAMES.base
        super().__init__(arr, **kwargs)

    # noinspection PyNestedDecorators
    @field_validator('arr', mode='before')
    @classmethod
    def normalise_to_uint8(cls, data: npt.NDArray[Any]) -> npt.NDArray[np.uint8]:
        return normalize_uint8(data)

    @property
    def red(self) -> Vector_Uint8_T:
        return self.arr[:, 0]

    @property
    def r(self) -> Vector_Uint8_T:
        return self.red

    @property
    def green(self) -> Vector_Uint8_T:
        return self.arr[:, 1]

    @property
    def g(self) -> Vector_Uint8_T:
        return self.green

    @property
    def blue(self) -> Vector_Uint8_T:
        return self.arr[:, 2]

    @property
    def b(self) -> Vector_Uint8_T:
        return self.blue

    def as_normalised_float32(self) -> Array_Float32_T:
        return normalize_min_max(self.arr, 0, 1, np.float32)


class NormalFields(ScalarFieldTriplet):
    arr: Array_Nx3_Float32_T
    name: LowerStr = NORMAL_NAMES.base

    def __init__(self, arr: Array_Nx3_Float_T|Self, **kwargs: Unpack[ScalarKwargT]):
        kwargs['name'] = NORMAL_NAMES.base
        super().__init__(arr, **kwargs)

    # noinspection PyNestedDecorators
    @field_validator('arr', mode='before')
    @classmethod
    def ensure_unit_vector(cls, array: Array_Nx3_Float_T) -> Array_Nx3_Float32_T:
        if not (np.issubdtype(array.dtype, np.floating) or np.issubdtype(array.dtype, np.signedinteger)):
            raise TypeError("Dtype of normals array must be of type floating or signed integer}")

        array /= np.linalg.norm(array, axis=1).reshape(-1, 1)
        result = Array_Nx3_Float32_T(array.astype(np.float32))

        return result

    @property
    def nx(self) -> Vector_Float32_T:
        return self.arr[:, 0]

    @property
    def ny(self) -> Vector_Float32_T:
        return self.arr[:, 1]

    @property
    def nz(self) -> Vector_Float32_T:
        return self.arr[:, 2]

    @classmethod
    def initialize(cls, size: int, value: Array_Nx3_Float32_T | None = None, name: str = "") -> Self:
        dtype = cls.model_fields['arr'].annotation.__dict__['__args__'][1]
        if value is None:
            value = np.zeros((size, 3), dtype=dtype)
            value[:, 2] = 1
        return cls(value, name=name)


class SegmentationMap(ScalarField):
    arr: Vector_Uint8_T | Vector_Uint16_T

    def __init__(self, arr: Vector_Uint8_T | Vector_Uint16_T | Self, **kwargs: Unpack[ScalarKwargT]):
        super().__init__(arr, **kwargs)

    @classmethod
    def initialize(cls, name: LowerStr, pt_cloud_sizes: list[int]) -> Self:
        vector_length = sum(pt_cloud_sizes)

        if len(pt_cloud_sizes) <= 2**8 - 1:
            arr = np.zeros(vector_length, dtype=np.uint8)

        elif len(pt_cloud_sizes) <= 2**16 - 1:
            arr = np.zeros(vector_length, dtype=np.uint16)

        else:
            raise ValueError(f"Segmentation map for more than {2 ** 16} classes {len(pt_cloud_sizes)} not supported.")

        return cls(arr, name=name)


class ScalarFieldUint8(ScalarField):
    arr: Vector_Uint8_T

    def __init__(self, arr: Vector_Uint8_T | Self, **kwargs: Unpack[ScalarKwargT]):
        super().__init__(arr, **kwargs)


class ScalarFieldBoolean(ScalarField):
    arr: Vector_Bool_T

    def __init__(self, arr: Vector_Bool_T | Self, **kwargs: Unpack[ScalarKwargT]):
        super().__init__(arr, **kwargs)


class ScalarFieldFloat32(ScalarField):
    arr: Vector_Float32_T

    def __init__(self, arr: Vector_Float32_T | Self, **kwargs: Unpack[ScalarKwargT]):
        super().__init__(arr, **kwargs)


class NormalisedInt16ScalarField(ScalarField):
    """
    Class to support importing reflectance or intensity values as they are often in a range larger than Uint8
    """
    arr: Annotated[Vector_Int16_T, BeforeValidator(normalize_int16)]

    def __init__(self, arr: VectorT | Self, **kwargs: Unpack[ScalarKwargT]):
        super().__init__(arr, **kwargs)

    def to_uint8(self) -> ScalarFieldUint8:
        return ScalarFieldUint8(normalize_uint8(self.arr), name=self.name, origin_dtype=self.origin_dtype)


SF_T = TypeVar("SF_T", bound=AbstractScalarField)