from __future__ import annotations

import logging
from typing import Annotated, NamedTuple, Self, Any, TypeVar

import numpy as np
from numpy.typing import DTypeLike, NDArray
from pydantic import StringConstraints, model_validator, Field

from ..base_arrays import BaseVector
from ..base_types import (Uint8VectorT, Float32VectorT, Array_Nx3_uint8_T, Array_Nx3_float32_T, VectorT,
                          Uint16VectorT, Int16VectorT, Int32VectorT)
from ..validators import linear_map_dtype, extract_array
from ..constants import RGB_FIELD, NORMALS_FIELD


logger = logging.getLogger(__name__.split(".")[0])


LowerStr = Annotated[str, StringConstraints(strip_whitespace=True, to_lower=True),]



class DtypeState(NamedTuple):
    dtype: DTypeLike
    lower: NDArray[np.number]|float|int|None
    upper: NDArray[np.number]|float|int|None

class ScalarField(BaseVector):
    name: LowerStr
    operations_performed: list[DtypeState] = Field(default_factory=list)
    original_dtype: DtypeState|None = None

    @model_validator(mode='before')
    @classmethod
    def initialise_dtype(cls, kwargs: Any) -> dict[str, Any]:

        original_dtype = cls._get_original_dtype(kwargs)

        operations = kwargs.get('operations_performed', [])
        if len(operations) == 0:
            operations.append(original_dtype)

        kwargs['operations_performed'] = operations

        return kwargs

    @staticmethod
    def _get_original_dtype(kwargs: dict) -> DtypeState:
        arr = kwargs.get('arr')
        arr = extract_array(arr)

        original = kwargs.get('original_dtype', None)

        if original is None:
            original = DtypeState(
                dtype=arr.dtype,
                lower=arr.min(),
                upper=arr.max()
            )

        return original


    # # FIXME Reimplement once other decisions made
    # def rollback_data_type(self):
    #     data = self.arr.copy()
    #     for operation, op_parameters in self.operations_performed[::-1]:
    #         match operation:
    #
    #             case "normalize":
    #                 np.multiply(self.arr, upper - lower, out=data)
    #                 np.add(data, lower, out=data)
    #             case "dtype_conversion":
    #                 data = data.astype(dt)
    #             case _:
    #                 return ValueError(f"Operation {operation} not supported.")
    #     assert data.dtype == self.original_range.dtype
    #
    #     logger.debug(f"Converted scalar field `{self.name}` to original bounds and dtype.")
    #     return data



class ScalarFieldUInt8(ScalarField):
    arr: Uint8VectorT

class ScalarFieldUInt16(ScalarField):
    arr: Uint16VectorT

class ScalarFieldInt8(ScalarField):
    arr: Uint8VectorT

class ScalarFieldInt16(ScalarField):
    arr: Int16VectorT

class ScalarFieldInt32(ScalarField):
    arr: Int32VectorT

class ScalarFieldFloat32(ScalarField):
    arr: Float32VectorT

class ScalarFieldBool(ScalarField):
    arr: Float32VectorT


class RGBFields(ScalarField):
    arr: Array_Nx3_uint8_T|Array_Nx3_float32_T
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

    def to_float(self, lower: float = 0.0, upper: float = 1.0):
        raise NotImplementedError("Use the dtype normalising funcs here")

    def to_uint8(self, lower: int = 0, upper: int = 255):
        raise NotImplementedError("Use the normalising functions")


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


def unpack_npydantic_dtype(cls: type[ScalarField]) -> tuple[DTypeLike, ...]:
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