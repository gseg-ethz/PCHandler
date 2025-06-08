from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import Optional, Annotated, NamedTuple, Self, Union

import numpy as np
from numpy.typing import DTypeLike
from pydantic import ConfigDict, StringConstraints, model_validator, field_validator, BeforeValidator, Field

from ..base_arrays import (
    BaseVector, Array_Nx3_T, Vector_N_uint8_T, Vector_N_float32_T, Array_Nx3_uint8_T, Array_Nx3_float32_T, Vector_N_T, Vector_N_uint16_T,
    Vector_N_bool_T, BaseArray, _FixedLengthArray, Vector_N_int16_T, Vector_N_int32_T, Vector_N_int8_T)
from ..validators import linear_map_dtype, extract_array
from ..custom_types import DataRange


logger = logging.getLogger(__name__.split(".")[0])

DEFAULT_CONFIG = ConfigDict(arbitrary_types_allowed=True)

RGB_FIELD = 'rgb'
NORMALS_FIELD = 'normals'
# INTENSITY_FIELD = 'intensity'

RGB_POTENTIAL_NAMES = ('r', 'g', 'b', 'rgb', 'bgr', 'red', 'green', 'blue', 'rgba')
NORMAL_POTENTIAL_NAMES = ('normal', 'normals', 'normal_fields')
INTENSITY_POTENTIAL_NAMES = ('intensity', 'reflectance', 'intensities')

# DECIDE on consistent names for rgb/colour and intensity/reflectance and supported names
# DECIDE on default dtypes for classes -> e.g. intensities (int16? uint8?)
# DECIDE on default normalisation strategy and which methods for automatic coercion
#  e..g RGB from [0.0, 1.0] to [0, 255]
# DECIDE when and where to apply normalisation by default. Helper functions must exist though (e.g. utils)


LowerStr = Annotated[str, StringConstraints(strip_whitespace=True, to_lower=True),]


class AbstractScalarField(_FixedLengthArray, ABC):
    name: LowerStr
    operations_performed: list[tuple[str, DataRange]] = Field(default_factory=list)
    original_range: DTypeLike|None = None

    @model_validator(mode='before')
    @classmethod
    def get_original_dtype(cls, kwargs):
        arr = kwargs.get('arr')
        arr = extract_array(arr)
        original = kwargs.get('original_range', None)

        if original is None:
            DataRange(
                dtype = arr.dtype,
                min = arr.min(),
                max = arr.max()
            )

        operations = kwargs.get('operations', [])
        if len(operations) == 0:
            operations.append(original)


        return kwargs

    # FIXME Reimplement once other decisions made
    def rollback_data_type(self):
        data = self.arr.copy()
        for operation, op_parameters in self.operations_performed[::-1]:
            match operation:

                case "normalize":
                    np.multiply(self.arr, upper - lower, out=data)
                    np.add(data, lower, out=data)
                case "dtype_conversion":
                    data = data.astype(dt)
                case _:
                    return ValueError(f"Operation {operation} not supported.")
        assert data.dtype == self.original_range.dtype

        logger.debug(f"Converted scalar field `{self.name}` to original bounds and dtype.")
        return data


class ScalarField(AbstractScalarField):
    arr: Vector_N_T|Self

class ScalarFieldUInt8(AbstractScalarField):
    arr: Vector_N_uint8_T

class ScalarFieldUInt16(AbstractScalarField):
    arr: Vector_N_uint8_T

class ScalarFieldInt8(AbstractScalarField):
    arr: Vector_N_uint8_T

class ScalarFieldInt16(AbstractScalarField):
    arr: Vector_N_int16_T

class ScalarFieldInt32(AbstractScalarField):
    arr: Vector_N_int32_T

class ScalarFieldFloat32(AbstractScalarField):
    arr: Vector_N_float32_T

class ScalarFieldBool(AbstractScalarField):
    arr: Vector_N_float32_T


class RGBFields(AbstractScalarField):
    arr: Array_Nx3_uint8_T
    name: LowerStr = RGB_FIELD
    @property
    def rgb(self) -> Array_Nx3_uint8_T: return self.arr
    @property
    def r(self) -> Vector_N_uint8_T: return self.arr[:, 0]
    @property
    def g(self) -> Vector_N_uint8_T: return self.arr[:, 1]
    @property
    def b(self) -> Vector_N_uint8_T: return self.arr[:, 2]

    @classmethod
    def initialize(cls, size: int, value: Array_Nx3_uint8_T | None = None) -> RGBFields:
        if value is None:
            value = np.zeros((size, 3), dtype=np.uint8)
        return RGBFields(arr=value)


class NormalFields(AbstractScalarField):
    arr: Array_Nx3_float32_T
    name: str = NORMALS_FIELD
    @property
    def normals(self) -> Array_Nx3_float32_T: return self.arr
    @property
    def nx(self) -> Vector_N_float32_T: return self.arr[:, 0]
    @property
    def ny(self) -> Vector_N_float32_T: return self.arr[:, 1]
    @property
    def nz(self) -> Vector_N_float32_T: return self.arr[:, 2]

    @classmethod
    def initialize(cls, size: int, value: Array_Nx3_float32_T | None=None) -> NormalFields:
        if value is None:
            value = np.zeros((size, 3), dtype=np.float32)
        return NormalFields(arr=value)

# TODO is there a uniform consensus on Intensity / Reflectance?
#  Should we create these based on instruments / manufacturer specs we know

# DISCUSS should we support a segmentation
class SegmentationMap(ScalarField):
    arr: Vector_N_uint8_T | Vector_N_uint16_T

    @classmethod
    def initialize(cls, pt_cloud_sizes: list[int]) -> NormalFields:
        vector_length = sum(pt_cloud_sizes)
        if len(pt_cloud_sizes) >= 2**8:
            arr = np.zeros(vector_length, dtype=np.uint8)
        elif len(pt_cloud_sizes) >= 2**16:
            arr = np.zeros(vector_length, dtype=np.uint16)
        else:
            raise ValueError(f"Creating segmentation map for more than {2**16} point {len(pt_cloud_sizes)} not supported.")

        return NormalFields(arr=arr)


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
    except Exception:
        all_types.append(a)
    return tuple(all_types)