from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import Optional, Annotated, TYPE_CHECKING, Any, NamedTuple

import numpy as np
from numpy.typing import DTypeLike
from pydantic import ConfigDict, StringConstraints, model_validator, field_validator, BeforeValidator

from ..base_arrays import (
    BaseVector, Array_Nx3_T, Vector_N_u1_T, Vector_N_f4_T, Array_Nx3_u1_T, Array_Nx3_f4_T, Vector_N_T, Vector_N_u2_T,
    Vector_N_b_T, BaseArray, _FixedLengthArray)
from ..validators import linear_map_dtype
from ..custom_types import OriginalFieldState


logger = logging.getLogger(__name__.split(".")[0])

DEFAULT_CONFIG = ConfigDict(arbitrary_types_allowed=True)

RGB_FIELD = 'rgb'
NORMALS_FIELD = 'normals'
INTENSITY_FIELD = 'intensity'

RGB_POTENTIAL_NAMES = ('r', 'g', 'b', 'rgb', 'bgr', 'red', 'green', 'blue', 'rgba')
NORMAL_POTENTIAL_NAMES = ('normal', 'normals', 'normal_fields')
INTENSITY_POTENTIAL_NAMES = ('intensity', 'reflectance', 'intensities')

# DECIDE on consistent names for rgb/colour and intensity/reflectance and supported names
# DECIDE on default dtypes for classes -> e.g. intensities (int16? uint8?)
# DECIDE on default normalisation strategy and which methods for automatic coercion
#  e..g RGB from [0.0, 1.0] to [0, 255]
# DECIDE when and where to apply normalisation by default. Helper functions must exist though (e.g. utils)


LowerStr = Annotated[str, StringConstraints(strip_whitespace=True, to_lower=True),]


class NormaliseOptions(NamedTuple):
    target_dtype: DTypeLike


class AbstractScalarField(_FixedLengthArray, ABC):
    name: LowerStr
    operations_performed: Optional[list[str]] = None
    original_state: OriginalFieldState|None = None


    @field_validator('arr', mode='before')
    @classmethod
    def coerce_to_target_type(cls, arr: np.ndarray) -> np.ndarray:
        target_types: DTypeLike = unpack_npydantic_dtype(cls)
        # DISCUSS what would be the approach? If float, normalise [0,1] then linear_map_dtype?

        if arr.dtype in target_types:
            return arr
        else:
            return linear_map_dtype(arr, target_types[0])


    @model_validator(mode='before')
    @classmethod
    def get_original_dtype(cls, kwargs):
        arr: np.ndarray = kwargs.get('arr')

        kwargs['original_state'] = OriginalFieldState(
            dtype=arr.dtype,
            upper=arr.max(),
            lower=arr.min(),
        )

        return kwargs


    # FIXME Reimplement once other decisions made
    def create_rollback(self) -> NDArray[np.generic]:
        data = self.data.copy()
        for operation, operation_parameters in self.operations_performed[::-1]:
            match operation:
                case "normalize":
                    lower, upper = cast(tuple[float, float],operation_parameters)
                    np.multiply(self.data, upper - lower, out=data)
                    np.add(data, lower, out=data)
                case "dtype_conversion":
                    data = data.astype(cast(DTypeLike,operation_parameters[0]))
                case _:
                    return ValueError(f"Operation {operation} not supported.")
        assert self.original_dtype is None or data.dtype == self.original_dtype

        logger.debug(f"Converted scalar field `{self.name}` to original bounds and dtype.")
        return data

class ScalarField(AbstractScalarField, BaseVector):
    pass

class ScalarFieldTriplet(AbstractScalarField):
    arr: Array_Nx3_T


class IntensityField(ScalarField):
    name: LowerStr = INTENSITY_FIELD
    arr: Vector_N_u2_T|Vector_N_u1_T

    @classmethod
    def initialize(cls, size: int, value: Vector_N_u2_T|Vector_N_u1_T|None=None, dtype=np.uint16):
        if value is None:
            value = np.zeros(size, dtype=dtype)
        return IntensityField(arr=value)


class RGBFields(ScalarFieldTriplet):
    name: LowerStr = RGB_FIELD
    arr: Array_Nx3_u1_T
    @property
    def rgb(self) -> Array_Nx3_u1_T: return self.arr
    @property
    def r(self) -> Vector_N_u1_T: return self.arr[:, 0]
    @property
    def g(self) -> Vector_N_u1_T: return self.arr[:, 1]
    @property
    def b(self) -> Vector_N_u1_T: return self.arr[:, 2]

    @classmethod
    def initialize(cls, size: int, value: Array_Nx3_u1_T|None = None) -> RGBFields:
        if value is None:
            value = np.zeros((size, 3), dtype=np.uint8)
        return RGBFields(arr=value)


class NormalFields(ScalarFieldTriplet):
    name: str = NORMALS_FIELD
    @property
    def normals(self) -> Array_Nx3_f4_T: return self.arr
    @property
    def nx(self) -> Vector_N_f4_T: return self.arr[:, 0]
    @property
    def ny(self) -> Vector_N_f4_T: return self.arr[:, 1]
    @property
    def nz(self) -> Vector_N_f4_T: return self.arr[:, 2]

    @classmethod
    def initialize(cls, size: int, value: Array_Nx3_f4_T|None=None) -> NormalFields:
        if value is None:
            value = np.zeros((size, 3), dtype=np.float32)
        return NormalFields(arr=value)

# DISCUSS should we have a mask/boolean type to use for the indices?
class BooleanField(ScalarField):
    arr: Vector_N_b_T


# DISCUSS should we support a segmentation
class SegmentationMap(ScalarField):
    arr: Vector_N_u1_T|Vector_N_u2_T

@classmethod
    def initialize(cls, pt_cloud_sizes: list[int]) -> NormalFields:
        vector_length = sum(pt_cloud_sizes)
        if len(pt_cloud_sizes) >= 2**8:
            arr = np.zeros(vector_length, dtype=np.uint8)
        elif len(pt_cloud_sizes) >= 2**16:
            arr = np.zeros(vector_length, dtype=np.uint16)
        else:
            raise ValueError(f"Creating segmentation map for more than {2**16} point {len(pt_cloud_sizes)} not supported.")


        if value is None:
            value = np.zeros((size, 3), dtype=np.float32)
        return NormalFields(arr=value)

def get_npydantic_dtype(cls: type):
    return cls.model_fields['arr'].annotation.__dict__['__args__'][1]


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