from __future__ import annotations

import logging
from typing import Annotated, NamedTuple, Self, TypeVar, Union

import numpy as np
from numpy.typing import DTypeLike, NDArray
from pydantic import StringConstraints, AfterValidator

from base_arrays import BaseVector
from validators import extract_array
from constants import RGB_FIELD, NORMALS_FIELD
from base_types import (VectorT_Uint8, VectorT_Float32, Array_Nx3_uint8_T, Array_Nx3_float32_T,
                        VectorT_Uint16, VectorT_Int16, VectorT_Int32, VectorT_Int8, VectorT_Bool)


logger = logging.getLogger(__name__.split(".")[0])

LowerStr = Annotated[str, StringConstraints(strip_whitespace=True, to_lower=True),]

def linear_map_dtype(array: np.ndarray, target_dtype: DTypeLike) -> np.ndarray:

    def get_dtype_min_max(dt: np.dtype) -> tuple[float, float]:
        if np.issubdtype(dt, np.integer):
            info = np.iinfo(dt)
            return info.min, info.max
        elif np.issubdtype(dt, np.floating):
            return 0.0, 1.0
        else:
            raise TypeError(f'Invalid dtype detected: {dt}')

    # Types match, exit
    if array.dtype == target_dtype:
        return array

    # Get the corresponding min and max from the type info
    origin_min, origin_max = get_dtype_min_max(array.dtype)
    target_min, target_max = get_dtype_min_max(target_dtype)

    # Ensure precision is not lost
    array = array.astype(np.float64)

    # normalised
    # TODO catch divide by 0
    np.divide(array - origin_min, origin_max - origin_min, out=array)

    if np.issubdtype(target_dtype, np.floating):
        return array.astype(np.float32)

    mapped = np.floor(array * float(target_max - target_min) + target_min)
    return mapped.astype(target_dtype).flatten()

def normalize_array(array: np.ndarray, target_state: DtypeState = None) -> np.ndarray:
    """
    General normalisation function.Normalise array by default to the bounds [0, 1] alternatively a custom range as defined by lower and upper.

    Default -> [0.0, 1.0] np.float32
    Custom target state will normalise to range of [0, 1] then scale to lower and upper values

    If desired normalisation is a mapping from one dtype to another e.g. uint8 to uint16, use 'linear_map_dtype' as this
    respects the datatype limit bounds
    """

    original_dtype = array.dtype
    array = array.astype(np.float64)

    arr_min, arr_max = array.min(axis=0), array.max(axis=0)

    if target_state is None:
        lower, upper = 0, 1
    else:
        lower, upper = target_state.lower, target_state.upper

    DtypeState.validate(target_state)

    np.divide(array - arr_min, arr_max - arr_min, out=array)
    np.add(array * (upper-lower), lower, out=array)

    if target_state is not None:
        # Floating point return as is
        if np.issubdtype(target_state.dtype, np.floating):
            return array.astype(np.float32)
        else:
            # Integers need to be floored
            np.floor(array, out=array)
            return array.astype(original_dtype)
    logger.debug("No dtype defined for normalisation. Set astype np.float32 by default")
    return array.astype(np.float32)

def normalise_self(array: np.ndarray) -> np.ndarray:
    """
    Normalise values to the min and max values of the associated integer type
    """

    if np.dtype(array.dtype).kind not in ["u", "i"]:
        logger.debug(f"Scalar field is floating. Converting to [0.0, 1.0].")
        target_state=DtypeState(np.float32, 0.0, 1.0)
    else:
        target_state = DtypeState(dtype=array.dtype,
                                  lower=np.iinfo(array.dtype).min,
                                  upper=np.iinfo(array.dtype).max)

    return normalize_array(array=array, target_state=target_state)


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

    @staticmethod
    def validate(obj: DtypeState):
        if obj is not None and obj.lower >= obj.upper:
            raise ValueError(f"lower must be less than upper. {obj=}")


class ScalarField(BaseVector):
    name: LowerStr
    origin_dtype: DtypeState | None = None

    def __init__(self, arr: np.ndarray|ScalarField, name: LowerStr = None, *args, **kwargs):
        if isinstance(arr, ScalarField) and name is None:
            kwargs['arr'] = extract_array(arr)
            kwargs['name'] = arr.name
            kwargs['origin_dtype'] = arr.origin_dtype
            super().__init__(**kwargs)
            return

        if arr is not None:
            kwargs['arr'] = extract_array(arr)
        kwargs['name'] = name
        if 'origin_dtype' not in kwargs:
            kwargs['origin_dtype'] = DtypeState.generate(kwargs['arr'])

        super().__init__(**kwargs)

    def __getitem__(self, key):
        return self.sample(key)

    def get_original_data(self):
        current_dtype_state = DtypeState.generate(self.arr)
        if current_dtype_state == self.origin_dtype:
            logger.info('No changes to the data as no prior conversions made')
            return self.arr.copy()

        data = self.arr.copy().astype(np.float64)
        return normalize_array(data, self.origin_dtype)


class ScalarFieldUInt8(ScalarField):
    arr: VectorT_Uint8
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)

class ScalarFieldUInt16(ScalarField):
    arr: VectorT_Uint16
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)

class ScalarFieldInt8(ScalarField):
    arr: VectorT_Int8
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)

class ScalarFieldInt16(ScalarField):
    arr: VectorT_Int16
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)


class ScalarFieldInt32(ScalarField):
    arr: VectorT_Int32
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)


class ScalarFieldFloat32(ScalarField):
    arr: VectorT_Float32
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)


class ScalarFieldBool(ScalarField):
    arr: VectorT_Bool
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)


class RGBFields(ScalarField):
    arr: Array_Nx3_uint8_T
    name: Annotated[LowerStr, AfterValidator(lambda _: RGB_FIELD)] = RGB_FIELD

    def __init__(self, arr: np.ndarray | RGBFields = None, name: LowerStr = RGB_FIELD, **kwargs):
        super().__init__(arr, name, **kwargs)

    @property
    def r(self) -> VectorT_Uint8: return self.arr[:, 0]
    @property
    def g(self) -> VectorT_Uint8: return self.arr[:, 1]
    @property
    def b(self) -> VectorT_Uint8: return self.arr[:, 2]

    @classmethod
    def initialize(cls, size: int, value: Array_Nx3_uint8_T | None = None) -> RGBFields:
        if value is None:
            value = np.zeros((size, 3), dtype=np.uint8)
        return RGBFields(value)

    def get_normalized(self, lower: float = 0.0, upper: float = 1.0):
        return normalize_array(self.arr.astype(np.float32, copy=True),
                               target_state=DtypeState(np.float32, lower, upper))


class NormalFields(ScalarField):
    arr: Array_Nx3_float32_T
    name: Annotated[LowerStr, AfterValidator(lambda _: NORMALS_FIELD)] = NORMALS_FIELD

    def __init__(self, arr: np.ndarray, name: LowerStr = NORMALS_FIELD, **kwargs):
        super().__init__(arr, name, **kwargs)

    @property
    def nx(self) -> VectorT_Float32: return self.arr[:, 0]
    @property
    def ny(self) -> VectorT_Float32: return self.arr[:, 1]
    @property
    def nz(self) -> VectorT_Float32: return self.arr[:, 2]

    @classmethod
    def initialize(cls, size: int, value: Array_Nx3_float32_T | None=None) -> NormalFields:
        if value is None:
            value = np.zeros((size, 3), dtype=np.float32)
        return NormalFields(value)


SF_T = TypeVar('SF_T', bound=Union[ScalarField, RGBFields, NormalFields])


class SegmentationMap(ScalarField):
    arr: VectorT_Uint8 | VectorT_Uint16

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def initialize(cls, name: LowerStr, pt_cloud_sizes: list[int]) -> Self:
        vector_length = sum(pt_cloud_sizes)
        if len(pt_cloud_sizes) <= 2**8 - 1:
            arr = np.zeros(vector_length, dtype=np.uint8)
        elif len(pt_cloud_sizes) <= 2**16 - 1:
            arr = np.zeros(vector_length, dtype=np.uint16)
        else:
            raise ValueError(f"Creating segmentation map for more than {2**16} point {len(pt_cloud_sizes)} not supported.")

        return cls(arr, name=name)


