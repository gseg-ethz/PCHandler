from __future__ import annotations

import logging
from typing import Optional, Any, TypeVar, TYPE_CHECKING

import numpy as np
from numpy import typing as npt

from .constants import HALF_PI, PI, TWO_PI

if TYPE_CHECKING:
    from pchandler.geometry import AbstractScalarField

logger = logging.getLogger(__name__.split(".")[0])

T = TypeVar('T', bound=npt.DTypeLike)


def validate_spherical_angles(array: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Input values should be an ndarray not : {type(array)}")

    array[:, 0] = validate_radius(array[:, 0])
    array[:, 1] = validate_horizontal_angles(array[:, 1])
    array[:, 2] = validate_zenith_angles(array[:, 2])
    return array


def validate_radius(array: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Input values should be an ndarray not : {type(array)}")

    if np.any(array < 0):
        raise ValueError("Radius must be positive")
    return array


def validate_azimuth_angles(array: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Input values should be an ndarray not : {type(array)}")

    if 0 <= array.min() and array.max() <= TWO_PI:
        return array
    else:
        if -PI <= array.min() and array.max() <= PI:
            arr_min, arr_max = -PI, PI
        else:
            arr_min, arr_max = array.min(), array.max()

        raise ValueError(f"Azimuths must be between [0, 2*pi] not [{arr_min}, {arr_max}]")


def validate_horizontal_angles(array: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Input values should be an ndarray not : {type(array)}")

    if -PI <= array.min() and array.max() <= PI:
        return array
    else:
        if 0 <= array.min() and array.max() <= PI * 2:
            arr_min, arr_max = -PI, PI
        else:
            arr_min, arr_max = array.min(), array.max()

        raise ValueError(f"Horizontal angles must be between [-pi, +pi] not [{arr_min}, {arr_max}]")


def validate_zenith_angles(array: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Input values should be an ndarray not : {type(array)}")

    if 0 <= array.min() and array.max() <= PI:
        return array
    else:
        if -HALF_PI <= array.min() and array.max() <= HALF_PI:
            raise ValueError("Input Angles in [-pi/2, +pi/2] but should be [0, +pi]")
        raise ValueError(f"Zenith angles should be in [0, +pi] not [{array.min()}, {array.max()}]")


def validate_inclination_angles(array: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Input values should be an ndarray not : {type(array)}")

    if -HALF_PI <= array.min() and array.max() <= HALF_PI:
        return array
    else:
        if 0 <= array.min() and array.max() <= PI:
            array_min, array_max = 0, PI
        else:
            array_min, array_max = array.min(), array.max()
        raise ValueError(f"Inclination angles should be between [-pi/2, +pi/2] not [{array_min}, {array_max}]")


def coerce_wrapped_azimuth_angles(array: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    array[array < 0] += TWO_PI
    array[array > TWO_PI] -= TWO_PI
    return array


def coerce_wrapped_horizontal_angles(array: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    array[array <= -PI] += TWO_PI
    array[array > PI] -= TWO_PI
    return array



def validate_transposed_2d_array(array: npt.NDArray[Any], cols: int) -> npt.NDArray[Any]:
    """
    Validats the  transpose of 2D arrays with known fixed cols
    :param array:
    :param cols:
    :return:
    """
    if array.ndim != 2:
        raise ValueError(f"Input array must be 2-dimensional of Nx{cols} or {cols}xN shape. Received: {array.shape}")

    if array.shape[1] == cols:
        return array

    if array.shape[0] == cols and array.shape[1] != cols:
        return array.T
    else:
        raise ValueError(f"Array does not appear to be an Nx{cols} array nor it's transpose.")


def check_in_range(value: npt.ArrayLike|npt.NDArray[Any], target_min: float, target_max: float) -> None:
    value = np.asarray(value)
    val_min: float | int = value.min()
    val_max: float | int = value.max()

    if (val_min < target_min) and (val_max > target_max):
        raise ValueError(f"Min and max values [{val_min},{val_max}] exceeds bounds [{target_min},{target_max}].")

    elif val_min < target_min:
        raise ValueError(f"Min value {val_min} exceeds lower limit {target_min}.")

    elif val_max > target_max:
        raise ValueError(f"Max value {val_max} exceeds upper limit {target_max}.")


def normalize_min_max(array: npt.NDArray,
                      lower: float|int|np.number,
                      upper: float|int|np.number,
                      target_dtype: T,
                      v_min: Optional[float|int] = None,
                      v_max: Optional[float|int] = None) -> npt.NDArray[T]:

    if (not np.issubdtype(array.dtype, np.floating) and
            not np.issubdtype(array.dtype, np.integer) and
            not np.issubdtype(array.dtype, np.bool)):
        raise TypeError(f"Cannot convert numpy array of type {array.dtype}")

    array = array.astype(np.float64)

    if v_min is None:
        v_min = array.min()

    if v_max is None:
        v_max = array.max()

    array = (array - v_min) / (v_max - v_min)
    array = np.add(array * (upper - lower), lower)
    return np.clip(array, lower, upper).astype(target_dtype)


def linear_map_dtype(array: npt.NDArray[Any], target_dtype: npt.DTypeLike) -> npt.NDArray[Any]:

    def get_dtype_min_max(dt: np.dtype) -> tuple[float, float]:
        if np.issubdtype(dt, np.integer):
            return np.iinfo(dt).min, np.iinfo(dt).max
        elif np.issubdtype(dt, np.floating):
            return 0.0, 1.0
        else:
            raise TypeError(f"Invalid dtype detected: {dt}")

    # Types match, exit
    if array.dtype == target_dtype:
        return array

    # Get the corresponding min and max from the type info
    origin_min, origin_max = get_dtype_min_max(array.dtype)
    target_min, target_max = get_dtype_min_max(target_dtype)

    return normalize_min_max(array=array,
                             lower=target_min,
                             upper=target_max,
                             target_dtype=target_dtype,
                             v_min=origin_min,
                             v_max=origin_max)


def normalize_self(array: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """
    Normalise values to the min and max values of the associated data type or [0, 1] for floating point
    """
    if np.dtype(array.dtype).kind not in ["u", "i"]:
        logger.debug(f"Scalar field is floating. Converting to [0.0, 1.0].")
        lower, upper = 0, 1
    else:
        lower, upper = np.iinfo(array.dtype).min, np.iinfo(array.dtype).max

    return normalize_min_max(array, lower, upper, array.dtype)


def _normalize_base(array: npt.NDArray[Any], dtype: npt.DTypeLike) -> npt.NDArray[Any]:
    if hasattr(array, 'arr'):
        array = array.arr

    if array.dtype != dtype:
        if np.issubdtype(dtype, np.floating):
            return normalize_min_max(array, 0, 1, dtype)
        if 0 <= array.min() <= array.max() <= 1:
            return normalize_min_max(array, np.iinfo(dtype).min, np.iinfo(dtype).max, dtype, 0, 1)
        return normalize_min_max(array, np.iinfo(dtype).min, np.iinfo(dtype).max, dtype)
    return array


normalize_uint8 = lambda array: _normalize_base(array, np.uint8)
normalize_uint16 = lambda array: _normalize_base(array, np.uint16)
normalize_int8 = lambda array: _normalize_base(array, np.int8)
normalize_int16 = lambda array: _normalize_base(array, np.int16)
normalize_int32 = lambda array: _normalize_base(array, np.int32)
normalize_int64 = lambda array: _normalize_base(array, np.int64)
normalize_float32 = lambda array: _normalize_base(array, np.float32)
normalize_float64 = lambda array: _normalize_base(array, np.float64)


def ensure_unit_vector(array: npt.NDArray[Any]) -> npt.NDArray[Any]:
    if not (np.issubdtype(array.dtype, np.floating) or np.issubdtype(array.dtype, np.signedinteger)):
        raise TypeError("Dtype of normals array must be of type floating or signed integer}")

    array /= np.linalg.norm(array, axis=1).reshape(-1, 1)
    array = array.astype(np.float32)

    return array
