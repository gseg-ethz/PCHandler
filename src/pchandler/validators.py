
import logging
import numpy as np
from numpy.typing import DTypeLike, ArrayLike

from .constants import PI, TWO_PI, HALF_PI

logger = logging.getLogger(__name__.split(".")[0])

def validate_spherical_angles(array: np.ndarray) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        raise TypeError(f'Input values should be an ndarray not : {type(array)}')

    array[:, 0] = validate_radius(array[:, 0])
    array[:, 1] = validate_horizontal_angles(array[:, 1])
    array[:, 2] = validate_zenith_angles(array[:, 2])
    return array

def validate_radius(array: np.ndarray) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        raise TypeError(f'Input values should be an ndarray not : {type(array)}')

    if np.any(array < 0):
        raise ValueError("Radius must be positive")
    return array

def validate_azimuth_angles(array: np.ndarray) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        raise TypeError(f'Input values should be an ndarray not : {type(array)}')

    if 0 <= array.min() and array.max() <= TWO_PI:
        return array
    else:
        if -PI <= array.min() and array.max() <= PI:
            arr_min, arr_max = -PI, PI
        else:
            arr_min, arr_max = array.min(), array.max()

        raise ValueError(f'Azimuths must be between [0, 2*pi] not [{arr_min}, {arr_max}]')

def validate_horizontal_angles(array: np.ndarray) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        raise TypeError(f'Input values should be an ndarray not : {type(array)}')

    if -PI <= array.min() and array.max() <= PI:
        return array
    else:
        if 0 <= array.min() and array.max() <= PI * 2:
            arr_min, arr_max = -PI, PI
        else:
            arr_min, arr_max = array.min(), array.max()

        raise ValueError(f'Horizontal angles must be between [-pi, +pi] not [{arr_min}, {arr_max}]')


def validate_zenith_angles(array: np.ndarray) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        raise TypeError(f'Input values should be an ndarray not : {type(array)}')

    if 0 <= array.min() and array.max() <= PI:
        return array
    else:
        if -HALF_PI <= array.min() and array.max() <= HALF_PI:
            raise ValueError("Input Angles in [-pi/2, +pi/2] but should be [0, +pi]")
        raise ValueError(f'Zenith angles should be in [0, +pi] not [{array.min()}, {array.max()}]')


def validate_inclination_angles(array: np.ndarray) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        raise TypeError(f'Input values should be an ndarray not : {type(array)}')

    if -HALF_PI <= array.min() and array.max() <= HALF_PI:
        return array
    else:
        if 0 <= array.min() and array.max() <= PI:
            array_min, array_max = 0, PI
        else:
            array_min, array_max = array.min(), array.max()
        raise ValueError(f'Inclination angles should be between [-pi/2, +pi/2] not [{array_min}, {array_max}]')

def coerce_wrapped_azimuth_angles(array: np.ndarray) -> np.ndarray:
    array[array < 0] += TWO_PI
    array[array > TWO_PI] -= TWO_PI
    return array

def coerce_wrapped_horizontal_angles(array: np.ndarray) -> np.ndarray:
    array[array <= -PI] += TWO_PI
    array[array > PI] -= TWO_PI
    return array


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
    normalised = (array - origin_min) / (origin_max - origin_min)

    if np.issubdtype(target_dtype, np.floating):
        return normalised.astype(np.float32)

    mapped = np.floor(normalised * float(target_max - target_min) + target_min)
    return mapped.astype(target_dtype).flatten()


def extract_array(value: np.ndarray | tuple[np.ndarray | object] | object | dict[str, np.ndarray]) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value

    if hasattr(value, 'arr'):
        return value.arr

    if isinstance(value, tuple):
        if len(value) != 1:
            raise TypeError(f'Value to unpack from a tuple > 1 is ambiguous: {value}')

        if isinstance(value[0], np.ndarray):
            return value[0]
        elif hasattr(value[0], 'arr'):
            return value[0].arr
        else:
            raise TypeError(f'Input value is an unsupported type: {type(value[0])} ')
    elif isinstance(value, dict):
        if 'arr' in value:
            return value['arr']
        else:
            raise TypeError(f"'arr' is not in the passed dictionary.")
    else:
        raise TypeError(f'Input value is an unsupported type: {type(value)} ')

def normalize_array(array: np.ndarray,
                    lower: float|np.ndarray|None = None,
                    upper: float|np.ndarray|None = None) -> np.ndarray:
    """
    Normalise array by default to the bounds [0, 1] alternatively a custom range as defined by lower and upper.

    Clipping will be applied if a custom bound is provided and values exceed this bound.
    """

    original_dtype = array.dtype
    array = array.astype(np.float64)

    clip = lower is not None or upper is not None

    lower = array.min(axis=0) if lower is None else lower
    upper = array.max(axis=0) if upper is None else upper

    if isinstance(lower, np.ndarray):
        if np.any(lower >= upper):
            raise ValueError("lower must be less than upper")
    elif lower >= upper:
        raise ValueError("lower must be less than upper")

    np.divide(array-lower, upper-lower, out=array)
    if clip:
        np.clip(array, a_min=lower, a_max=upper, out=array)

    return array.astype(original_dtype)

def normalise_to_integer_dtype(array: np.ndarray, name: str, target_integer_dtype: DTypeLike) -> np.ndarray:
    """
    Normalise values to the min and max values associated with the existing dtype
    Parameters
    ----------
    array
    name
    target_integer_dtype

    Returns
    -------

    """

    if array.dtype == target_integer_dtype:
        logger.debug(f"Scalar field `{name}` hasn't changed. No operation performed.")
        return array

    if np.dtype(target_integer_dtype).kind not in ["u", "i"]:
        logger.debug(f"Scalar field `{name}` is floating. Not converting to dtype limits. Consider [0.0, 1.0].")
        return array

    lower = np.iinfo(target_integer_dtype).min
    upper = np.iinfo(target_integer_dtype).max

    return normalize_array(array=array, lower=lower, upper=upper)

# TODO ensure error thrown with single point but a get_point function exists
def validate_transposed_vector(array: np.ndarray) -> np.ndarray:
    return np.atleast_1d(array.squeeze())

def validate_n_by_3_transposed(array: np.ndarray) -> np.ndarray:
    return validate_transposed(array, cols=3)

def validate_n_by_2_transposed(array: np.ndarray) -> np.ndarray:
    return validate_transposed(array, cols=2)

def validate_transposed(array: np.ndarray, cols: int) -> np.ndarray:
    if array.ndim != 2:
        raise ValueError(f"Input array must be 2-dimensional of Nx{cols} or {cols}xN shape. Received: {array.shape}")

    if array.shape[1] == cols:
        return array

    if array.shape[0] == cols and array.shape[1] != cols:
        return array.T
    else:
        raise ValueError(f"Array does not appear to be an Nx{cols} array nor it's transpose.")


def check_in_range(value: ArrayLike, target_min: float, target_max: float) -> None:
    value = np.asarray(value)
    val_min: float | int = value.min()
    val_max: float | int = value.max()

    if (val_min < target_min) and (val_max > target_max):
        raise ValueError(f'Min and max values [{val_min},{val_max}] exceeds bounds [{target_min},{target_max}].')

    elif val_min < target_min:
        raise ValueError(f'Min value {val_min} exceeds lower limit {target_min}.')

    elif val_max > target_max:
        raise ValueError(f'Max value {val_max} exceeds upper limit {target_max}.')
