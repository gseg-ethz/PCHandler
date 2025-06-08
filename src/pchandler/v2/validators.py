
import logging
import numpy as np
from numpy.typing import DTypeLike

from .custom_types import DataRange

PI = np.pi
TWO_PI = 2 * PI
HALF_PI = 0.5 * PI

logger = logging.getLogger(__name__.split(".")[0])

def validate_spherical_angles(array: np.ndarray) -> np.ndarray:
    array[:, 0] = validate_radius(array[:, 0])
    array[:, 1] = validate_azimuthal_angles(array[:, 1])
    array[:, 2] = validate_zenith_angles(array[:, 2])
    return array

def validate_radius(array: np.ndarray) -> np.ndarray:
    if np.any(array < 0):
        raise ValueError("Radius must be positive")
    return array

def validate_azimuthal_angles(array: np.ndarray) -> np.ndarray:
    if -PI <= array.min() and array.max() <= PI:
        return array
    else:
        if 0 <= array.min() and array.max() <= PI * 2:
            raise ValueError("Azimuthal angles must be between [-pi, +pi] not [0 and +2*pi]. Please convert.")
        raise ValueError(f'Azimuthal angles should be in [-pi, +pi] not [{array.min()}, {array.max()}]')


def validate_zenith_angles(array: np.ndarray) -> np.ndarray:
    if 0 <= array.min() and array.max() <= PI:
        return array
    else:
        if -HALF_PI <= array.min() and array.max() <= HALF_PI:
            raise ValueError("Detected angles in [-pi/2, +pi/2] but should be [0, +pi]")
        raise ValueError(f'Zenith angles should be in [0, +pi] not [{array.min()}, {array.max()}]')

def coerce_wrapped_azimuths(array: np.ndarray) -> np.ndarray:
    array[array <= -PI] += TWO_PI
    array[array > PI] -= TWO_PI
    return array


# TODO need to add some extra error handling / error or warning raises
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
    else:
        raise TypeError(f'Input value is an unsupported type: {type(value)} ')

def normalize_array(array: np.ndarray, name: str,
                    lower: float|np.ndarray|None = None,
                    upper: float|np.ndarray|None = None,
                    operations_performed: list[str]|None = None) -> np.ndarray:

    array = array.astype(np.float64)

    lower = array.min(axis=0) if lower is None else lower
    upper = array.max(axis=0) if upper is None else upper

    if isinstance(lower, np.ndarray):
        if np.any(lower >= upper):
            raise ValueError("lower must be less than upper")
    elif lower >= upper:
        raise ValueError("lower must be less than upper")

    array -= lower
    array /= (upper - lower)

    # TODO remove the operations performed to class
    operations_performed.append(f"normalize: [{lower}, {upper}]")
    logger.debug(f"Normalized scalar field `{name}` from (original) span [{lower}, {upper}] to [0, 1].")
    return array

def normalise_to_dtype_limits(array: np.ndarray, name: str, original_state: DataRange, operations: list) -> np.ndarray:
    if array.dtype == original_state.dtype:
        logger.debug(f"Scalar field `{name}` hasn't changed. No operation performed.")
        return array

    if np.dtype(original_state.dtype).kind not in ["u", "i"]:
        logger.debug(f"Scalar field `{name}` is floating. Not converting to dtype limits. Consider [0.0, 1.0].")
        return array

    lower = np.iinfo(original_state.dtype).min
    upper = np.iinfo(original_state.dtype).max

    return normalize_array(array=array, name=name, lower=lower, upper=upper, operations_performed=operations)

# TODO Decision against supporting squeezing of arrays in case a single point is selected from the array
def validate_transposed_vector(array: np.ndarray) -> np.ndarray:
    return np.atleast_1d(array.squeeze())

def validate_Nx3_transposed(array: np.ndarray) -> np.ndarray:
    return validate_transposed(array, cols=3)

def validate_Nx2_transposed(array: np.ndarray) -> np.ndarray:
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
