import numpy as np
from numpy import ndarray, floating, asarray, issubdtype


def validate_spherical_angles(array: np.ndarray) -> np.ndarray:
    array[:, 0] = validate_radius(array[:, 0])
    array[:, 1] = validate_azimuthal_angles(array[:, 1])
    array[:, 2] = validate_zenith_angles(array[:, 2])
    return array

def validate_radius(array: np.ndarray) -> np.ndarray:
    if array.min() < 0:
        raise ValueError("Radius must be positive")
    return array

def validate_azimuthal_angles(array: np.ndarray) -> np.ndarray:
    if -np.pi <= array.min() and array.max() <= np.pi:
        return array

    elif 0 <= array.min() and array.max() <= np.pi*2:
        # TODO show warning
        return array - np.pi
    else:
        raise ValueError(f'The range of the azimuthal (horizontal) angles should be in [-pi, pi].'
                         f'Instead the range of [{array.min()}, {array.max()}] was received.')


def validate_zenith_angles(array: np.ndarray) -> np.ndarray:
    if 0 <= array.min() and array.max() <= np.pi:
        return array

    elif -np.pi/2 <= array.min() and array.max() <= np.pi/2:
        # TODO show warning
        return array - np.pi

    else:
        raise ValueError(f'The range of the zenith (vertical) angles should be in [0, pi].'
                         f'Instead the range of [{array.min()}, {array.max()}] was received.')


