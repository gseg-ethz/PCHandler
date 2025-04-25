import numpy as np
from typing import TypeAlias

NumOrArray: TypeAlias  = np.ndarray|float|int|list|tuple


def check_in_range(value: NumOrArray, target_min: float, target_max: float):
    value: np.ndarray = np.asarray(value)
    val_min: float|int = value.min()
    val_max: float|int = value.max()

    if (val_min < target_min) and (val_max > target_max):
        raise ValueError(f'Min and max values [{val_min},{val_max}] exceeds bounds [{target_min},{target_max}].')

    elif val_min < target_min:
        raise ValueError(f'Min value {val_min} exceeds lower limit {target_min}.')

    elif val_max > target_max:
        raise ValueError(f'Max value {val_max} exceeds upper limit {target_max}.')

def check_hz_angles(array: NumOrArray):
    check_in_range(array, -np.pi, np.pi)

def check_zenith_angles(array: NumOrArray):
    check_in_range(array, 0, np.pi)

def check_azimuth_angles(array: NumOrArray):
    check_in_range(array, 0, 2*np.pi)

def check_radial_distances(array: NumOrArray):
    check_in_range(array, 0, np.inf)

def check_inclination_angles(array: NumOrArray):
    check_in_range(array, -np.pi/2, np.pi/2)

def check_spherical_coordinates(array: np.ndarray):
    check_radial_distances(array[:, 0])
    check_zenith_angles(array[:, 1])
    check_hz_angles(array[:, 2])
