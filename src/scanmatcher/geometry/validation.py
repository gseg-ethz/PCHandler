from __future__ import annotations

import numpy as np
from typing import TypeAlias

from src.scanmatcher.validation import check_in_range

NumOrArray: TypeAlias  = np.ndarray|float|int|list|tuple



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