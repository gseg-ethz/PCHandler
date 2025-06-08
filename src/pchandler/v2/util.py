"""
`pchandler.util`

This module provides utility functions and constants for angle conversion and numerical operations,
along with an enumeration for specifying angle units.

Features:
---------
- **AngleUnit Enum**: Defines supported angular units: radians (RAD), degrees (DEGREE), and gradians (GON).
- **Angle Conversion**: Convert angles between radians, degrees, and gradians efficiently.
- **Numerical Constants**: Includes `EPS`, the smallest representable positive number for 32-bit floating-point values.

Dependencies:
-------------
- ``numpy``: Used for numerical computations and array operations.

Usage:
------
Typical usage patterns include:
1. Converting an array of angles from degrees to radians:

.. code-block:: python

    from pchandler.util import convert_angles, AngleUnit
    angles_deg = np.array([0, 90, 180, 360])
    angles_rad = convert_angles(angles_deg, AngleUnit.DEGREE, AngleUnit.RAD)

2. Accessing the smallest floating-point epsilon for numerical stability:

.. code-block:: python

    from pchandler.util import EPS
    print(f"Floating-point epsilon: {EPS}")
"""

import logging
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__.split(".")[0])

EPS = np.finfo(np.float32).eps
"""
The smallest positive number such that `1.0 + EPS != 1.0` for 32-bit floating-point values.

Type
----
float
"""


class AngleUnit(Enum):
    """
    An enumeration for angular units.

    Attributes
    ----------
    RAD : str
        Radians, the standard angular unit in mathematical computations.
    DEGREE : str
        Degrees, commonly used in geographic and engineering applications.
    GON : str
        Gradians (also known as gons), where a full circle is divided into 400 units.

    Usage
    -----
    This enum is used to specify the angular unit for conversions and operations.
    """

    RAD = "rad"
    DEGREE = "deg"
    GON = "gon"


def convert_angles(
    values: np.ndarray, source_unit: AngleUnit, target_unit: AngleUnit, out: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Converts an array of angles from one unit to another.

    Parameters
    ----------
    values : np.ndarray
        The array of angles to convert.
    source_unit : AngleUnit
        The unit of the input angles (e.g., `AngleUnit.RAD`, `AngleUnit.DEGREE`, `AngleUnit.GON`).
    target_unit : AngleUnit
        The unit to convert the angles to.
    out : Optional[np.ndarray], default=None
        An optional output array to store the results. If provided, must be the same shape as `values`.

    Returns
    -------
    np.ndarray
        The converted angles in the target unit.

    Notes
    -----
    - If `source_unit` and `target_unit` are the same, the function returns a copy of the input `values` unless
      `out` is provided, in which case it writes the result to `out`.
    - Supported conversions:
      - Radians ↔ Degrees
      - Radians ↔ Gradians
      - Degrees ↔ Gradians

    Examples
    --------
    Convert an array of angles from degrees to radians:
    >>> import numpy as np
    >>> from pchandler.util import convert_angles, AngleUnit
    >>> angles_deg = np.array([0, 90, 180, 360])
    >>> convert_angles(angles_deg, AngleUnit.DEGREE, AngleUnit.RAD)
    array([0.        , 1.57079633, 3.14159265, 6.28318531])

    Convert angles from radians to gradians:
    >>> angles_rad = np.array([0, np.pi/2, np.pi, 2*np.pi])
    >>> convert_angles(angles_rad, AngleUnit.RAD, AngleUnit.GON)
    array([  0., 100., 200., 400.])
    """
    if source_unit == target_unit:
        if out is None:
            return values.copy()
        else:
            out = values
            return out

    match source_unit:
        case AngleUnit.RAD:
            match target_unit:
                case AngleUnit.DEGREE:
                    return np.rad2deg(values, out=out)
                case AngleUnit.GON:
                    return np.multiply(values, 200 / np.pi, out=out)
        case AngleUnit.DEGREE:
            match target_unit:
                case AngleUnit.RAD:
                    return np.deg2rad(values, out=out)
                case AngleUnit.GON:
                    return np.multiply(values, 200 / 180, out=out)
        case AngleUnit.GON:
            match target_unit:
                case AngleUnit.RAD:
                    return np.multiply(values, np.pi / 200, out=out)
                case AngleUnit.DEGREE:
                    return np.multiply(values, 180 / 200, out=out)


def cartesian_to_spherical(xyz: np.ndarray, origin: np.ndarray = None) -> np.ndarray:
    """
    Converts Cartesian coordinates to spherical coordinates (range, elevation, azimuth).

    Parameters
    ----------
    xyz : np.ndarray
        An (N x 3) array of Cartesian coordinates.
    origin : np.ndarray, optional
        An optional (3,) array to subtract from xyz before conversion.

    Returns
    -------
    np.ndarray
        An (N x 3) array of spherical coordinates.
    """
    if origin is not None:
        xyz = xyz - origin
    sph = np.zeros_like(xyz, dtype=np.float32)
    xy_sq = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
    sph[:, 0] = np.sqrt(xy_sq + xyz[:, 2] ** 2)
    sph[:, 1] = np.arctan2(np.sqrt(xy_sq), xyz[:, 2])
    sph[:, 2] = -np.arctan2(xyz[:, 1], xyz[:, 0])
    return sph


def spherical_to_cartesian(spherical_coords: np.ndarray, origin: np.ndarray = None) -> np.ndarray:
    """
    Converts spherical coordinates (range, elevation, azimuth) to Cartesian coordinates.

    Parameters
    ----------
    spherical_coords : np.ndarray
        An (N x 3) array of spherical coordinates.
    origin : np.ndarray, optional
        An optional (3,) array to add to the result.

    Returns
    -------
    np.ndarray
        An (N x 3) array of Cartesian coordinates.
    """
    xyz = np.zeros((spherical_coords.shape[0], 3), dtype=np.float32)
    xyz[:, 0] = spherical_coords[:, 0] * np.sin(spherical_coords[:, 1]) * np.cos(spherical_coords[:, 2])
    xyz[:, 1] = -spherical_coords[:, 0] * np.sin(spherical_coords[:, 1]) * np.sin(spherical_coords[:, 2])
    xyz[:, 2] = spherical_coords[:, 0] * np.cos(spherical_coords[:, 1])
    if origin is not None:
        xyz += origin
    return xyz


def unique_rows_fast(bin_idx: np.ndarray):
    """
    bin_idx: 2D int32 array of shape (N, D)
    returns (unique_rows, inverse_indices) exactly like
    np.unique(bin_idx, axis=0, return_inverse=True)
    but ~5–10× faster for large N.
    """
    # make sure data is contiguous so the view trick works
    arr = np.ascontiguousarray(bin_idx)

    # view each row as a single opaque blob of bytes
    byte_dt = np.dtype((np.void, arr.dtype.itemsize * arr.shape[1]))
    blob = arr.view(byte_dt).ravel()

    # unique over the 1D blob array
    uniq_blob, inv = np.unique(blob, return_inverse=True)

    # turn the blobs back into an (M, D) int32 array
    uniq = uniq_blob.view(arr.dtype).reshape(-1, arr.shape[1])

    return uniq, inv
