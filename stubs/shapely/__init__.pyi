# SPDX-License-Identifier: BSD-3-Clause
"""Narrow handwritten stub for shapely 2.x — covers only the surface used by pchandler."""

from typing import Any

import numpy.typing as npt
from shapely.geometry import Polygon as Polygon

NDArray = npt.NDArray[Any]

def contains_xy(geom: Any, x: Any, y: Any = ...) -> NDArray: ...
