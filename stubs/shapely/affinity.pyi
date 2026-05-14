# SPDX-License-Identifier: BSD-3-Clause
"""Narrow handwritten stub for shapely.affinity — covers only the surface used by pchandler."""

from typing import Any

def translate(geom: Any, xoff: Any = 0, yoff: Any = 0, zoff: Any = 0) -> Any: ...
def scale(geom: Any, xfact: Any = 1, yfact: Any = 1, zfact: Any = 1, origin: Any = "center") -> Any: ...
