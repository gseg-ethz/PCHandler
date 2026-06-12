# SPDX-License-Identifier: BSD-3-Clause
"""Narrow handwritten stub for shapely.geometry — Polygon and MultiPolygon."""

from typing import Any

class Polygon:
    def __init__(self, shell: Any = None, holes: Any = None) -> None: ...
    @property
    def exterior(self) -> Any: ...

class MultiPolygon:
    def __init__(self, polygons: Any = None) -> None: ...
