# SPDX-License-Identifier: BSD-3-Clause
"""Narrow handwritten stub for plyfile 1.x — covers only the surface used by pchandler."""

from typing import Any

import numpy as np

class PlyElement:
    count: int
    properties: list[Any]

    @classmethod
    def describe(
        cls,
        data: np.ndarray,
        name: str,
        comments: list[str] | None = ...,
    ) -> "PlyElement": ...

class PlyData:
    @classmethod
    def read(cls, path: Any) -> "PlyData": ...
    def __init__(self, elements: list[PlyElement], text: bool = False) -> None: ...
    def __getitem__(self, name: str) -> PlyElement: ...
    def write(self, path: Any) -> None: ...
