# SPDX-License-Identifier: BSD-3-Clause
"""Narrow handwritten stub for py4dgeo — covers only the surface used by pchandler."""

from typing import Any

import numpy.typing as npt

NDArray = npt.NDArray[Any]

class Epoch:
    def __init__(
        self,
        cloud: NDArray,
        normals: NDArray | None = None,
        additional_dimensions: NDArray | None = None,
        timestamp: Any = None,
        scanpos_info: dict[str, Any] | None = None,
    ) -> None: ...
    @property
    def cloud(self) -> NDArray: ...
    @property
    def normals(self) -> NDArray | None: ...
    @property
    def additional_dimensions(self) -> NDArray: ...
