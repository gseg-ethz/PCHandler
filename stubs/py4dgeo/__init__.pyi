# SPDX-License-Identifier: BSD-3-Clause
"""Narrow handwritten stub for py4dgeo — covers only the surface used by pchandler."""

from typing import Any

import numpy as np

class Epoch:
    def __init__(
        self,
        cloud: np.ndarray,
        normals: np.ndarray | None = None,
        additional_dimensions: np.ndarray | None = None,
        timestamp: Any = None,
        scanpos_info: dict[str, Any] | None = None,
    ) -> None: ...
    @property
    def cloud(self) -> np.ndarray: ...
    @property
    def normals(self) -> np.ndarray | None: ...
    @property
    def additional_dimensions(self) -> np.ndarray: ...
