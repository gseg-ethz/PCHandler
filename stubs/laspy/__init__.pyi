# SPDX-License-Identifier: BSD-3-Clause
"""Narrow handwritten stub for laspy 2.x — covers only the surface used by pchandler."""

from typing import Any

import numpy as np

class ExtraBytesParams:
    def __init__(
        self,
        name: str,
        type: Any,
        description: str = ...,
        offsets: Any = ...,
        scales: Any = ...,
        no_data: Any = ...,
    ) -> None: ...

class LasData:
    xyz: np.ndarray
    intensity: np.ndarray

    @property
    def header(self) -> Any: ...
    @property
    def _points(self) -> Any: ...
    def change_scaling(self, scales: Any = ..., offsets: Any = ...) -> None: ...
    def add_extra_dim(self, params: ExtraBytesParams) -> None: ...
    def write(self, path: str) -> None: ...
    def __getattr__(self, name: str) -> np.ndarray: ...
    def __setattr__(self, name: str, value: Any) -> None: ...

def read(path: Any) -> LasData: ...
def create(point_format: Any = ..., file_version: Any = ...) -> LasData: ...
