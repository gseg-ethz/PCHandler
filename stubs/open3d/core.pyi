# SPDX-License-Identifier: BSD-3-Clause
"""Narrow handwritten stub for open3d.core — Tensor class."""

from typing import Any

import numpy.typing as npt

NDArray = npt.NDArray[Any]

class Tensor:
    def __init__(self, arr: NDArray) -> None: ...
    def numpy(self) -> NDArray: ...
