# SPDX-License-Identifier: BSD-3-Clause
"""Narrow handwritten stub for open3d.core — Tensor class."""

import numpy as np

class Tensor:
    def __init__(self, arr: np.ndarray) -> None: ...
    def numpy(self) -> np.ndarray: ...
