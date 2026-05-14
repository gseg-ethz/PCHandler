# SPDX-License-Identifier: BSD-3-Clause
"""Narrow handwritten stub for open3d.utility — Vector3dVector."""

from typing import Any

import numpy as np

class Vector3dVector:
    def __init__(self, arr: np.ndarray) -> None: ...
    def __len__(self) -> int: ...
    def __array__(self, dtype: Any = None) -> np.ndarray: ...
