# SPDX-License-Identifier: BSD-3-Clause
"""Narrow handwritten stub for open3d.t.geometry — tensor-based PointCloud class."""

from typing import Any, Iterable

from open3d import core

class TensorMap:
    positions: core.Tensor

    def items(self) -> Iterable[tuple[str, core.Tensor]]: ...
    def __setattr__(self, name: str, value: Any) -> None: ...

class PointCloud:
    point: TensorMap
