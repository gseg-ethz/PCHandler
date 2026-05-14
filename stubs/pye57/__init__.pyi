# SPDX-License-Identifier: BSD-3-Clause
"""Narrow handwritten stub for pye57 0.4.x — covers only the surface used by pchandler."""

from typing import Any

import numpy as np

class ScanHeader:
    point_fields: list[str]
    has_pose: bool
    rotation: np.ndarray
    translation: np.ndarray
    rotation_matrix: np.ndarray

class E57:
    scan_count: int

    def __init__(self, path: str, mode: str = "r") -> None: ...
    def get_header(self, index: int | None) -> ScanHeader: ...
    def read_scan(
        self,
        index: int | None,
        *,
        intensity: bool = False,
        colors: bool = False,
        row_column: bool = False,
        transform: bool = True,
        ignore_missing_fields: bool = False,
    ) -> dict[str, np.ndarray]: ...
    def write_scan_raw(
        self,
        data: dict[str, np.ndarray],
        *,
        name: str | None = None,
        rotation: np.ndarray | None = None,
        translation: np.ndarray | None = None,
        scan_header: ScanHeader | None = None,
    ) -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> "E57": ...
    def __exit__(self, *args: Any) -> None: ...
