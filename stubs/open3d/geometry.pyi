# SPDX-License-Identifier: BSD-3-Clause
"""Narrow handwritten stub for open3d.geometry — legacy PointCloud class."""

from . import utility

class PointCloud:
    points: utility.Vector3dVector
    colors: utility.Vector3dVector
    normals: utility.Vector3dVector

    def remove_statistical_outlier(
        self,
        nb_neighbors: int,
        std_ratio: float,
        print_progress: bool = False,
    ) -> tuple["PointCloud", list[int]]: ...
