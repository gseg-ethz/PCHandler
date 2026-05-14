# pchandler – Toolbox for point-cloud handling, processing and analysis
#
# Copyright (c) 2022–2026 ETH Zurich
# Department of Civil, Environmental and Geomatic Engineering (D-BAUG)
# Institute of Geodesy and Photogrammetry
# Geosensors and Engineering Geodesy
#
# Authors:
#   Nicholas Meyer
#   Jon Allemand
#
# SPDX-License-Identifier: BSD-3-Clause

# pchandler - Toolbox for point-cloud handling, processing and analysis
#
# Copyright (c) 2025, Nicholas Meyer, Geosensors and Engineering Geodesy,
# Institute of Geodesy and Photogrammetry, ETH Zurich, Switzerland
# SPDX-License-Identifier: BSD-3-Clause
#
# Author: Nicholas Meyer (meyernic@ethz.ch)

"""Statistical outlier removal filters."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated

import numpy as np
from GSEGUtils.base_types import Vector_Bool_T
from pydantic import Field, PositiveInt

from pchandler import PointCloudData
from pchandler._optional import ensure_open3d_available
from pchandler.filters import PointCloudFilter

if TYPE_CHECKING:
    import open3d as o3d

logger = logging.getLogger(__name__.split(".")[0])


class BaseOutlierFilter(PointCloudFilter):
    """Base class for filters that remove statistical outliers via Open3D."""

    def __init__(self, std_ratio: Annotated[float, Field(gt=0, le=1)] = 0.95, number_of_neighbours: PositiveInt = 13):
        """Build a statistical-outlier-removal filter.

        The underlying algorithm uses a standard-deviation ratio and a
        configurable neighborhood size.

        Parameters
        ----------
        std_ratio : float, default=0.95
            Standard-deviation ratio threshold (``0 < std_ratio <= 1``).
        number_of_neighbours : int, default=13
            Number of nearest neighbors used to estimate local statistics.
        """
        ensure_open3d_available()
        self.std_ratio = std_ratio
        self.number_of_neighbours = number_of_neighbours

    def mask(self, pcd: o3d.geometry.PointCloud) -> Vector_Bool_T:
        """Create a boolean mask of inliers (non-outlier points).

        Parameters
        ----------
        pcd : o3d.geometry.PointCloud
            Open3D point cloud to evaluate.

        Returns
        -------
        Vector_Bool_T
            Boolean mask, ``True`` for inliers.
        """
        mask = np.zeros(len(pcd.points), dtype=np.bool_)
        _, inliers = pcd.remove_statistical_outlier(self.number_of_neighbours, self.std_ratio, True)
        mask[inliers] = True
        return mask


class SphericalOutlierFilter(BaseOutlierFilter):
    """Outlier filter for point clouds in spherical coordinates."""

    def mask(self, pcd: PointCloudData) -> Vector_Bool_T:
        """Create a boolean mask of inliers using spherical-coordinate statistics.

        Parameters
        ----------
        pcd : PointCloudData
            Source point cloud.

        Returns
        -------
        Vector_Bool_T
            Boolean mask, ``True`` for inliers.
        """
        ensure_open3d_available()
        import open3d as _o3d  # noqa: PLC0415

        sp_pcd = _o3d.geometry.PointCloud()
        sp_pcd.points = _o3d.utility.Vector3dVector(
            np.hstack((pcd.spher[:, [1, 2]], np.zeros((len(pcd), 1), dtype=np.float32)))
        )
        return super().mask(sp_pcd)


class CartesianOutlierFilter(BaseOutlierFilter):
    """Outlier filter for point clouds in cartesian coordinates."""

    def mask(self, pcd: PointCloudData) -> Vector_Bool_T:
        """Create a boolean mask of inliers using cartesian-coordinate statistics.

        Parameters
        ----------
        pcd : PointCloudData
            Source point cloud.

        Returns
        -------
        Vector_Bool_T
            Boolean mask, ``True`` for inliers.
        """
        return super().mask(pcd.to_o3d())
