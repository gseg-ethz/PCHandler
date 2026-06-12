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

"""Spherical-coordinate-based filters."""

import logging

import numpy as np
from GSEGUtils.base_types import Vector_Bool_T
from shapely import contains_xy
from shapely.geometry import Polygon

from pchandler import PointCloudData
from pchandler.filters import PointCloudFilter
from pchandler.geometry.spherical import FoV

logger = logging.getLogger(__name__.split(".")[0])


class FoVFilter(PointCloudFilter):
    """Filter points by a field-of-view (FoV) angular region."""

    def __init__(self, fov: FoV):
        """Filter points based on a given field of view (FoV).

        Parameters
        ----------
        fov : FoV
            Field of view object defining the top, bottom, left, and right
            boundaries.
        """
        self.fov = fov

    def mask(self, pcd: PointCloudData) -> Vector_Bool_T:
        """Create a boolean mask for a point cloud based on a given field of view (FoV).

        Parameters
        ----------
        pcd : PointCloudData

        Returns
        -------
        Vector_Bool_T
        """
        return self.fov.find_points_inside(horizontal=pcd.hz, vertical=pcd.v)


class RangeFilter(PointCloudFilter):
    """Filter points by a minimum/maximum spherical-range threshold."""

    def __init__(self, low: float = 0.0, high: float = np.inf):
        """Filter points based on minimum and maximum range thresholds.

        Parameters
        ----------
        low : float, default=0.0
            Lower (inclusive) range threshold.
        high : float, default=``np.inf``
            Upper (inclusive) range threshold.
        """
        self.low = low
        self.high = high

    def mask(self, pcd: PointCloudData) -> Vector_Bool_T:
        """Create a boolean mask for a point cloud based on given range limits.

        Parameters
        ----------
        pcd : PointCloudData

        Returns
        -------
        Vector_Bool_T
        """
        mask = np.logical_and(pcd.r >= self.low, pcd.r <= self.high)
        return mask


class SphericalPolygonFilter(PointCloudFilter):
    """Filter points by a polygon defined in spherical (horizontal × vertical) coordinates."""

    def __init__(self, polygon: Polygon):
        """Filter points based on a polygon defined in spherical coordinates.

        Parameters
        ----------
        polygon : Polygon
            Polygon defining the filter region in (horizontal, vertical)
            spherical-angle coordinates.
        """
        self.polygon = polygon

    def mask(self, pcd: PointCloudData) -> Vector_Bool_T:
        """Create a boolean mask for a point cloud based on a polygon defined in spherical coordinates.

        Points inside the polygon are marked as `True`.

        Parameters
        ----------
        pcd : PointCloudData

        Returns
        -------
        Vector_Bool_T
        """
        mask = contains_xy(self.polygon, pcd.hz, pcd.v)
        return mask
