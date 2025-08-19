# pchandler - Toolbox for point-cloud handling, processing and analysis
#
# Copyright (c) 2025, Nicholas Meyer, Geosensors and Engineering Geodesy,
# Institute of Geodesy and Photogrammetry, ETH Zurich, Switzerland
# SPDX-License-Identifier: BSD-3-Clause
#
# Author: Nicholas Meyer (meyernic@ethz.ch)

"""
GPU module for pchandler.filters

Provides functions that use GPU acceleration (via cudf and cuspatial) to filter point clouds.
"""

import copy
import gc
import logging
from typing import Optional

_HAS_GPU: bool = False
_GPU_IMPORT_ERROR: Optional[Exception] = None

try:
    import cudf
    import cuspatial
    import geopandas as gpd
except (ImportError, RuntimeError) as e:
    _GPU_IMPORT_ERROR = e
else:
    _HAS_GPU = True

from GSEGUtils.constants import validate_variables
from numpy.typing import NDArray
from shapely.geometry import Polygon

from pchandler import PointCloudData
from pchandler.filters.cartesian_filters import PlaneStrings
from pchandler.filters.core import PointCloudFilter, ValidatedPolygonT

logger = logging.getLogger(__name__.split(".")[0])


def is_available() -> bool:
    """Quick check whether the GPU stack was successfully imported."""
    return _HAS_GPU


def ensure_available():
    """Raise a clear ImportError if GPU support isn’t actually usable."""
    if not _HAS_GPU:
        msg = (
            "GPU support is not available. "
            "Install with `pip install pchandler[cuda11]` or `pip install pchandler[cuda12]`."
        )
        if _GPU_IMPORT_ERROR is not None:
            raise ImportError(msg) from _GPU_IMPORT_ERROR
        else:
            raise ImportError(msg)


class PolygonFilterGPU(PointCloudFilter):
    @validate_variables
    def __init__(self, polygon: ValidatedPolygonT, plane: PlaneStrings = "xy") -> None:
        ensure_available()
        self.polygon: Polygon = polygon
        self.plane = plane

    def mask(self, pcd: PointCloudData) -> NDArray:
        # if self.plane not in ('xy', 'xz', 'yz'): # Handled by validation
        #     raise ValueError("Invalid plane. Choose 'xy', 'xz', or 'yz'.")

        proj_pts = cudf.DataFrame(
            {axis_char: getattr(pcd, axis_char).astype(float) for axis_char in self.plane}
        ).interleave_columns()

        polygon = copy.deepcopy(self.polygon)
        if pcd.numerical_optimization_shift is not None:
            if self.plane == "xy":
                global_shift = -pcd.numerical_optimization_shift.value[:2]
            elif self.plane == "xz":
                global_shift = -pcd.numerical_optimization_shift.value[[0, 2]]
            else:
                global_shift = -pcd.numerical_optimization_shift.value[1:]
            polygon = gpd.GeoSeries([polygon]).translate(*global_shift).iloc[0]

        polygon_gpu = cuspatial.GeoSeries(gpd.GeoSeries([polygon]))
        proj_pts_gs = cuspatial.GeoSeries.from_points_xy(proj_pts)
        proj_pts_in = cuspatial.point_in_polygon(proj_pts_gs, polygon_gpu)
        mask = proj_pts_in[0].to_numpy()

        del polygon_gpu, proj_pts, proj_pts_gs, proj_pts_in
        gc.collect()

        return mask


class SphericalPolygonFilterGPU(PointCloudFilter):
    @validate_variables
    def __init__(self, polygon: ValidatedPolygonT):
        ensure_available()
        self.polygon: Polygon = polygon

    def mask(self, pcd: PointCloudData) -> NDArray:
        proj_pts = cudf.DataFrame({"x": pcd.hz.astype(float), "y": pcd.v.astype(float)}).interleave_columns()

        polygon_gpu = cuspatial.GeoSeries(gpd.GeoSeries([self.polygon]))
        proj_pts_gs = cuspatial.GeoSeries.from_points_xy(proj_pts)
        proj_pts_in = cuspatial.point_in_polygon(proj_pts_gs, polygon_gpu)
        mask = proj_pts_in[0].to_numpy()

        del polygon_gpu, proj_pts, proj_pts_gs, proj_pts_in
        gc.collect()

        return mask
