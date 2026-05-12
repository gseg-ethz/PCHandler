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

"""GPU-accelerated filters backed by cudf and cuspatial."""

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

# Imports placed below the GPU availability probe so that ImportError on optional
# RAPIDS deps is detected before the rest of the module is parsed.
from GSEGUtils.base_types import Vector_Bool_T  # noqa: E402
from GSEGUtils.constants import validate_variables  # noqa: E402
from shapely.geometry import Polygon  # noqa: E402

from pchandler import PointCloudData  # noqa: E402
from pchandler.filters.cartesian_filters import PlaneStrings  # noqa: E402
from pchandler.filters.core import PointCloudFilter, ValidatedPolygonT  # noqa: E402

logger = logging.getLogger(__name__.split(".")[0])


def is_available() -> bool:
    """Check whether GPU support (cudf + cuspatial + geopandas) is importable.

    Returns
    -------
    bool
        ``True`` if the GPU dependencies imported cleanly, ``False`` otherwise.
    """
    return _HAS_GPU


def ensure_available():
    """Raise ``ImportError`` if GPU support is not available.

    Raises
    ------
    ImportError
        If GPU support is unavailable; chained from the original
        :class:`ImportError` or :class:`RuntimeError` when one was captured.
    """
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
    """GPU-accelerated polygon filter projected on a specified plane."""

    @validate_variables
    def __init__(self, polygon: ValidatedPolygonT, plane: PlaneStrings = "xy") -> None:
        """Filter points based on a polygon projected on a specified plane (GPU backend).

        Parameters
        ----------
        polygon : ValidatedPolygonT
            Polygon defining the filter region.
        plane : PlaneStrings, default="xy"
            Plane on which the polygon is projected.
        """
        ensure_available()
        self.polygon: Polygon = polygon
        self.plane = plane

    def mask(self, pcd: PointCloudData) -> Vector_Bool_T:
        """Create a boolean mask from the points inside the projected polygon.

        Parameters
        ----------
        pcd : PointCloudData

        Returns
        -------
        Vector_Bool_T
        """
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
    """GPU-accelerated polygon filter defined in spherical-angle coordinates."""

    @validate_variables
    def __init__(self, polygon: ValidatedPolygonT):
        """Filter points based on a polygon defined in spherical-angle coordinates (GPU backend).

        Parameters
        ----------
        polygon : ValidatedPolygonT
            Polygon defining the filter region in (horizontal, vertical)
            spherical-angle coordinates.
        """
        ensure_available()
        self.polygon: Polygon = polygon

    def mask(self, pcd: PointCloudData) -> Vector_Bool_T:
        """Create a mask of points inside the spherical-angle polygon.

        Parameters
        ----------
        pcd : PointCloudData
            The input point cloud data containing points to be processed.

        Returns
        -------
        Vector_Bool_T
            Boolean mask, ``True`` for points inside the polygon.
        """
        proj_pts = cudf.DataFrame({"x": pcd.hz.astype(float), "y": pcd.v.astype(float)}).interleave_columns()

        polygon_gpu = cuspatial.GeoSeries(gpd.GeoSeries([self.polygon]))
        proj_pts_gs = cuspatial.GeoSeries.from_points_xy(proj_pts)
        proj_pts_in = cuspatial.point_in_polygon(proj_pts_gs, polygon_gpu)
        mask = proj_pts_in[0].to_numpy()

        del polygon_gpu, proj_pts, proj_pts_gs, proj_pts_in
        gc.collect()

        return mask
