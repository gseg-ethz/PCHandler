"""
GPU module for pchandler.geometry.fitlers

Provides functions that use GPU acceleration (via cudf and cuspatial) to filter point clouds.
"""

import gc
import logging

try:
    import cudf
    import cuspatial
    import geopandas as gpd
except ImportError:
    raise ImportError("pchandler.geometry.filters.gpu requires the ‘cuda11’ or ‘cuda12’ extra (`pip install pchandler[cuda11]`).")

from numpy.typing import NDArray
from shapely.geometry import Polygon

from ..core import PointCloudData
from .core import PointCloudFilter

logger = logging.getLogger(__name__.split(".")[0])


class PolygonFilterGPU(PointCloudFilter):
    def __init__(self, polygon: Polygon, plane: str = "xy"):
        self.polygon = polygon
        self.plane = plane

    def mask(self, pcd: PointCloudData) -> NDArray:
        if self.plane == "xy":
            proj_pts = cudf.DataFrame(
                {"x": pcd.xyz[:, 0].astype(float), "y": pcd.xyz[:, 1].astype(float)}
            ).interleave_columns()
        elif self.plane == "xz":
            proj_pts = cudf.DataFrame(
                {"x": pcd.xyz[:, 0].astype(float), "y": pcd.xyz[:, 2].astype(float)}
            ).interleave_columns()
        elif self.plane == "yz":
            proj_pts = cudf.DataFrame(
                {"x": pcd.xyz[:, 1].astype(float), "y": pcd.xyz[:, 2].astype(float)}
            ).interleave_columns()
        else:
            raise ValueError("Invalid plane. Choose 'xy', 'xz', or 'yz'.")

        if pcd.global_coordinate_shift is not None:
            if self.plane == "xy":
                gs = -pcd.global_coordinate_shift[:2]
            elif self.plane == "xz":
                gs = -pcd.global_coordinate_shift[[0, 2]]
            elif self.plane == "yz":
                gs = -pcd.global_coordinate_shift[1:]
            polygon = gpd.GeoSeries([self.polygon]).translate(*gs).iloc[0]

        polygon_gpu = cuspatial.GeoSeries(gpd.GeoSeries([polygon]))
        proj_pts_gs = cuspatial.GeoSeries.from_points_xy(proj_pts)
        proj_pts_in = cuspatial.point_in_polygon(proj_pts_gs, polygon_gpu)
        mask = proj_pts_in[0].to_numpy()

        del polygon_gpu, proj_pts, proj_pts_gs, proj_pts_in
        gc.collect()

        return mask


class SphericalPolygonFilterGPU(PointCloudFilter):
    def __init__(self, polygon: Polygon):
        self.polygon = polygon

    def mask(self, pcd: PointCloudData) -> NDArray:
        proj_pts = cudf.DataFrame(
            {"x": pcd.spherical_coordinates[:, 1].astype(float), "y": pcd.spherical_coordinates[:, 2].astype(float)}
        ).interleave_columns()

        polygon_gpu = cuspatial.GeoSeries(gpd.GeoSeries([self.polygon]))
        proj_pts_gs = cuspatial.GeoSeries.from_points_xy(proj_pts)
        proj_pts_in = cuspatial.point_in_polygon(proj_pts_gs, polygon_gpu)
        mask = proj_pts_in[0].to_numpy()

        del polygon_gpu, proj_pts, proj_pts_gs, proj_pts_in
        gc.collect()

        return mask
