"""
GPU module for pchandler.geometry.filters

Provides functions that use GPU acceleration (via cudf and cuspatial) to filter point clouds.
# """
#
# import gc
# import logging
#
# from typing import Annotated, Any
#
# try:
#     import cudf
#     import cuspatial
#     import geopandas as gpd
# except RuntimeError:
#     pass
# except ImportError:
#     raise ImportError("pchandler.geometry.filters.gpu requires the ‘cuda11’ or ‘cuda12’ extra (`pip install pchandler[cuda11]`).")
#
# import numpy as np
# from numpy.typing import NDArray
# from shapely.geometry import Polygon
# from pydantic import BeforeValidator
#
# from ..geometry.core import PointCloudData
# from .core import PointCloudFilter
# from .cartesian_filters import PlaneStrings
# from ..constants import validate_variables
# from ..base_types import ValidatedPolygonT
#
# logger = logging.getLogger(__name__.split(".")[0])
#
#
# class PolygonFilterGPU(PointCloudFilter):
#     @validate_variables
#     def __init__(self, polygon: ValidatedPolygonT, plane: PlaneStrings = "xy") -> None:
#         self.polygon: Polygon = polygon
#         self.plane = plane
#
#     def mask(self, pcd: PointCloudData) -> NDArray:
#         if self.plane not in ('xy', 'xz', 'yz'):
#             raise ValueError("Invalid plane. Choose 'xy', 'xz', or 'yz'.")
#
#         proj_pts = cudf.DataFrame(
#             {character: getattr(pcd, character).astype(float) for character in self.plane}
#         )
#
#         if pcd.global_coordinate_shift is not None:
#             if self.plane == "xy":
#                 global_shift = -pcd.global_coordinate_shift[:2]
#             elif self.plane == "xz":
#                 global_shift = -pcd.global_coordinate_shift[[0, 2]]
#             else:
#                 global_shift = -pcd.global_coordinate_shift[1:]
#             self.polygon = gpd.GeoSeries([self.polygon]).translate(*global_shift).iloc[0]
#
#         polygon_gpu = cuspatial.GeoSeries(gpd.GeoSeries([self.polygon]))
#         proj_pts_gs = cuspatial.GeoSeries.from_points_xy(proj_pts)
#         proj_pts_in = cuspatial.point_in_polygon(proj_pts_gs, polygon_gpu)
#         mask = proj_pts_in[0].to_numpy()
#
#         del polygon_gpu, proj_pts, proj_pts_gs, proj_pts_in
#         gc.collect()
#
#         return mask
#
#
# class SphericalPolygonFilterGPU(PointCloudFilter):
#     @validate_variables
#     def __init__(self, polygon: ValidatedPolygonT):
#         self.polygon: Polygon = polygon
#
#     def mask(self, pcd: PointCloudData) -> NDArray:
#         proj_pts = cudf.DataFrame(
#             {"x": pcd.hz.astype(float), "y": pcd.v.astype(float)}
#         ).interleave_columns()
#
#         polygon_gpu = cuspatial.GeoSeries(gpd.GeoSeries([self.polygon]))
#         proj_pts_gs = cuspatial.GeoSeries.from_points_xy(proj_pts)
#         proj_pts_in = cuspatial.point_in_polygon(proj_pts_gs, polygon_gpu)
#         mask = proj_pts_in[0].to_numpy()
#
#         del polygon_gpu, proj_pts, proj_pts_gs, proj_pts_in
#         gc.collect()
#
#         return mask
