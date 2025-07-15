from pathlib import Path
from typing import Unpack, NotRequired, Any
import logging
from datetime import datetime

import numpy as np
import numpy.typing as npt
import laspy                # type: ignore[import-untyped]

from .core import AbstractIOHandler, _LoadConfigType, _SaveConfigType, SaveConfig, LoadConfig
from ..geometry import PointCloudData
from ..geometry.scalar_field_manager import ScalarFieldManager
from ..geometry.scalar_fields import ScalarFieldBoolean, ScalarField

logger = logging.getLogger(__name__.split(".")[0])


class _LASLoadConfigType(_LoadConfigType):
    pass


class _LASSaveConfigType(_SaveConfigType):
    pass


class LasHandler(AbstractIOHandler):
    FORMATS = ['.las', '.laz']

    @classmethod
    def load(cls, /, path: str | Path, **config: Unpack[_LASLoadConfigType]) -> PointCloudData:
        cfg = LoadConfig(**config)
          # TODO: Extend usage from `dimension_names` to `extra_dimension_names`
        logger.info(f"Loading LAZ file: {path}")
        las = laspy.read(path)
        scalar_field_names = set(las.point_format.dimension_names).difference({"X", "Y", "Z", "x", "y", "z"})
        lower_sf_names = {name.lower(): name for name in scalar_field_names}
        pcd = PointCloudData(las.xyz)

        # Update the abstractIOhandler class to clean this up and be more DRY
        if cfg.retain_rgb:
            if rgb_field_names := cls._get_rgb_field_names(set(lower_sf_names.values())):
                pcd.rgb = cls.extract_rgb(las.points.array, len(pcd),
                                          [lower_sf_names[name] for name in rgb_field_names])
                cls.remove_field_names(scalar_field_names, *[lower_sf_names.pop(i) for i in rgb_field_names])

        if cfg.retain_normals:
            if normal_field_names := cls._get_normals_field_names(set(lower_sf_names.keys())):
                pcd.normals = cls.extract_normals(las.points.array, len(pcd),
                                                    [lower_sf_names[name] for name in normal_field_names])
                cls.remove_field_names(scalar_field_names, *[lower_sf_names.pop(i) for i in normal_field_names])

        if cfg.retain_intensity:
            if intensity_field_names := cls._get_intensity_field_names(set(lower_sf_names.values())):
                pcd.intensity = cls.extract_intensity(las.points.array)
                cls.remove_field_names(scalar_field_names, *[lower_sf_names.pop(i) for i in intensity_field_names])

        if cfg.retain_extra_scalar_fields:
            field_names = tuple(set(cfg.retain_extra_scalar_fields) & set(lower_sf_names))
        else:
            field_names = tuple(scalar_field_names)

        lower_sf_names = {k: lower_sf_names[k] for k in field_names}

        for lower_name, name in lower_sf_names.items():
            if isinstance(las[name], np.ndarray):
                pcd.scalar_fields[lower_name] = ScalarField(las[name], name=lower_name)

            elif isinstance(las[name], laspy.point.dims.SubFieldView):
                if name in [
                    "scan_direction_flag",
                    "edge_of_flight_line",
                    "synthetic",
                    "key_point",
                    "withheld",
                    "overlap",
                ]:
                    pcd.scalar_fields[lower_name] = ScalarFieldBoolean(las[name].array.astype(np.bool_), name=lower_name)
                else:
                    pcd.scalar_fields[lower_name] = ScalarField(las[name].array, name=lower_name)

            else:
                raise NotImplementedError

        logger.info(f"Successfully loaded LAZ file: {path}")
        return pcd

    @classmethod
    def save(cls, /, pcd: PointCloudData, path: str | Path, **config: Unpack[_LASSaveConfigType]) -> None:
        cfg = SaveConfig(**config)

        offsets: npt.NDArray[np.float32|np.float64] = pcd.min(axis=0)
        scales: npt.NDArray[np.float32|np.float64] = np.array([0.0001, 0.0001, 0.0001])

        las = laspy.create()
        las.header.offsets = offsets
        las.header.scales = scales

        las.X = (pcd.x - offsets[0]) / scales[0]
        las.Y = (pcd.y - offsets[1]) / scales[1]
        las.Z = (pcd.z - offsets[2]) / scales[2]

        if cfg.retain_intensity and pcd.intensity:
            las.intensity = pcd.intensity

        if cfg.retain_rgb and pcd.rgb:
            las.red = pcd.rgb.r
            las.green = pcd.rgb.g
            las.blue = pcd.rgb.b

        if cfg.retain_normals and pcd.normals:
            for val in ('nx', 'ny', 'nz'):
                las.add_extra_dim(laspy.ExtraBytesParams(val, pcd.normals.dtype))
                setattr(las, val, getattr(pcd.normals, val))

        if cfg.retain_reflectance and pcd.reflectance:
            las.reflectance = pcd.reflectance

        if cfg.retain_extra_scalar_fields:
            extra_fields = cfg.retain_extra_scalar_fields & pcd.scalar_fields.keys()
            for field in extra_fields:
                las.add_extra_dim(laspy.ExtraBytesParams(field, pcd.scalar_fields[field].dtype))
                setattr(las, field, pcd.scalar_fields[field])

        las.write(path)

