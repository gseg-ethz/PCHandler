from pathlib import Path
from typing import Unpack, NotRequired
import logging
from datetime import datetime

import numpy as np
import laspy

from .core import AbstractIOHandler, _BaseLoadConfigType, _BaseSaveConfigType, BaseSaveConfig, BaseLoadConfig
from ..geometry import PointCloudData
from ..geometry.scalar_field_manager import ScalarFieldManager
from ..geometry.scalar_fields import BooleanScalarField, ScalarField

logger = logging.getLogger(__name__.split(".")[0])


class _LASLoadConfigType(_BaseLoadConfigType):
    pass


class _LASSaveConfigType(_BaseSaveConfigType):
    pass


class LASLoadConfig(BaseLoadConfig):
    pass


class LASSaveConfig(BaseSaveConfig):
    pass


class LasHandler(AbstractIOHandler):
    FORMATS = ['.las', '.laz']
    LOAD_CONFIG: type[LASLoadConfig] = LASLoadConfig
    SAVE_CONFIG: type[LASSaveConfig] = LASSaveConfig

    @classmethod
    def load(cls, /, path: str | Path, **config: Unpack[_LASLoadConfigType]) -> PointCloudData:
        config = cls.get_config(config)
          # TODO: Extend usage from `dimension_names` to `extra_dimension_names`
        logger.info(f"Loading LAZ file: {path}")
        las = laspy.read(path)
        scalar_field_names = set(las.point_format.dimension_names).difference({"X", "Y", "Z", "x", "y", "z"})
        lower_sf_names = {name.lower(): name for name in scalar_field_names}
        pcd = PointCloudData(las.xyz)

        if config.keep_rgb:
            if field_names := cls._get_rgb_field_names(set(lower_sf_names.values())):
                pcd.rgb = cls.extract_rgb(las.points.array, len(pcd),
                                          [lower_sf_names[name] for name in field_names])
                cls.remove_field_names(scalar_field_names, *[lower_sf_names.pop(i) for i in field_names])

        if config.keep_normals:
            if field_names := cls._get_normals_field_names(set(lower_sf_names.keys())):
                pcd.normals = cls.extract_normals(las.points.array, len(pcd),
                                                    [lower_sf_names[name] for name in field_names])
                cls.remove_field_names(scalar_field_names, *[lower_sf_names.pop(i) for i in field_names])

        if config.keep_intensity:
            if field_names := cls._get_intensity_field_names(set(lower_sf_names.values())):
                pcd.intensity = cls.extract_intensity(las.points.array)
                cls.remove_field_names(scalar_field_names, *[lower_sf_names.pop(i) for i in field_names])

        if config.keep_extra_scalar_fields:
            field_names = tuple(set(config.keep_extra_scalar_fields) & set(lower_sf_names))
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
                    pcd.scalar_fields[lower_name] = BooleanScalarField(las[name].array.astype(np.bool), name=lower_name)
                else:
                    pcd.scalar_fields[lower_name] = ScalarField(las[name].array, name=lower_name)

            else:
                raise NotImplementedError

        logger.info(f"Successfully loaded LAZ file: {path}")
        return pcd

    @classmethod
    def save(cls, /, pcd: PointCloudData, path: str | Path, **config: Unpack[_LASSaveConfigType]):
        config = cls.get_config(load=False)

        scale = np.array([0.0001, 0.0001, 0.0001])
        offsets = pcd.min(axis=0)

        las = laspy.create()
        las.header.offsets = np.min(pcd.xyz, axis=0)
        las.header.scales = np.array([0.0001, 0.0001, 0.0001])

        las.X = (pcd.x - offsets[0]) / scale[0]
        las.Y = (pcd.y - offsets[1]) / scale[1]
        las.Z = (pcd.z - offsets[2]) / scale[2]

        if config.keep_intensity and pcd.intensity:
            las.intensity = pcd.intensity

        if config.keep_rgb and pcd.rgb:
            las.red = pcd.r
            las.green = pcd.g
            las.blue = pcd.b

        if config.keep_normals and pcd.normals:
            las.add_extra_dim(laspy.ExtraBytesParams('nx', pcd.normals.dtype))
            las.add_extra_dim(laspy.ExtraBytesParams('ny', pcd.normals.dtype))
            las.add_extra_dim(laspy.ExtraBytesParams('nz', pcd.normals.dtype))
            las.nx = pcd.normals.nx
            las.ny = pcd.normals.ny
            las.nz = pcd.normals.nz

        if config.keep_reflectance and pcd.reflectance:
            las.reflectance = pcd.reflectance

        if config.keep_extra_scalar_fields:
            extra_fields = config.keep_extra_scalar_fields & pcd.scalar_fields.keys()
            for field in extra_fields:
                las.add_extra_dim(laspy.ExtraBytesParams(field, pcd.scalar_fields[field].dtype))
                setattr(las, field, pcd.scalar_fields[field])

        las.write(path)

