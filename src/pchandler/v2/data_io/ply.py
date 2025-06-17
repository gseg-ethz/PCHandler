from pathlib import Path
from typing import Any, NamedTuple, Unpack
import logging
from datetime import datetime

import numpy as np
from plyfile import PlyData, PlyElement

from .core import AbstractIOHandler, _BaseLoadConfigType, _BaseSaveConfigType
from ..geometry import PointCloudData
from ..constants import (
    RGB_PARTIAL_NAMES,
    RGB_FIELD,
    NORMALS_FIELD,
    INTENSITY_FIELD,
    REFLECTANCE_FIELD,
    RGB_WORD,
    NORMAL_PARTIAL_NAMES
)
from ..geometry.scalar_field_manager import ScalarFieldManager

logger = logging.getLogger(__name__.split(".")[0])


class _PlyLoadConfigType(_BaseLoadConfigType):
    pass

class _PlySaveConfigType(_BaseSaveConfigType):
    pass

class PlyHandler(AbstractIOHandler):
    FORMATS = ['.ply']

    @classmethod
    def load(cls, /, path: str | Path, **config: Unpack[_PlyLoadConfigType]):
        cfg = cls._get_config(**config)

        logger.info(f"Loading PLY file: {path}")

        try:
            with open(path, "rb") as f:
                plydata = PlyData.read(f)
        except Exception as e:
            logger.error(f"Failed to read PLY file {path}: {e}")
            raise e

        num_points = plydata["vertex"].count
        logger.debug(f"PLY file {path} contains {num_points} points")

        ply_scalar_fields = set([pe.name.lower() for pe in plydata["vertex"].properties])

        pcd = PointCloudData(cls._extract_xyz(plydata["vertex"], num_points, ply_scalar_fields))

        if cfg.keep_rgb:
            pcd.rgb = cls._extract_rgb(plydata["vertex"], num_points, ply_scalar_fields, cfg.normalize_rgb)

        if cfg.keep_normals:
            pcd.normals = cls._extract_normals(plydata["vertex"], num_points, ply_scalar_fields)

        if cfg.keep_intensity:
            pcd.intensity = cls._extract_intensity(plydata["vertex"], num_points, cfg.normalize_intensity)

        if cfg.keep_reflectance:
            pcd.reflectance = cls._extract_reflectance(plydata["vertex"], num_points, cfg.normalize_reflectance)

        ply_scalar_fields &= cfg.keep_extra_scalar_fields

        for name in ply_scalar_fields:
            pcd.scalar_fields.create_field(name, plydata["vertex"][name])

        return pcd

    @classmethod
    def save(cls, /, pcd: PointCloudData, path: str | Path, **config: Unpack[_PlySaveConfigType]):
        cfg = cls._get_config(load=False, **config)

        xyz_dtype = np.dtype(np.float64).str if pcd.optimized is not None else pcd.xyz.dtype.str
        dtype_list = [(name, str(xyz_dtype)) for name in ('x', 'y', 'z')]

        pcd_scalar_fields = set(pcd.scalar_fields.keys()).difference(
            {RGB_FIELD, NORMALS_FIELD, INTENSITY_FIELD, REFLECTANCE_FIELD})

        if cfg.keep_rgb and pcd.rgb is not None:
            sf_dtype = cls._get_sf_save_dtype(cfg, pcd.rgb)
            dtype_list.extend((name, sf_dtype) for name in RGB_WORD)

        if cfg.keep_normals and pcd.normals is not None:
            sf_dtype = cls._get_sf_save_dtype(cfg, pcd.normals)
            dtype_list.extend((name, sf_dtype) for name in NORMAL_PARTIAL_NAMES)

        if cfg.keep_intensity and pcd.intensity is not None:
            sf_dtype = cls._get_sf_save_dtype(cfg, pcd.intensity)
            dtype_list.append((INTENSITY_FIELD, sf_dtype))

        if cfg.keep_reflectance and pcd.reflectance is not None:
            sf_dtype = cls._get_sf_save_dtype(cfg, pcd.reflectance)
            dtype_list.append((REFLECTANCE_FIELD, sf_dtype))

        pcd_scalar_fields = list(cfg.keep_extra_scalar_fields & pcd_scalar_fields)

        for sf_name in pcd_scalar_fields:
            sf_dtype = cls._get_sf_save_dtype(cfg, pcd[sf_name])
            dtype_list.append((sf_name, sf_dtype))

        pcd_structured_array = np.empty((len(pcd),), dtype=dtype_list)
        base_shift = np.zeros(3, dtype=np.float32) if pcd.optimal_shift is None else pcd.optimal_shift.optimal_shift

        pcd_structured_array["x"] = pcd.x + base_shift[0]
        pcd_structured_array["y"] = pcd.y + base_shift[1]
        pcd_structured_array["z"] = pcd.z + base_shift[2]

        if cfg.keep_rgb and pcd.rgb is not None:
            rgb = pcd.rgb.get_original_data() if cfg.revert_sf_types else pcd.rgb

            pcd_structured_array["red"] = rgb.r
            pcd_structured_array["green"] = rgb.g
            pcd_structured_array["blue"] = rgb.b

        if cfg.keep_normals and pcd.normals is not None:
            normals = pcd.normals.get_original_data() if cfg.revert_sf_types else pcd.normals

            pcd_structured_array["nx"] = normals.nx
            pcd_structured_array["ny"] = normals.ny
            pcd_structured_array["nz"] = normals.nz

        if cfg.keep_intensity and pcd.intensity is not None:
            pcd_structured_array[INTENSITY_FIELD] = pcd.intensity

        if cfg.keep_reflectance and pcd.reflectance is not None:
            pcd_structured_array[REFLECTANCE_FIELD] = pcd.reflectance

        for name in pcd_scalar_fields:
            sf = pcd.scalar_fields[name]
            pcd_structured_array[name] = sf.get_original_data() if cfg.revert_sf_types else sf.arr

        element = PlyElement.describe(
            pcd_structured_array,
            name="vertex",
            comments=[
                "Created with dranjan/python-plyfile in gseg-ethz/pchandler",
                f"Created {datetime.now():%Y-%m-%dT%H:%M:%S%z}",
            ],
        )

        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created {path.parent} folder")

        PlyData([element]).write(f"{path}")
        logger.info(f"PLY file saved successfully: {path}")