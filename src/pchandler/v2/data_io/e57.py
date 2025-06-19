from pathlib import Path
from typing import Unpack, NotRequired, Generator, Optional
import logging
from datetime import datetime

import numpy as np
import laspy
import pye57

from .core import AbstractIOHandler, _BaseLoadConfigType, _BaseSaveConfigType, BaseSaveConfig, BaseLoadConfig
from ..geometry import PointCloudData
from ..constants import (
    RGB_FIELD,
    NORMALS_FIELD,
    INTENSITY_FIELD,
    REFLECTANCE_FIELD,
    RGB_WORD,
    NORMAL_PARTIAL_NAMES
)

logger = logging.getLogger(__name__.split(".")[0])


class _E57LoadConfigType(_BaseLoadConfigType):
    ignore_missing_fields: NotRequired[bool]


class _E57SaveConfigType(_BaseSaveConfigType):
    pass


class E57LoadConfig(BaseLoadConfig):
    ignore_missing_fields: bool = True


class E57SaveConfig(BaseSaveConfig):
    pass

class E57Handler(AbstractIOHandler):
    FORMATS = ['.e57']
    LOAD_CONFIG: type[E57LoadConfig] = E57LoadConfig
    SAVE_CONFIG: type[E57SaveConfig] = E57SaveConfig

    @classmethod
    def load(cls, /,
             path: str | Path,
             pcd_index: Optional[int] = None,
             **config: Unpack[_E57LoadConfigType]
             ) -> PointCloudData | Generator[PointCloudData, None, None]:

        logger.info(f"Loading E57 file: {path}")
        e57 = pye57.E57(str(path), mode="r")
        number_of_scans = e57.scan_count
        e57.close()

        assert pcd_index is None or (0 <= pcd_index < number_of_scans)

        point_cloud_index = 0 if number_of_scans == 1 else pcd_index

        if point_cloud_index is None:
            logger.debug(f"Loading {number_of_scans} scans from E57 file.")
            return cls._load_all_e57_scans(path, **config)
        else:
            logger.debug(f"Loading scan index {point_cloud_index} from E57 file.")
            return cls._load_single_e57(path, point_cloud_index, **config)

    @classmethod
    def save(cls, /, pcd: PointCloudData, path: str | Path, **config: Unpack[_E57SaveConfigType]):
        pass

    @classmethod
    def _load_all_e57_scans(cls, pcd_path: Path, **config: Unpack[_E57LoadConfigType]) -> Generator[PointCloudData, None, None]:
        logger.debug(f"Loading multiple scans from E57 file: {pcd_path}")
        e57 = pye57.E57(str(pcd_path), mode="r")
        number_of_scans = e57.scan_count

        for i in range(number_of_scans):
            yield cls._load_single_e57(pcd_path, i, **config)

    @classmethod
    def _load_single_e57(cls, pcd_path: Path, pcd_index: int, **config: Unpack[_E57LoadConfigType]) -> PointCloudData:
        # TODO pye57 library seems limited
        cfg = cls.get_config(**config)
        logger.debug(f"Loading single scan {pcd_index} from E57 file: {pcd_path}")

        e57 = pye57.E57(str(pcd_path), mode="r")
        data = e57.read_scan(pcd_index, ignore_missing_fields=True, intensity=cfg.keep_intensity, colors=cfg.keep_rgb)
        header = e57.get_header(pcd_index)

        xyz = np.column_stack((data["cartesianX"], data["cartesianY"], data["cartesianZ"]))

        pcd = PointCloudData(xyz)
        if "colorRed" in data:
            pcd.rgb = np.column_stack((data["colorRed"], data["colorGreen"], data["colorBlue"]))

        if "intensity" in data:
            pcd.intensity = data["intensity"]

        if "normals" in data:
            pcd.normals = data["normals"]

        if "reflectance" in data:
            pcd.reflectance = data["reflectance"]

        # TODO investigate file format to fin other fields

        e57.close()
        logger.info(f"Successfully loaded scan {pcd_index} from E57 file: {pcd_path}")
        return pcd