from pathlib import Path
from typing import Unpack, NotRequired, Generator, Optional
import logging

import numpy as np
import pye57        # type: ignore[import-untyped]

from .core import AbstractIOHandler, _LoadConfigType, _SaveConfigType, SaveConfig, LoadConfig
from ..geometry import PointCloudData

logger = logging.getLogger(__name__.split(".")[0])


class _E57LoadConfigType(_LoadConfigType):
    ignore_missing_fields: NotRequired[bool]
    pcd_index: NotRequired[int]


class _E57SaveConfigType(_SaveConfigType):
    pass


class E57Handler(AbstractIOHandler):
    FORMATS = ['.e57']

    @classmethod
    def load(cls, /,
             path: str | Path,
             **config: Unpack[_E57LoadConfigType]) -> PointCloudData | Generator[PointCloudData, None, None]:

        cfg = LoadConfig(**config)

        path = Path(path)

        logger.info(f"Loading E57 file: {path}")
        e57 = pye57.E57(str(path), mode="r")
        number_of_scans = e57.scan_count
        e57.close()

        assert cfg.pcd_index is None or (0 <= cfg.pcd_index < number_of_scans)

        point_cloud_index = 0 if number_of_scans == 1 else cfg.pcd_index

        if point_cloud_index is None:
            logger.debug(f"Loading {number_of_scans} scans from E57 file.")
            return cls._load_all_e57_scans(path, **config)
        else:
            logger.debug(f"Loading scan index {point_cloud_index} from E57 file.")
            config['pcd_index'] = point_cloud_index
            return cls._load_single_e57(path, **config)

    @classmethod
    def save(cls, /, pcd: PointCloudData, path: str | Path, **config: Unpack[_E57SaveConfigType]) -> None:
        raise NotImplementedError

    @classmethod
    def _load_all_e57_scans(cls, pcd_path: Path, **config: Unpack[_E57LoadConfigType]) -> Generator[PointCloudData, None, None]:
        logger.debug(f"Loading multiple scans from E57 file: {pcd_path}")
        e57 = pye57.E57(str(pcd_path), mode="r")
        number_of_scans = e57.scan_count

        for i in range(number_of_scans):
            config['pcd_index'] = i
            yield cls._load_single_e57(pcd_path, **config)

    @classmethod
    def _load_single_e57(cls, pcd_path: Path, **config: Unpack[_E57LoadConfigType]) -> PointCloudData:
        cfg = LoadConfig(**config)
        logger.debug(f"Loading single scan {cfg.pcd_index} from E57 file: {pcd_path}")

        e57 = pye57.E57(str(pcd_path), mode="r")
        data = e57.read_scan(cfg.pcd_index, ignore_missing_fields=True, intensity=cfg.keep_intensity, colors=cfg.keep_rgb)
        header = e57.get_header(cfg.pcd_index)

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

        e57.close()
        logger.info(f"Successfully loaded scan {cfg.pcd_index} from E57 file: {pcd_path}")
        return pcd