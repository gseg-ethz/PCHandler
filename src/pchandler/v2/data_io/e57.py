from pathlib import Path
from typing import Unpack, NotRequired, Generator, Optional
import logging

import numpy as np
import pye57        # type: ignore[import-untyped]

from .core import AbstractIOHandler, _LoadConfigType, _SaveConfigType, SaveConfig, LoadConfig
from ..geometry import PointCloudData

logger = logging.getLogger(__name__.split(".")[0])



class E57Handler(AbstractIOHandler):
    FORMATS = ['.e57']

    @classmethod
    def load(cls,
             path: str | Path,
             /,
             scalar_fields: Optional[list[str]] = None,
             remove_prefix: bool = True,
             prefix: str = 'scalar_',
             revert_sf_types: bool = False,
             pcd_index: Optional[int] = None,
             ignore_missing_fields: bool = True) -> PointCloudData | Generator[PointCloudData, None, None]:

        path = Path(path)

        logger.info(f"Loading E57 file: {path}")
        e57 = pye57.E57(str(path), mode="r")
        number_of_scans = e57.scan_count
        e57.close()

        assert pcd_index is None or (0 <= pcd_index < number_of_scans)
        point_cloud_index = 0 if number_of_scans == 1 else pcd_index

        if point_cloud_index is None:
            logger.debug(f"Loading {number_of_scans} scans from E57 file.")
            return cls._load_all_e57_scans(path)
        else:
            logger.debug(f"Loading scan index {point_cloud_index} from E57 file.")
            return cls._load_single_e57(path, pcd_index=point_cloud_index)

    @classmethod
    def save(cls, /, pcd: PointCloudData, path: str | Path, **config) -> None:
        raise NotImplementedError

    @classmethod
    def _load_all_e57_scans(cls,
                            path: str|Path,
                            /,
                            scalar_fields: Optional[list[str]] = None,
                            remove_prefix: bool = True,
                            prefix: str = 'scalar_',
                            revert_sf_types: bool = False,
                            pcd_index: Optional[int] = None,
                            ignore_missing_fields: bool = True) -> Generator[PointCloudData, None, None]:

        logger.debug(f"Loading multiple scans from E57 file: {path}")
        e57 = pye57.E57(str(path), mode="r")
        number_of_scans = e57.scan_count

        for i in range(number_of_scans):
            pcd_index = i
            yield cls._load_single_e57(
                path, scalar_fields, remove_prefix, prefix, revert_sf_types, pcd_index, ignore_missing_fields
            )

    @classmethod
    def _load_single_e57(cls, pcd_path: Path,
                         /,
                         scalar_fields: Optional[list[str]] = None,
                         remove_prefix: bool = True,
                         prefix: str = 'scalar_',
                         revert_sf_types: bool = False,
                         pcd_index: Optional[int] = None,
                         ignore_missing_fields: bool = True
                         ) -> PointCloudData:
        logger.debug(f"Loading single scan {pcd_index} from E57 file: {pcd_path}")



        e57 = pye57.E57(str(pcd_path), mode="r")
        data = e57.read_scan(pcd_index, ignore_missing_fields=True, intensity=True, colors=True)
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

        e57.close()
        logger.info(f"Successfully loaded scan {cfg.pcd_index} from E57 file: {pcd_path}")
        return pcd