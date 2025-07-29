from pathlib import Path
from typing import Generator, Optional
import logging

import numpy as np
import pye57        # type: ignore[import-untyped]

from pchandler.data_io.core import AbstractIOHandler
from pchandler.constants import INTENSITY_NAMES, RGB_NAMES
from pchandler.geometry import PointCloudData

logger = logging.getLogger(__name__.split(".")[0])


class E57Handler(AbstractIOHandler):
    FORMATS = ['.e57']

    SUPPORTED_FIELDS = {
        'x': 'cartesianX',
        'y': 'cartesianY',
        'z': 'cartesianZ'
    }

    SUPPORTED_SCALAR_FIELDS_MAP = {
        'intensity': 'intensity',
        'r': 'colorRed',
        'g': 'colorGreen',
        'b': 'colorBlue'
    }

    @classmethod
    def load(cls,
             path: str | Path, /,
             retain_rgb: bool = True,
             retain_intensity: bool = True,
             pcd_index: Optional[int] = None,
             read_transform: bool = True,
             ignore_missing_fields: bool = True,
             **kwargs) -> PointCloudData | Generator[PointCloudData, None, None]:

        path = Path(path)
        kwargs = {
            'retain_rgb': retain_rgb,
            'retain_intensity': retain_intensity,
            'pcd_index': pcd_index,
            'ignore_missing_fields': ignore_missing_fields,
            'read_transform': read_transform
        }

        logger.info(f"Loading E57 file: {path}")

        e57 = pye57.E57(str(path), mode="r")
        number_of_scans = e57.scan_count
        e57.close()

        point_cloud_index = 0 if number_of_scans == 1 else pcd_index

        if point_cloud_index is None:
            logger.debug(f"Loading {number_of_scans} scans from E57 file.")
            return cls._load_all_e57_scans(path, **kwargs)

        elif 0 <= point_cloud_index < number_of_scans:
            logger.debug(f"Loading scan index {point_cloud_index} from E57 file.")
            kwargs['pcd_index'] = point_cloud_index
            return cls._load_single_e57(path, **kwargs)

        else:
            raise ValueError(f"Input point cloud index passed is outside of the range [0, num_scans). Got {pcd_index}")

    @classmethod
    def save(cls, path: str | Path, /, pcd: PointCloudData, **config) -> None:
        # TODO need to decide if to implement this / extend the pye57 library to support other fields and writing
        #  or extend the PLY format header to contain metadata. E.g. transforms
        raise NotImplementedError

    @classmethod
    def _load_all_e57_scans(cls, path, **kwargs) -> Generator[PointCloudData, None, None]:
        logger.debug(f"Loading multiple scans from E57 file: {path}")
        e57 = pye57.E57(str(path), mode="r")
        number_of_scans = e57.scan_count

        for i in range(number_of_scans):
            kwargs['pcd_index'] = i
            yield cls._load_single_e57(path, **kwargs)

    @classmethod
    def _load_single_e57(cls,
             path: str | Path, /,
             retain_rgb: bool = True,
             retain_intensity: bool = True,
             pcd_index: Optional[int] = None,
             read_transform: bool = True,
             ignore_missing_fields: bool = True,) -> PointCloudData:
        logger.debug(f"Loading single scan {pcd_index} from E57 file: {path}")

        logger.debug(f"Reading fields:"
                     f"{'\n    ' + RGB_NAMES.base if retain_rgb else ''}"
                     f"{'\n    ' + INTENSITY_NAMES.base if retain_intensity else ''}")

        e57 = pye57.E57(str(path), mode="r")

        try:
            header = e57.get_header(pcd_index)

            expected_fields = tuple()

            if retain_rgb:
                expected_fields += ('colorRed', 'colorGreen', 'colorBlue')

            if retain_intensity:
                expected_fields += ('intensity',)

            unsupported_fields = set(header.point_fields).difference(expected_fields)

            if len(unsupported_fields) > 0:
                logger.warning(f"Fields discovered in file but are not supported by pye57: "
                               f"{('\n' + field for field in unsupported_fields)}")

            data = e57.read_scan(pcd_index,
                                 ignore_missing_fields=ignore_missing_fields,
                                 intensity=retain_intensity,
                                 colors=retain_rgb,
                                 transform=read_transform)

            pcd = PointCloudData(np.column_stack((data["cartesianX"],
                                                  data["cartesianY"],
                                                  data["cartesianZ"])))

            if retain_rgb:
                if 'colorRed' in data:
                    pcd.rgb = np.column_stack((data["colorRed"], data["colorGreen"], data["colorBlue"]))
                else:
                    logger.warning('Could not read colour information from point cloud')

            if retain_intensity:
                if "intensity" in data:
                    pcd.intensity = data["intensity"]
                else:
                    logger.warning('Could not read intensity information from point cloud')

        except Exception as e:
            raise e
        else:
            logger.info(f"Successfully loaded scan {pcd_index} from E57 file: {path}")
        finally:
            e57.close()

        return pcd