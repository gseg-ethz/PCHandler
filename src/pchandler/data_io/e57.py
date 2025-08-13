"""CSV / ASCII file format handler class"""
import logging
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import pye57  # type: ignore[import-untyped]

from pchandler import PointCloudData
from pchandler.constants import INTENSITY_NAMES, RGB_NAMES
from pchandler.data_io.core import AbstractIOHandler

logger = logging.getLogger(__name__.split(".")[0])

__all__ = ['E57Handler']


class E57Handler(AbstractIOHandler):
    """Handles E57 file input and output.

    Currently limited to reading and writing single scans with and only RGB and intensity scalar field types.
    All other scalar fields are ignored (due to library limitations).

    Supported file extensions:

    * .e57
    """
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
    def load(cls,   # type: ignore[override]
             path: str | Path, /,
             retain_rgb: bool = True,
             retain_intensity: bool = True,
             pcd_index: Optional[int] = None,
             read_transform: bool = True,
             ignore_missing_fields: bool = True,
             **kwargs) -> PointCloudData | Generator[PointCloudData, None, None]:
        """Load one or more point cloud from an E57 file.

        Parameters
        ----------
        path: str or Path
        retain_rgb: bool, default=True
            Flag if RGB values should be loaded (if exists)
        retain_intensity: bool, default=True
            Flag if intensity values should be loaded (if exists)
        pcd_index: int or None, default=None
            Index of the specific point cloud to load. If None, loads all scans,
            by default None.
        read_transform: bool, default=True
            Indicates if the transformation information should be read, by default True.
        ignore_missing_fields: bool, default=True
            If true, no errors are raised if fields are missing from the point cloud.
        kwargs: dict

        Returns
        -------
        PointCloudData or Generator[PointCloudData, None, None]
            Returns a single point cloud or a generator of point clouds depending
            on the value of `pcd_index`.

        Raises
        ------
        ValueError
            If `pcd_index` is provided and is out of the range [0, num_scans).

        Notes
        -----
        This is a class method intended for loading E57 point cloud data based on
        the provided parameters.
        """

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
    def save(cls, path: str | Path, /, pcd: PointCloudData, **config) -> None:   # type: ignore[override]
        """Save the point cloud data to an E57 file.

        ** NOT CURRENTLY IMPLEMENTED **

        Parameters
        ----------
        path : str or Path
        pcd : PointCloudData
        **config : dict
        """
        raise NotImplementedError

    # TODO implement tests on file with multiple scans
    @classmethod
    def _load_all_e57_scans(cls, path, **kwargs) -> Generator[PointCloudData, None, None]:
        """Load all E57 scans from a file.

        Parameters
        ----------
        path : str
        **kwargs : dict

        Yields
        ------
        Generator[PointCloudData, None, None]
        """
        logger.debug(f"Loading multiple scans from E57 file: {path}")
        e57 = pye57.E57(str(path), mode="r")
        number_of_scans = e57.scan_count

        for i in range(number_of_scans):
            kwargs['pcd_index'] = i
            yield cls._load_single_e57(path, **kwargs)

    @classmethod
    def _load_single_e57(cls,   # type: ignore[override]
             path: str | Path, /,
             retain_rgb: bool = True,
             retain_intensity: bool = True,
             pcd_index: Optional[int] = None,
             read_transform: bool = True,
             ignore_missing_fields: bool = True,) -> PointCloudData:
        """Load a single scan from an E57 file as a PointCloudData object"""
        logger.debug(f"Loading single scan {pcd_index} from E57 file: {path}")

        logger.debug(f"Reading fields:"
                     f"{'\n    ' + RGB_NAMES.base if retain_rgb else ''}"
                     f"{'\n    ' + INTENSITY_NAMES.base if retain_intensity else ''}")


        with pye57.E57(str(path), mode="r") as e57:
            header = e57.get_header(pcd_index)

            expected_fields: tuple = tuple()
            if retain_rgb:
                expected_fields += ('colorRed', 'colorGreen', 'colorBlue')
            if retain_intensity:
                expected_fields += ('intensity',)

            unsupported_fields = set(header.point_fields).difference(expected_fields)

            if len(unsupported_fields) > 0:
                logger.warning(f"Fields discovered in file but are not supported by pye57 and will not be loaded: "
                               f"{('\n' + field for field in unsupported_fields)}")

            data = e57.read_scan(pcd_index,
                                 ignore_missing_fields=ignore_missing_fields,
                                 intensity=retain_intensity,
                                 colors=retain_rgb,
                                 transform=read_transform)

            pcd = PointCloudData(np.column_stack((data["cartesianX"], data["cartesianY"], data["cartesianZ"])))

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

            logger.info(f"Successfully loaded scan {pcd_index} from E57 file: {path}")

        return pcd
