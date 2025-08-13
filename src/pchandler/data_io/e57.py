import logging
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import pye57  # type: ignore[import-untyped]

from pchandler import PointCloudData
from pchandler.constants import INTENSITY_NAMES, RGB_NAMES
from pchandler.data_io.core import AbstractIOHandler

logger = logging.getLogger(__name__.split(".")[0])


class E57Handler(AbstractIOHandler):
    """
    Handler for processing E57 point cloud files.

    This class provides functionality for loading point cloud data from E57 files.
    It supports extracting coordinate data, intensity values, and RGB color information
    from E57 formatted scans. The handler also facilitates reading single or multiple
    scans from E57 files.

    Parameters
    ----------
    FORMATS : list of str
        List of file formats supported by the handler.
    SUPPORTED_FIELDS : dict
        Mapping of supported field names to corresponding E57 field names.
    SUPPORTED_SCALAR_FIELDS_MAP : dict
        Mapping for supported scalar fields to specific E57 field names.
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
        """
        Load one or more point cloud data from an E57 file.

        Parameters
        ----------
        path : str or Path
            Path to the E57 file.
        retain_rgb : bool, optional
            Specifies if RGB data should be retained, by default True.
        retain_intensity : bool, optional
            Specifies if intensity data should be retained, by default True.
        pcd_index : int or None, optional
            Index of the specific point cloud to load. If None, loads all scans,
            by default None.
        read_transform : bool, optional
            Indicates if the transformation information should be read, by default True.
        ignore_missing_fields : bool, optional
            If True, handles missing fields gracefully during loading, by default True.
        kwargs : dict
            Additional arguments for loading E57 scans.

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
        """
        Saves the given point cloud data to the specified path using provided configurations.

        Parameters
        ----------
        path : str or Path
            The file path where the point cloud data will be saved.
        pcd : PointCloudData
            The point cloud data to be saved.
        **config : dict
            Additional configuration options for saving the point cloud data.
        """
        raise NotImplementedError

    # TODO implement tests on file with multiple scans
    @classmethod
    def _load_all_e57_scans(cls, path, **kwargs) -> Generator[PointCloudData, None, None]:
        """
        Load all E57 scans from a file.

        This method loads multiple scans from an E57 file, yielding a PointCloudData
        object for each scan.

        Parameters
        ----------
        path : str
            Path to the E57 file.
        **kwargs : dict
            Additional arguments to be passed for loading individual scans.

        Yields
        ------
        Generator[PointCloudData, None, None]
            A generator yielding PointCloudData objects for each scan in the file.
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
        """
        Load a single scan from an E57 file as a PointCloudData object.

        Parameters
        ----------
        path : str or Path
            Path to the E57 file.
        retain_rgb : bool, optional
            Whether to retain RGB color data, by default True.
        retain_intensity : bool, optional
            Whether to retain intensity data, by default True.
        pcd_index : int or None, optional
            Index of the specific point cloud to load from the E57 file. Defaults to None.
        read_transform : bool, optional
            Whether to read and apply transformation data, by default True.
        ignore_missing_fields : bool, optional
            Whether to ignore missing fields when loading the E57 scan, by default True.

        Returns
        -------
        PointCloudData
            The loaded point cloud data from the specified E57 scan, including its geometry and, optionally, RGB
            and intensity data.
        """
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
