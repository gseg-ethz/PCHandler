import logging
from pathlib import Path
from typing import Optional

import laspy  # type: ignore[import-untyped]
import numpy as np
from GSEGUtils.base_types import Vector_3_T
from GSEGUtils.validators import normalize_uint16

from pchandler import PointCloudData
from pchandler.constants import INTENSITY_NAMES, NORMAL_NAMES, RGB_NAMES, XYZ_NAMES
from pchandler.data_io.core import AbstractIOHandler, _get_rgb_or_normal_field_names
from pchandler.geometry import OptimizedShift

logger = logging.getLogger(__name__.split(".")[0])


class LasHandler(AbstractIOHandler):
    """
    Handles loading and saving of LAS/LAZ files for point cloud data management.

    This class provides methods to load LAS/LAZ files into PointCloudData objects
    and to save PointCloudData objects back to LAS/LAZ format. Includes support for
    extracting scalar fields, applying numerical optimizations, and managing additional
    attributes like RGB, intensity, and normals.

    Parameters
    ----------
    FORMATS : list of str
        Supported file extensions for this handler (".las" and ".laz").
    """
    FORMATS = [".las", ".laz"]

    @classmethod
    def load(
        cls,  # type: ignore[override]
        path: str | Path,
        /,
        scalar_fields: Optional[list[str]] = None,
        remove_prefix: bool = True,
        prefix: str = "scalar_",
        force_no_numerical_shift: bool = False,
        **config,
    ) -> PointCloudData:
        """
        Loads a LAZ file and initializes a PointCloudData object. Extracts scalar fields as specified
        and applies numerical optimization based on the header offsets if not disabled.

        Parameters
        ----------
        path : str or Path
            The file path of the LAZ file to be loaded.
        scalar_fields : list of str, optional
            List of scalar field names to extract, by default None.
        remove_prefix : bool
            Whether to remove a prefix from scalar field names, by default True.
        prefix : str
            The prefix to be removed if `remove_prefix` is True, by default "scalar_".
        force_no_numerical_shift : bool
            If True, disables numerical optimization shift, by default False.
        config : dict
            Additional configuration parameters.

        Returns
        -------
        PointCloudData
            A PointCloudData object containing the loaded 3D point cloud alongside scalar fields.
        """

        logger.info(f"Loading LAZ file: {path}")

        las = laspy.read(path)
        header_names = las._points.array.dtype.names

        data = las._points.array
        field_names = cls._validate_field_selection(scalar_fields, header_names, remove_prefix, prefix)

        if force_no_numerical_shift:
            pcd = PointCloudData(las.xyz, numerical_optimization_shift=None)
        else:
            pcd = PointCloudData(las.xyz, numerical_optimization_shift=OptimizedShift(las.header.offsets))
        cls.extract_scalar_fields(pcd, data, data.size, field_names)

        logger.info(f"Successfully loaded LAZ file: {path}")
        return pcd

    @classmethod
    def save(
        cls,  # type: ignore[override]
        pcd: PointCloudData,
        path: str | Path,
        /,
        scalar_fields: Optional[list[str]] = None,
        add_prefix: bool = True,
        prefix: str = "scalar_",
        revert_sf_types: bool = False,
        scales: Vector_3_T = np.array([0.0001, 0.0001, 0.0001]),
        **config,
    ) -> None:
        """
        Save the PointCloudData to a LAS/LAZ file.

        This method exports point cloud data to a LAS/LAZ file, including its coordinates, scalar fields,
        and additional attributes like RGB values, intensities, and normals. It also allows for optional
        scaling, offset adjustments, and prefix handling for scalar fields.

        Parameters
        ----------
        pcd : PointCloudData
            The point cloud data containing information such as coordinates, scalars, and other attributes.
        path : str or Path
            The output file path for the LAS/LAZ file.
        scalar_fields : list of str, optional
            A list of scalar fields to include in the exported file. If None, all scalar fields from the
            input data are included.
        add_prefix : bool, optional
            If True, a prefix is added to the scalar field names in the output file.
        prefix : str, optional
            The prefix to add to scalar field names when `add_prefix` is True.
        revert_sf_types : bool, optional
            Whether to revert scalar field types to their original format before exporting.
        scales : Vector_3_T, optional
            Scaling factors for the x, y, and z coordinates in the LAS/LAZ file.
        config : dict
            Additional configuration options passed as keyword arguments.

        Returns
        -------
        None
        """

        logger.info(f"Attempting to write to LAS/LAZ file: {path}")
        if pcd.numerical_optimization_shift is None:
            offsets: Vector_3_T = np.zeros(3)
        else:
            offsets = np.array(pcd.numerical_optimization_shift)

        if scalar_fields is None:
            scalar_fields = list(pcd.scalar_fields.keys())

        scalar_fields = list(scalar_fields)

        # Can use the check on the base name as named scalar_fields should match the pcd object names
        if RGB_NAMES.base in scalar_fields:
            index = scalar_fields.index(RGB_NAMES.base)
            scalar_fields = scalar_fields[:index] + list(RGB_NAMES.char) + scalar_fields[index + 1 :]

        if NORMAL_NAMES.base in scalar_fields:
            index = scalar_fields.index(NORMAL_NAMES.base)
            scalar_fields = scalar_fields[:index] + list(NORMAL_NAMES.char) + scalar_fields[index + 1 :]

        # Set the base coordinates of the LAS point cloud as well as offsets and scales
        las = laspy.create()
        las.change_scaling(scales=scales, offsets=offsets)
        las.xyz = pcd.xyz + offsets

        # RGB values
        if (rgb_fields := _get_rgb_or_normal_field_names(scalar_fields, RGB_NAMES)) and pcd.rgb:
            for field in rgb_fields:
                index = RGB_NAMES.get_position(field)
                setattr(las, RGB_NAMES.words[index], getattr(pcd.rgb, RGB_NAMES.char[index]))

        # Intensities - LAS expects unsigned 16bit (Uint16)
        if (intensity_fields := set(scalar_fields).intersection(INTENSITY_NAMES.all)) and pcd.intensity:
            # Case 1 - Leave data as is for Uint8 and Uint16
            if pcd.intensity.dtype == np.uint8 or pcd.intensity.dtype == np.uint16:
                las.intensity = pcd.intensity.copy()

            # Case 2: Linear map values in range [0, 1] to [0, (2**16)-1]
            # Case 3: Any other combination, normalize to [0, 1] then scale to Uint16 range
            else:
                logger.info(
                    f"Values range [{pcd.intensity.min()}, {pcd.intensity.max()}] has been normalized and scaled to"
                    f"Uint16 range required by LAS format: [0, {np.iinfo(np.uint16).max}]."
                )
                las.intensity = normalize_uint16(pcd.intensity)

        # Clear the previous sfs used
        for name in XYZ_NAMES.char + tuple(rgb_fields) + tuple(intensity_fields):
            if name in scalar_fields:
                scalar_fields.remove(name)

        # Remaining fields including normals as these are extra dimensions
        for field in scalar_fields:
            value = np.asarray(pcd.scalar_fields[field])

            if not hasattr(las, field):
                las.add_extra_dim(laspy.ExtraBytesParams(field, value.dtype))

            setattr(las, field, value)

        las.write(path)

        logger.info(f"Successfully wrote to  LAS/LAZ file: {path}")
