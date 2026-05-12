# pchandler - Toolbox for point-cloud handling, processing and analysis
#
# Copyright (c) 2025, Nicholas Meyer, Geosensors and Engineering Geodesy,
# Institute of Geodesy and Photogrammetry, ETH Zurich, Switzerland
# SPDX-License-Identifier: BSD-3-Clause
#
# Author: Nicholas Meyer (meyernic@ethz.ch)

"""LAS/LAZ file format handler class"""

import logging
from pathlib import Path
from typing import Optional, Unpack

import laspy  # type: ignore[import-untyped]
import numpy as np
from GSEGUtils.base_types import Vector_3_T
from GSEGUtils.validators import normalize_uint16

from pchandler import PointCloudData
from pchandler.constants import INTENSITY_NAMES, NORMAL_NAMES, RGB_NAMES, XYZ_NAMES
from pchandler.data_io.core import AbstractIOHandler, PointCloudDataKW, _get_rgb_or_normal_field_names
from pchandler.geometry import OptimizedShift

__all__ = ["LasHandler"]

logger = logging.getLogger(__name__.split(".")[0])


class LasHandler(AbstractIOHandler):
    """Handles LAS/LAZ file input and output.

    Supported file extensions:

     * .las
     * .laz
    """

    FORMATS = [".las", ".laz"]

    # TODO check how socs_origin is being passed to PointCloudData. Should support to stop optimal shift be shared?
    @classmethod
    def load(
        cls,  # type: ignore[override]
        path: str | Path,
        /,
        scalar_fields: Optional[list[str]] = None,
        remove_prefix: bool = True,
        prefix: str = "scalar_",
        force_no_numerical_shift: bool = False,
        **pcd_kw: Unpack[PointCloudDataKW],
    ) -> PointCloudData:
        """Load a point cloud from a LAS/LAZ file.

        Parameters
        ----------
        path : str | Path
            Input LAS or LAZ file path.
        scalar_fields : list[str]], default=None
            List of specific scalar fields to extract from the PLY file.
            Setting `None` will retrieve all scalar fields. Setting to `[]` will ignore scalar fields in the file.
        remove_prefix : bool, default=True
            Flag to remove prefixes on scalar field names.
        prefix : str, default="scalar_"
            Prefix to strip from scalar field names if `remove_prefix` is True.
        force_no_numerical_shift : bool, default=False
            Flag determining if the optimal shifts should be overridden and original coordinates used
        config : dict

        Returns
        -------
        PointCloudData
        """

        logger.info(f"Loading LAZ file: {path}")

        las = laspy.read(path)
        header_names = las._points.array.dtype.names

        data = las._points.array
        field_names = cls._validate_field_selection(scalar_fields, header_names, remove_prefix, prefix)

        if force_no_numerical_shift:
            pcd = PointCloudData(las.xyz, numerical_optimization_shift=None, **pcd_kw)
        else:
            # TODO error multiple "numerical_optimization_shift" passed in ("pcd_kw")
            pcd = PointCloudData(las.xyz, numerical_optimization_shift=OptimizedShift(las.header.offsets), **pcd_kw)
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
        """Save the point cloud data to a LAS/LAZ file.

        Parameters
        ----------
        pcd : PointCloudData
            Point cloud object
        path : str | Path
            Path to save the LAS/LAZ file to. File extension must be ".las" or ".laz".
        scalar_fields: list[str], default=None
            List of specific scalar fields to extract from the PLY file.
            Setting `None` will retrieve all scalar fields. Setting to `[]` will ignore scalar fields in the file.
        add_prefix: bool, default=False
            Flag to add prefixes on scalar field names
        prefix: str, default="scalar_"
            Prefix to strip from scalar field names if `remove_prefix` is True.
        revert_sf_types: bool, default=False
            Flag to revert scalar field values to their original types or not
        scales : Vector_3_T, default=[0.0001, 0.0001, 0.0001]
            Scaling factors for (x,y,z). Relates to data precision and reducing memory footprint in file
        config : dict[str, Any]

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

        # TODO the add prefix and revert_sf are not yet supported here
        # Remaining fields including normals as these are extra dimensions
        for field in scalar_fields:
            value = np.asarray(pcd.scalar_fields[field])

            if not hasattr(las, field):
                las.add_extra_dim(laspy.ExtraBytesParams(field, value.dtype))

            setattr(las, field, value)

        las.write(str(path))

        logger.info(f"Successfully wrote to  LAS/LAZ file: {path}")
