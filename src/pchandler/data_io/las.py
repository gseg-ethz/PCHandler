# pchandler – Toolbox for point-cloud handling, processing and analysis
#
# Copyright (c) 2022–2026 ETH Zurich
# Department of Civil, Environmental and Geomatic Engineering (D-BAUG)
# Institute of Geodesy and Photogrammetry
# Geosensors and Engineering Geodesy
#
# Authors:
#   Nicholas Meyer
#   Jon Allemand
#
# SPDX-License-Identifier: BSD-3-Clause

# pchandler - Toolbox for point-cloud handling, processing and analysis
#
# Copyright (c) 2025, Nicholas Meyer, Geosensors and Engineering Geodesy,
# Institute of Geodesy and Photogrammetry, ETH Zurich, Switzerland
# SPDX-License-Identifier: BSD-3-Clause
#
# Author: Nicholas Meyer (meyernic@ethz.ch)

"""LAS/LAZ file-format handler class."""

import logging
from pathlib import Path
from typing import Optional, Unpack

import laspy
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

        # BUG-08 (D-16): caller wins on numerical_optimization_shift. If the caller
        # passed it in pcd_kw, that value is used; otherwise the LAS-header-derived
        # default applies (unless force_no_numerical_shift suppresses it entirely).
        nos: Optional[OptimizedShift] = pcd_kw.pop(
            "numerical_optimization_shift",
            None if force_no_numerical_shift else OptimizedShift(las.header.offsets),
        )
        pcd = PointCloudData(las.xyz, numerical_optimization_shift=nos, **pcd_kw)
        cls.extract_scalar_fields(pcd, data, data.size, field_names)

        logger.info(f"Successfully loaded LAZ file: {path}")
        return pcd

    @classmethod
    def save(  # noqa: C901  # Multi-format LAS write — branching tracks laspy's per-field path; refactor deferred to Phase 6.
        cls,  # type: ignore[override]
        pcd: PointCloudData,
        path: str | Path,
        /,
        scalar_fields: Optional[list[str]] = None,
        add_prefix: bool = True,
        prefix: str = "scalar_",
        revert_sf_types: bool = False,
        scales: Optional[Vector_3_T] = None,
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
        scales : Vector_3_T | None, default=None
            Scaling factors for (x,y,z). Relates to data precision and reducing memory footprint in file.
            Defaults to ``[0.0001, 0.0001, 0.0001]`` when ``None``.
        config : dict[str, Any]

        Returns
        -------
        None
        """
        if scales is None:
            scales = np.array([0.0001, 0.0001, 0.0001])
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

        # RGB values — walrus always binds rgb_fields so the cleanup loop below
        # can unconditionally remove those names from scalar_fields regardless
        # of whether pcd.rgb is present.
        rgb_fields: list[str] = _get_rgb_or_normal_field_names(scalar_fields, RGB_NAMES)
        if rgb_fields and pcd.rgb is not None:
            for field in rgb_fields:
                index = RGB_NAMES.get_position(field)
                setattr(las, RGB_NAMES.words[index], getattr(pcd.rgb, RGB_NAMES.char[index]))

        # Intensities - LAS expects unsigned 16bit (Uint16)
        intensity_fields: set[str] = set(scalar_fields).intersection(INTENSITY_NAMES.all)
        if intensity_fields and pcd.intensity is not None:
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
                las.intensity = normalize_uint16(pcd.intensity, source_range=(0.0, 1.0))

        # Clear the previous sfs used
        for name in XYZ_NAMES.char + tuple(rgb_fields) + tuple(intensity_fields):
            if name in scalar_fields:
                scalar_fields.remove(name)

        # Remaining fields including normals as extra dimensions.
        # Use the shared helper so add_prefix and revert_sf_types are honoured identically
        # to PLY/CSV save (CONTEXT D-08 / PATTERNS.md S-3).
        residual_scalar_fields: list[str] = scalar_fields
        if residual_scalar_fields:
            structured_array = cls._generate_structured_array(
                pcd, residual_scalar_fields, add_prefix, prefix, revert_sf_types
            )
            # _generate_structured_array always includes XYZ columns — skip them here
            # since LAS handles XYZ via las.xyz above.
            xyz_col_set = set(XYZ_NAMES.char)
            for col_name in structured_array.dtype.names:
                if col_name in xyz_col_set:
                    continue
                # 31-char guard: LAS 1.4 spec restricts extra-dim names to 32 bytes
                # including null terminator → max 31 ASCII chars (RESEARCH §"Pitfall 2").
                if len(col_name) > 31:
                    raise ValueError(
                        f"LasHandler.save: extra-dim name {col_name!r} exceeds laspy's 31-char limit "
                        f"(LAS 1.4 spec: 32-byte field, null-terminated). "
                        f"Consider add_prefix=False or a shorter scalar field name."
                    )
                column = structured_array[col_name]
                # Only register the column as an extra dim if it is not already a
                # standard LAS point-format dimension (e.g. scan_angle_rank, gps_time).
                if not hasattr(las, col_name):
                    las.add_extra_dim(laspy.ExtraBytesParams(col_name, column.dtype))
                setattr(las, col_name, column)

        las.write(str(path))

        logger.info(f"Successfully wrote to  LAS/LAZ file: {path}")
