# pchandler - Toolbox for point-cloud handling, processing and analysis
#
# Copyright (c) 2025, Nicholas Meyer, Geosensors and Engineering Geodesy,
# Institute of Geodesy and Photogrammetry, ETH Zurich, Switzerland
# SPDX-License-Identifier: BSD-3-Clause
#
# Author: Nicholas Meyer (meyernic@ethz.ch)

"""PLY file format handler class"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Unpack

from plyfile import PlyData, PlyElement  # type: ignore

from pchandler import PointCloudData
from pchandler.data_io.core import AbstractIOHandler, PointCloudDataKW

__all__ = ["PlyHandler"]

logger = logging.getLogger(__name__.split(".")[0])


class PlyHandler(AbstractIOHandler):
    """Handles PLY file input and output.

    Supported file extensions:

    * .ply

    """

    FORMATS = [".ply"]

    @classmethod
    def load(  # type: ignore[override]
        cls,
        path: str | Path,
        /,
        scalar_fields: Optional[list[str]] = None,
        remove_prefix: bool = True,
        prefix: str = "scalar_",
        **pcd_kw: Unpack[PointCloudDataKW],
    ) -> PointCloudData:
        """Load a point cloud from a PLY file

        Parameters
        ----------
        path : str or Path
            Input PLY file path.
        scalar_fields : list of str, default=None
            List of specific scalar fields to extract from the PLY file.
            Setting `None` will retrieve all scalar fields. Setting to `[]` will ignore scalar fields in the file.
        remove_prefix : bool, default=True
            Flag to remove prefixes on scalar field names.
        prefix : str, default="scalar_"
            Prefix to strip from scalar field names if `remove_prefix` is True.
        **config : dict of str, Any

        Returns
        -------
        PointCloudData
        """
        logger.info(f"Loading PLY file: {path}")

        plydata = PlyData.read(path)

        num_points = plydata["vertex"].count
        logger.debug(f"PLY file {path} contains {num_points} points")
        file_fields = [pe.name for pe in plydata["vertex"].properties]

        field_names = cls._validate_field_selection(scalar_fields, file_fields, remove_prefix, prefix)

        pcd = PointCloudData(cls.extract_xyz(plydata["vertex"], num_points), **pcd_kw)
        cls.extract_scalar_fields(pcd, plydata["vertex"], num_points, field_names)

        return pcd

    @classmethod
    def save(  # type: ignore[override]
        cls,
        /,
        pcd: PointCloudData,
        path: str | Path,
        scalar_fields: Optional[list[str]] = None,
        add_prefix: bool = False,
        prefix: str = "scalar_",
        revert_sf_types: bool = False,
        as_ascii: bool = False,
        **config: dict[str, Any],
    ) -> None:
        """Save the point cloud data to a PLY file

        Parameters
        ----------
        pcd: PointCloudData
            Point cloud object
        path: str | Path
            Path to save the PLY file to. File extension must be ".ply".
        scalar_fields: list[str], default=None
            List of specific scalar fields to extract from the PLY file.
            Setting `None` will retrieve all scalar fields. Setting to `[]` will ignore scalar fields in the file.
        add_prefix: bool, default=False
            Flag to add prefixes on scalar field names
        prefix: str, default="scalar_"
            Prefix to strip from scalar field names if `remove_prefix` is True.
        revert_sf_types: bool, default=False
            Flag to revert scalar field values to their original types or not
        as_ascii: bool, default=False
            Write to ASCII file if `as_ascii` is True, otherwise write to binary file.
        config: dict[str, Any]
        """

        path = Path(path)

        prefix = prefix if add_prefix else ""

        structured_array = cls._generate_structured_array(pcd, scalar_fields, add_prefix, prefix, revert_sf_types)

        element = PlyElement.describe(
            structured_array,  # type: ignore
            name="vertex",
            comments=[
                "Created with dranjan/python-plyfile in gseg-ethz/pchandler",
                f"Created {datetime.now():%Y-%m-%dT%H:%M:%S%z}",
            ],
        )

        PlyData([element], text=as_ascii).write(path)

        logger.info(f"PLY file saved successfully: {path}")
