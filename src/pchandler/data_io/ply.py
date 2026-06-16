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

"""PLY file-format handler class."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Unpack

from plyfile import PlyData, PlyElement

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
        r"""Load a point cloud from a PLY file.

        Parameters
        ----------
        path : str | Path
            Input PLY file path.
        scalar_fields : list[str], default=None
            Specific scalar fields to extract from the PLY file.
            ``None`` retrieves all scalar fields; ``[]`` ignores scalar fields.
        remove_prefix : bool, default=True
            If ``True``, strip ``prefix`` from scalar-field names.
        prefix : str, default="scalar\_"
            Prefix to strip from scalar-field names if ``remove_prefix`` is ``True``.
        **pcd_kw : Unpack[PointCloudDataKW]
            Additional keyword arguments forwarded to :class:`PointCloudData`.

        Returns
        -------
        PointCloudData
            The loaded point cloud.
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
        r"""Save the point cloud data to a PLY file.

        Parameters
        ----------
        pcd : PointCloudData
            Point cloud to save.
        path : str | Path
            Output PLY file path (suffix must be ``.ply``).
        scalar_fields : list[str], default=None
            Specific scalar fields to write. ``None`` writes every field on
            the cloud; ``[]`` writes XYZ only.
        add_prefix : bool, default=False
            If ``True``, prepend ``prefix`` to scalar-field column names.
        prefix : str, default="scalar\_"
            Prefix to prepend when ``add_prefix`` is ``True``.
        revert_sf_types : bool, default=False
            If ``True``, restore each scalar field's original on-disk dtype.
        as_ascii : bool, default=False
            Write as ASCII PLY if ``True``, otherwise write a binary PLY.
        **config : Any
            Additional configuration (currently unused).
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
