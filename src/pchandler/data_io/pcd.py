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

"""PCD file-format handler class (not yet implemented)."""

import logging
from pathlib import Path

from pchandler import PointCloudData
from pchandler.data_io.core import AbstractIOHandler

logger = logging.getLogger(__name__.split(".")[0])


class PcdHandler(AbstractIOHandler):
    """Handles PCD file input and output.

    Supported file extensions:

    * .pcd
    """

    FORMATS = [".pcd"]

    @classmethod
    def load(cls, /, path: str | Path, **config) -> PointCloudData:  # type: ignore[override]
        """Load a point cloud from a PCD file.

        *Not yet implemented.*

        Parameters
        ----------
        path : str | Path
            Input PCD file path.
        **config : Any
            Additional configuration (currently unused).

        Returns
        -------
        PointCloudData
            Would return the loaded point cloud.

        Raises
        ------
        NotImplementedError
            Always — PCD loading is not yet implemented.
        """
        raise NotImplementedError()

    @classmethod
    def save(cls, /, pcd: PointCloudData, path: str | Path, **config) -> None:  # type: ignore[override]
        """Save a point cloud to a PCD file.

        *Not yet implemented.*

        Parameters
        ----------
        pcd : PointCloudData
            Point cloud to save.
        path : str | Path
            Output PCD file path.
        **config : Any
            Additional configuration (currently unused).

        Raises
        ------
        NotImplementedError
            Always — PCD saving is not yet implemented.
        """
        raise NotImplementedError()
