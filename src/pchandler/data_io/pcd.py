# pchandler - Toolbox for point-cloud handling, processing and analysis
#
# Copyright (c) 2025, Nicholas Meyer, Geosensors and Engineering Geodesy,
# Institute of Geodesy and Photogrammetry, ETH Zurich, Switzerland
# SPDX-License-Identifier: BSD-3-Clause
#
# Author: Nicholas Meyer (meyernic@ethz.ch)

"""PCD file format handler class"""

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
        """Load a point cloud from a PCD file

        ** NOT IMPLEMENTED YET **

        Parameters
        ----------
        path : str or Path
            Input PCD file path.
        **config : dict

        Returns
        -------
        PointCloudData

        Raises
        ------
        NotImplementedError
            Raised when the method is not implemented.
        """
        raise NotImplementedError()

    @classmethod
    def save(cls, /, pcd: PointCloudData, path: str | Path, **config) -> None:  # type: ignore[override]
        """Save the point cloud data to a PCD file

        ** NOT IMPLEMENTED YET **

        Parameters
        ----------
        pcd : PointCloudData
        path : str or Path
        **config

        Returns
        -------

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError()
