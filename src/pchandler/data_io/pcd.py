# pchandler - Toolbox for point-cloud handling, processing and analysis
#
# Copyright (c) 2025, Nicholas Meyer, Geosensors and Engineering Geodesy,
# Institute of Geodesy and Photogrammetry, ETH Zurich, Switzerland
# SPDX-License-Identifier: BSD-3-Clause
#
# Author: Nicholas Meyer (meyernic@ethz.ch)

import logging
from pathlib import Path

from pchandler import PointCloudData
from pchandler.data_io.core import AbstractIOHandler

logger = logging.getLogger(__name__.split(".")[0])


class PcdHandler(AbstractIOHandler):
    FORMATS = [".pcd"]

    @classmethod
    def load(cls, /, path: str | Path, **config) -> PointCloudData:  # type: ignore[override]
        raise NotImplementedError()

    @classmethod
    def save(cls, /, pcd: PointCloudData, path: str | Path, **config) -> None:  # type: ignore[override]
        raise NotImplementedError()
