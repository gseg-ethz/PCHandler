from pathlib import Path
from typing import Unpack, NotRequired
import logging
from datetime import datetime

import numpy as np
# TODO decide if to use https://github.com/MapIV/pypcd4
#  It has already done similar work on "merge" / concatenating point clouds

from .core import AbstractIOHandler
from ..geometry import PointCloudData
from ..constants import (
    RGB_NAMES,
    NORMAL_NAMES,
    INTENSITY_NAMES,
    REFLECTANCE_NAMES
)

logger = logging.getLogger(__name__.split(".")[0])


class PcdHandler(AbstractIOHandler):
    FORMATS = ['.pcd']

    @classmethod
    def load(cls, /, path: str | Path, **config: Unpack[_PcdLoadConfigType]) -> PointCloudData:
        raise NotImplementedError

    @classmethod
    def save(cls, /, pcd: PointCloudData, path: str | Path, **config: Unpack[_PcdSaveConfigType]) -> None:
        raise NotImplementedError