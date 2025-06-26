from pathlib import Path
from typing import Unpack, NotRequired
import logging
from datetime import datetime

import numpy as np
# TODO decide if to use https://github.com/MapIV/pypcd4
#  It has already done similar work on "merge" / concatenating point clouds

from .core import AbstractIOHandler, _LoadConfigType, _SaveConfigType, SaveConfig, LoadConfig
from ..geometry import PointCloudData
from ..constants import (
    RGB_FIELD,
    NORMALS_FIELD,
    INTENSITY_FIELD,
    REFLECTANCE_FIELD,
    RGB_WORD,
    NORMAL_PARTIAL_NAMES
)

logger = logging.getLogger(__name__.split(".")[0])


class _PcdLoadConfigType(_LoadConfigType):
    pass

class _PcdSaveConfigType(_SaveConfigType):
    pass


class PcdHandler(AbstractIOHandler):
    FORMATS = ['.pcd']

    @classmethod
    def load(cls, /, path: str | Path, **config: Unpack[_PcdLoadConfigType]) -> PointCloudData:
        raise NotImplementedError

    @classmethod
    def save(cls, /, pcd: PointCloudData, path: str | Path, **config: Unpack[_PcdSaveConfigType]) -> None:
        raise NotImplementedError