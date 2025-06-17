from pathlib import Path
from typing import Unpack, NotRequired
import logging
from datetime import datetime

import numpy as np
import laspy

from .core import AbstractIOHandler, _BaseLoadConfigType, _BaseSaveConfigType, BaseSaveConfig, BaseLoadConfig
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


class _LasLoadConfigType(_BaseLoadConfigType):
    pass

class _LasSaveConfigType(_BaseSaveConfigType):
    pass


class LasLoadConfig(BaseLoadConfig):
    pass

class LasSaveConfig(BaseSaveConfig):
    pass

class LasHandler(AbstractIOHandler):
    FORMATS = ['.las', '.laz']

    @classmethod
    def load(cls, /, path: str | Path, **config: Unpack[_LasLoadConfigType]):
        pass

    @classmethod
    def save(cls, /, pcd: PointCloudData, path: str | Path, **config: Unpack[_LasSaveConfigType]):
        pass