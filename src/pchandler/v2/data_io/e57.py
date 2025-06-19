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


class _E57LoadConfigType(_BaseLoadConfigType):
    pass


class _E57SaveConfigType(_BaseSaveConfigType):
    pass


class E57LoadConfig(BaseLoadConfig):
    pass


class E57SaveConfig(BaseSaveConfig):
    pass

class E57Handler(AbstractIOHandler):
    FORMATS = ['.e57']
    LOAD_CONFIG: type[E57LoadConfig] = E57LoadConfig
    SAVE_CONFIG: type[E57SaveConfig] = E57SaveConfig

    @classmethod
    def load(cls, /, path: str | Path, **config: Unpack[_E57LoadConfigType]):
        pass

    @classmethod
    def save(cls, /, pcd: PointCloudData, path: str | Path, **config: Unpack[_E57SaveConfigType]):
        pass