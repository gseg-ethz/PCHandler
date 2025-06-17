import csv
from pathlib import Path
from typing import Unpack, NotRequired
import logging
from datetime import datetime

import numpy as np

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

# TODO implement sniffer to derive delimiter... not a priority
def delimiter_sniffer(file: Path, sample_size=1024, delimiters=' ,;'):
    pass

# TODO implement function to get header based on comment string. Returns number of lines and the header lines
def check_for_header(file: Path, sample_size=1024) -> tuple[int, list[str]]:
    pass

# TODO
def get_column_names(file: Path, header_row: int = 0, comment: str = "//") -> tuple[str]:
    pass



class _CsvLoadConfigType(_BaseLoadConfigType):
    skip_rows: NotRequired[int]

class _CsvSaveConfigType(_BaseSaveConfigType):
    pass


class CsvLoadConfig(BaseLoadConfig):
    pass

class CsvSaveConfig(BaseSaveConfig):
    pass

class CsvHandler(AbstractIOHandler):
    FORMATS = ['.csv', '.txt', '.xyz', '.asc', '.ascii', '.pts']

    @classmethod
    def load(cls, /, path: str | Path, **config: Unpack[_CsvLoadConfigType]):
        pass

    @classmethod
    def save(cls, /, pcd: PointCloudData, path: str | Path, **config: Unpack[_CsvSaveConfigType]):
        pass