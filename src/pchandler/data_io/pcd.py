from pathlib import Path
import logging

# TODO decide if to use https://github.com/MapIV/pypcd4
#  It has already done similar work on "merge" / concatenating point clouds

from pchandler.data_io.core import AbstractIOHandler
from pchandler.geometry import PointCloudData

logger = logging.getLogger(__name__.split(".")[0])


class PcdHandler(AbstractIOHandler):
    FORMATS = ['.pcd']

    @classmethod
    def load(cls, /, path: str | Path, **config) -> PointCloudData:
        raise NotImplementedError

    @classmethod
    def save(cls, /, pcd: PointCloudData, path: str | Path, **config) -> None:
        raise NotImplementedError