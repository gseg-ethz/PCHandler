from pathlib import Path
from typing import Unpack
import logging
from datetime import datetime

from plyfile import PlyData, PlyElement

from .core import AbstractIOHandler, _BaseLoadConfigType, _BaseSaveConfigType, BaseLoadConfig, BaseSaveConfig
from ..geometry import PointCloudData

logger = logging.getLogger(__name__.split(".")[0])


class _PlyLoadConfigType(_BaseLoadConfigType):
    pass

class _PlySaveConfigType(_BaseSaveConfigType):
    pass

class PlyLoadConfig(BaseLoadConfig):
    pass

class PlySaveConfig(BaseSaveConfig):
    pass

class PlyHandler(AbstractIOHandler):
    FORMATS = ['.ply']
    LOAD_CONFIG: type[BaseLoadConfig] = PlyLoadConfig
    SAVE_CONFIG: type[BaseSaveConfig] = PlySaveConfig

    @classmethod
    def load(cls, /, path: str | Path, **config: Unpack[_PlyLoadConfigType]):
        cfg = cls.get_config(**config)

        logger.info(f"Loading PLY file: {path}")

        try:
            with open(path, "rb") as f:
                plydata = PlyData.read(f)
        except Exception as e:
            logger.error(f"Failed to read PLY file {path}: {e}")
            raise e

        num_points = plydata["vertex"].count
        logger.debug(f"PLY file {path} contains {num_points} points")

        ply_scalar_fields = set([pe.name.lower() for pe in plydata["vertex"].properties])

        pcd = PointCloudData(cls._extract_xyz(plydata["vertex"], num_points, ply_scalar_fields))
        cls.extract_common_fields(pcd, plydata["vertex"], cfg, num_points, ply_scalar_fields)
        cls.extract_extra_fields(pcd, plydata["vertex"], cfg, ply_scalar_fields)

        return pcd


    # DISCUSS is it worth saving the optimised state with the np.float64 shift written in a header?
    @classmethod
    def save(cls, /, pcd: PointCloudData, path: str | Path, **config: Unpack[_PlySaveConfigType]):
        cfg = cls.get_config(**config, load=False)

        structured_array = cls.generate_structured_array(pcd, cfg)

        element = PlyElement.describe(
            structured_array,
            name="vertex",
            comments=[
                "Created with dranjan/python-plyfile in gseg-ethz/pchandler",
                f"Created {datetime.now():%Y-%m-%dT%H:%M:%S%z}",
            ],
        )

        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created {path.parent} folder")

        PlyData([element]).write(f"{path}")
        logger.info(f"PLY file saved successfully: {path}")