from pathlib import Path
from typing import Optional
import logging
from datetime import datetime

from plyfile import PlyData, PlyElement     # type: ignore

from .core import AbstractIOHandler, _LoadConfigType, _SaveConfigType, LoadConfig, SaveConfig
from ..geometry import PointCloudData

logger = logging.getLogger(__name__.split(".")[0])


class PlyHandler(AbstractIOHandler):
    FORMATS = ['.ply']

    @classmethod
    def load(cls, /, path: str | Path,
             scalar_fields: Optional[list[str]] = None,
             remove_scalar_prefix: bool = True,
             **config) -> PointCloudData:
        logger.info(f"Loading PLY file: {path}")

        try:
            with open(path, "rb") as f:
                plydata = PlyData.read(f)
        except Exception as e:
            logger.error(f"Failed to read PLY file {path}: {e}")
            raise e

        num_points = plydata["vertex"].count
        logger.debug(f"PLY file {path} contains {num_points} points")
        file_fields = [pe.name for pe in plydata["vertex"].properties]

        field_names = cls._validate_field_selection(scalar_fields, file_fields)

        pcd = PointCloudData(cls.extract_xyz(plydata["vertex"], num_points))
        cls.extract_scalar_fields(pcd, plydata["vertex"], num_points, field_names)

        return pcd


    # DISCUSS is it worth saving the optimised state with the np.float64 shift written in a header?
    @classmethod
    def save(cls, /,
             pcd: PointCloudData,
             path: str | Path,
             scalar_fields: Optional[list[str]] = None,
             revert_sf_types: bool = False,
             add_scalar_prefix: bool = False,
             **config) -> None:
        # TODO change bool scalar prefix to string with scalar_ as default
        path = Path(path)
        cfg = SaveConfig(**config)

        structured_array = cls.generate_structured_array(pcd, scalar_fields, revert_sf_types, add_scalar_prefix)

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