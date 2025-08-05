import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from plyfile import PlyData, PlyElement  # type: ignore

from pchandler import PointCloudData
from pchandler.data_io.core import AbstractIOHandler

logger = logging.getLogger(__name__.split(".")[0])


class PlyHandler(AbstractIOHandler):
    FORMATS = [".ply"]

    @classmethod
    def load(  # type: ignore[override]
        cls,
        path: str | Path,
        /,
        scalar_fields: Optional[list[str]] = None,
        remove_prefix: bool = True,
        prefix: str = "scalar_",
        **config: dict[str, Any],
    ) -> PointCloudData:
        logger.info(f"Loading PLY file: {path}")

        plydata = PlyData.read(path)

        num_points = plydata["vertex"].count
        logger.debug(f"PLY file {path} contains {num_points} points")
        file_fields = [pe.name for pe in plydata["vertex"].properties]

        field_names = cls._validate_field_selection(scalar_fields, file_fields, remove_prefix, prefix)

        pcd = PointCloudData(cls.extract_xyz(plydata["vertex"], num_points))
        cls.extract_scalar_fields(pcd, plydata["vertex"], num_points, field_names)

        return pcd

    @classmethod
    def save(  # type: ignore[override]
        cls,
        /,
        pcd: PointCloudData,
        path: str | Path,
        scalar_fields: Optional[list[str]] = None,
        add_prefix: bool = False,
        prefix: str = "scalar_",
        revert_sf_types: bool = False,
        as_ascii: bool = False,
        **config: dict[str, Any],
    ) -> None:

        path = Path(path)

        prefix = prefix if add_prefix else ""

        structured_array = cls._generate_structured_array(pcd, scalar_fields, add_prefix, prefix, revert_sf_types)

        element = PlyElement.describe(
            structured_array,  # type: ignore
            name="vertex",
            comments=[
                "Created with dranjan/python-plyfile in gseg-ethz/pchandler",
                f"Created {datetime.now():%Y-%m-%dT%H:%M:%S%z}",
            ],
        )

        PlyData([element], text=as_ascii).write(path)

        logger.info(f"PLY file saved successfully: {path}")
