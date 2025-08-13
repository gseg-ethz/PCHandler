import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from plyfile import PlyData, PlyElement  # type: ignore

from pchandler import PointCloudData
from pchandler.data_io.core import AbstractIOHandler

logger = logging.getLogger(__name__.split(".")[0])


class PlyHandler(AbstractIOHandler):
    """
    Handles PLY file input and output.

    This class provides methods for loading and saving PLY files as point cloud data. It supports
    various configurations for managing scalar fields, prefixes, and file format (ASCII or binary).
    PLY files must contain vertex data to properly function with this handler.

    Parameters
    ----------
    FORMATS : list of str
        Supported file formats for the handler.
    """
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
        """
        Loads a PointCloudData object from a PLY file, supporting optional scalar field extraction
        and field name customization.

        Parameters
        ----------
        path : str or Path
            Path to the PLY file to load.
        scalar_fields : list of str, optional
            List of scalar fields to extract from the PLY file. If None, no scalar fields
            are extracted.
        remove_prefix : bool, optional
            Indicates whether prefixes in scalar field names should be removed while extracting.
        prefix : str, optional
            Prefix to identify scalar field names. Used in combination with `remove_prefix`.
        **config : dict of str, Any
            Additional configuration parameters.

        Returns
        -------
        PointCloudData
            A PointCloudData instance containing the loaded point cloud data.
        """
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
        """
        Saves the point cloud data to a PLY file.

        This class method processes the given point cloud data and writes it to a
        PLY file format at the specified path. It can selectively include scalar
        fields and customize naming conventions for the fields.

        Parameters
        ----------
        pcd : PointCloudData
            Input point cloud data to be saved.
        path : str or Path
            The destination file path for the saved PLY file.
        scalar_fields : list of str, optional
            List of scalar field names to include in the file. If None, all fields
            are included.
        add_prefix : bool, optional
            Whether to add a prefix to scalar field names.
        prefix : str, optional
            The prefix string to be added to the field names, if `add_prefix` is True.
        revert_sf_types : bool, optional
            Specifies whether to revert scalar fields to their original data types.
        as_ascii : bool, optional
            If True, the PLY file will be written in ASCII format; otherwise, it
            will use binary format.
        config : dict of str, Any
            Additional configuration for customization during file saving.
        """

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
