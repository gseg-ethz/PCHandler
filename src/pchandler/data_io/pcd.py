import logging
from pathlib import Path

from pchandler import PointCloudData
from pchandler.data_io.core import AbstractIOHandler

logger = logging.getLogger(__name__.split(".")[0])


class PcdHandler(AbstractIOHandler):
    """
    Handles operations for loading and saving Point Cloud Data (.pcd) files.

    This class defines methods to load point cloud data from .pcd files and save point cloud
    data to .pcd files. It provides functionality to handle .pcd format through standardized
    methods for input and output operations.

    Parameters
    ----------
    FORMATS : list of str
        List of file formats supported by the handler.
    """
    FORMATS = [".pcd"]

    @classmethod
    def load(cls, /, path: str | Path, **config) -> PointCloudData:  # type: ignore[override]
        """
        Loads point cloud data from a specified file path with optional configuration.

        Parameters
        ----------
        path : str or Path
            Path to the file containing point cloud data.
        **config : dict
            Additional configuration options for loading the point cloud data.

        Returns
        -------
        PointCloudData
            An instance of PointCloudData loaded from the specified path.

        Raises
        ------
        NotImplementedError
            Raised when the method is not implemented.
        """
        raise NotImplementedError()

    @classmethod
    def save(cls, /, pcd: PointCloudData, path: str | Path, **config) -> None:  # type: ignore[override]
        """
        Save a point cloud data object to a specified file path, using a given configuration.

        Parameters
        ----------
        pcd : PointCloudData
            The point cloud data to be saved.
        path : str or Path
            The file path where the point cloud data should be saved.
        **config
            Additional configuration options for saving the point cloud.

        Returns
        -------
        None

        Raises
        ------
        NotImplementedError
            Indicates that the method is not implemented.
        """
        raise NotImplementedError()
