"""Top-level convenience loader that dispatches to the per-format handler by suffix."""

from pathlib import Path

from pchandler import PointCloudData
from pchandler.data_io.core import SUPPORTED_TYPES


def load_file(file_path: str | Path, **kwargs) -> PointCloudData:
    """Load a point cloud from disk by dispatching on the file suffix.

    Recognised suffixes are listed in :data:`SUPPORTED_TYPES`
    (``.las`` / ``.laz`` / ``.txt`` / ``.asc`` / ``.csv`` / ``.pts`` /
    ``.e57`` / ``.ply``).

    Parameters
    ----------
    file_path : str | Path
        Path to the point-cloud file to load.
    **kwargs : Any
        Additional keyword arguments forwarded to the per-format handler's
        ``load`` method.

    Returns
    -------
    PointCloudData
        The loaded point cloud.

    Raises
    ------
    ValueError
        If the file's suffix is not in :data:`SUPPORTED_TYPES`.
    """
    file_path = Path(file_path)

    if file_path.suffix not in SUPPORTED_TYPES:
        raise ValueError(f"File suffix {file_path.suffix} is not supported. It should be in {SUPPORTED_TYPES}")

    # Lazy import only the required handler based on file extension
    suffix_to_handler = {
        ".las": ("Las", "load"),
        ".laz": ("Las", "load"),
        ".txt": ("Csv", "load"),
        ".asc": ("Csv", "load"),
        ".csv": ("Csv", "load"),
        ".pts": ("Csv", "load"),
        ".e57": ("E57", "load"),
        ".ply": ("Ply", "load"),
    }

    handler_name, method_name = suffix_to_handler[file_path.suffix]

    # Import only the required handler using the lazy loading mechanism
    from pchandler import data_io

    handler = getattr(data_io, handler_name)
    loader = getattr(handler, method_name)

    return loader(file_path, **kwargs)
