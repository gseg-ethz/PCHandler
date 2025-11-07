from pathlib import Path

from pchandler import PointCloudData
from pchandler.data_io.core import SUPPORTED_TYPES
from pchandler.data_io import las, csv, e57, ply


def load_file(file_path: str |Path) -> PointCloudData:
    file_path = Path(file_path)

    data_loaders = {
        '.las': las.LasHandler.load,
        '.laz': las.LasHandler.load,
        '.txt': csv.CsvHandler.load,
        '.asc': csv.CsvHandler.load,
        '.csv': csv.CsvHandler.load,
        '.pts': csv.CsvHandler.load,
        '.e57': e57.E57Handler.load,
        '.ply': ply.PlyHandler.load,
    }

    if file_path.suffix not in SUPPORTED_TYPES:
        raise ValueError(f"File suffix {file_path.suffix} is not supported. It should be in {SUPPORTED_TYPES}")

    return data_loaders[file_path.suffix](file_path)

