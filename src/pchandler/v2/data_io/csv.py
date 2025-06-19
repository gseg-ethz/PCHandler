from pathlib import Path
from typing import Unpack, NotRequired, Iterable, Optional, Sequence
import logging

import numpy as np
from pydantic import Field

from .core import AbstractIOHandler, _BaseLoadConfigType, _BaseSaveConfigType, BaseSaveConfig, BaseLoadConfig
from ..geometry import PointCloudData

logger = logging.getLogger(__name__.split(".")[0])


class _CsvLoadConfigType(_BaseLoadConfigType):
    comment: NotRequired[str]
    skip_rows: NotRequired[int]
    delimiter: NotRequired[str]
    num_points_line: NotRequired[bool]
    column_names: NotRequired[Iterable[str]]
    column_names_row: NotRequired[int]


class _CsvSaveConfigType(_BaseSaveConfigType):
    delimiter: NotRequired[str]
    number_points: NotRequired[bool]
    header_lines: NotRequired[Iterable[str]]


class CsvLoadConfig(BaseLoadConfig):
    comment: str = '//'
    skip_rows: Optional[int] = None
    delimiter: Optional[str] = None
    num_points_line: Optional[bool] = None
    column_names: Sequence[str] = Field(default_factory=list)
    column_names_row: int = Field(default=-1, le=-1, ge=100)


class CsvSaveConfig(BaseSaveConfig):
    delimiter: str = ' '
    number_points_line: bool = False
    header_lines: Sequence[str] = Field(default_factory=list)


class CsvHandler(AbstractIOHandler):
    FORMATS = ['.csv', '.txt', '.xyz', '.asc', '.ascii', '.pts']
    LOAD_CONFIG: type[CsvLoadConfig] = CsvLoadConfig
    SAVE_CONFIG: type[CsvSaveConfig] = CsvSaveConfig

    @classmethod
    def load(cls, /, path: str | Path, **config: Unpack[_CsvLoadConfigType]):
        cfg: CsvLoadConfig = cls.get_config(**config)

        header, delimiter, column_names, num_cols, num_points = sniff_file(path, comment=cfg.comment)

        # Use the detected delimiter if none defined
        delimiter = cfg.delimiter or delimiter

        # Check for the number points line (first after header from cloud compare export)
        cfg.num_points_line = True if num_points else False

        use_columns = _get_col_indexes(cfg, column_names, num_cols)
        dtype = _generate_csv_load_dtype(cfg)
        skip_rows = len(header) + cfg.num_points_line

        data = np.loadtxt(fname=path,
                          dtype=dtype,
                          comments=cfg.comment,
                          delimiter=delimiter,
                          skiprows=skip_rows,
                          usecols=use_columns
                          )

        num_points = data.size

        field_names = set([name.lower() for name in cfg.column_names])

        pcd = PointCloudData(cls._extract_xyz(data, num_points, field_names))
        cls.extract_common_fields(pcd, data, cfg, num_points, field_names)
        cls.extract_extra_fields(pcd, data, cfg, field_names)

        return pcd

    @classmethod
    def save(cls, /, pcd: PointCloudData, path: str | Path, **config: Unpack[_CsvSaveConfigType]):
        cfg = cls.get_config(**config, load=False)
        array = cls.generate_structured_array(pcd, cfg)

        header = f"// {cfg.delimiter.join(array.dtype.names)}"

        fmt_map = {
            "f": "%.6f",  # Default float format
            "u": "%u",  # Unsigned integer
            "i": "%d",  # Signed integer
        }

        # Use `"%s"` for fallback
        fmt = cfg.delimiter.join([fmt_map.get(field[1][1], "%s") for field in array.dtype.descr])
        array = np.stack([array[field] for field in array.dtype.names], axis=-1)

        np.savetxt(
            path,
            array,
            fmt=fmt,
            delimiter=cfg.delimiter,
            header=header,
            comments="",  # Avoid prepending '#' to the header
        )

        logger.info(f"CSV file saved successfully: {path}")


def sniff_file(file,
               delimiters = (' ', ';', '\t', ','),
               names_row: int = -1,
               lines_to_check: int = 10,
               minimum_columns: int = 3,
               comment: str = "//") -> tuple[list[str], str, tuple[str, ...]|None, int, int|None]:

    header, num_points = _get_header(file, comment)
    delimiter, number_fields = _delimiter_sniffer(file, delimiters, lines_to_check, minimum_columns, comment)

    if len(header) == 0:
        return header, delimiter, None, number_fields, num_points
    line: str = header[names_row]

    for i in (' ', delimiter):
        column_names = line.split(i)
        if len(column_names) == number_fields:
            return header, delimiter, tuple(column_names), number_fields, num_points

    raise ValueError(f"Header line does not appear to have column_names at row number {names_row}: {line=}")

def _get_field_counts(file: Path, character: str, lines_to_check: int = 10, minimum_columns: int = 3, comment: str = "//"):

    header, number_points = _get_header(file, comment)
    skip_lines = len(header)

    if number_points is not None:
        skip_lines += 1

    field_counts = set()

    with open(file, "r") as f:
        for i in range(skip_lines):
            f.readline()

        for i in range(lines_to_check):
            line = f.readline()

            if not line:
                break

            line = line.rstrip('\n\r')

            # Case for empty line
            if not line.strip():
                continue

            if character not in line:
                return False

            fields = line.split(character)
            field_counts.add(len(fields))

        # Ensure number of fields per line are consistent
        if len(field_counts) > 1:
            return False

        # Check all lines have same number of columns
        num_fields = field_counts.pop()
        if num_fields < minimum_columns:
            return False

        return num_fields

def _delimiter_sniffer(file: Path,
                       delimiters = (' ', ';', '\t', ','),
                       lines_to_check: int = 10,
                       minimum_columns: int = 3,
                       comment: str = "//"):

    for delimiter in delimiters:
        number_fields = _get_field_counts(file, delimiter, lines_to_check, minimum_columns, comment)
        if number_fields:
            return delimiter, number_fields

    else:
        raise ValueError(f"No valid delimiter was not found in selection: {repr(delimiters)}")


def _get_header(file: Path, comment: str = '//') -> tuple[list[str], int|None]:
    with open(file, "r") as f:
        header = []
        while True:
            line = f.readline()

            if not line.startswith(comment):
                line = line.strip("\n\r").strip()
                number_points = int(line) if line.isdigit() else None
                break

            header.append(line.lstrip('//').strip('\n\r').strip())

    return header, number_points

def _generate_csv_load_dtype(cfg):
    elements = []
    for name in cfg.column_names:
        if name.lower() in ('x', 'y', 'z'):
            elements.append((name.lower(), 'f8'))
        else:
            elements.append((name.lower(), 'f4'))
    return np.dtype(elements)

def _get_col_indexes(cfg: CsvLoadConfig, header_names: Sequence[str], num_cols: int) -> Sequence[int] | None:
    # TODO by default, header_names starts with {'x', 'y', 'z'}
    if header_names is None or not cfg.column_names:
        if num_cols == 3:
            cfg.column_names = ['x', 'y', 'z']
            return 0, 1, 2
        elif len(header_names) == num_cols:
            cfg.column_names = header_names
            return tuple([i for i in range(num_cols)])
        else:
            raise ValueError(f"Unknown fields in text file detect. {num_cols=} > 3.\n"
                             f"Please define the column names")

    if len(cfg.column_names) > num_cols:
        logger.warning(f"More columns are detected than field names provided. "
                       f"Extracting the first {len(cfg.column_names)} fields as {cfg.column_names}")
        cfg.column_names = cfg.column_names[:num_cols]
        return tuple([i for i in range(num_cols)])

    if len(set(cfg.column_names)) > len(set(header_names)):
        raise ValueError("There are more user defined column names than ")

    return tuple([header_names.index(name) for name in cfg.column_names])