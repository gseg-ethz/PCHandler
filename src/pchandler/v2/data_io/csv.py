from pathlib import Path
from typing import Unpack, NotRequired, Iterable, Sequence, Any, Collection
import logging

import numpy as np
import numpy.typing as npt

from .core import AbstractIOHandler, _LoadConfigType, _SaveConfigType, SaveConfig, LoadConfig
from ..geometry import PointCloudData

logger = logging.getLogger(__name__.split(".")[0])


class _CsvLoadConfigType(_LoadConfigType):
    comment: NotRequired[str]
    skip_rows: NotRequired[int]
    delimiter: NotRequired[str]
    num_points_line: NotRequired[bool]
    column_names: NotRequired[Collection[str]]
    column_names_row: NotRequired[int]


class _CsvSaveConfigType(_SaveConfigType):
    delimiter: NotRequired[str]
    number_points_line: NotRequired[bool]
    header_lines: NotRequired[Collection[str]]


class CsvHandler(AbstractIOHandler):
    FORMATS = ['.csv', '.txt', '.xyz', '.asc', '.ascii', '.pts']

    @classmethod
    def load(cls, path: str | Path, **config: Unpack[_CsvLoadConfigType]) -> PointCloudData:
        cfg = LoadConfig(**config)
        path = Path(path)

        header, delimiter, column_names, num_cols, num_points = sniff_file(path, comment=cfg.comment)

        # Use the detected delimiter if none defined

        # Check for the number points line (first after header from cloud compare export)
        cfg.num_points_line = True if num_points else False
        use_columns = _get_col_indexes(cfg, column_names, num_cols)

        load_params: dict[str, Any] = {'fname': path}
        load_params['delimiter'] = cfg.delimiter or delimiter
        load_params['dtype'] = _generate_csv_load_dtype(cfg)
        load_params['comments'] = cfg.comment
        load_params['skiprows'] = len(header) + cfg.num_points_line

        if len(use_columns) != 0:
            load_params['usecols'] = use_columns

        data = np.loadtxt(**load_params)

        num_points = data.size

        field_names = set([name.lower() for name in cfg.column_names])

        pcd = PointCloudData(cls._extract_xyz(data, num_points, field_names))
        cls.extract_common_fields(pcd, data, cfg, num_points, field_names)
        cls.extract_extra_fields(pcd, data, cfg, field_names)

        return pcd

    @classmethod
    def save(cls, /, pcd: PointCloudData, path: str | Path, **config: Unpack[_CsvSaveConfigType]) -> None:
        cfg = SaveConfig(**config)
        array = cls.generate_structured_array(pcd, cfg)

        header = f"// {cfg.delimiter.join(list(array.dtype.names or ''))}"

        fmt_map = {
            "f": "%.6f",  # Default float format
            "u": "%u",  # Unsigned integer
            "i": "%d",  # Signed integer
        }

        # Use `"%s"` for fallback
        fmt = cfg.delimiter.join([fmt_map.get(field[1][1], "%s") for field in array.dtype.descr])
        if array.dtype.names is not None:
            array = np.stack([array[field] for field in list(array.dtype.names)], axis=-1)

            np.savetxt(
                path,
                array,
                fmt=fmt,
                delimiter=cfg.delimiter,
                header=header,
                comments="",  # Avoid prepending '#' to the header
            )

            logger.info(f"CSV file saved successfully: {path}")
        else:
            raise ValueError("No fields passed")


def sniff_file(file: Path,
               delimiters: tuple[str, ...] = (' ', ';', '\t', ','),
               names_row: int = -1,
               lines_to_check: int = 10,
               minimum_columns: int = 3,
               comment: str = "//") -> tuple[list[str], str, Sequence[str], int, int|None]:

    header, num_points = _get_header(file, comment)
    delimiter, number_fields = _delimiter_sniffer(file, delimiters, lines_to_check, minimum_columns, comment)

    if len(header) == 0:
        return header, delimiter, [], number_fields, num_points
    line: str = header[names_row]

    for i in (' ', delimiter):
        column_names = line.split(i)
        if len(column_names) == number_fields:
            return header, delimiter, column_names, number_fields, num_points

    raise ValueError(f"Header line does not appear to have column_names at row number {names_row}: {line=}")

def _get_field_counts(file: Path,
                      character: str,
                      lines_to_check: int = 10,
                      minimum_columns: int = 3,
                      comment: str = "//") -> int:

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
                return 0

            fields = line.split(character)
            field_counts.add(len(fields))

        # Ensure number of fields per line are consistent
        if len(field_counts) > 1:
            return 0

        # Check all lines have same number of columns
        num_fields = field_counts.pop()
        if num_fields < minimum_columns:
            return 0

        return num_fields

def _delimiter_sniffer(file: Path,
                       delimiters: str|Iterable[str] = (' ', ';', '\t', ','),
                       lines_to_check: int = 10,
                       minimum_columns: int = 3,
                       comment: str = "//") -> tuple[str, int]:

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

def _generate_csv_load_dtype(cfg: LoadConfig) -> npt.DTypeLike:
    elements = []
    for name in cfg.column_names:
        if name.lower() in ('x', 'y', 'z'):
            elements.append((name.lower(), 'f8'))
        else:
            elements.append((name.lower(), 'f4'))
    return np.dtype(elements)

def _get_col_indexes(cfg: LoadConfig, header_names: Sequence[str], num_cols: int) -> Sequence[int]:
    # TODO by default, header_names starts with {'x', 'y', 'z'}
    if len(header_names) == 0 or not cfg.column_names:
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
        cfg.column_names = list(cfg.column_names)[:num_cols]
        return tuple([i for i in range(num_cols)])

    if len(set(cfg.column_names)) > len(set(header_names)):
        raise ValueError("There are more user defined column names than ")

    return tuple([header_names.index(name) for name in cfg.column_names])