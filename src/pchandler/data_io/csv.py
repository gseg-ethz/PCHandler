from pathlib import Path
from typing import Iterable, Optional, NamedTuple, Any
import logging

import numpy as np
import numpy.typing as npt

from pchandler.data_io.core import AbstractIOHandler
from pchandler.constants import XYZ_NAMES, RGB_NAMES
from pchandler.geometry import PointCloudData

logger = logging.getLogger(__name__.split(".")[0])


class AsciiInfo(NamedTuple):
    header: list[str]
    delimiter: str
    fields: list[str]
    num_fields: int
    num_points: int|None


class CsvHandler(AbstractIOHandler):
    FORMATS = ['.txt', '.csv', '.xyz', '.asc', '.ascii', '.pts']

    @classmethod
    def load(cls,
             path: str | Path, /,
             scalar_fields: Optional[list[str]] = None,
             remove_prefix: bool = True,
             prefix: str = 'scalar_',
             column_names_row: int = -1,
             comment: str = '//',
             delimiter: Optional[str] = None,) -> PointCloudData:

        # TODO need to simplify the case when only a 3 column array is passed (with columns 'X Y Z').

        # Get general file structure information
        file_info = sniff_file(path, comment=comment, field_names_row_index=column_names_row)
        num_points_line = True if file_info.num_points else False

        # Validate the fields to be output
        field_names = cls._validate_field_selection(scalar_fields, file_info.fields, remove_prefix, prefix)

        # Define load config parameters

        if tuple([k for k in field_names.keys()]) == XYZ_NAMES.char:
            dt = generate_ascii_load_dtype([k.lower() for k in field_names.values()])
        else:
            dt = generate_ascii_load_dtype(['x', 'y', 'z'] + list(field_names.values()))

        load_config: dict[str, Any] = {
            'fname': Path(path), 'skiprows': len(file_info.header) + num_points_line,
            'delimiter': delimiter or file_info.delimiter,
            'dtype': dt,
            'usecols': None
        }

        # When number of scalar_fields match, assumes all fields are in the same order
        if len(field_names) + 3 <= file_info.num_fields:
            load_config['usecols'] = [0, 1, 2] + [file_info.fields.index(name) for name in field_names.values()]
        elif len(field_names) == 3:
            load_config['usecols'] = [0, 1, 2]

        # Load all data
        data = np.loadtxt(**load_config)
        pcd = PointCloudData(cls.extract_xyz(data, data.size))

        if not tuple([k for k in field_names.keys()]) == XYZ_NAMES.char:
            cls.extract_scalar_fields(pcd, data, data.size, field_names)

        return pcd

    @classmethod
    def save(cls,
             pcd: PointCloudData,
             path: str | Path,
             scalar_fields: Optional[list[str]] = None,
             add_prefix: bool = True,
             prefix: str = 'scalar_',
             revert_sf_types: bool = False,
             delimiter: Optional[str] = ',') -> None:

        array = cls._generate_structured_array(pcd, scalar_fields, add_prefix, prefix, revert_sf_types)

        header = f"// {delimiter.join(list(array.dtype.names or ''))}"

        fmt_map = { "f": "%.6f", "u": "%u", "i": "%d", }
        # Use `"%s"` for fallback
        fmt = delimiter.join([fmt_map.get(field[1][1], "%s") for field in array.dtype.descr])

        array = np.stack([array[field] for field in list(array.dtype.names)], axis=-1)

        # Avoid prepending '#' to the header with comments
        np.savetxt( path, array, fmt=fmt, delimiter=delimiter, header=header, comments="" )
        logger.info(f"CSV file saved successfully: {path}")


def sniff_file(file: Path,
               delimiters: tuple[str, ...] = (' ', ';', '\t', ','),
               field_names_row_index: int = -1,
               lines_to_check: int = 10,
               minimum_columns: int = 3,
               comment: str = "//") -> AsciiInfo:

    # Read the header information defined by the comment section of the Ascii File and check if number of points is the
    # first line
    header, num_points = _get_header(file, comment)
    delimiter, number_fields = _delimiter_sniffer(file, delimiters, lines_to_check, minimum_columns, comment)

    # Get the field names based on the defined row_index. default is the last line of the header (-1)
    line: str = header[field_names_row_index].removeprefix(comment)

    # Test both the delimiter defined and white space as to the joining character between field name header info
    for i in (' ', delimiter):
        field_names = line.split(i)
        if len(field_names) == number_fields:
            return AsciiInfo(header, delimiter, field_names, number_fields, num_points)

    # Log warning if no field names found
    logger.info(f"Header row number '{field_names_row_index}' does not contain column_names in the line : {line=}")
    return AsciiInfo(header, delimiter, [], number_fields, num_points)

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

def generate_ascii_load_dtype(column_names: list[str]) -> npt.DTypeLike:
    names = []; formats = []
    for name in column_names:
        if name in XYZ_NAMES.char:
            names.append(name.lower())
            formats.append(np.float64)
        else:
            names.append(name)
            if name in RGB_NAMES.char or name in RGB_NAMES.words:
                formats.append(np.uint8)
            else:
                formats.append(np.float32)
    return np.dtype({'names': names, 'formats': formats})

