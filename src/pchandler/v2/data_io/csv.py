from pathlib import Path
from typing import Unpack, NotRequired, Iterable, Sequence, Any, Collection, Optional, NamedTuple
import logging


import numpy as np
import numpy.typing as npt

from .core import AbstractIOHandler, _clean_header_name, _clean_field_name
from ..constants import XYZ_FIELDS
from ..geometry import PointCloudData

logger = logging.getLogger(__name__.split(".")[0])

class AsciiInfo(NamedTuple):
    header: list[str]
    delimiter: str
    fields: list[str]
    num_fields: int
    num_points: int|None


class CsvHandler(AbstractIOHandler):
    FORMATS = ['.csv', '.txt', '.xyz', '.asc', '.ascii', '.pts']

    @classmethod
    def load(cls, path: str | Path, /,
             scalar_fields: Optional[list[str]] = None,
             column_names_row: int = -1,
             comment: str = '//',
             delimiter: Optional[str] = None,
             remove_scalar_prefix: bool = True) -> PointCloudData:

        path = Path(path)

        # Sniff the file for header comments, column_names, delimiters, etc and update parameters
        file_info = sniff_file(path, comment=comment, field_names_row_index=column_names_row)
        num_points_line = True if file_info.num_points else False
        skip_rows = len(file_info.header) + num_points_line
        delimiter = delimiter or file_info.delimiter

        # Sort out the right fields to read or write from
        field_names = cls._validate_field_selection(scalar_fields, file_info.fields)

        # When number of scalar_fields match, assumes all fields are in the same order
        if len(field_names) + 3 == file_info.num_fields:
            use_cols = None
        else:
            use_cols = [0, 1, 2] + [file_info.fields.index(name) for name in field_names.values()]

        data = np.loadtxt(fname=path,
                          delimiter=delimiter,
                          dtype=_generate_csv_load_dtype(['x', 'y' ,'z'] + list(field_names.values())),
                          comments=comment,
                          skiprows=skip_rows,
                          usecols=use_cols)

        num_points = data.size

        pcd = PointCloudData(cls.extract_xyz(data, num_points))
        cls.extract_scalar_fields(pcd, data, num_points, field_names)

        return pcd

    @classmethod
    def save(cls,
             pcd: PointCloudData,
             path: str | Path,
             delimiter: Optional[str] = ',',
             scalar_fields: Optional[list[str]] = None,
             prefix_with_scalar: bool = False,
             revert_sf_types: bool = True) -> None:

        array = cls.generate_structured_array(pcd, scalar_fields, revert_sf_types)

        header = f"// {delimiter.join(list(array.dtype.names or ''))}"

        fmt_map = { "f": "%.6f", "u": "%u", "i": "%d", }
        # Use `"%s"` for fallback
        fmt = delimiter.join([fmt_map.get(field[1][1], "%s") for field in array.dtype.descr])

        if array.dtype.names is not None:
            array = np.stack([array[field] for field in list(array.dtype.names)], axis=-1)

            # Avoid prepending '#' to the header with comments
            np.savetxt( path, array, fmt=fmt, delimiter=delimiter, header=header, comments="" )
            logger.info(f"CSV file saved successfully: {path}")
        else:
            raise ValueError("No fields passed")


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
    line: str = header[field_names_row_index]

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

# TODO FIX / REPLACE This as it's mostly repetition
def _generate_csv_load_dtype(column_names: list[str]) -> npt.DTypeLike:
    elements = []
    for name in column_names:
        if name.lower() in XYZ_FIELDS:
            elements.append((name.lower(), np.float64))
        else:
            elements.append((name, np.float32))
    return np.dtype(elements)

def get_column_names(scalar_fields: list[str],
                     detected_column_names: list[str],
                     num_cols: int,
                     use_cols: Optional[list[int]]) -> tuple[list[str], list[int]] :
    if scalar_fields is None:
        if len(detected_column_names) == 0:  # No column names detected
            if num_cols == 3:
                column_names = ['x', 'y', 'z']
            else:
                raise ValueError("Number of columns detected is greater than 3 but no "
                                 "scalar_fields were passed or column names detected.")

        elif len(detected_column_names) == num_cols:
            column_names = detected_column_names

        else:
            raise ValueError(f'Number of columns detected do not match the number of column names '
                             f'read from the header \n {detected_column_names=}')

    elif len(scalar_fields) + 3 == num_cols:  # Exact number of scalar_fields
        column_names = ['x', 'y', 'z'] + scalar_fields

    elif len(scalar_fields) + 3 < num_cols:  # Select number of scalar_fields
        if len(detected_column_names) != num_cols:  # Invalid column header
            raise ValueError(f"Could not resolve the scalar_fields ({scalar_fields}) in the file as the "
                             f"detected column names ({detected_column_names}) did not match the number of "
                             f"columns: {num_cols}")

        if not set(scalar_fields).issubset(set(detected_column_names)):  # Unmatched scalar_field
            raise ValueError(f"The following scalar_fields could not be found in the column names:"
                             f"\n{set(scalar_fields) - set(detected_column_names)}"
                             f"\nColumn names: ({detected_column_names})")

        column_names = ['x', 'y', 'z'] + scalar_fields
        use_cols = [0, 1, 2]
        for name in scalar_fields:
            use_cols.append(detected_column_names.index(name))

    else:  # Num scalar_fields greater than num additional columns
        raise ValueError(f"The input scalar fields should exclude the coordinate names ('x', 'y', 'z') and not "
                         f"exceed the number of columns-3 (compensating for coordinate columns)")

    return column_names, use_cols
