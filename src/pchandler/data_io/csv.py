# pchandler - Toolbox for point-cloud handling, processing and analysis
#
# Copyright (c) 2025, Nicholas Meyer, Geosensors and Engineering Geodesy,
# Institute of Geodesy and Photogrammetry, ETH Zurich, Switzerland
# SPDX-License-Identifier: BSD-3-Clause
#
# Author: Nicholas Meyer (meyernic@ethz.ch)

"""CSV / ASCII file format handler class"""
import logging
from pathlib import Path
from typing import Any, Iterable, NamedTuple, Optional, Unpack

import numpy as np
import numpy.typing as npt

from pchandler import PointCloudData
from pchandler.constants import RGB_NAMES, XYZ_NAMES
from pchandler.data_io.core import AbstractIOHandler, PointCloudDataKW

__all__ = ["CsvHandler"]

logger = logging.getLogger(__name__.split(".")[0])


class AsciiInfo(NamedTuple):
    """Summary of what the class does.

    Represents information related to an ASCII file, including its header,
    delimiter, fields, and other properties.

    Parameters
    ----------
    header : list[str]
        Contains header information of the ASCII file.
    delimiter : str
        Specifies the delimiter used in the ASCII file.
    fields : list[str]
        Names of the fields in the ASCII file.
    num_fields : int
        Number of fields described in the file.
    num_points : int | None
        Number of data points in the file or None if unspecified.
    """
    header: list[str]
    delimiter: str
    fields: list[str]
    num_fields: int
    num_points: int | None

# TODO write tests for the pcd_kwargs in each of the load functions
class CsvHandler(AbstractIOHandler):
    """Handles TXT and CSV like file input and output

    Supported file extensions:

    * .txt
    * .csv
    * .xyz
    * .asc
    * .ascii
    * .pts

    """
    FORMATS = [".txt", ".csv", ".xyz", ".asc", ".ascii", ".pts"]

    @classmethod
    def load(
        cls,  # type: ignore[override]
        /,
        path: str | Path,
        scalar_fields: Optional[list[str]] = None,
        remove_prefix: bool = True,
        prefix: str = "scalar_",
        column_names_row: int = -1,
        comment: str = "//",
        delimiter: Optional[str] = None,
        **pcd_kw: Unpack[PointCloudDataKW],
    ) -> PointCloudData:
        """Load a point cloud from a CSV-like file.

        Parameters
        ----------
        path : str or Path
        scalar_fields : list of str, default=None
            List of specific scalar fields to extract from the PLY file.
            Setting `None` will retrieve all scalar fields. Setting to `[]` will ignore scalar fields in the file.
        remove_prefix : bool, default=True
            Flag to remove prefixes on scalar field names.
        prefix : str, default="scalar_"
            Prefix to strip from scalar field names if `remove_prefix` is True.
        column_names_row : int, default=-1
            Header row index where column names are defined. Default is -1 for the last line of the header.
        comment : str, default = "//"
            Character used for comments or header in the file
        delimiter : str, default=None
            Delimiting character(s) type. If None is set, the file will be sniffed and attempt to automatically it.
        config : dict

        Returns
        -------
        PointCloudData
        """

        # Get general file structure information
        file_info = sniff_file(path := Path(path), comment=comment, field_names_row_index=column_names_row)
        num_points_line = True if file_info.num_points else False

        # Validate the fields to be output
        field_names = cls._validate_field_selection(scalar_fields, file_info.fields, remove_prefix, prefix)

        # Define load config parameters

        if tuple([k for k in field_names.keys()]) == XYZ_NAMES.char:
            dt = generate_ascii_load_dtype([k.lower() for k in field_names.values()])
        else:
            dt = generate_ascii_load_dtype(["x", "y", "z"] + list(field_names.values()))

        load_config: dict[str, Any] = {
            "fname": path,
            "skiprows": len(file_info.header) + num_points_line,
            "delimiter": delimiter or file_info.delimiter,
            "dtype": dt,
            "usecols": None,
        }

        # Todo: Rethink this logic
        # When a number of scalar_fields match, assumes all fields are in the same order
        if len(field_names) + 3 <= file_info.num_fields:
            if not file_info.fields:
                load_config["usecols"] = list(range(len(field_names) + 3))
            else:
                load_config["usecols"] = [0, 1, 2] + [file_info.fields.index(name) for name in field_names.values()]
        elif len(field_names) == 3:
            load_config["usecols"] = [0, 1, 2]

        # Load all data
        data = np.loadtxt(**load_config)
        pcd = PointCloudData(cls.extract_xyz(data, data.size), **pcd_kw)

        if not tuple([k for k in field_names.keys()]) == XYZ_NAMES.char:
            cls.extract_scalar_fields(pcd, data, data.size, field_names)

        return pcd

    @classmethod
    def save(
        cls,  # type: ignore[override]
        /,
        pcd: PointCloudData,
        path: str | Path,
        scalar_fields: Optional[list[str]] = None,
        add_prefix: bool = True,
        prefix: str = "scalar_",
        revert_sf_types: bool = False,
        delimiter: str = ",",
        **config: dict[str, Any],
    ) -> None:
        """Save the point cloud data to a text-delimited file format

        Alternatively, save to TXT or other

        Parameters
        ----------
        pcd : PointCloudData
            Point cloud object
        path : str or Path
            File path
        scalar_fields: list[str], default=None
            List of specific scalar fields to extract from the PLY file.
            Setting `None` will retrieve all scalar fields. Setting to `[]` will ignore scalar fields in the file.
        add_prefix: bool, default=False
            Flag to add prefixes on scalar field names
        prefix: str, default="scalar_"
            Prefix to strip from scalar field names if `remove_prefix` is True.
        revert_sf_types: bool, default=False
            Flag to revert scalar field values to their original types or not
        delimiter: str, default=","
            Delimiter to separate fields in the file.
        config : dict of str, Any
        """

        array = cls._generate_structured_array(pcd, scalar_fields, add_prefix, prefix, revert_sf_types)

        header = f"// {delimiter.join(list(array.dtype.names or ''))}"
        fmt_map = {
            "f": "%.6f",
            "u": "%u",
            "i": "%d",
        }
        fmt = delimiter.join([fmt_map.get(field[1][1], "%s") for field in array.dtype.descr])  # use %s as fallback

        np.savetxt(path, array, fmt=fmt, delimiter=delimiter, header=header, comments="")
        logger.info(f"CSV file saved successfully: {path}")


def sniff_file(
    file: Path,
    delimiters: tuple[str, ...] = (" ", ";", "\t", ","),
    field_names_row_index: int = -1,
    lines_to_check: int = 10,
    minimum_columns: int = 3,
    comment: str = "//",
) -> AsciiInfo:
    """Attempts to read part of the file and determine some information about it's structure

    Parameters
    ----------
    file : Path
        File path
    delimiters : tuple[str, ...], default=(" ", ";", "\t", ",")
        Possible delimiters to search for
    field_names_row_index : int, default=-1
        Index number for the row in the header that contains field names. Defaults to the last row (-1).
    lines_to_check : int, default=10
        Number of lines to analyze for detecting the delimiter and number of columns. Defaults to 10.
    minimum_columns : int, default=3
        Minimum number of columns required for a file to be valid (X,Y,Z)
    comment : str, default="//"
        String indicating line comments

    Returns
    -------
    AsciiInfo
        Object contains detected header, delimiter, field names, number of fields, and number of points in the file.
    """

    # Read the header information defined by the comment section of the Ascii File and check if
    # a number of points is on the first line
    header, num_points = _get_header(file, comment)
    delimiter, number_fields = _delimiter_sniffer(file, delimiters, lines_to_check, minimum_columns, comment)

    if not header:
        return AsciiInfo([], delimiter, [], number_fields, num_points)

    # Get the field names based on the defined row_index. Default is the last line of the header (-1)
    line: str = header[field_names_row_index].removeprefix(comment)

    # Test both the delimiter defined and white space as to the joining character between field name header info
    for i in (" ", delimiter):
        field_names = line.split(i)
        if len(field_names) == number_fields:
            return AsciiInfo(header, delimiter, field_names, number_fields, num_points)

    # Log warning if no field names found
    logger.info(f"Header row number '{field_names_row_index}' does not contain column_names in the line : {line=}")
    return AsciiInfo(header, delimiter, [], number_fields, num_points)


def _get_field_counts(
    file: Path, character: str, lines_to_check: int = 10, minimum_columns: int = 3, comment: str = "//"
) -> int:
    """Determine the number of fields (including XYZ coordinates) in a file

    Parameters
    ----------
    file : Path
    character : str
        Delimiter character
    lines_to_check : int, default=10
    minimum_columns : int, default=3
    comment : str, default="//"

    Returns
    -------
    int
    """

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

            line = line.rstrip("\n\r")

            # Case for empty line
            if not line.strip():
                continue

            if character not in line:
                return 0

            fields = line.split(character)
            field_counts.add(len(fields))

        # Ensure the number of fields per line are consistent
        if len(field_counts) > 1:
            return 0

        # Check all lines have the same number of columns
        num_fields = field_counts.pop()
        if num_fields < minimum_columns:
            return 0

        return num_fields


def _delimiter_sniffer(
    file: Path,
    delimiters: str | Iterable[str] = (" ", ";", "\t", ","),
    lines_to_check: int = 10,
    minimum_columns: int = 3,
    comment: str = "//",
) -> tuple[str, int]:
    """Automatically try to determine the delimiter in a file.

    Parameters
    ----------
    file: Path
    delimiters: str or Iterable[str], default=(" ", ";", "\t", ",")
    lines_to_check: int, default=10
    minimum_columns: int, default=3
    comment : str, default="//"

    Returns
    -------
    tuple[str, int]
        delimiter and number of fields found in the file
    """

    for delimiter in delimiters:
        number_fields = _get_field_counts(file, delimiter, lines_to_check, minimum_columns, comment)
        if number_fields:
            return delimiter, number_fields

    raise ValueError(f"No valid delimiter was not found in selection: {repr(delimiters)}")


def _get_header(file: Path, comment: str = "//") -> tuple[list[str], int | None]:
    """Extracts the header and an optional points number from a file.

    Will only get the number of points if it is represented as a standalone integer on the first line
    after the header/comments section.

    Parameters
    ----------
    file : Path
    comment : str, default="//"

    Returns
    -------
    tuple[list[str], int | None]
    """
    with open(file, "r") as f:
        header = []
        while True:
            line = f.readline()

            if not line.startswith(comment):
                line = line.strip("\n\r").strip()
                number_points = int(line) if line.isdigit() else None
                break

            header.append(line.lstrip("//").strip("\n\r").strip())

    return header, number_points


def generate_ascii_load_dtype(column_names: list[str]) -> npt.DTypeLike:
    """Generates a dtype object for loading ASCII data based on column names.

    Parameters
    ----------
    column_names : list of str

    Returns
    -------
    numpy.typing.DTypeLike
        dtype object used for creating a structured array
    """
    names: list[str] = []
    formats: list[npt.DTypeLike] = []
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
    return np.dtype({"names": names, "formats": formats})
