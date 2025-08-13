import logging
from pathlib import Path
from typing import Any, Iterable, NamedTuple, Optional

import numpy as np
import numpy.typing as npt

from pchandler import PointCloudData
from pchandler.constants import RGB_NAMES, XYZ_NAMES
from pchandler.data_io.core import AbstractIOHandler

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


class CsvHandler(AbstractIOHandler):
    """
    Handler for reading and writing point cloud data in CSV and ASCII-like formats.

    This class provides methods to load and save point cloud data from and to CSV or similar
    formats. It supports scalar fields and provides customization options for field names,
    prefixes, delimiters, and more.

    Parameters
    ----------
    FORMATS : list of str
        Supported file formats by the handler.
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
        **config,
    ) -> PointCloudData:
        """
        Load point cloud data from a specified file using provided configurations.

        This method processes the file structure, validates the fields, determines
        the data load configuration, and extracts the relevant data into a PointCloudData
        object.

        Parameters
        ----------
        path : str or Path
            The path to the file that contains the point cloud data.
        scalar_fields : list of str, optional
            A list of scalar field names to extract. Default is None.
        remove_prefix : bool
            Indicates if the prefix should be removed from scalar field names. Default is True.
        prefix : str
            The prefix to be removed if 'remove_prefix' is True. Default is "scalar_".
        column_names_row : int
            The row index of column names in the file. Use -1 if not applicable. Default is -1.
        comment : str
            The comment character(s) in the file used to skip lines. Default is "//".
        delimiter : str, optional
            The delimiter used to separate fields in the file. If None, it will attempt
            auto-detection. Default is None.
        config : dict
            Additional configuration parameters passed as keyword arguments.

        Returns
        -------
        PointCloudData
            Object containing the extracted point cloud data.
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

        # When a number of scalar_fields match, assumes all fields are in the same order
        if len(field_names) + 3 <= file_info.num_fields:
            load_config["usecols"] = [0, 1, 2] + [file_info.fields.index(name) for name in field_names.values()]
        elif len(field_names) == 3:
            load_config["usecols"] = [0, 1, 2]

        # Load all data
        data = np.loadtxt(**load_config)
        pcd = PointCloudData(cls.extract_xyz(data, data.size))

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
        """
        Saves point cloud data to a CSV file with optional configurations for scalar fields
        and formatting.

        Parameters
        ----------
        pcd : PointCloudData
            The point cloud data to save.
        path : str or Path
            The file path where the CSV will be saved.
        scalar_fields : list of str, optional
            List of scalar fields to include. Include all if None.
        add_prefix : bool, default=True
            Whether to prepend a prefix to scalar field names in the CSV.
        prefix : str, default='scalar_'
            The prefix to add to scalar field names if `add_prefix` is True.
        revert_sf_types : bool, default=False
            Whether to revert scalar field types to their original format.
        delimiter : str, default=','
            The delimiter to use in the CSV file.
        config : dict of str, Any
            Additional configuration options for CSV generation.

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
    """
    Reads and analyzes a file to determine its structure, extracting header information,
    data delimiter, field names, and the number of data points.

    Parameters
    ----------
    file : Path
        Path to the file to be analyzed.
    delimiters : tuple of str, optional
        Possible delimiters to test for splitting lines. Defaults to (" ", ";", "\t", ",").
    field_names_row_index : int, optional
        Index of the row in the header that contains field names. Defaults to -1 (last header row).
    lines_to_check : int, optional
        Number of lines to analyze for detecting the delimiter and number of columns. Defaults to 10.
    minimum_columns : int, optional
        Minimum expected number of columns in the data. Rows with fewer columns are ignored. Defaults to 3.
    comment : str, optional
        String prefix used to denote comment lines in the file. Defaults to "//".

    Returns
    -------
    AsciiInfo
        Object containing detected header, delimiter, field names, number of fields,
        and the total number of data points in the file.
    """

    # Read the header information defined by the comment section of the Ascii File and check if
    # a number of points is on the first line
    header, num_points = _get_header(file, comment)
    delimiter, number_fields = _delimiter_sniffer(file, delimiters, lines_to_check, minimum_columns, comment)

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
    """
    Determine the number of consistent delimited fields in a file.

    Parses a given file to analyze a specified number of lines to identify consistent
    field counts based on a delimiter. Ensures all considered lines have the same
    number of fields and meet the minimum column requirement. The function disregards
    lines starting with a specified comment prefix and processes a specified header
    skipping mechanism.

    Parameters
    ----------
    file : Path
        Path to the input file to be processed.
    character : str
        Delimiter character used to split fields in each line.
    lines_to_check : int, optional
        Number of lines to analyze for field consistency (default: 10).
    minimum_columns : int, optional
        Minimum required number of columns in a line (default: 3).
    comment : str, optional
        Prefix for comment lines to ignore in the file (default: "//").

    Returns
    -------
    int
        The determined number of fields per line if consistent and meeting the minimum
        required columns; otherwise, 0.
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
    """
    Analyzes a file to detect the most appropriate delimiter based on the provided
    criteria, such as acceptable delimiters, number of lines to check, minimum column
    requirements, and comment exclusions. Returns the detected delimiter and field count.

    Parameters
    ----------
    file : Path
        The path of the file to examine for delimiter detection.
    delimiters : str or Iterable[str], optional
        A single delimiter or multiple potential delimiters to test. Defaults to
        (" ", ";", "\t", ",").
    lines_to_check : int, optional
        Number of lines from the start of the file to analyze for delimiter suitability.
        Defaults to 10.
    minimum_columns : int, optional
        Minimum number of columns required for a delimiter to be considered valid.
        Defaults to 3.
    comment : str, optional
        String pattern indicating lines to be treated as comments, which should be
        ignored during analysis. Defaults to "//".

    Returns
    -------
    tuple[str, int]
        The detected delimiter and the number of fields it identifies in the file.
    """

    for delimiter in delimiters:
        number_fields = _get_field_counts(file, delimiter, lines_to_check, minimum_columns, comment)
        if number_fields:
            return delimiter, number_fields

    raise ValueError(f"No valid delimiter was not found in selection: {repr(delimiters)}")


def _get_header(file: Path, comment: str = "//") -> tuple[list[str], int | None]:
    """
    Extracts the header and an optional points number from a file.

    The function reads a file line by line to extract comments at the beginning of
    the file, treating them as header content. It looks for the first non-comment
    line and determines if it can be interpreted as an integer number.

    Parameters
    ----------
    file : Path
        The path to the input file to extract the header and points number from.
    comment : str, optional
        The string prefix used to identify comment lines in the file. Defaults to "//".

    Returns
    -------
    tuple[list[str], int or None]
        A tuple where the first element is a list of header lines (strings without
        the comment prefix) and the second element is an integer if the first non-
        comment line is interpretable as a number, otherwise None.
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
    """
    Generates a numpy dtype object for loading ASCII data based on column names.

    This function maps given column names to appropriate data types based on
    their category or type definitions in predefined constants (e.g., XYZ_NAMES
    and RGB_NAMES). It is used to define structured arrays for loading data.

    Parameters
    ----------
    column_names : list of str
        A list of column names to create the dtype for.

    Returns
    -------
    numpy.typing.DTypeLike
        A numpy dtype object with structured names and corresponding formats
        based on the input column names.
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
