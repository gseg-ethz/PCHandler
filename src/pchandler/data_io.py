"""
``pchandler.data_io``

This module provides functionality for reading, writing, and processing point cloud data (PCD) files in various formats.
It includes methods for loading and saving PCDs in common formats such as ``.ply``, ``.csv``, ``.las``, and ``.laz``.
Additionally, it provides utility functions for locating PCD files within directories.

Features:
---------
- **PCD Formats Supported**:
  - PLY (``.ply``)
  - LAS/LAZ (``.las``, ``.laz``)
  - ASCII/CSV (``.txt``, ``.csv``)

- **Key Functionalities**:
  - Find PCD files in a directory with optional subdirectory inclusion.
  - Save PCD data to `.ply` format with configurable retention of colors, normals, and scalar fields.
  - Load PCD data from `.ply`, `.csv`, or `.laz` files into `PointCloudData` objects.
  - Normalize intensity values for scalar fields when loading.

- **Future Improvements**:
  - Implement an `Enum` for defining supported file types.
  - Refactor functions to use generators (``yield``) for improved memory efficiency.

Dependencies:
-------------
- **Python Standard Library**:
  - ``csv``, ``datetime``, ``enum``, ``itertools``, ``pathlib``

- **Third-Party Libraries**:
  - ``numpy``: For efficient numerical operations.
  - ``laspy``: For handling ``.las`` and ``.laz`` files.
  - ``pye57``: For handling ``.e57`` files.
  - ``plyfile``: For handling ``.ply`` files.
  - ``pchandler.geometry.PointCloudData``: The internal representation for 3D point cloud data.

Usage:
------
Typical usage patterns include:

1. Finding PCD files in a directory:

.. code-block:: python

    from pchandler.data_io import find_pcd_in_directory
    pcd_files = find_pcd_in_directory(directory_path, pcd_file_types=[".ply", ".las"], include_subdirectories=True)

2. Saving a ``PointCloudData`` object to a ``.ply`` file:

.. code-block:: python

    from pchandler.data_io import save_ply
    save_ply(pcd_path=Path("output.ply"), pcd=point_cloud, retain_colors=True, retain_normals=True)

3. Loading a `.ply` or `.csv` file into a `PointCloudData` object:

.. code-block:: python

    from pchandler.data_io import load_ply, load_csv
    point_cloud = load_ply(pcd_path=Path("input.ply"))
    point_cloud_csv = load_csv(pcd_path=Path("input.csv"), delimeter=",")
"""



## TODO: Rework to use yield instead of generating a list and iterating over it

import csv
from datetime import datetime
from enum import Enum
from itertools import compress
from pathlib import Path
from typing import Any, Callable, Optional, Generator

import numpy as np
import laspy
from plyfile import PlyElement, PlyData
import pye57
from yaml import warnings

from pchandler.geometry import PointCloudData

# TODO: Implement `Enum` for FileTypes
# class PCDFileTypes(Enum):
#     E57 = ".e57"  # Not yet implemented
#     PLY = ".ply"
#     LAS = ".las"
#     LAZ = ".laz"
#     ASCII = ".txt"


def find_pcd_in_directory(directory_path, pcd_file_types: list[str], include_subdirectories: bool = True) -> list[Path]:
    """
    (Recursively) searches a directory for point cloud files with specific extensions.

    Parameters
    ----------
    directory_path : Path
        The directory to search for PCD files.
    pcd_file_types : list[str]
        A list of file extensions to search for (e.g., [".ply", ".las", ".txt"]).
    include_subdirectories : bool, default=True
        Whether to include subdirectories in the search.

    Returns
    -------
    list[Path]
        A list of `Path` objects representing the found PCD files.
    """
    file_list = [file_path for file_path in directory_path.iterdir() if file_path.suffix.lower() in pcd_file_types]

    if include_subdirectories:
        for file_path in directory_path.iterdir():
            if file_path.is_dir():
                file_list.extend(find_pcd_in_directory(file_path, pcd_file_types, include_subdirectories))
    return file_list


def save_ply(pcd_path: Path, pcd: PointCloudData, retain_colors: bool = True, retain_normals: bool = True,
             scalar_fields: list[str] = None) -> None:
    """
    Saves a `PointCloudData` object to a `.ply` file.

    Parameters
    ----------
    pcd_path : Path
        The output path for the `.ply` file.
    pcd : PointCloudData
        The point cloud data to save.
    retain_colors : bool, default=True
        Whether to include color information in the saved file.
    retain_normals : bool, default=True
        Whether to include normal vectors in the saved file.
    scalar_fields : list[str], optional
        A list of scalar fields to include in the saved file. If `None`, all scalar fields are saved.
    """
    nb_points = pcd.xyz.shape[0]
    if pcd.global_coordinate_shift is None:
        dtype_list = [("x", "f4"), ("y", "f4"), ("z", "f4"), ]
    else:
        dtype_list = [("x", "f8"), ("y", "f8"), ("z", "f8"), ]

    if retain_colors and pcd.color is not None:
        assert pcd.color.shape == (nb_points, 3)
        dtype_list.extend([("red", "u1"), ("green", "u1"), ("blue", "u1"), ])

    if retain_normals and pcd.normals is not None:
        assert pcd.normals.shape == (nb_points, 3)
        dtype_list.extend([("nx", "f8"), ("ny", "f8"), ("nz", "f8"), ])

    pcd_scalar_fields = pcd.scalar_fields.keys()
    common_scalar_fields = pcd_scalar_fields if scalar_fields is None else list(set(scalar_fields) &
                                                                                set(pcd_scalar_fields))

    for sf in common_scalar_fields:
        assert pcd.scalar_fields[sf].shape == (nb_points, )
        dtype_list.append((sf, pcd.scalar_fields[sf].dtype.str))

    pcd_np_st = np.empty((nb_points,), dtype=dtype_list)

    pcd_np_st["x"] = pcd.xyz[:, 0]
    pcd_np_st["y"] = pcd.xyz[:, 1]
    pcd_np_st["z"] = pcd.xyz[:, 2]

    if pcd.global_coordinate_shift is not None:
        pcd_np_st["x"] += pcd.global_coordinate_shift[0]
        pcd_np_st["y"] += pcd.global_coordinate_shift[1]
        pcd_np_st["z"] += pcd.global_coordinate_shift[2]

    if retain_colors and pcd.color is not None:
        pcd_np_st["red"] = pcd.color[:, 0]
        pcd_np_st["green"] = pcd.color[:, 1]
        pcd_np_st["blue"] = pcd.color[:, 2]

    if retain_normals and pcd.normals is not None:
        pcd_np_st["nx"] = pcd.normals[:, 0]
        pcd_np_st["ny"] = pcd.normals[:, 1]
        pcd_np_st["nz"] = pcd.normals[:, 2]

    for sf in common_scalar_fields:
        pcd_np_st[sf] = pcd.scalar_fields[sf]

    # TODO: Rename program in comment
    el = PlyElement.describe(pcd_np_st, "vertex", comments=["Created with dranjan/python-plyfile with PCHANDLER",
                                                            f"Created {datetime.now():%Y-%m-%dT%H:%M:%S}"])

    if not pcd_path.parent.exists():
        pcd_path.parent.mkdir(parents=True, exist_ok=True)

    PlyData([el]).write(f"{pcd_path}")


def save_csv(pcd_path: Path, pcd: PointCloudData, delimiter: str = " ", add_header: bool = False, retain_colors: bool = True,
             retain_normals: bool = True, scalar_fields: list[str] = None) -> None:
    """
    Saves a `PointCloudData` object to a CSV file, accounting for different data types.

    Parameters
    ----------
    pcd_path : Path
        The output path for the CSV file.
    pcd : PointCloudData
        The point cloud data to save.
    delimiter : str, default=" "
        The delimiter to use in the CSV file.
    retain_colors : bool, default=True
        Whether to include color information in the saved file.
    retain_normals : bool, default=True
        Whether to include normal vectors in the saved file.
    scalar_fields : list[str], optional
        A list of scalar fields to include in the saved file. If `None`, all scalar fields are saved.
    """
    nb_points = pcd.nbPoints

    # Define structured dtype for the output data
    dtype_list = [("x", "f4"), ("y", "f4"), ("z", "f4"), ] if pcd.global_coordinate_shift is None else [("x", "f8"), ("y", "f8"), ("z", "f8"), ]

    if retain_colors and pcd.color is not None:
        assert pcd.color.shape == (nb_points, 3)
        dtype_list.extend([("r", "u1"), ("g", "u1"), ("b", "u1")])

    if retain_normals and pcd.normals is not None:
        assert pcd.normals.shape == (nb_points, 3)
        dtype_list.extend([("nx", pcd.normals.dtype.str), ("ny", pcd.normals.dtype.str), ("nz", pcd.normals.dtype.str)])

    pcd_scalar_fields = pcd.scalar_fields.keys()
    common_scalar_fields = pcd_scalar_fields if scalar_fields is None else list(set(scalar_fields) & set(pcd_scalar_fields))

    for sf in common_scalar_fields:
        assert pcd.scalar_fields[sf].shape == (nb_points,)
        dtype_list.append((sf, pcd.scalar_fields[sf].dtype.str))

    # Create a structured array with the defined dtype
    data = np.empty(nb_points, dtype=dtype_list)

    # Fill structured array with data
    data["x"] = pcd.xyz[:, 0]
    data["y"] = pcd.xyz[:, 1]
    data["z"] = pcd.xyz[:, 2]

    if pcd.global_coordinate_shift is not None:
        data["x"] += pcd.global_coordinate_shift[0]
        data["y"] += pcd.global_coordinate_shift[1]
        data["z"] += pcd.global_coordinate_shift[2]

    if retain_colors and pcd.color is not None:
        data["r"] = pcd.color[:, 0]
        data["g"] = pcd.color[:, 1]
        data["b"] = pcd.color[:, 2]

    if retain_normals and pcd.normals is not None:
        data["nx"] = pcd.normals[:, 0]
        data["ny"] = pcd.normals[:, 1]
        data["nz"] = pcd.normals[:, 2]

    for sf in common_scalar_fields:
        data[sf] = pcd.scalar_fields[sf]

    # Convert structured array to a plain 2D array
    plain_data = np.stack([data[field] for field in data.dtype.names], axis=-1)

    # Write header and data using numpy.savetxt
    header = delimiter.join(data.dtype.names) if add_header else ""
    if not pcd_path.parent.exists():
        pcd_path.parent.mkdir(parents=True, exist_ok=True)

    # Build dynamic format string for `numpy.savetxt`
    fmt_map = {
        "f": "%.6f",  # Default float format
        "u": "%u",  # Unsigned integer
        "i": "%d",  # Signed integer
    }
    fmt = [fmt_map.get(field[1][0], "%s") for field in dtype_list]  # Use `"%s"` for fallback

    np.savetxt(
        pcd_path,
        plain_data,
        delimiter=delimiter,
        header=header,
        comments="",  # Avoid prepending '#' to the header
        fmt=fmt
    )




def load_csv(pcd_path: Path, delimeter: str = " ", scalar_fields: list[str] = None, normalize_intensities: bool = True,
             **kwargs) -> PointCloudData:
    """
    Loads a point cloud from a CSV or ASCII file.

    Parameters
    ----------
    pcd_path : Path
        The path to the input file.
    delimeter : str, default=" "
        The delimiter used in the CSV file.
    scalar_fields : list[str], optional
        A list of scalar fields to parse from the file.
    normalize_intensities : bool, default=True
        Whether to normalize intensity values if they are present in the scalar fields.
    **kwargs : dict
        Additional parameters to pass to the `PointCloudData` constructor.

    Returns
    -------
    PointCloudData
        A `PointCloudData` object created from the loaded data.
    """
    with open(pcd_path, "r") as f:
        reader = csv.reader(f, delimiter=delimeter)
        data = list(reader)
    xyz = np.array([row[:3] for row in data], dtype=np.float32)

    colors = np.empty(shape=(len(data), 3,), dtype=np.uint8) if (scalar_fields is not None and
            len(set(scalar_fields).intersection({"r", "g", "b"})) == 3) else None

    normals = np.empty(shape=(len(data), 3,), dtype=np.float32) if (scalar_fields is not None and
            len(set(scalar_fields).intersection({"nx", "ny", "nz"})) == 3) else None

    scalar_fields_dict = {}
    if scalar_fields is not None:

        for i, sf in enumerate(scalar_fields):
            if sf is None:
                continue
            elif colors and sf in {"r", "g", "b"}:
                if sf == "r":
                    colors[:, 0] = np.array([row[3+i] for row in data], dtype=np.uint8)
                elif sf == "g":
                    colors[:, 1] = np.array([row[3 + i] for row in data], dtype=np.uint8)
                else:
                    colors[:, 2] = np.array([row[3 + i] for row in data], dtype=np.uint8)
                continue
            elif normals and sf in {"nx", "ny", "nz"}:
                if sf == "nx":
                    normals[:, 0] = np.array([row[3 + i] for row in data], dtype=np.float32)
                elif sf == "ny":
                    normals[:, 1] = np.array([row[3 + i] for row in data], dtype=np.float32)
                else:
                    normals[:, 2] = np.array([row[3 + i] for row in data], dtype=np.float32)
                continue
            scalar_fields_dict[sf] = np.array([row[3+i] for row in data], dtype=np.float32)

        if "scalar_Intensity" in scalar_fields_dict and normalize_intensities:
            scalar_fields_dict["scalar_Intensity"] = (
                    scalar_fields_dict["scalar_Intensity"] / scalar_fields_dict["scalar_Intensity"].max())

    return PointCloudData(xyz, color=colors, normals=normals, scalar_fields=scalar_fields_dict, **kwargs)

# def load_e57(pcd_path: Path, point_cloud_index: Optional[int] = None, retain_intensity: bool = True, retain_colors: bool = True,
#              normalize_intensities: bool = False, **kwargs) -> [PointCloudData | Generator[PointCloudData, None, None]]:
#     e57 = pye57.E57(str(pcd_path))
#     number_of_scans = e57.scan_count
#
#     assert point_cloud_index is None or (0 <= point_cloud_index < number_of_scans)
#
#     point_cloud_index = 0 if number_of_scans == 1 else point_cloud_index
#
#     if point_cloud_index is None:
#         e57.close()
#         for i in range(number_of_scans):
#             yield load_e57(pcd_path, i, retain_intensity, retain_colors, normalize_intensities, **kwargs)
#
#
#     data = e57.read_scan(point_cloud_index, ignore_missing_fields=True, intensity=retain_intensity, colors=retain_colors)
#     header = e57.get_header(point_cloud_index)
#
#     # xyz = np.hstack((data["cartesianX"][:,np.newaxis],data["cartesianY"][:,np.newaxis],data["cartesianZ"][:,np.newaxis]))
#     xyz = np.column_stack((data["cartesianX"], data["cartesianY"], data["cartesianZ"]))
#     colors = np.column_stack((data["colorRed"], data["colorGreen"], data["colorBlue"])) if "colorRed" in data else None
#     if "intensity" in data:
#         intensity = data["intensity"]
#         if normalize_intensities:
#             try:
#                 intensity = (intensity - header.intensityMinimum) / (header.intensityMaximum - header.intensityMinimum)
#             except:
#                 intensity = intensity / intensity.max()
#     else:
#         intensity = None
#
#     scalar_field_dict = dict() if intensity is None else {"scalar_Intensity": intensity}
#
#     return PointCloudData(xyz, color=colors, scalar_fields=scalar_field_dict)

def load_e57(pcd_path: Path, point_cloud_index: Optional[int] = None, retain_intensity: bool = True,
             retain_colors: bool = True, normalize_intensities: bool = False, **kwargs) -> PointCloudData | Generator[PointCloudData, None, None]:

    e57 = pye57.E57(str(pcd_path), mode="r")
    number_of_scans = e57.scan_count
    e57.close()

    assert point_cloud_index is None or (0 <= point_cloud_index < number_of_scans)

    point_cloud_index = 0 if number_of_scans == 1 else point_cloud_index

    if point_cloud_index is None:
        return _load_all_e57_scans(pcd_path, retain_intensity, retain_colors, normalize_intensities, **kwargs)
    else:
        return _load_single_e57(pcd_path, point_cloud_index, retain_intensity, retain_colors, normalize_intensities, **kwargs)



def _load_all_e57_scans(pcd_path: Path, retain_intensity: bool = True, retain_colors: bool = True,
                       normalize_intensities: bool = False, **kwargs) -> Generator[PointCloudData, None, None]:
    e57 = pye57.E57(str(pcd_path), mode="r")
    number_of_scans = e57.scan_count

    for i in range(number_of_scans):
        yield _load_single_e57(pcd_path, i, retain_intensity, retain_colors, normalize_intensities, **kwargs)


def _load_single_e57(pcd_path: Path, point_cloud_index: int, retain_intensity: bool = True, retain_colors: bool = True,
                    normalize_intensities: bool = False, **kwargs) -> PointCloudData:
    e57 = pye57.E57(str(pcd_path), mode="r")
    data = e57.read_scan(point_cloud_index, ignore_missing_fields=True, intensity=retain_intensity, colors=retain_colors)
    header = e57.get_header(point_cloud_index)

    xyz = np.column_stack((data["cartesianX"], data["cartesianY"], data["cartesianZ"]))
    colors = np.column_stack((data["colorRed"], data["colorGreen"], data["colorBlue"])) if "colorRed" in data else None
    if "intensity" in data:
        intensity = data["intensity"]
        if normalize_intensities:
            try:
                intensity = (intensity - header.intensityMinimum) / (header.intensityMaximum - header.intensityMinimum)
            except:
                intensity = intensity / intensity.max()
    else:
        intensity = None

    scalar_field_dict = dict() if intensity is None else {"scalar_Intensity": intensity}

    e57.close()
    return PointCloudData(xyz, color=colors, scalar_fields=scalar_field_dict)





def load_ply(pcd_path: Path, retain_colors: bool = True, retain_normals: bool = True, scalar_fields: list[str] = None,
             normalize_intensities: bool = False, **kwargs) -> PointCloudData:
    """
    Loads a point cloud from a `.ply` file.

    Parameters
    ----------
    pcd_path : Path
        The path to the input `.ply` file.
    retain_colors : bool, default=True
        Whether to load color information from the file.
    retain_normals : bool, default=True
        Whether to load normal vectors from the file.
    scalar_fields : list[str], optional
        A list of scalar fields to retain from the file. If `None`, all available scalar fields are loaded.
    normalize_intensities : bool, default=False
        Whether to normalize intensity values if they are present in the scalar fields.
    **kwargs : dict
        Additional parameters to pass to the `PointCloudData` constructor.

    Returns
    -------
    PointCloudData
        A `PointCloudData` object created from the loaded `.ply` file.
    """
    with open(pcd_path, "rb") as f:
        plydata = PlyData.read(f)
    xyz = np.empty((plydata['vertex'].count, 3,), dtype=float)
    xyz[:, 0] = plydata["vertex"]["x"]
    xyz[:, 1] = plydata["vertex"]["y"]
    xyz[:, 2] = plydata["vertex"]["z"]

    ply_scalar_fields = [pe.name for pe in plydata["vertex"].properties]

    # ply_scalar_fields_lower = [ply_sf.lower() for ply_sf in ply_scalar_fields]
    # scalar_fields = None if scalar_fields is None else [sf for sf in scalar_fields]

    colors = None
    if retain_colors and len(set(ply_scalar_fields) & set(["r", "g", "b", "red", "green", "blue"])) == 3:
        colors = np.empty((plydata['vertex'].count, 3,), dtype=np.uint8)
        colors[:, 0] = plydata["vertex"]["r"] if "r" in ply_scalar_fields else plydata["vertex"]["red"]
        colors[:, 1] = plydata["vertex"]["g"] if "g" in ply_scalar_fields else plydata["vertex"]["green"]
        colors[:, 2] = plydata["vertex"]["b"] if "b" in ply_scalar_fields else plydata["vertex"]["blue"]

    normals = None
    if retain_normals and len(set(ply_scalar_fields) & set(["nx", "ny", "nz"])) == 3:
        normals = np.empty((plydata['vertex'].count, 3,), dtype=float)
        normals[:, 0] = plydata["vertex"]["nx"]
        normals[:, 1] = plydata["vertex"]["ny"]
        normals[:, 2] = plydata["vertex"]["nz"]

    common_scalar_fields = ply_scalar_fields if scalar_fields is None else list(set(scalar_fields) &
                                                                                set(ply_scalar_fields))

    scalar_fields_dict = dict()
    for sf in common_scalar_fields:
        if sf.lower() not in ["x", "y", "z", "r", "g", "b", "red", "green", "blue", "nx", "ny", "nz"]:
            scalar_fields_dict[sf] = np.array(plydata["vertex"][sf]).squeeze()

    if "scalar_Intensity" in scalar_fields_dict and normalize_intensities:
        scalar_fields_dict["scalar_Intensity"] = (
                scalar_fields_dict["scalar_Intensity"] / scalar_fields_dict["scalar_Intensity"].max())

    return PointCloudData(xyz, color=colors, normals=normals, scalar_fields=scalar_fields_dict, **kwargs)


def load_laz(pcd_path, retain_colors: bool = True, scalar_fields: list[str] = None) -> PointCloudData:
    """
    Loads a point cloud from a `.las` or `.laz` file using `laspy`.

    Parameters
    ----------
    pcd_path : Path
        The path to the input `.las` or `.laz` file.
    retain_colors : bool, default=True
        Whether to load color information from the file.
    scalar_fields : list[str], optional
        A list of scalar fields to retain from the file. If `None`, all available scalar fields are loaded.

    Returns
    -------
    PointCloudData
        A `PointCloudData` object created from the loaded `.las` or `.laz` file.
    """

    # TODO: Extend usage from `dimension_names` to `extra_dimension_names`

    pcd = laspy.read(pcd_path)
    laz_scalar_fields = list(pcd.point_format.dimension_names)


    colors = None
    if retain_colors and len(set(laz_scalar_fields) & set(["red", "green", "blue"])) == 3:
        colors = np.empty((pcd.header.point_count, 3,), dtype=np.uint8)
        colors[:, 0] = (pcd["red"] / 256).astype(np.uint8)
        colors[:, 1] = (pcd["green"] / 256).astype(np.uint8)
        colors[:, 2] = (pcd["blue"] / 256).astype(np.uint8)

    common_scalar_fields = laz_scalar_fields if scalar_fields is None else list(set(scalar_fields) &
                                                                                set(laz_scalar_fields))

    scalar_fields_dict = dict()
    for sf in common_scalar_fields:
        if sf.lower() not in ["x", "y", "z", "red", "green", "blue"]:
            if isinstance(pcd[sf], np.ndarray):
                scalar_fields_dict[sf] = pcd[sf]
            elif isinstance(pcd[sf], laspy.point.dims.SubFieldView):
                if sf in ["scan_direction_flag", "edge_of_flight_line", "synthetic", "key_point", "withheld",
                          "overlap"]:
                    scalar_fields_dict[sf] = np.array(pcd[sf], dtype=bool)
                else:
                    scalar_fields_dict[sf] = np.array(pcd[sf])

            else:
                raise NotImplementedError

    print(f"{pcd.header.point_count:,d} points added for file '{pcd_path.name:s}'.")

    return PointCloudData(xyz=pcd.xyz, color=colors, scalar_fields=scalar_fields_dict)
