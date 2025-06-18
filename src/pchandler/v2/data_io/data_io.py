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
  - E57 (``.e57``)

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
import logging
import warnings
from datetime import datetime
from enum import IntEnum, auto
from pathlib import Path
from typing import Generator, Optional, Any
from abc import ABC

import laspy
import numpy as np
import pye57
from plyfile import PlyData, PlyElement

from pchandler.v2.geometry.core import PointCloudData
from pchandler.v2.geometry.scalar_field_manager import ScalarFieldManager
from pchandler.v2.geometry.scalar_fields import (
    ScalarField,
    linear_map_dtype,
    normalize_self,
    normalize_array,
    DtypeState,
)
from pchandler.v2.constants import RGB_FIELD, NORMALS_FIELD

logger = logging.getLogger(__name__.split(".")[0])




def load_ply(
    pcd_path: Path,
    retain_colors: bool = True,
    retain_normals: bool = True,
    scalar_fields: list[str] = None,
    normalize_intensities: bool = False,
    **kwargs,
) -> PointCloudData:
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
    logger.info(f"Loading PLY file: {pcd_path}")

    try:
        with open(pcd_path, "rb") as f:
            plydata = PlyData.read(f)
    except Exception as e:
        logger.error(f"Failed to read PLY file {pcd_path}: {e}")
        raise

    num_points = plydata["vertex"].count
    normals = None
    rgb = None
    logger.debug(f"PLY file {pcd_path} contains {num_points} points")

    xyz = np.empty((num_points, 3), dtype=float)
    xyz[:, 0] = plydata["vertex"]["x"]
    xyz[:, 1] = plydata["vertex"]["y"]
    xyz[:, 2] = plydata["vertex"]["z"]

    # DECIDE should there be a flag for cloud_compare exported files with scalar_ prefix?
    ply_scalar_fields = set([pe.name for pe in plydata["vertex"].properties])
    for name in ('x', 'y', 'z'):
        ply_scalar_fields.remove(name)

    rgb_field_names = ply_scalar_fields & {"r", "g", "b", "red", "green", "blue"}
    if retain_colors and len(rgb_field_names) == 3:
        color_dtype = plydata["vertex"]["r"].dtype if "r" in ply_scalar_fields else plydata["vertex"]["red"].dtype

        rgb = np.empty((num_points, 3), dtype=color_dtype)
        rgb[:, 0] = plydata["vertex"]["r"] if "r" in ply_scalar_fields else plydata["vertex"]["red"]
        rgb[:, 1] = plydata["vertex"]["g"] if "g" in ply_scalar_fields else plydata["vertex"]["green"]
        rgb[:, 2] = plydata["vertex"]["b"] if "b" in ply_scalar_fields else plydata["vertex"]["blue"]

        # DECIDE check values and normalise
        rgb = linear_map_dtype(rgb, np.uint8)
        logger.debug("Color information extracted successfully.")

        for name in rgb_field_names:
            ply_scalar_fields.remove(name)

    normals_field_names = ply_scalar_fields & {"nx", "ny", "nz"}

    if retain_normals and len(normals_field_names) == 3:
        normals = np.empty((num_points, 3), dtype=np.float32)
        normals[:, 0] = plydata["vertex"]["nx"]
        normals[:, 1] = plydata["vertex"]["ny"]
        normals[:, 2] = plydata["vertex"]["nz"]

        normals = normalize_self(normals)
        logger.debug("Normal vectors extracted successfully.")

        for name in normals_field_names:
            ply_scalar_fields.remove(name)

    scalar_fields = ScalarFieldManager()

    for name in ply_scalar_fields:
        scalar_fields.add_field(ScalarField(plydata["vertex"][name], name=name))

    return PointCloudData(xyz, rgb=rgb, normals=normals, scalar_fields=scalar_fields)


def save_ply(
    pcd_path: Path,
    pcd: PointCloudData,
    retain_colors: bool = True,
    retain_normals: bool = True,
    scalar_fields: list[str]|set[str] = None,
    revert_scalar_field_type: bool = False,
) -> None:
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
    revert_scalar_field_type : bool, default=False
        Whether to convert scalar fields to original dtype and undo normalization.
    """
    logger.info(f"Saving point cloud to PLY file: {pcd_path}")

    xyz_dtype = np.dtype(np.float64).str if pcd.optimized is not None else pcd.xyz.dtype.str

    dtype_list = [
        ("x", xyz_dtype),
        ("y", xyz_dtype),
        ("z", xyz_dtype),
    ]

    pcd_scalar_fields = set(pcd.scalar_fields.keys()).difference({RGB_FIELD, NORMALS_FIELD})

    if retain_colors and pcd.rgb is not None:
        color_dtype = pcd.rgb.dtype.str
        dtype_list.extend(
            [
                ("red", color_dtype),
                ("green", color_dtype),
                ("blue", color_dtype),
            ]
        )

    if retain_normals and pcd.normals is not None:
        normal_dtype = pcd.normals.dtype.str
        dtype_list.extend(
            [
                ("nx", normal_dtype),
                ("ny", normal_dtype),
                ("nz", normal_dtype),
            ]
        )

    common_scalar_field_names = (
        pcd_scalar_fields if scalar_fields is None else list(set(scalar_fields) & set(pcd_scalar_fields))
    )

    # DECIDE How to handle the fact cloud compare exports all SFs with 'scalar_'
    #  Is this required by the file format definition

    for name in common_scalar_field_names:
        if revert_scalar_field_type is not None:
            dtype_list.append((name, pcd.scalar_fields[name].origin_dtype.dtype.str))
        else:
            dtype_list.append((name, pcd.scalar_fields[name].dtype.str))

    pcd_np_st = np.empty((len(pcd),), dtype=dtype_list)

    base_shift = np.zeros(3, dtype=np.float32) if pcd.optimal_shift is None else pcd.optimal_shift._optimal_shift

    pcd_np_st["x"] = pcd.x + base_shift[0]
    pcd_np_st["y"] = pcd.y + base_shift[1]
    pcd_np_st["z"] = pcd.z + base_shift[2]

    if retain_colors and pcd.rgb is not None:
        pcd_np_st["red"] = pcd.rgb.r
        pcd_np_st["green"] = pcd.rgb.g
        pcd_np_st["blue"] = pcd.rgb.b

    if retain_normals and pcd.normals is not None:
        pcd_np_st["nx"] = pcd.normals.nx
        pcd_np_st["ny"] = pcd.normals.ny
        pcd_np_st["nz"] = pcd.normals.nz

    for name in common_scalar_field_names:

        if revert_scalar_field_type:
            pcd_np_st[name] = pcd.scalar_fields[name].get_original_data()
        else:
            pcd_np_st[name] = pcd.scalar_fields[name].arr

    # TODO: Rename program in comment
    # DECIDE: project transformation
    el = PlyElement.describe(
        pcd_np_st,
        name="vertex",
        comments=[
            "Created with dranjan/python-plyfile in gseg-ethz/pchandler",
            f"Created {datetime.now():%Y-%m-%dT%H:%M:%S%z}",
        ],
    )

    if not pcd_path.parent.exists():
        pcd_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created {pcd_path.parent} folder")

    PlyData([el]).write(f"{pcd_path}")
    logger.info(f"PLY file saved successfully: {pcd_path}")


def load_csv(
    pcd_path: Path, delimeter: str = " ", scalar_fields: list[str] = None, normalize_intensities: bool = True, **kwargs
) -> PointCloudData:
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
    logger.info(f"Loading CSV file: {pcd_path}")

    with open(pcd_path, "r") as f:
        reader = csv.reader(f, delimiter=delimeter)
        data = list(reader)

    if scalar_fields is None:
        header = data[0]
        header[0] = header[0].strip('//')
        scalar_fields = set(name.lower() for name in header)
        data = data[1:]

    xyz = np.array([row[:3] for row in data], dtype=np.float32)
    for name in ('x', 'y', 'z'):
        scalar_fields.remove(name)

    rgb = (
        np.empty(
            shape=(
                len(data),
                3,
            ),
            dtype=np.float32,
        )
        if (scalar_fields is not None and len(set(scalar_fields).intersection({"r", "g", "b"})) == 3)
        else None
    )


    normals = ( np.empty( shape=( len(data), 3,
            ),
            dtype=np.float32,
        )
        if (scalar_fields is not None and len(set(scalar_fields).intersection({"nx", "ny", "nz"})) == 3)
        else None
    )

    sfm = ScalarFieldManager()
    if scalar_fields is not None:

        for i, sf_name in enumerate(scalar_fields):
            if sf_name is None:
                continue
            elif rgb is not None and sf_name in {"r", "g", "b"}:
                if sf_name == "r":
                    rgb[:, 0] = np.array([row[3 + i] for row in data])
                elif sf_name == "g":
                    rgb[:, 1] = np.array([row[3 + i] for row in data])
                else:
                    rgb[:, 2] = np.array([row[3 + i] for row in data])
                continue
            elif normals is not None and sf_name in {"nx", "ny", "nz"}:
                if sf_name == "nx":
                    normals[:, 0] = np.array([row[3 + i] for row in data], dtype=np.float32)
                elif sf_name == "ny":
                    normals[:, 1] = np.array([row[3 + i] for row in data], dtype=np.float32)
                else:
                    normals[:, 2] = np.array([row[3 + i] for row in data], dtype=np.float32)
                continue

            sfm.add_field(ScalarField(np.array([row[3 + i] for row in data], dtype=np.float32), name=sf_name))

    if "Intensity" in sfm and normalize_intensities:
        warnings.warn(
            "normalize_intensities has been deprecated, and will be removed in future relases. Please call the "
            "ScalarField.normalize() function on the individual scalar fields after loading instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        sfm["Intensity"].normalize()

    target_state = DtypeState(np.uint8, 0, 255)
    rgb = normalize_array(rgb, target_state)

    logger.info(f"CSV file loaded successfully: {pcd_path}")
    return PointCloudData(xyz, rgb=rgb, normals=normals, scalar_fields=sfm, **kwargs)


def save_csv(
    pcd_path: Path,
    pcd: PointCloudData,
    delimiter: str = " ",
    add_header: bool = False,
    retain_colors: bool = True,
    retain_normals: bool = True,
    scalar_fields: list[str] = None,
) -> None:
    """
    Saves a `PointCloudData` object to a CSV file, accounting for different data types.

    Parameters
    ----------
    pcd_path : Path
        The output path for the CSV file.
    pcd : PointCloudData
        The point cloud data to save.
    add_header : bool, default=False
        Create a header line for the CSV file with the column info.
    delimiter : str, default=" "
        The delimiter to use in the CSV file.
    retain_colors : bool, default=True
        Whether to include color information in the saved file.
    retain_normals : bool, default=True
        Whether to include normal vectors in the saved file.
    scalar_fields : list[str], optional
        A list of scalar fields to include in the saved file. If `None`, all scalar fields are saved.
    """
    logger.info(f"Saving point cloud to CSV file: {pcd_path}")
    nb_points = pcd.nbPoints

    # Define structured dtype for the output data
    dtype_list = (
        [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
        ]
        if pcd.global_coordinate_shift is None
        else [
            ("x", "f8"),
            ("y", "f8"),
            ("z", "f8"),
        ]
    )

    if retain_colors and pcd.rgb is not None:
        assert pcd.rgb.shape == (nb_points, 3)
        dtype_list.extend([("r", pcd.rgb.dtype.str), ("g", pcd.rgb.dtype.str), ("b", pcd.rgb.dtype.str)])

    if retain_normals and pcd.normals is not None:
        assert pcd.normals.shape == (nb_points, 3)
        dtype_list.extend([("nx", pcd.normals.dtype.str), ("ny", pcd.normals.dtype.str), ("nz", pcd.normals.dtype.str)])

    pcd_scalar_fields = pcd.scalar_fields.keys()
    common_scalar_fields = (
        pcd_scalar_fields if scalar_fields is None else list(set(scalar_fields) & set(pcd_scalar_fields))
    )

    for sf in common_scalar_fields:
        dtype_list.append((sf, pcd.scalar_fields[sf].data.dtype.str))

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

    if retain_colors and pcd.rgb is not None:
        data["r"] = pcd.rgb[:, 0]
        data["g"] = pcd.rgb[:, 1]
        data["b"] = pcd.rgb[:, 2]

    if retain_normals and pcd.normals is not None:
        data["nx"] = pcd.normals[:, 0]
        data["ny"] = pcd.normals[:, 1]
        data["nz"] = pcd.normals[:, 2]

    for sf in common_scalar_fields:
        data[sf] = pcd.scalar_fields[sf].data

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
        fmt=fmt,
    )

    logger.info(f"CSV file saved successfully: {pcd_path}")


def load_e57(
    pcd_path: Path,
    point_cloud_index: Optional[int] = None,
    retain_intensity: bool = True,
    retain_colors: bool = True,
    normalize_intensities: bool = False,
    **kwargs,
) -> PointCloudData | Generator[PointCloudData, None, None]:
    logger.info(f"Loading E57 file: {pcd_path}")
    e57 = pye57.E57(str(pcd_path), mode="r")
    number_of_scans = e57.scan_count
    e57.close()

    assert point_cloud_index is None or (0 <= point_cloud_index < number_of_scans)

    point_cloud_index = 0 if number_of_scans == 1 else point_cloud_index

    if point_cloud_index is None:
        logger.debug(f"Loading {number_of_scans} scans from E57 file.")
        return _load_all_e57_scans(pcd_path, retain_intensity, retain_colors, normalize_intensities, **kwargs)
    else:
        logger.debug(f"Loading scan index {point_cloud_index} from E57 file.")
        return _load_single_e57(
            pcd_path, point_cloud_index, retain_intensity, retain_colors, normalize_intensities, **kwargs
        )


def _load_all_e57_scans(
    pcd_path: Path,
    retain_intensity: bool = True,
    retain_colors: bool = True,
    normalize_intensities: bool = False,
    **kwargs,
) -> Generator[PointCloudData, None, None]:
    logger.debug(f"Loading multiple scans from E57 file: {pcd_path}")
    e57 = pye57.E57(str(pcd_path), mode="r")
    number_of_scans = e57.scan_count

    for i in range(number_of_scans):
        yield _load_single_e57(pcd_path, i, retain_intensity, retain_colors, normalize_intensities, **kwargs)


def _load_single_e57(
    pcd_path: Path,
    point_cloud_index: int,
    retain_intensity: bool = True,
    retain_colors: bool = True,
    normalize_intensities: bool = False,
    **kwargs,
) -> PointCloudData:
    logger.debug(f"Loading single scan {point_cloud_index} from E57 file: {pcd_path}")
    e57 = pye57.E57(str(pcd_path), mode="r")
    data = e57.read_scan(
        point_cloud_index, ignore_missing_fields=True, intensity=retain_intensity, colors=retain_colors
    )
    header = e57.get_header(point_cloud_index)

    xyz = np.column_stack((data["cartesianX"], data["cartesianY"], data["cartesianZ"]))
    colors = np.column_stack((data["colorRed"], data["colorGreen"], data["colorBlue"])) if "colorRed" in data else None
    sfm = ScalarFieldManager(expected_length=xyz.shape[0])
    if "intensity" in data:
        sfm.create_field("intensity", data["intensity"])
        if normalize_intensities:
            sfm["Intensity"].normalize()

    e57.close()
    logger.info(f"Successfully loaded scan {point_cloud_index} from E57 file: {pcd_path}")
    return PointCloudData(xyz, color=colors, scalar_fields=sfm, **kwargs)


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
    logger.info(f"Loading LAZ file: {pcd_path}")
    pcd = laspy.read(pcd_path)
    laz_scalar_fields = list(pcd.point_format.dimension_names)

    colors = None
    if retain_colors and len(set(laz_scalar_fields) & set(["red", "green", "blue"])) == 3:
        colors = np.empty(
            (
                pcd.header.point_count,
                3,
            ),
            dtype=np.uint8,
        )
        colors[:, 0] = (pcd["red"] / 256).astype(np.uint8)
        colors[:, 1] = (pcd["green"] / 256).astype(np.uint8)
        colors[:, 2] = (pcd["blue"] / 256).astype(np.uint8)

    common_scalar_fields = (
        laz_scalar_fields if scalar_fields is None else list(set(scalar_fields) & set(laz_scalar_fields))
    )

    # scalar_fields_dict = dict()
    sfm = ScalarFieldManager(expected_length=pcd.header.point_count)
    for sf in common_scalar_fields:
        if sf.lower() not in ["x", "y", "z", "red", "green", "blue"]:
            if isinstance(pcd[sf], np.ndarray):
                scalar_fields_dict[sf] = pcd[sf]
            elif isinstance(pcd[sf], laspy.point.dims.SubFieldView):
                if sf in [
                    "scan_direction_flag",
                    "edge_of_flight_line",
                    "synthetic",
                    "key_point",
                    "withheld",
                    "overlap",
                ]:
                    scalar_fields_dict[sf] = np.array(pcd[sf], dtype=bool)
                else:
                    scalar_fields_dict[sf] = np.array(pcd[sf])

            else:
                raise NotImplementedError

    print(f"{pcd.header.point_count:,d} points added for file '{pcd_path.name:s}'.")
    logger.info(f"Successfully loaded LAZ file: {pcd_path}")
    return PointCloudData(xyz=pcd.xyz, color=colors, scalar_fields=scalar_fields_dict)

