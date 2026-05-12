# pchandler - Toolbox for point-cloud handling, processing and analysis
#
# Copyright (c) 2025, Nicholas Meyer, Geosensors and Engineering Geodesy,
# Institute of Geodesy and Photogrammetry, ETH Zurich, Switzerland
# SPDX-License-Identifier: BSD-3-Clause
#
# Author: Nicholas Meyer (meyernic@ethz.ch)

"""
Base module for I/O handlers and supporting methods
"""

import copy
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generator,
    Mapping,
    Optional,
    Sequence,
    TypedDict,
    Unpack,
    cast,
)

import numpy as np
import numpy.typing as npt
from GSEGUtils.base_types import Array_Nx3_T, DtypeDict, Vector_3_T
from numpy._typing._dtype_like import _DTypeDict

from pchandler import PointCloudData
from pchandler.constants import (
    COMMON_FIELD_NAMES,
    INTENSITY_NAMES,
    NORMAL_NAMES,
    REFLECTANCE_NAMES,
    RGB_NAMES,
    XYZ_NAMES,
    _NameConstantsSingle,
    _NameConstantsTriplet,
)
from pchandler.geometry import OptimizedShift
from pchandler.scalar_fields import (
    SF_T,
    NormalFields,
    RGBFields,
    ScalarField,
    ScalarFieldTriplet,
)
from pchandler.scalar_fields.scalar_fields import DtypeState

logger = logging.getLogger(__name__.split(".")[0])

BaseDataT = Mapping[str, npt.NDArray[Any]] | npt.NDArray[Any]
SUPPORTED_TYPES = (".ply", ".las", ".laz", ".txt", ".csv", ".pts", ".e57")


class PointCloudDataKW(TypedDict, total=False):
    socs_origin: Optional[Vector_3_T]
    numerical_optimization_shift: Optional[OptimizedShift]


def find_point_cloud_in_directory(
    directory_path: Path, pcd_file_types: Sequence[str] = SUPPORTED_TYPES, include_subdirectories: bool = True
) -> list[Path]:
    """Search a directory for point cloud files with specific extensions.
    Parameters
    ----------
    directory_path : Path
        Path of directory to search.
    pcd_file_types : list[str]
        List of extension names (e.g. ['.ply', '.las', '.txt']).
    include_subdirectories : bool, default=True
        Flag to include a subdirectory search

    Returns
    -------
    list[Path]
    """

    if not directory_path.is_dir():
        if directory_path.is_file():
            raise IOError(f"{directory_path} is file not a directory.")
        raise IOError(f"{directory_path} is not a directory.")

    logger.debug(f"Starting search for {directory_path}")
    file_list = [file_path for file_path in directory_path.iterdir() if file_path.suffix.lower() in pcd_file_types]

    if include_subdirectories:
        for file_path in directory_path.iterdir():
            if file_path.is_dir():
                file_list.extend(find_point_cloud_in_directory(file_path, pcd_file_types, include_subdirectories))
    logger.info(f"Found {len(file_list)} PCD files in {directory_path}")
    return file_list


class AbstractIOHandler(ABC):
    """Abstract I/O handler class"""

    FORMATS: list[str] = []

    @classmethod
    def find_pcds_in_directory(cls, directory_path: str | Path, include_subdirectories: bool = True) -> list[Path]:
        """Find all point cloud files in a specified directory

        Parameters
        ----------
        directory_path : str or Path
        include_subdirectories : bool, default=True

        Returns
        -------
        list of Path
        """
        return find_point_cloud_in_directory(Path(directory_path), cls.FORMATS, include_subdirectories)

    @classmethod
    @abstractmethod
    def load(
        cls,
        /,
        path: str | Path,
        scalar_fields: Optional[list[str]] = None,
        remove_prefix: bool = True,
        prefix: str = "scalar_",
        **pcd_kw: Unpack[PointCloudDataKW],
    ) -> PointCloudData | Generator[PointCloudData, None, None]: ...

    @classmethod
    @abstractmethod
    def save(
        cls,
        /,
        pcd: PointCloudData,
        path: str | Path,
        scalar_fields: Optional[list[str]] = None,
        add_prefix: bool = False,
        prefix: str = "scalar_",
        revert_sf_types: bool = False,
        **config: dict[str, Any],
    ) -> None: ...

    @classmethod
    def _validate_field_selection(
        cls, input_selection: list[str] | None, header_fields: list[str], remove_prefix: bool, prefix: str
    ) -> dict[str, str]:
        """Validates and resolves field selection based on user input and header fields.

        Parameters
        ----------
        input_selection : list[str] or None
            List of user-selected field names to validate. Pass `None` to use all fields from the header.
        header_fields : list[str]
            List of field names extracted from the file header.
        remove_prefix : bool
            Whether to remove the specified prefix from field names during validation.
        prefix : str
            The prefix to remove from field names, depending on `remove_prefix`.

        Returns
        -------
        dict[str, str]
            A dictionary mapping input field names to corresponding resolved field names.
        """

        prefix = prefix if remove_prefix else ""

        headers = _clean_field_names(header_fields, _clean_header_name, prefix=prefix)

        if input_selection is not None:
            selection = _clean_field_names(input_selection, _clean_header_name, prefix=prefix)
        else:
            if len(headers) > 0:
                return headers

            raise ValueError("Unable to resolve field names without header info or user selection.")

        # Todo: Check if this logic is sound
        if len(selection) == 0 or set(selection.values()).issubset(headers.values()) or not headers:
            return selection

        # User input is a subset of the 'cleaned' names - lowercase, prefix removed etc.
        elif set(selection.values()).issubset(headers.keys()):
            return {k: headers[v] for k, v in selection.items()}

        # Unmatched keys
        else:
            raise ValueError(
                f"Unhandled combination of selected fields and those found in the file header:\n"
                f"   {headers=}, {selection=}"
            )

    @classmethod
    def extract_xyz(cls, data: BaseDataT, num_points: int) -> Array_Nx3_T:
        """Extract XYZ components from the given structured array or dict

        Parameters
        ----------
        data : BaseDataT
        num_points : int

        Returns
        -------
        Array_Nx3_T
        """
        xyz = np.empty((num_points, 3), dtype=data["x"].dtype)

        for i, name in enumerate(XYZ_NAMES.char):
            xyz[:, i] = data[name]

        return xyz

    @classmethod
    def extract_scalar_fields(
        cls, pcd: PointCloudData, data: BaseDataT, num_points: int, field_names: dict[str, str]
    ) -> None:
        """Extract scalar fields from the given structured array or dict

        Parameters
        ----------
        pcd : PointCloudData
        data : BaseDataT
        num_points : int
        field_names : dict of str, str
            Mapping of scalar field names to their corresponding data keys in the dataset.
        """

        sf_keys = list(field_names.keys())

        # RGB Cases (e.g. 'rgb' or 'r', 'g', 'b')
        if rgb_fields := _get_rgb_or_normal_field_names(sf_keys, RGB_NAMES):
            pcd.rgb = cls._extract_scalar_field_triplet(data, num_points, rgb_fields, RGBFields, field_names)

        # Normals Cases (e.g. 'normals' or 'nx', 'ny', 'nz')
        if normal_fields := _get_rgb_or_normal_field_names(sf_keys, NORMAL_NAMES):
            pcd.normals = cls._extract_scalar_field_triplet(data, num_points, normal_fields, NormalFields, field_names)

        # Intensity
        if intensity_fields := set(sf_keys).intersection(INTENSITY_NAMES.all):
            pcd.intensity = cls._extract_scalar_field(data, INTENSITY_NAMES.base, field_names)

        # Reflectance
        if reflectance_fields := set(sf_keys).intersection(REFLECTANCE_NAMES.all):
            pcd.reflectance = cls._extract_scalar_field(data, REFLECTANCE_NAMES.base, field_names)

        remaining_keys = set(sf_keys).difference(
            rgb_fields + normal_fields + list(intensity_fields) + list(reflectance_fields)
        )

        # All others
        for field in remaining_keys:
            pcd.scalar_fields.create_field(field, data[field_names[field]])

    @staticmethod
    def _extract_scalar_field_triplet(
        data: BaseDataT, n: int, field_names: list[str], sf_class: type[SF_T], field_name_map: dict[str, str]
    ) -> SF_T | None:
        """Extract scalar field triplet types from the structured array or dict.

        Parameters
        ----------
        data: BaseDataT
        n: int
        field_names: list of str
        sf_class: type[SF_T]
        field_name_map: dict of str to str

        Returns
        -------
        SF_T or None
        """
        array = np.empty((n, 3), dtype=data[field_name_map[field_names[0]]].dtype)

        for i, name in enumerate(field_names):
            array[:, i] = data[field_name_map[name]]

        if np.all(array == 0):
            logger.info(f"All values in the '{field_names}' field are zero. Skipping.")
            return None

        return sf_class(array)

    @staticmethod
    def _extract_scalar_field(data: BaseDataT, name: str, field_name_map: dict[str, str]) -> ScalarField | None:
        """Extract scalar field from the structured array or dict.

        Parameters
        ----------
        data: BaseDataT
        name: str
        field_name_map: dict[str, str]

        Returns
        -------
        ScalarField | None
        """
        arr = data[field_name_map[name]]

        if np.all(arr == 0):
            logger.info(f"All values in the '{field_name_map[name]}' field are zero. Skipping.")
            return None

        return ScalarField(arr, name=name, origin_dtype=DtypeState.generate(arr))

    @classmethod
    def _generate_struct_dtype(cls, pcd: PointCloudData, scalar_fields: list[str], revert_sf_types: bool) -> DtypeDict:
        """Generate a numpy dtype for initialising a structured array

        Parameters
        ----------
        pcd: PointCloudData
        scalar_fields: list of str
        revert_sf_types: bool

        Returns
        -------
        DtypeDict
        """

        # Leverage dict to avoid any duplicates of using 'rgb' or 'r', 'g', 'b', for example
        dtype_dict = DtypeDict(names=[], formats=[])

        if pcd.numerical_optimization_shift is None:
            xyz_dtype: npt.DTypeLike = np.float64
        else:
            xyz_dtype = pcd.xyz.dtype

        for name in XYZ_NAMES.char:
            dtype_dict["names"].append(name)
            dtype_dict["formats"].append(xyz_dtype)

        for field in scalar_fields:
            name_set = None

            for FIELD_NAMES in COMMON_FIELD_NAMES:
                if field in FIELD_NAMES.all:
                    name_set = FIELD_NAMES
                    break

            if isinstance(name_set, _NameConstantsTriplet):
                if field in (RGB_NAMES.names + NORMAL_NAMES.names) and getattr(pcd, name_set.base) is not None:
                    for name in name_set.char:
                        dtype_dict["names"].append(name)
                        dtype_dict["formats"].append(_get_sf_dtype(getattr(pcd, name_set.base), revert_sf_types))

                # r, g, b, red, green, blue
                elif field in (RGB_NAMES.char + RGB_NAMES.words) and pcd.rgb is not None:
                    dtype_dict["names"].append(field[0])  # Force it to 'r', 'g', 'b'
                    dtype_dict["formats"].append(_get_sf_dtype(pcd.rgb, revert_sf_types))

                # rf, gf, bf
                elif field in RGB_NAMES.float and pcd.rgb is not None:
                    dtype_dict["names"].append(field)  # Force it to 'r', 'g', 'b'
                    dtype_dict["formats"].append(np.float32)

                elif field in (NORMAL_NAMES.char + NORMAL_NAMES.words) and pcd.normals is not None:
                    dtype_dict["names"].append(field[0] + field[-1])
                    dtype_dict["formats"].append(_get_sf_dtype(pcd.normals, revert_sf_types))

            elif isinstance(name_set, _NameConstantsSingle):
                dtype_dict["names"].append(field)
                dtype_dict["formats"].append(_get_sf_dtype(cast(SF_T, pcd.scalar_fields[field]), revert_sf_types))

        for field in scalar_fields:
            name_set = None

            for FIELD_NAMES in COMMON_FIELD_NAMES:
                if field in FIELD_NAMES.all:
                    name_set = FIELD_NAMES
                    break

            if name_set is None:
                dtype_dict["names"].append(field)
                dtype_dict["formats"].append(_get_sf_dtype(cast(SF_T, pcd.scalar_fields[field]), revert_sf_types))

        return dtype_dict

    @classmethod
    def _generate_structured_array(
        cls,
        pcd: PointCloudData,
        scalar_fields: Optional[list[str]],
        add_prefix: bool,
        prefix: str,
        revert_sf_types: bool,
    ) -> npt.NDArray[Any]:
        """Generate a structured array from the existing data for I/O purposes

        Parameters
        ----------
        pcd: PointCloudData
        scalar_fields: list of str or None
        add_prefix: bool
        prefix: str
        revert_sf_types: bool

        Returns
        -------
        npt.NDArray[Any]
        """

        prefix = prefix if add_prefix else ""

        if scalar_fields is None:
            scalar_fields = list(pcd.scalar_fields.keys())

        dtype_dict = cls._generate_struct_dtype(pcd, scalar_fields, revert_sf_types)

        sf_names = sorted(copy.deepcopy(dtype_dict["names"]))
        for i, name in enumerate(dtype_dict["names"]):
            if (name not in RGB_NAMES.all) and (name not in NORMAL_NAMES.all) and (name not in XYZ_NAMES.all):
                dtype_dict["names"][i] = prefix + name

        array = np.empty((len(pcd),), dtype=cast(_DTypeDict, dtype_dict))

        for i, coord in enumerate(XYZ_NAMES.char):
            if pcd.numerical_optimization_shift is None:
                array[coord] = getattr(pcd, coord)
            else:
                array[coord] = getattr(pcd, coord) + pcd.numerical_optimization_shift.value[i]

            sf_names.remove(coord)

        for name in sf_names:
            if (name not in RGB_NAMES.all) and (name not in NORMAL_NAMES.all):
                array[prefix + name] = pcd.scalar_fields[name]

            elif name in RGB_NAMES.float:
                array[name] = cast(RGBFields, pcd.scalar_fields[name]) / 255

            else:
                array[name] = pcd.scalar_fields[name]

        return array


def _get_rgb_or_normal_field_names(input_names: list[str], target_field: _NameConstantsTriplet) -> list[str]:
    """Get matching rgb or normal field names from eligible names in the target field.

    Ensures the output fields are still in order and are a subset of the appropriate target field

    Parameters
    ----------
    input_names: list[str]
    target_field: _NameConstantsTriplet

    Returns
    -------
    list[str]
    """
    if target_field not in (RGB_NAMES, NORMAL_NAMES):
        raise ValueError(f"Field '{target_field}' is not a valid target field constant set.")

    valid_names_set = tuple([set(names) for names in target_field.triplets])
    identified_names = set(input_names) & set(target_field.all)

    if identified_names in valid_names_set:
        identified_triplet = list(target_field.triplets[valid_names_set.index(identified_names)])

    elif len(identified_names := identified_names.intersection(target_field.all)) >= 1:
        logger.debug(
            f"Only a full list of RGB or Normal triplet fields supported. "
            f"A partial or mixed list was passed: {identified_names}."
        )
        return list()

    else:
        logger.debug(f"No valid {target_field} found in [{target_field.triplets}]. Only :{identified_names}")
        return list()

    # Removes the fields from the current list
    for name in identified_triplet:
        input_names.remove(name)

    return identified_triplet


def _get_sf_dtype(scalar_field: ScalarField | ScalarFieldTriplet, revert_sf_types: bool) -> npt.DTypeLike:
    """Get the dtype of a scalar field

    Parameters
    ----------
    scalar_field: ScalarField | ScalarFieldTriplet
    revert_sf_types: bool

    Returns
    -------
    npt.DTypeLike
    """
    if revert_sf_types:
        return scalar_field.origin_dtype.dtype
    return scalar_field.dtype


def _clean_field_names(column_names: list[str], func: Callable, **kwargs) -> dict[str, str]:
    """Clean field names of restricted field names (X, Y, Z) and a dedicated function passed

    Parameters
    ----------
    column_names : list of str
    func : Callable
    kwargs : dict[str, Any]

    Returns
    -------
    dic[str, str]]
        A dictionary where the keys are cleaned names and the values are the original names.
    """
    cleaned_names = dict()
    for name in column_names:
        cleaned_names[func(name, **kwargs)] = name

    # Remove cartesian coordinates from the field names. These are always assumed to be the first three columns
    if tuple([k for k in cleaned_names.keys()]) != XYZ_NAMES.char:
        # Keep cleaned_names if 'x', 'y', 'z' are the only fields
        for name in XYZ_NAMES.char:
            if name in cleaned_names:
                del cleaned_names[name]
    else:
        logger.debug("Only X, Y, Z fields in the file.")

    return cleaned_names


def _clean_string(name: str) -> str:
    """Clean string of whitespace and convert to lowercase

    Parameters
    ----------
    name : str

    Returns
    -------
    str
    """
    return name.strip().lower()


def _clean_header_name(original_name: str, prefix: str) -> str:
    """Recursively cleans and modifies a header name based on a given prefix.

    Parameters
    ----------
    original_name : str
    prefix : str

    Returns
    -------
    str
    """
    cleaned_name = _clean_string(original_name)
    prefix = prefix.lower()

    if cleaned_name != prefix:
        updated_name = cleaned_name.removeprefix(prefix).strip()

        if updated_name == cleaned_name:
            # No further change
            return updated_name

        # Return the recursive result
        return _clean_header_name(updated_name, prefix)

    # Return if it matches the prefix to be stripped
    else:
        return cleaned_name
