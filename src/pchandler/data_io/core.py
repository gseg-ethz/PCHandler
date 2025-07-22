import logging

from abc import ABC, abstractmethod
from pathlib import Path

from typing import Mapping, Callable, Optional, Any, Generator

import numpy as np
import numpy.typing as npt

from pchandler.base_types import Array_Nx3_T
from pchandler.constants import (
    INTENSITY_NAMES,
    RGB_NAMES,
    NORMAL_NAMES,
    REFLECTANCE_NAMES,
    XYZ_NAMES,
    NameConstantsTriplet,
    NameConstantsSingle,
    COMMON_FIELD_NAMES
)
from pchandler.geometry.core import PointCloudData
from pchandler.geometry.scalar_fields import (
    ScalarField,
    ScalarFieldTriplet,
    RGBFields,
    NormalFields,
    DtypeState,
    SF_T
)


logger = logging.getLogger(__name__.split(".")[0])

BaseDataT =  Mapping[str, npt.NDArray[Any]] | npt.NDArray[Any]
SUPPORTED_TYPES = (".ply", ".las", ".laz", ".txt", ".csv", ".ply", ".e57")


def find_pcd_in_directory(directory_path: Path,
                          pcd_file_types: list[str],
                          include_subdirectories: bool = True) -> list[Path]:
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
    logger.debug(f"Starting search for {directory_path}")
    file_list = [file_path for file_path in directory_path.iterdir() if file_path.suffix.lower() in pcd_file_types]

    if include_subdirectories:
        for file_path in directory_path.iterdir():
            if file_path.is_dir():
                file_list.extend(find_pcd_in_directory(file_path, pcd_file_types, include_subdirectories))
    logger.info(f"Found {len(file_list)} PCD files in {directory_path}")
    return file_list


class AbstractIOHandler(ABC):
    FORMATS: list[str] = []

    @classmethod
    def find_pcds_in_directory(cls, directory_path: str|Path, include_subdirectories: bool = True) -> list[Path]:
        return find_pcd_in_directory(Path(directory_path), cls.FORMATS, include_subdirectories)

    @classmethod
    @abstractmethod
    def load(cls, /,
             path: str|Path,
             scalar_fields: Optional[list[str]] = None,
             remove_prefix: bool = True,
             prefix: str = 'scalar_',
             **config: dict[str, Any]
             ) -> PointCloudData | Generator[PointCloudData, None, None]: ...

    @classmethod
    @abstractmethod
    def save(cls, /,
             pcd: PointCloudData,
             path: str | Path,
             scalar_fields: Optional[list[str]] = None,
             add_prefix: bool = False,
             prefix: str = 'scalar_',
             revert_sf_types: bool = False,
             **config: dict[str, Any]) -> None: ...

    @classmethod
    def _validate_field_selection(
            cls,
            user_fields: list[str] | None,
            header_fields: list[str],
            remove_prefix: bool,
            prefix: str
    ) -> dict[str, str]:

        prefix = prefix if remove_prefix else ''

        header_fields = _clean_field_names(header_fields, _clean_header_name, prefix=prefix)

        if user_fields is None:
            # Case 1 - Retain all information from the header fields
            if len(header_fields) > 0:
                return header_fields

            # Case 2 - Not enough information to resolve fields
            raise ValueError(f'Unable to resolve field names without any header information or scalar fields defined\n'
                             f'  {header_fields=}, {user_fields,}')

        # If user fields exist, convert to lower case and remove whitespace first
        else:
            user_fields = _clean_field_names(user_fields, _clean_string)

        # Case 3 - Empty list passed, keep only coordinates
        if len(user_fields) == 0:
            return {}

        # Case 4 - User input is a subset of the unedited scalar field names
        elif set(user_fields.values()).issubset(header_fields.values()):
            return user_fields

        # Case 5 - User input is a subset of the edited field names (lowercase, and potentially scalar_ removed)
        elif set(user_fields.values()).issubset(header_fields.keys()):
            return {k: header_fields[v] for k, v in user_fields.items()}

        else:
            raise ValueError(f"Unhandled combination of user_defined_fields and detected fields in file header. \n"
                             f"Potentially user entered scalar_fields that do not match the field/property names:\n"
                             f"   {header_fields=}, {user_fields=}")

    @classmethod
    def extract_xyz(cls, data: BaseDataT, num_points: int) -> Array_Nx3_T:
        xyz = np.empty((num_points, 3), dtype=data["x"].dtype)

        for i, name in enumerate(XYZ_NAMES.char):
            xyz[:, i] = data[name]

        return xyz

    @classmethod
    def extract_scalar_fields(
            cls, pcd: PointCloudData, data: BaseDataT, num_points: int, field_names: dict[str, str]
    ) -> None:

        sf_keys = list(field_names.keys())

        # RGB Cases (e.g. 'rgb' or 'r', 'g', 'b')
        if rgb_fields := _get_field_names(sf_keys, RGB_NAMES):
            pcd.rgb = cls._extract_scalar_field_triplet(data, num_points, rgb_fields, RGBFields, field_names)

        # Normals Cases (e.g. 'normals' or 'nx', 'ny', 'nz')
        if normal_fields := _get_field_names(sf_keys, NORMAL_NAMES):
            pcd.normals = cls._extract_scalar_field_triplet(data, num_points, normal_fields, NormalFields, field_names)

        # Intensity
        if set(sf_keys).intersection(INTENSITY_NAMES.all):
            pcd.intensity = cls._extract_scalar_field(data, INTENSITY_NAMES.base, field_names)

        # Reflectance
        if set(sf_keys).intersection(REFLECTANCE_NAMES.all):
            pcd.reflectance = cls._extract_scalar_field(data, REFLECTANCE_NAMES.base, field_names)

        # All others
        for field in sf_keys:
            pcd.scalar_fields.create_field(field, data[field_names[field]])

    @staticmethod
    def _extract_scalar_field_triplet(data: BaseDataT,
                                      n: int, field_names: list[str],
                                      sf_class: type[SF_T],
                                      field_name_map: dict[str, str]) -> SF_T:
        array = np.empty((n, 3), dtype=data[field_name_map[field_names[0]]].dtype)

        for i, name in enumerate(field_names):
            array[:, i] = data[field_name_map[name]]

        return sf_class(array)

    @staticmethod
    def _extract_scalar_field(data: BaseDataT, name: str, field_name_map: dict[str, str]):
        arr = data[field_name_map[name]]
        return ScalarField(arr, name=name, origin_dtype=DtypeState.generate(arr))

    @classmethod
    def _generate_struct_dtype(cls,
                               pcd: PointCloudData,
                               scalar_fields: list[str],
                               revert_sf_types: bool) -> npt.DTypeLike:

        # Leverage dict to avoid any duplicates of using 'rgb' or 'r', 'g', 'b' for example
        dtype_dict: dict = {'names': [], 'formats': []}

        xyz_dtype = np.float64 if pcd.optimized else pcd.xyz.dtype

        for name in XYZ_NAMES.char:
            dtype_dict['names'].append(name)
            dtype_dict['formats'].append(xyz_dtype)

        for field in scalar_fields:
            name_set = None

            for FIELD_NAMES in COMMON_FIELD_NAMES:
                if field in FIELD_NAMES.all:
                    name_set = FIELD_NAMES
                    break

            if isinstance(name_set, NameConstantsTriplet):
                if field in (RGB_NAMES.names + NORMAL_NAMES.names) and getattr(pcd, name_set.base) is not None:
                    for name in name_set.char:
                        dtype_dict['names'].append(name)
                        dtype_dict['formats'].append(_get_sf_dtype(getattr(pcd, name_set.base), revert_sf_types))

                # r, g, b, red, green, blue
                elif field in (RGB_NAMES.char + RGB_NAMES.words) and pcd.rgb is not None:
                    dtype_dict['names'].append(field[0])   # Force it to 'r', 'g', 'b'
                    dtype_dict['formats'].append(_get_sf_dtype(pcd.rgb, revert_sf_types))

                # rf, gf, bf
                elif field in RGB_NAMES.float and pcd.rgb is not None:
                    dtype_dict['names'].append(field)   # Force it to 'r', 'g', 'b'
                    dtype_dict['formats'].append(np.float32)

                elif field in (NORMAL_NAMES.char + NORMAL_NAMES.words)and pcd.normals is not None:
                    dtype_dict['names'].append(field[0]+field[-1])
                    dtype_dict['formats'].append(_get_sf_dtype(pcd.normals, revert_sf_types))

            # General scalar fields
            elif field in pcd.scalar_fields.fields:
                dtype_dict['names'].append(field)
                dtype_dict['formats'].append(_get_sf_dtype(pcd.scalar_fields[field], revert_sf_types))

            else:
                logger.warning(f"Could not find '{field}' as a scalar field in the point cloud")

        return dtype_dict

    @classmethod
    def _generate_structured_array(cls,
                                   pcd: PointCloudData,
                                   scalar_fields: Optional[list[str]],
                                   add_prefix: bool,
                                   prefix: str,
                                   revert_sf_types: bool
                                   ) -> npt.NDArray[Any]:

        prefix = prefix if add_prefix else ''

        if scalar_fields is None:
            scalar_fields = list(pcd.scalar_fields.keys())

        dtype_dict = cls._generate_struct_dtype(pcd, scalar_fields, revert_sf_types)

        array = np.empty((len(pcd),), dtype=np.dtype(dtype_dict))

        shift = pcd.optimized_shift.value

        scalar_fields = dtype_dict['names']
        for i, coord in enumerate(XYZ_NAMES.char):
            array[coord] = getattr(pcd, coord) + shift[i] if pcd.optimized else getattr(pcd, coord)
            scalar_fields.remove(coord)

        for name in scalar_fields:
            if name not in (RGB_NAMES.all, NORMAL_NAMES.all):
                array[prefix + name] = pcd.scalar_fields[name]

            elif name in RGB_NAMES.float:
                array[name] = pcd.scalar_fields[name] / 255

            else:
                array[name] = pcd.scalar_fields[name]

        return array

def _get_field_names(input_names: list[str], target_field: NameConstantsSingle|NameConstantsTriplet ) -> list[str]:
    """ Get matching input field names from eligible names of target field.

    Ensures the output list of fields are still in order and are a subset of the appropriate target field

    Parameters
    ----------
    input_names
    target_field

    Returns
    -------

    """
    if target_field not in (RGB_NAMES, NORMAL_NAMES):
        raise ValueError(f"Field '{target_field}' is not a valid target field constant set")

    valid_names_set = tuple([set(names) for names in target_field.triplets])
    identified_names = set(input_names) & set(target_field.all)

    if not (identified_names in valid_names_set):
        logger.warning(f"No valid {target_field} found in [{target_field.triplets}]. Only :{identified_names}")
        return list()

    else:
        # Get the ordered names again
        identified_names = list(target_field.triplets[valid_names_set.index(identified_names)])
        if isinstance(identified_names, str):
            identified_names = [identified_names]

    # Removes the fields from the current list
    for name in identified_names:
        input_names.remove(name)

    return identified_names

def _get_sf_dtype(scalar_field: ScalarField|ScalarFieldTriplet, revert_sf_types: bool) -> npt.DTypeLike:
    if revert_sf_types:
        return scalar_field.dtype

    if scalar_field.origin_dtype is not None:
        return scalar_field.origin_dtype.dtype
    else:
        logger.warning(f'Scalar field "{scalar_field.name}" has no dtype. Using the sf dtype: {scalar_field.dtype}')
        return scalar_field.dtype

def _clean_field_names(column_names: list[str], func: Callable, **kwargs) -> dict[str, str]:
    cleaned_names = dict()
    for name in column_names:
        cleaned_names[func(name, **kwargs)] = name

    # Remove cartesian coordinates from the field names. These are always assumed to be the first three columns
    for name in XYZ_NAMES.char:
        del cleaned_names[name]

    return cleaned_names

def _clean_string(name: str) -> str:
    return name.strip().lower()

def _clean_header_name(original_name: str, prefix: str) -> str:
    cleaned_name = _clean_string(original_name)

    if cleaned_name != prefix:
        cleaned_name = cleaned_name.removeprefix(prefix).strip()

    return cleaned_name
