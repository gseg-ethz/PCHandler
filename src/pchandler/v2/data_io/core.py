
import logging

from abc import ABC, abstractmethod
from enum import IntEnum, auto
from pathlib import Path

from typing import Mapping, TypedDict, Callable, Optional, NotRequired, Annotated, Sequence, Any, Collection, Generator

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, BeforeValidator

from ..base_types import Array_Nx3_T
from ..constants import (
    INTENSITY_ALL_NAMES,
    REFLECTANCE_ALL_NAMES,
    RGB_CHAR,
    RGB_WORD,
    RGB_FLOAT,
    RGB_FULL_NAMES,
    RGB_ALL_NAMES,
    NORMALS_CHAR,
    NORMALS_WORD,
    NORMAL_FULL_NAMES,
    NORMAL_ALL_NAMES,
    RGB_FIELD,
    NORMALS_FIELD,
    INTENSITY_FIELD,
    REFLECTANCE_FIELD,
    XYZ_FIELDS
)
from ..geometry.core import PointCloudData
from ..geometry.scalar_fields import (
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


class NormaliseEnum(IntEnum):
    NONE = auto()
    MINMAX = auto()
    DTYPE = auto()
    MAPPING = auto()

def find_pcd_in_directory(directory_path: Path, pcd_file_types: list[str], include_subdirectories: bool = True) -> list[Path]:
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


def skip_main_fields(field_names: Sequence[str]|set[str]) -> set[str]:
    return set(field_names).difference({RGB_FIELD, INTENSITY_FIELD, NORMALS_FIELD, REFLECTANCE_FIELD})


class _BaseConfigType(TypedDict):
    retain_normals: NotRequired[bool]
    retain_rgb: NotRequired[bool]
    retain_intensity: NotRequired[bool]
    retain_reflectance: NotRequired[bool]
    retain_extra_scalar_fields: NotRequired[set[str]]
    cloud_compare_exported: NotRequired[bool]


class _LoadConfigType(_BaseConfigType):
    pass


class _SaveConfigType(_BaseConfigType):
    revert_sf_types: NotRequired[bool]


class _BaseConfig(BaseModel):
    model_config = ConfigDict(extra='ignore', arbitrary_types_allowed=True)
    retain_rgb: bool = True
    retain_normals: bool = True
    retain_intensity: bool = True
    retain_reflectance: bool = True
    retain_extra_scalar_fields: Annotated[set[str], BeforeValidator(skip_main_fields), Field(default_factory=set)]
    cloud_compare_exported: bool = False


class LoadConfig(_BaseConfig):
    comment: str = '//'
    skip_rows: Optional[int] = None
    delimiter: Optional[str] = None
    num_points_line: Optional[bool] = None
    column_names: Collection[str] = Field(default_factory=list)
    column_names_row: int = Field(default=-1, le=-1, ge=100)
    ignore_missing_fields: bool = True
    pcd_index: int = 0


class SaveConfig(_BaseConfig):
    revert_sf_types: bool = False
    delimiter: str = ' '
    number_points_line: bool = False
    header_lines: Collection[str] = Field(default_factory=list)


class AbstractIOHandler(ABC):
    FORMATS: list[str] = []

    @classmethod
    def find_pcds_in_directory(cls, directory_path: str|Path, include_subdirectories: bool = True) -> list[Path]:
        return find_pcd_in_directory(Path(directory_path), cls.FORMATS, include_subdirectories)

    @staticmethod
    def _clean_field_names(column_names: list[str], func: Callable) -> dict[str, str]:
        cleaned_names = dict()
        for name in column_names:
            cleaned_names[func(name)] = name

        # Remove cartesian coordinates from the field names. These are always assumed to be the first three columns
        for name in XYZ_FIELDS:
            del cleaned_names[name]

        return cleaned_names

    @staticmethod
    @abstractmethod
    def _get_scalar_fields_from_header(data: Any) -> set[str]: ...

    @classmethod
    @abstractmethod
    def load(cls, /, path: str|Path, **config: dict[str, Any]) -> PointCloudData | Generator[PointCloudData, None, None]: ...

    @classmethod
    @abstractmethod
    def save(cls, /, pcd: PointCloudData, path: str | Path, **config: dict[str, Any]) -> None: ...

    @classmethod
    def _validate_field_selection(cls, user_fields: list[str] | None, header_fields: list[str]) -> dict[str, str]:
        header_fields = cls._clean_field_names(header_fields, _clean_header_name)

        if user_fields is None:
            # Case 1 - Not enough information to resolve fields
            if len(header_fields) == 0:
                raise ValueError(f'Unable to resolve field names without any fields in the header or defined fields.\n'
                                 f'  {header_fields=}, {user_fields,}')

            # Case 2 - Retain all information from the header fields
            else:
                return header_fields
        else:
            user_fields = cls._clean_field_names(user_fields, _clean_field_name)

        # Case 3 - Empty list passed, keep only coordinates
        if len(user_fields) == 0:
            return {}

        # Case 4 - User input is a subset of the unedited scalar field names
        elif set(user_fields.values()).issubset(header_fields.values()):
            # This will remove any duplicated fields by user input and order will change

            return user_fields

        # Case 5 - User input is a subset of the edited field names (lowercase, and potentially scalar_ removed)
        elif set(user_fields.values()).issubset(header_fields.keys()):
            return {k: header_fields[v] for k, v in user_fields.items()}

        # Other cases not handled
        else:
            raise ValueError(f"Unhandled combination of user_defined_fields and detected fields in file header. \n"
                             f"Potentially user entered scalar_fields that do not match the field/property names:\n"
                             f"   {header_fields=}, {user_fields=}")

    @classmethod
    def extract_xyz(cls, data: BaseDataT, num_points: int) -> Array_Nx3_T:
        xyz = np.empty((num_points, 3), dtype=data["x"].dtype)

        for i, name in enumerate(XYZ_FIELDS):
            xyz[:, i] = data[name]

        return xyz

    @classmethod
    def extract_scalar_fields(
            cls, pcd: PointCloudData, data: BaseDataT, num_points: int, field_names: dict[str, str]
    ) -> None:

        sf_keys = list(field_names.keys())

        # Get the common scalar field names 'rgb', 'normals', 'intensity', 'reflectance'
        # The field names are then popped from the list
        if rgb_fields := cls._get_field_names(sf_keys, RGB_FIELD):
            pcd.rgb = cls._extract_scalar_field_triplet(data, num_points, rgb_fields, RGBFields, field_names)

        if normal_fields := cls._get_field_names(sf_keys, NORMALS_FIELD):
            pcd.normals = cls._extract_scalar_field_triplet(data, num_points, normal_fields, NormalFields, field_names)

        if cls._get_field_names(sf_keys, INTENSITY_FIELD):
            pcd.intensity = cls._extract_scalar_field(data, INTENSITY_FIELD, field_names)

        if cls._get_field_names(sf_keys, REFLECTANCE_FIELD):
            pcd.reflectance = cls._extract_scalar_field(data, REFLECTANCE_FIELD, field_names)

        for field in sf_keys:
            pcd.scalar_fields.create_field(field, data[field_names[field]])

    @staticmethod
    def _extract_scalar_field_triplet(data: BaseDataT,  n: int, field_names: list[str], sf_class: type[SF_T], field_name_map: dict[str, str]) -> SF_T:
        array = np.empty((n, 3), dtype=data[field_name_map[field_names[0]]].dtype)

        for i, name in enumerate(field_names):
            array[:, i] = data[field_name_map[name]]

        return sf_class(array)

    @staticmethod
    def _extract_scalar_field(data: BaseDataT, name: str, field_name_map: dict[str, str]):
        arr = data[field_name_map[name]]
        return ScalarField(arr, name=name, origin_dtype=DtypeState.generate(arr))

    @staticmethod
    def _get_field_names(input_names: list[str], target_field: str ) -> list[str]:
        if target_field == RGB_FIELD:
            potential_names = RGB_ALL_NAMES
            valid_names = (RGB_CHAR, RGB_WORD, RGB_FLOAT)

        elif target_field == NORMALS_FIELD:
            potential_names = NORMAL_ALL_NAMES
            valid_names = (NORMALS_CHAR, NORMALS_WORD)

        elif target_field == INTENSITY_FIELD:
            potential_names = INTENSITY_ALL_NAMES
            valid_names = (INTENSITY_FIELD,)

        elif target_field == REFLECTANCE_FIELD:
            potential_names = REFLECTANCE_ALL_NAMES
            valid_names = (REFLECTANCE_FIELD,)

        else:
            raise ValueError(f"Unsupported Target Field Name: {target_field}")

        valid_names_set = tuple([set(names) for names in valid_names])
        identified_names = set(input_names) & set(potential_names)

        if not (identified_names in valid_names_set):
            logger.warning(
                f"No valid {target_field} found in [{potential_names}]. Only :{identified_names} - Skipping ...")

            return list()

        else:
            # Get the ordered names again
            identified_names = list(valid_names[valid_names_set.index(identified_names)])
            if isinstance(identified_names, str):
                identified_names = [identified_names]

        # Removes the fields from the current list
        for name in identified_names:
            input_names.remove(name)

        return identified_names

    @staticmethod
    def _get_sf_dtype(revert_sf_types: bool, scalar_field: ScalarField|ScalarFieldTriplet) -> npt.DTypeLike:
        if scalar_field.origin_dtype is not None:
            dt: npt.DTypeLike = scalar_field.origin_dtype

            if revert_sf_types:
                return dt.dtype

            return scalar_field.dtype
        else:
            raise ValueError('Origin Dtype is set to None')

    @classmethod
    def generate_struct_dtype(cls,
                              pcd: PointCloudData,
                              scalar_fields: list[str],
                              revert_sf_types: bool) -> npt.DTypeLike:

        # Leverage dict to avoid any duplicates of using 'rgb' or 'r', 'g', 'b' for example
        dtype_dict: npt.DTypeLike = {'names': [], 'formats': []}

        xyz_dtype = np.float64 if pcd.optimized is not None else pcd.xyz.dtype

        for name in XYZ_FIELDS:
            dtype_dict['names'].append(name)
            dtype_dict['formats'].append(xyz_dtype)

        for sf_name in scalar_fields:
            # rgb,
            if sf_name in RGB_FULL_NAMES and pcd.rgb is not None:
                for name in RGB_CHAR:
                    dtype_dict['names'].append(name)
                    dtype_dict['formats'].append(cls._get_sf_dtype(revert_sf_types, pcd.rgb))

            # r, g, b, red, green, blue
            elif sf_name in (RGB_CHAR + RGB_WORD) and pcd.rgb is not None:
                dtype_dict['names'].append(sf_name[0])   # Force it to 'r', 'g', 'b'
                dtype_dict['formats'].append(cls._get_sf_dtype(revert_sf_types, pcd.rgb))


            elif sf_name in NORMAL_FULL_NAMES and pcd.normals is not None:
                for name in NORMALS_CHAR:
                    dtype_dict['names'].append(name)
                    dtype_dict['formats'].append(cls._get_sf_dtype(revert_sf_types, pcd.normals))

            elif sf_name in (NORMALS_CHAR + NORMALS_WORD)and pcd.normals is not None:
                dtype_dict['names'].append(sf_name[0]+sf_name[-1])
                dtype_dict['formats'].append(cls._get_sf_dtype(revert_sf_types, pcd.normals))

            # General scalar fields
            elif sf_name in pcd.scalar_fields.fields:
                dtype_dict['names'].append(sf_name)
                dtype_dict['formats'].append(cls._get_sf_dtype(revert_sf_types, pcd.scalar_fields[sf_name]))

            else:
                logger.warning(f"Could not find '{sf_name}' as a scalar field in the point cloud")

        return dtype_dict

    @classmethod
    def generate_structured_array(cls,
                                  pcd: PointCloudData,
                                  scalar_fields: Optional[list[str]],
                                  revert_sf_types: bool = False,
                                  add_scalar_prefix: bool = False) -> npt.NDArray[Any]:
        if scalar_fields is None:
            scalar_fields = list(pcd.scalar_fields.keys())

        prefix = 'scalar_' if add_scalar_prefix else ''

        dtype_dict = cls.generate_struct_dtype(pcd, scalar_fields, revert_sf_types)

        array = np.empty((len(pcd),), dtype=np.dtype(dtype_dict))

        if pcd.optimized_shift:
            shift = pcd.optimized_shift.value
            array["x"] = pcd.x + shift[0]
            array["y"] = pcd.y + shift[1]
            array["z"] = pcd.z + shift[2]
        else:
            array["x"] = pcd.x
            array["y"] = pcd.y
            array["z"] = pcd.z


        extra_field_names = dtype_dict['names']
        for name in XYZ_FIELDS:
            extra_field_names.remove(name)

        for name in extra_field_names:
            out_name = name
            if name not in (RGB_ALL_NAMES, NORMAL_ALL_NAMES):
                out_name = prefix + name
            array[out_name] = pcd.scalar_fields[name]

        return array

def _clean_field_name(name: str) -> str:
    return name.strip().lower()

def _clean_header_name(original_name: str) -> str:
    cleaned_name = _clean_field_name(original_name)

    scalar_prefixes = ('scalar_', 'scalar')

    if cleaned_name not in scalar_prefixes:
        cleaned_name = cleaned_name.removeprefix('scalar').strip().strip('_')

    return cleaned_name
