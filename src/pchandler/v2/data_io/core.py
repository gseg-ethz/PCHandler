
import logging

from abc import ABC, abstractmethod
from enum import IntEnum, auto
from pathlib import Path

from typing import Mapping, TypedDict, Iterable, Unpack, Optional, NotRequired, Annotated

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, BeforeValidator

from ..constants import (
    INTENSITY_POTENTIAL_NAMES,
    REFLECTANCE_POTENTIAL_NAMES,
    RGB_PARTIAL_NAMES,
    RGB_CHAR,
    RGB_WORD,
    NORMAL_PARTIAL_NAMES,
    RGB_FIELD,
    NORMALS_FIELD,
    INTENSITY_FIELD,
    REFLECTANCE_FIELD
)
from ..geometry.core import PointCloudData
from ..geometry.scalar_fields import (
    ScalarField,
    RGBFields,
    NormalFields,
    DtypeState,
    NormalisedInt16ScalarField,
)


logger = logging.getLogger(__name__.split(".")[0])

BaseDataT =  Mapping[str, np.ndarray] | np.ndarray
SUPPORTED_TYPES = (".ply", ".las", ".laz", ".txt", ".csv", ".ply", ".e57")


class NormaliseEnum(IntEnum):
    NONE = auto()
    MINMAX = auto()
    DTYPE = auto()
    MAPPING = auto()

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
    logger.debug(f"Starting search for {directory_path}")
    file_list = [file_path for file_path in directory_path.iterdir() if file_path.suffix.lower() in pcd_file_types]

    if include_subdirectories:
        for file_path in directory_path.iterdir():
            if file_path.is_dir():
                file_list.extend(find_pcd_in_directory(file_path, pcd_file_types, include_subdirectories))
    logger.info(f"Found {len(file_list)} PCD files in {directory_path}")
    return file_list




class _BaseConfigType(TypedDict):
    keep_normals: NotRequired[bool]
    keep_rgb: NotRequired[bool]
    keep_intensity: NotRequired[bool]
    keep_reflectance: NotRequired[bool]
    keep_extra_scalar_fields: NotRequired[Iterable[str]]
    cloud_compare_exported: NotRequired[bool]


class _BaseLoadConfigType(_BaseConfigType):
    pass


class _BaseSaveConfigType(_BaseConfigType):
    revert_sf_types: NotRequired[bool]


class _BaseConfig(BaseModel):
    model_config = ConfigDict(extra='ignore')
    keep_rgb: Optional[bool] = True
    keep_normals: Optional[bool] = True
    keep_intensity: Optional[bool] = True
    keep_reflectance: Optional[bool] = True
    keep_extra_scalar_fields: Annotated[set[str], BeforeValidator(lambda value: set(value)), Field(default_factory=set)]
    cloud_compare_exported: bool = False


class BaseLoadConfig(_BaseConfig):
    pass


class BaseSaveConfig(_BaseConfig):
    revert_sf_types: bool = False


class AbstractIOHandler(ABC):
    FORMATS: list[str] = None
    LOAD_CONFIG: type[BaseLoadConfig] = BaseLoadConfig
    SAVE_CONFIG: type[BaseSaveConfig] = BaseSaveConfig

    @classmethod
    @abstractmethod
    def load(cls, /, path: str|Path, **config: Unpack[_BaseLoadConfigType]): ...

    @classmethod
    @abstractmethod
    def save(cls, /, pcd: PointCloudData, path: str | Path, **config: Unpack[_BaseSaveConfigType]): ...

    @classmethod
    def find_pcds_in_directory(cls, directory_path: str|Path, include_subdirectories: bool = True):
        return find_pcd_in_directory(directory_path, cls.FORMATS, include_subdirectories)

    @classmethod
    def _extract_xyz(cls, data: BaseDataT, num_points: int, field_names: set[str]):
        xyz = np.empty((num_points, 3), dtype=data["x"].dtype)
        xyz[:, 0] = data["x"]
        xyz[:, 1] = data["y"]
        xyz[:, 2] = data["z"]

        cls._remove_field_names(field_names, 'x', 'y', 'z')

        return xyz

    @classmethod
    def extract_common_fields_to_pcd(cls,
                              pcd: PointCloudData,
                              data: BaseDataT,
                              cfg: BaseLoadConfig,
                              num_points: int,
                              field_names: set[str]):

        if cfg.keep_rgb:
            pcd.rgb = cls._extract_rgb(data, num_points, field_names)

        if cfg.keep_normals:
            pcd.normals = cls._extract_normals(data, num_points, field_names)

        if cfg.keep_intensity:
            pcd.intensity = cls._extract_intensity(data, field_names)

        if cfg.keep_reflectance:
            pcd.reflectance = cls._extract_reflectance(data, field_names)

    @classmethod
    def extract_extra_fields_to_pcd(cls,
                              pcd: PointCloudData,
                              data: BaseDataT,
                              cfg: BaseLoadConfig,
                              field_names: set[str]):

        field_names &= cfg.keep_extra_scalar_fields

        for name in field_names:
            pcd.scalar_fields.create_field(name, data)

    @classmethod
    def _extract_rgb(cls, data: BaseDataT, num_points: int, field_names: set[str], ) -> RGBFields:
        rgb_names = field_names & set(RGB_PARTIAL_NAMES)

        if rgb_names == set(RGB_CHAR):
            rgb_names = RGB_CHAR
        elif rgb_names == set(RGB_WORD):
            rgb_names = RGB_WORD
        else:
            raise ValueError(f"No full set of {RGB_CHAR} or {RGB_WORD} fields were found.\nOnly :{field_names=}")

        array = np.empty((num_points, 3), dtype=data[rgb_names[0]].dtype)
        for i, name in enumerate(rgb_names):
            array[:, i] = data[name]

        cls._remove_field_names(field_names, *rgb_names)

        return RGBFields(array)

    @classmethod
    def _extract_normals(cls, data: BaseDataT, num_points: int, field_names: set[str], ) -> NormalFields:
        normal_names = field_names & set(NORMAL_PARTIAL_NAMES)

        if not normal_names == set(NORMAL_PARTIAL_NAMES):
            raise ValueError(f"No set of {NORMAL_PARTIAL_NAMES} fields were found. \nOnly :{normal_names=}")

        array = np.empty((num_points, 3), dtype=np.float32)
        for i, name in enumerate(NORMAL_PARTIAL_NAMES):
            array[:, i] = data[name]

        cls._remove_field_names(field_names, *normal_names)

        return NormalFields(array)

    @classmethod
    def _extract_reflectance_or_intensity(cls,
                                          data: BaseDataT,
                                          field_names: set[str],
                                          potential_names: set[str],
                                          fixed_name: str) -> NormalisedInt16ScalarField:

        matched_names = list(field_names & set(potential_names))

        if len(matched_names) != 1:
            raise ValueError(f"No {fixed_name} like field name found in [{potential_names}] fields.")

        cls._remove_field_names(field_names, *matched_names)

        arr = data[matched_names[0]]
        origin_dtype = DtypeState.generate(arr)

        return NormalisedInt16ScalarField(arr, name=fixed_name, origin_dtype=origin_dtype)

    @classmethod
    def _extract_intensity(cls, data: BaseDataT, field_names: set[str], ) -> NormalisedInt16ScalarField:
        return cls._extract_reflectance_or_intensity(
            data, field_names, INTENSITY_POTENTIAL_NAMES, INTENSITY_FIELD
        )

    @classmethod
    def _extract_reflectance(cls, data: BaseDataT, field_names: set[str]) -> NormalisedInt16ScalarField:
        return cls._extract_reflectance_or_intensity(
            data, field_names, REFLECTANCE_POTENTIAL_NAMES, REFLECTANCE_FIELD
        )

    @classmethod
    def _get_config(cls, load=True, **kwargs):
        if load:
            return cls.LOAD_CONFIG(**kwargs)
        return cls.SAVE_CONFIG(**kwargs)

    @staticmethod
    def _remove_field_names(field_names: set, *args):
        for name in args:
            field_names.remove(name)

    @staticmethod
    def _get_sf_save_dtype(cfg: BaseSaveConfig, scalar_field: ScalarField) -> str:
        if cfg.revert_sf_types:
            return scalar_field.origin_dtype.dtype.str
        return scalar_field.dtype.str

    @classmethod
    def _generate_struct_dtype(cls, pcd: PointCloudData, cfg: BaseSaveConfig):
        xyz_dtype = np.dtype(np.float64).str if pcd.optimized is not None else pcd.xyz.dtype.str
        dtype_list = [(name, str(xyz_dtype)) for name in ('x', 'y', 'z')]

        pcd_scalar_fields = set(pcd.scalar_fields.keys()).difference(
            {RGB_FIELD, NORMALS_FIELD, INTENSITY_FIELD, REFLECTANCE_FIELD})

        if cfg.keep_rgb and pcd.rgb is not None:
            dtype_list.extend(
                (name, cls._get_sf_save_dtype(cfg, pcd.rgb)) for name in RGB_WORD
            )

        if cfg.keep_normals and pcd.normals is not None:
            dtype_list.extend(
                (name, cls._get_sf_save_dtype(cfg, pcd.normals)) for name in NORMAL_PARTIAL_NAMES
            )

        if cfg.keep_intensity and pcd.intensity is not None:
            dtype_list.append(
                (INTENSITY_FIELD, cls._get_sf_save_dtype(cfg, pcd.intensity))
            )

        if cfg.keep_reflectance and pcd.reflectance is not None:
            dtype_list.append(
                (REFLECTANCE_FIELD, cls._get_sf_save_dtype(cfg, pcd.reflectance))
            )

        extra_fields = list(cfg.keep_extra_scalar_fields & pcd_scalar_fields)

        for sf_name in extra_fields:
            dtype_list.append(
                (sf_name, cls._get_sf_save_dtype(cfg, pcd.scalar_fields[sf_name]))
            )

        return dtype_list, extra_fields

    @classmethod
    def generate_structured_array(cls, pcd: PointCloudData, cfg: BaseSaveConfig):
        dtype_list, extra_fields = cls._generate_struct_dtype(pcd, cfg)
        array = np.empty((len(pcd),), dtype=dtype_list)

        shift = pcd.optimized_shift.optimal_shift

        array["x"] = pcd.x + shift[0] if pcd.optimized_shift else pcd.x
        array["y"] = pcd.y + shift[1] if pcd.optimized_shift else pcd.y
        array["z"] = pcd.z + shift[2] if pcd.optimized_shift else pcd.z

        if cfg.keep_rgb and (pcd.rgb is not None):
            rgb = pcd.rgb.get_original_data() if cfg.revert_sf_types else pcd.rgb
            array["red"] = rgb.r
            array["green"] = rgb.g
            array["blue"] = rgb.b

        if cfg.keep_normals and pcd.normals is not None:
            normals = pcd.normals.get_original_data() if cfg.revert_sf_types else pcd.normals
            array["nx"] = normals.nx
            array["ny"] = normals.ny
            array["nz"] = normals.nz

        if cfg.keep_intensity and pcd.intensity is not None:
            array[INTENSITY_FIELD] = pcd.intensity

        if cfg.keep_reflectance and pcd.reflectance is not None:
            array[REFLECTANCE_FIELD] = pcd.reflectance

        for name in extra_fields:
            sf = pcd.scalar_fields[name]
            array[name] = sf.get_original_data() if cfg.revert_sf_types else sf.arr

        return array
