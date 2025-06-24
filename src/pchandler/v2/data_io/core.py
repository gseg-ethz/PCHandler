
import logging

from abc import ABC, abstractmethod
from enum import IntEnum, auto
from pathlib import Path

from typing import Mapping, TypedDict, Iterable, Unpack, Optional, NotRequired, Annotated, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, BeforeValidator

from ..constants import (
    INTENSITY_POTENTIAL_NAMES,
    REFLECTANCE_POTENTIAL_NAMES,
    RGB_PARTIAL_NAMES,
    RGB_CHAR,
    RGB_WORD,
    RGB_FLOAT,
    NORMALS_CHAR,
    NORMALS_WORD,
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


def skip_main_fields(field_names: Sequence[str]|set[str]):
    return set(field_names).difference({RGB_FIELD, INTENSITY_FIELD, NORMALS_FIELD, REFLECTANCE_FIELD})


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
    keep_extra_scalar_fields: Annotated[set[str], BeforeValidator(skip_main_fields), Field(default_factory=set)]
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
    def get_config(cls, load=True, **kwargs):
        if load:
            return cls.LOAD_CONFIG(**kwargs)
        return cls.SAVE_CONFIG(**kwargs)

    @classmethod
    def find_pcds_in_directory(cls, directory_path: str|Path, include_subdirectories: bool = True):
        return find_pcd_in_directory(directory_path, cls.FORMATS, include_subdirectories)

    @classmethod
    def _extract_xyz(cls, data: BaseDataT, num_points: int, field_names: set[str]):
        xyz = np.empty((num_points, 3), dtype=data["x"].dtype)
        xyz[:, 0] = data["x"]
        xyz[:, 1] = data["y"]
        xyz[:, 2] = data["z"]

        cls.remove_field_names(field_names, 'x', 'y', 'z')

        return xyz

    @classmethod
    def extract_common_fields(cls,
                              pcd: PointCloudData,
                              data: BaseDataT,
                              cfg: BaseLoadConfig,
                              num_points: int,
                              field_names: set[str]) -> None:

        for name in ('rgb', 'normals', 'intensity', 'reflectance'):
            if getattr(cfg, f'keep_{name}'):
                if current_field_names := getattr(cls, f"_get_{name}_field_names")(field_names):
                    if name in ('normals', 'rgb'):
                        setattr(pcd, name, getattr(cls, f"extract_{name}")(data, num_points, current_field_names))
                    else:
                        setattr(pcd, name, getattr(cls, f"extract_{name}")(data))

    @classmethod
    def extract_extra_fields(cls, pcd: PointCloudData, data: BaseDataT, cfg: BaseLoadConfig, field_names: set[str]):
        field_names &= cfg.keep_extra_scalar_fields

        for name in field_names:
            pcd.scalar_fields.create_field(name, data)

    @classmethod
    def extract_rgb(cls, data: BaseDataT, num_points: int, rgb_names: list[str]) -> RGBFields:
        array = np.empty((num_points, 3), dtype=data[rgb_names[0]].dtype)

        for i, name in enumerate(rgb_names):
            array[:, i] = data[name]

        return RGBFields(array)

    @classmethod
    def extract_normals(cls, data: BaseDataT, num_points: int, normals_names: list[str]) -> NormalFields:

        array = np.empty((num_points, 3), dtype=np.float32)

        for i, name in enumerate(normals_names):
            array[:, i] = data[name]

        return NormalFields(array)

    @classmethod
    def _extract_reflectance_or_intensity(cls, data: BaseDataT, fixed_name: str) -> NormalisedInt16ScalarField:

        arr = data[fixed_name]
        origin_dtype = DtypeState.generate(arr)

        return NormalisedInt16ScalarField(arr, name=fixed_name, origin_dtype=origin_dtype)

    @classmethod
    def extract_intensity(cls, data: BaseDataT, ) -> NormalisedInt16ScalarField:
        return cls._extract_reflectance_or_intensity(data, INTENSITY_FIELD)

    @classmethod
    def extract_reflectance(cls, data: BaseDataT) -> NormalisedInt16ScalarField:
        return cls._extract_reflectance_or_intensity(data, REFLECTANCE_FIELD)

    @staticmethod
    def remove_field_names(field_names: set[str], *args):
        for name in args:
            field_names.remove(name)

    @staticmethod
    def _get_sf_dtype(cfg: BaseSaveConfig, scalar_field: ScalarField) -> str:
        if cfg.revert_sf_types:
            return scalar_field.origin_dtype.dtype.str
        return scalar_field.dtype.str

    @classmethod
    def generate_struct_dtype(cls, pcd: PointCloudData, cfg: BaseSaveConfig):
        xyz_dtype = np.dtype(np.float64).str if pcd.optimized is not None else pcd.xyz.dtype.str
        dtype_list = [(name, str(xyz_dtype)) for name in ('x', 'y', 'z')]

        pcd_scalar_fields = set(pcd.scalar_fields.keys()).difference(
            {RGB_FIELD, NORMALS_FIELD, INTENSITY_FIELD, REFLECTANCE_FIELD})

        # TODO try to fix this and make more concise

        if cfg.keep_rgb and pcd.rgb is not None:
            dtype_list.extend(
                (name, cls._get_sf_dtype(cfg, pcd.rgb)) for name in RGB_WORD
            )

        if cfg.keep_normals and pcd.normals is not None:
            dtype_list.extend(
                (name, cls._get_sf_dtype(cfg, pcd.normals)) for name in NORMALS_CHAR
            )

        if cfg.keep_intensity and pcd.intensity is not None:
            dtype_list.append(
                (INTENSITY_FIELD, cls._get_sf_dtype(cfg, pcd.intensity))
            )

        if cfg.keep_reflectance and pcd.reflectance is not None:
            dtype_list.append(
                (REFLECTANCE_FIELD, cls._get_sf_dtype(cfg, pcd.reflectance))
            )

        extra_fields = list(cfg.keep_extra_scalar_fields & pcd_scalar_fields)

        for sf_name in extra_fields:
            dtype_list.append(
                (sf_name, cls._get_sf_dtype(cfg, pcd.scalar_fields[sf_name]))
            )

        return dtype_list, extra_fields

    @classmethod
    def generate_structured_array(cls, pcd: PointCloudData, cfg: BaseSaveConfig):
        dtype_list, extra_fields = cls.generate_struct_dtype(pcd, cfg)
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

    @staticmethod
    def _get_rgb_field_names(field_names: set[str]) -> list[str] | None:
        rgb_names = field_names & set(RGB_PARTIAL_NAMES)

        if rgb_names == set(RGB_CHAR):
            return list(RGB_CHAR)

        elif rgb_names == set(RGB_WORD):
            return list(RGB_WORD)

        elif rgb_names == set(RGB_FLOAT):
            return list(RGB_FLOAT)

        else:
            logger.warning(f"No full set of {RGB_CHAR} or {RGB_WORD} fields were found.\nOnly :{field_names=}")
            return None

    @staticmethod
    def _get_normals_field_names(field_names: set[str]) -> list[str] | None:
        normal_names = field_names & set(NORMAL_PARTIAL_NAMES)

        if normal_names == set(NORMALS_CHAR):
            return list(NORMALS_CHAR)

        elif normal_names == set(NORMALS_WORD):
            return list(NORMALS_WORD)

        else:
            logger.warning(f"No full set of {NORMALS_CHAR} or {NORMALS_WORD} fields were found. \n"
                           f"Only :{normal_names=}")
            return None


    @staticmethod
    def _get_intensity_or_reflectance_field_name(field_names: set[str], potential_names: set[str]) -> list[str]|None:
        matched_names = list(field_names & set(potential_names))

        if len(matched_names) != 1:
            logger.warning(f"No 'intensity' or 'reflectance' like field in [{potential_names}] fields.")
            return None

        return matched_names

    @classmethod
    def _get_intensity_field_names(cls, field_names) -> list[str] | None:
        return cls._get_intensity_or_reflectance_field_name(field_names, INTENSITY_POTENTIAL_NAMES)

    @classmethod
    def _get_reflectance_field_names(cls, field_names) -> list[str] | None:
        return cls._get_intensity_or_reflectance_field_name(field_names, REFLECTANCE_POTENTIAL_NAMES)