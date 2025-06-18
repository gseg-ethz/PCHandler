
import logging

from abc import ABC, abstractmethod
from enum import IntEnum, auto
from pathlib import Path

from typing import Any, Mapping, TypedDict, Iterable, Unpack, Optional, NotRequired

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..constants import (
    INTENSITY_POTENTIAL_NAMES,
    REFLECTANCE_POTENTIAL_NAMES,
    RGB_PARTIAL_NAMES,
    RGB_CHAR,
    RGB_WORD,
    NORMAL_PARTIAL_NAMES
)
from ..geometry.core import PointCloudData
from ..geometry.scalar_fields import (
    ScalarField,
    RGBFields,
    NormalFields,
    DtypeState,
)

from ..geometry.util import linear_map_dtype, normalize_min_max, normalize_self

logger = logging.getLogger(__name__.split(".")[0])


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


class _BaseLoadConfigType(_BaseConfigType):
    normalize_rgb: NotRequired[NormaliseEnum]
    normalize_intensity: NotRequired[NormaliseEnum]
    normalize_reflectance: NotRequired[NormaliseEnum]
    cloud_compare_exported: NotRequired[bool]

class _BaseSaveConfigType(_BaseConfigType):
    revert_sf_types: NotRequired[bool]


class _BaseConfig(BaseModel):
    model_config = ConfigDict(extra='ignore')
    keep_rgb: Optional[bool] = False
    keep_normals: Optional[bool] = False
    keep_intensity: Optional[bool] = False
    keep_reflectance: Optional[bool] = False
    keep_extra_scalar_fields: set[str] = Field(default_factory=set)

    @field_validator('keep_extra_scalar_fields', mode='before')
    @classmethod
    def convert_to_set(cls, value: Iterable[str]) -> set[str]:
        return set(value)


class BaseLoadConfig(_BaseConfig):
    normalize_rgb: NormaliseEnum = Field(default=NormaliseEnum.MINMAX)
    normalize_intensity: NormaliseEnum = Field(default=NormaliseEnum.NONE)
    normalize_reflectance: NormaliseEnum = Field(default=NormaliseEnum.NONE)
    keep_extra_scalar_fields: Iterable[str] = Field(default_factory=set)
    cloud_compare_exported: bool = False


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
    def _extract_xyz(cls, data: Mapping[str, np.ndarray], num_points: int, field_names: set[str]):
        xyz = np.empty((num_points, 3), dtype=float)
        xyz[:, 0] = data["x"]
        xyz[:, 1] = data["y"]
        xyz[:, 2] = data["z"]

        cls._remove_field_names(field_names, 'x', 'y', 'z')

        return xyz

    # TODO extract origin_dtype info and return as the scalar_field

    @classmethod
    def _extract_rgb(cls, data: Mapping[str, np.ndarray],
                    num_points: int,
                    field_names: set[str],
                    normalise_method: NormaliseEnum = NormaliseEnum.NONE,
                    ) -> RGBFields:

        rgb_names = field_names & set(RGB_PARTIAL_NAMES)

        if rgb_names == set(RGB_CHAR):
            rgb_names = RGB_CHAR

        elif rgb_names == set(RGB_WORD):
            rgb_names = RGB_WORD

        else:
            raise ValueError(
                f"No full set of {RGB_CHAR} or {RGB_WORD} fields were found."
                f"\nOnly :{field_names=}"
            )

        array = np.empty((num_points, 3), dtype=data[rgb_names[0]].dtype)

        for i, name in enumerate(rgb_names):
            array[:, i] = data[name]

        origin_dtype = DtypeState.generate(array)
        array = cls._normalise(array, method=normalise_method, target_dtype=np.uint8)
        cls._remove_field_names(field_names, *rgb_names)

        return RGBFields(array, origin_dtype=origin_dtype)

    @classmethod
    def _extract_normals(cls, data: Mapping[str, np.ndarray],
                    num_points: int,
                    field_names: set[str],
                    ) -> NormalFields:

        normal_names = field_names & set(NORMAL_PARTIAL_NAMES)

        if not normal_names == set(NORMAL_PARTIAL_NAMES):
            raise ValueError(
                f"No set of {NORMAL_PARTIAL_NAMES} fields were found. \nOnly :{normal_names=}"
            )

        array = np.empty((num_points, 3), dtype=np.float32)

        for i, name in enumerate(NORMAL_PARTIAL_NAMES):
            array[:, i] = data[name]

        origin_dtype = DtypeState.generate(array)
        dist = np.linalg.norm(array, axis=1).astype(np.float32)
        if not np.allclose(dist, 1):
            array /= dist
        array = NormalFields(array, origin_dtype = origin_dtype)

        cls._remove_field_names(field_names, *normal_names)

        return array

    @classmethod
    def _extract_intensity(cls,
                           data: Mapping[str, np.ndarray],
                           field_names: set[str],
                           normalise_method: NormaliseEnum = NormaliseEnum.MINMAX,
                           target_dtype: np.dtype|DtypeState = np.uint16
                           ) -> ScalarField:

        intensity_name = field_names & set(INTENSITY_POTENTIAL_NAMES)

        if len(intensity_name) != 1:
            raise ValueError(
                f"No intensity like field names {INTENSITY_POTENTIAL_NAMES} fields were found."
            )

        origin_dtype = DtypeState.generate(data[intensity_name[0]])
        array = cls._normalise(data[intensity_name[0]], method=normalise_method, target_dtype=target_dtype)
        array = ScalarField(array, intensity_name[0], origin_dtype=origin_dtype)
        cls._remove_field_names(field_names, *intensity_name)

        return array

    @classmethod
    def _extract_reflectance(cls,
                             data: Mapping[str, np.ndarray],
                             field_names: set[str],
                             normalise_method: NormaliseEnum = NormaliseEnum.NONE,
                             target_dtype: np.dtype|DtypeState = np.uint16
                             ) -> ScalarField:

        reflectance_name = field_names & set(REFLECTANCE_POTENTIAL_NAMES)

        if len(field_names) != 1:
            raise ValueError(
                f"No intensity like field names {REFLECTANCE_POTENTIAL_NAMES} fields were found."
            )

        origin_dtype = DtypeState.generate(data[reflectance_name[0]])
        array = cls._normalise(data[reflectance_name[0]], method=normalise_method, target_dtype=target_dtype)
        array = ScalarField(array, reflectance_name[0], origin_dtype=origin_dtype)

        cls._remove_field_names(field_names, *reflectance_name)

        return array

    @staticmethod
    def _normalise(array: np.ndarray,
                   method: NormaliseEnum,
                   target_dtype: np.typing.DTypeLike|DtypeState = None
                   ) -> np.ndarray:
        match method:
            case NormaliseEnum.NONE:
                return array

            case NormaliseEnum.DTYPE:
                # Normalizes to min and max of the integer dtype or 0.0 and 1.0 for floating point
                return normalize_self(array)

            case NormaliseEnum.MAPPING:
                # Linearly maps the current values to the target dtype based on the min_max values of the
                # array dtype.
                if not isinstance(target_dtype, DtypeState):
                    target_dtype = target_dtype.dtype
                return linear_map_dtype(array, target_dtype)

            case NormaliseEnum.MINMAX:
                # Normalises based on array min_max values to target
                if not isinstance(target_dtype, np.dtype):
                    target_dtype = DtypeState(dtype=target_dtype, lower=0, upper=1)
                return normalize_min_max(val=array,
                                         lower=target_dtype.lower,
                                         upper=target_dtype.upper,
                                         target_dtype=target_dtype.dtype)

            case _:
                raise ValueError(f"Unknown normalisation method passed.")

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