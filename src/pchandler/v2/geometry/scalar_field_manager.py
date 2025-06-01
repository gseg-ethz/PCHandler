from __future__ import annotations

import weakref
from collections import Counter
from typing import Iterator, MutableMapping, overload, Self, TYPE_CHECKING, Iterable, NamedTuple

import numpy as np
from pydantic import validate_call

if TYPE_CHECKING:
    from .core import PointCloudData

from ..custom_types import IndexLike
from ..base_arrays import (
    Vector_N_T, Array_Nx3_T, Vector_N_u1_T, Array_Nx3_u1_T, Vector_N_f4_T, Array_Nx3_f4_T, Vector_N_u2_T)
from .scalar_fields import (
    ScalarField, RGBFields, IntensityField, NormalFields, ScalarFieldTriplet, RGB_FIELD, NORMALS_FIELD, INTENSITY_FIELD,
    RGB_POTENTIAL_NAMES, NORMAL_POTENTIAL_NAMES, INTENSITY_POTENTIAL_NAMES, LowerStr, DEFAULT_CONFIG)

class IndexPosition(NamedTuple):
    start: int
    end: int


class ScalarFieldManager(MutableMapping[str, type[ScalarField]]):
    """
    Manages a collection of ScalarField objects, ensuring that all fields have the same
    number of data points. Also provides a mechanism to select subsets of the fields.
    """
    def __init__(self,
                 parent: PointCloudData|None,
                 fields: dict[str, ScalarField]|None = None,
                 partial_fields: dict[int: ScalarField]|None = None,
                 merged_map: MutableMapping[int, tuple[int, int]]|None=None) -> None:
        self._parent: weakref.ReferenceType[PointCloudData] = weakref.ref(parent) if parent is not None else None
        self._fields: dict[str, ScalarField] = fields or {}
        self._partial_fields: dict[int, ScalarField] = partial_fields or {}
        self._merged_map: MutableMapping[int, tuple[int, int]] = merged_map or {}

    def keys(self) -> list[str]:
        return list(self._fields.keys())

    def values(self) -> Iterator[ScalarField]:
        return iter(self._fields.values())

    def items(self) -> Iterator[tuple[str, ScalarField]]:
        return iter(self._fields.items())

    def __contains__(self, key: str) -> bool:
        return key.lower() in self._fields

    def __len__(self) -> int:
        return len(self._fields)

    def __iter__(self) -> Iterator[str]:
        return iter(self._fields)

    def __delitem__(self, key: str) -> None:
        del self._fields[key]

    def as_dict(self) -> dict[str, ScalarField]:
        return self._fields

    def add_field(self, sf_field: ScalarField) -> None:
        self[sf_field.name.lower()] = sf_field

    def remove_field(self, field_name: LowerStr) -> None:
        del self[field_name.lower()]

    def create_field(self, name: str, data: Vector_N_T|Array_Nx3_T) -> None:
        sf = ScalarField(name=name, arr=data)
        self.add_field(sf)

    @property
    def shape(self) -> tuple[int, int]:
        return self.num_points, len(self)

    @property
    def num_points(self) -> int:
        return len(self._parent())

    @property
    def rgb(self) -> RGBFields|None:
        return self._fields.get(RGB_FIELD, None)

    @property
    def normals(self) -> NormalFields|None:
        return self._fields.get(NORMALS_FIELD, None)

    @property
    def intensity(self) -> IntensityField|None:
        return self._fields.get(INTENSITY_FIELD, None)

    @property
    def reflectance(self) -> IntensityField|None:
        return self._fields.get(INTENSITY_FIELD, None)

    @overload
    def __getitem__(self, key: str) -> ScalarField: ...

    @overload
    def __getitem__(self, key: IndexLike) -> Self: ...

    @validate_call(config=DEFAULT_CONFIG)
    def __getitem__(self, key: LowerStr|IndexLike) -> (ScalarField | dict[str, ScalarField]):

        if isinstance(key, str):
            return self._fields[key]
        # TODO check partial dict if key not found

        return self.sample(key)

    @validate_call(config=DEFAULT_CONFIG)
    def __setitem__(self, name: LowerStr, value: ScalarField | Vector_N_T | Array_Nx3_T) -> None:
        if not isinstance(value, np.ndarray):
            value = value.arr

        if name in RGB_POTENTIAL_NAMES: return self._handle_rgb(name, value)
        if name in NORMAL_POTENTIAL_NAMES: return self._handle_normal(name, value)
        if name in INTENSITY_POTENTIAL_NAMES: return self._handle_intensity_reflectance(value)

        if self.num_points != value.shape[0]:
            raise ValueError(
                f"Scalar field length does not equal #points: {self.num_points} != {value.shape[0]}" )

        if isinstance(value, np.ndarray):
            if value.ndim == 2:
                self._fields[name] = ScalarFieldTriplet(name=name, arr=value)
            else:
                self._fields[name] = ScalarField(name=name, arr=value)

        return None

    @validate_call(config=DEFAULT_CONFIG)
    def _handle_rgb(self, name: LowerStr, value: Vector_N_u1_T|Array_Nx3_u1_T) -> None:
        # Set the whole field
        if name in ('rgb', 'rgba', 'color', 'colour', 'colors', 'colours'):
            self._fields[RGB_FIELD] = RGBFields(arr=value[:, [0, 1, 2]])
            return

        elif name in ('bgr', 'bgra'):
            self._fields[RGB_FIELD] = RGBFields(arr=value[[2, 1, 0], :])
            return

        if self.rgb is None:
            self._fields[RGB_FIELD] = RGBFields.initialize(self.num_points)

        if name in ('r', 'red'):
            self.rgb.arr[:, 0] = value

        elif name in ('g', 'green'):
            self.rgb.arr[:, 1] = value

        elif name in ('b', 'blue'):
            self.rgb.arr[:, 2] = value

        else:
            raise KeyError(f'Unknown key made it into _handle_rgb : {name}')

    @validate_call(config=DEFAULT_CONFIG)
    def _handle_normal(self, name: LowerStr, value: Vector_N_f4_T|Array_Nx3_f4_T) -> None:
        # Set the whole field
        if name == ('nxnynz', 'normals', 'normal'):
            self._fields[NORMALS_FIELD] = NormalFields(arr=value[:, [0, 1, 2]])
            return

        elif name == 'nznynx':
            self._fields[NORMALS_FIELD] = NormalFields(arr=value[[2, 1, 0], :])
            return

        if self.normals is None:
            self._fields[NORMALS_FIELD] = NormalFields.initialize(self.num_points)

        if name in ('nx', 'normal_x'):
            self.normals.arr[:, 0] = value

        elif name in ('ny', 'normal_y'):
            self.normals.arr[:, 1] = value

        elif name in ('nz', 'normal_z'):
            self.normals.arr[:, 2] = value

        else:
            raise KeyError(f'Unknown key made it into normals : {name}')

    @validate_call(config=DEFAULT_CONFIG)
    def _handle_intensity_reflectance(self, value: Vector_N_u2_T) -> None:
        self._fields[INTENSITY_FIELD] = IntensityField(arr=value)

    def sample(self, mask: IndexLike) -> dict[str, ScalarField]:
        sample = {}

        for name, value in self.items():
            if isinstance(value, ScalarFieldTriplet):
                sample[name] = value[mask, :]

            else:
                sample[name] = value[mask]

        return sample

    def extract(self, mask: IndexLike) -> dict[str, ScalarField]:
        sample = self.sample(mask)

        self.reduce(mask)

        return sample

    def reduce(self, mask: IndexLike) -> None:

        for name, value in self.items():
            self._fields[name] = value[mask]

    # def __and__(self, other: ScalarFieldManager) -> ScalarFieldManager:
    #     if not isinstance(other, ScalarFieldManager):
    #         raise TypeError(f"Can only merge with another ScalarFieldManager, got {type(other)}")
    #
    #     return self.merge(other)

    @staticmethod
    def generate_point_cloud_map(sfms: Iterable[Self]) -> dict[int, slice]:
        indexes = [0]
        for i, b in enumerate(sfms):
            indexes.append(indexes[i] + b.num_points)

        starting_indexes = tuple(indexes[:-1])
        ending_indexes = tuple(indexes[1:])

        index_map = {}
        for i, value in enumerate(starting_indexes):
            index_map[i] = slice(value, ending_indexes[i], None)
        return index_map


    # # TODO look at a method for dumping dicts and appending arrays
    # # DISCUSS what is the goal of merge -? same points, same sfms? joining separate pcd with same keys?
    # #  Do we extend the sfs if they're missing by initialising the other points
    # # TODO look at merge flags
    # @classmethod
    # def merge(cls, sfms: Iterable[Self]) -> Self:
    #     # DISCUSS Idea to create a partial_fields dict to store all those not linked
    #
    #     sfm_key_sets = (set(sfm) for sfm in sfms)
    #     all_keys = set.union(*sfm_key_sets)
    #     keys_in_common = set.intersection(*sfm_key_sets)
    #     partial_keys = list(set.difference(all_keys, keys_in_common))
    #
    #     partial_fields: dict[int: dict[str: ScalarField]] = {}
    #     pcds_index_map = cls.generate_point_cloud_map(sfms)
    #
    #     if len(all_keys) == 0:
    #         return ScalarFieldManager(parent=None, fields={})
    #
    #     new_sfm = ScalarFieldManager(parent=None)
    #     for common_key in keys_in_common:
    #         sfs: list[ScalarField] = [sfm[common_key] for sfm in sfms]
    #
    #         # DISCUSS Warn when there's a mismatch between keys and names.
    #         #  Should not be the case if implement robustly...
    #         sf_names: list[str] = [sf.name for sf in sfs]
    #         if len(set(sf_names)) != 1:
    #             logger.warning(f"While merging scalar field {common_key} different names were encountered.")
    #
    #         # Get the most occurring name
    #         name = max((counted_names := Counter(sf_names)), key=counted_names.get)
    #         # TODO ensures all the contents of the operations performed match between same named fields.
    #         #  This is definitely worth ensuring to ensure same normalisation etc.
    #         if len(set(map(tuple, [sf.operations_performed for sf in sfs]))) != 1:
    #             logger.warning(
    #                 f"While merging scalar field {common_key} different list of previously performed "
    #                 f"operations were encountered. Merged scalar field will have an empty record!"
    #             )
    #             partial_keys.append(common_key)
    #             continue
    #         else:
    #             operations_performed = sfs[0].operations_performed
    #         # TODO Most important is the coerced value is constant
    #         if len(set(sf.original_dtype for sf in sfs)) != 1:
    #             logger.warning(
    #                 f"While merging scalar field {common_key} different original dtypes were encountered. "
    #                 f"Merged scalar field will have an empty record!"
    #             )
    #             partial_keys.append(common_key)
    #             continue
    #         else:
    #             original_dtype = sfs[0].original_dtype
    #
    #         # TODO join all the data into a single scalarfield and add it to the sfm
    #         data = np.concatenate([sf.data for sf in sfs])
    #         sf = ScalarField(name=name, arr=data, original_dtype=original_dtype, operations_performed=operations_performed
    #         )
    #         new_sfm.add_field(sf)
    #
    #     return new_sfm