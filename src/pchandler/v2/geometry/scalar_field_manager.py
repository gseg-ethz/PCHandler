from __future__ import annotations

import logging
import weakref
from collections import Counter
from collections.abc import ItemsView, ValuesView
from typing import TYPE_CHECKING, Iterable, Iterator, MutableMapping, Self, overload

import numpy as np
from pydantic import validate_call

if TYPE_CHECKING:
    from .core import PointCloudData

from ..base_types import (
    Array_Nx3_float32_T,
    Array_Nx3_T,
    Array_Nx3_uint8_T,
    IndexLike,
    VectorT,
    VectorT_Float32,
    VectorT_Uint8,
)
from ..constants import DEFAULT_CONFIG, NORMAL_POTENTIAL_NAMES, RGB_POTENTIAL_NAMES
from .scalar_fields import (
    NORMALS_FIELD,
    RGB_FIELD,
    SF_T,
    LowerStr,
    NormalFields,
    RGBFields,
    ScalarField,
)

logger = logging.getLogger(__name__.split(".")[0])


class ScalarFieldManager(MutableMapping[str, SF_T]):
    """
    Manages a collection of ScalarField objects, ensuring that all fields have the same
    number of data points. Also provides a mechanism to select subsets of the fields.
    """

    def __init__(
        self, parent: PointCloudData | None = None, fields: dict[str, SF_T | np.ndarray] | Self | None = None
    ) -> None:
        self._parent: weakref.ReferenceType[PointCloudData] | None = weakref.ref(parent) if parent is not None else None

        if isinstance(fields, dict):
            for key, value in fields.items():
                if isinstance(value, np.ndarray):
                    value = ScalarField(value, name=key)

                elif isinstance(value, ScalarField):
                    value.name = key
                else:
                    raise TypeError(f"Type of input field is not of numpy array or ScalarField but {type(value)}")

                fields[key] = value

            self.fields: dict[str, SF_T] = fields
        elif fields is None:
            self.fields = {}
        elif isinstance(fields, type(self)):
            self.fields = fields.fields
        else:
            raise TypeError(f"Unknown fields type: {type(fields)}")

    def __iter__(self) -> Iterator[str]:
        return iter(self.fields)

    def __contains__(self, key: str) -> bool:
        return key.lower() in self.fields

    def __len__(self) -> int:
        return len(self.fields)

    def keys(self) -> list[str]:
        return list(self.fields.keys())

    def values(self) -> ValuesView[SF_T]:
        return self.fields.values()

    def items(self) -> ItemsView[str, SF_T]:
        return self.fields.items()

    @overload
    def __getitem__(self, key: str) -> ScalarField: ...

    @overload
    def __getitem__(self, key: IndexLike) -> Self: ...

    @validate_call(config=DEFAULT_CONFIG)
    def __getitem__(self, key: LowerStr | IndexLike) -> SF_T | dict[str, SF_T]:

        if isinstance(key, str):
            return self.fields[key]

        return self.sample(key)

    @validate_call(config=DEFAULT_CONFIG)
    def __setitem__(self, name: LowerStr, value: SF_T) -> None:
        if not isinstance(value, np.ndarray):
            value = value.arr

        if name in RGB_POTENTIAL_NAMES:
            return self._handle_rgb(name, value)
        if name in NORMAL_POTENTIAL_NAMES:
            return self._handle_normal(name, value)

        if self.num_points != value.shape[0]:
            raise ValueError(f"Scalar field length does not equal #points: {self.num_points} != {value.shape[0]}")

        if isinstance(value, np.ndarray):
            self.fields[name] = ScalarField(value, name=name)

        return None

    def __delitem__(self, key: str) -> None:
        del self.fields[key]

    def add_field(self, sf_field: ScalarField) -> None:
        self.fields[sf_field.name.lower()] = sf_field

    def remove_field(self, field_name: LowerStr) -> None:
        del self.fields[field_name.lower()]

    def create_field(self, name: str, data: VectorT | Array_Nx3_T) -> None:
        sf = ScalarField(data, name=name)
        self.add_field(sf)

    @property
    def shape(self) -> tuple[int, int]:
        return self.num_points, len(self)

    @property
    def num_points(self) -> int:
        return len(self._parent())

    @property
    def rgb(self) -> RGBFields | None:
        return self.fields.get(RGB_FIELD, None)

    @property
    def intensity(self):
        return self.fields.get("intensity", None)

    @property
    def reflectance(self):
        return self.fields.get("reflectance", None)

    @property
    def normals(self) -> NormalFields | None:
        return self.fields.get(NORMALS_FIELD, None)

    @validate_call(config=DEFAULT_CONFIG)
    def _handle_rgb(self, name: LowerStr, value: VectorT_Uint8 | Array_Nx3_uint8_T) -> None:
        # Set the whole field
        if name in ("rgb", "rgba", "color", "colour", "colors", "colours"):
            self.fields[RGB_FIELD] = RGBFields(arr=value[:, [0, 1, 2]])
            return

        elif name in ("bgr", "bgra"):
            self.fields[RGB_FIELD] = RGBFields(arr=value[[2, 1, 0], :])
            return

        if self.rgb is None:
            self.fields[RGB_FIELD] = RGBFields.initialize(self.num_points)

        if name in ("r", "red"):
            self.rgb.arr[:, 0] = value

        elif name in ("g", "green"):
            self.rgb.arr[:, 1] = value

        elif name in ("b", "blue"):
            self.rgb.arr[:, 2] = value

        else:
            raise KeyError(f"Unknown key made it into _handle_rgb : {name}")

    @validate_call(config=DEFAULT_CONFIG)
    def _handle_normal(self, name: LowerStr, value: VectorT_Float32 | Array_Nx3_float32_T) -> None:
        # Set the whole field
        if name == ("nxnynz", "normals", "normal"):
            self.fields[NORMALS_FIELD] = NormalFields(arr=value[:, [0, 1, 2]])
            return

        elif name == "nznynx":
            self.fields[NORMALS_FIELD] = NormalFields(arr=value[[2, 1, 0], :])
            return

        if self.normals is None:
            self.fields[NORMALS_FIELD] = NormalFields.initialize(self.num_points)

        if name in ("nx", "normal_x"):
            self.normals.arr[:, 0] = value

        elif name in ("ny", "normal_y"):
            self.normals.arr[:, 1] = value

        elif name in ("nz", "normal_z"):
            self.normals.arr[:, 2] = value

        else:
            raise KeyError(f"Unknown key made it into normals : {name}")

    def sample(self, mask: IndexLike) -> dict[str, ScalarField]:
        sample = {}

        for name, value in self.items():
            mask = value.create_mask(mask)
            sample[name] = value[mask]

        return sample

    def extract(self, mask: IndexLike) -> dict[str, ScalarField]:
        sample = self.sample(mask)
        self.reduce(mask)
        return sample

    def reduce(self, mask: IndexLike) -> None:
        for name, value in self.items():
            self.fields[name] = value[mask]

    # TODO pull out, maybe nice to have in future if requested
    @staticmethod
    def generate_point_cloud_map(sfms: Iterable[ScalarFieldManager]) -> dict[int, slice]:
        indexes = [0]
        for i, b in enumerate(sfms):
            indexes.append(indexes[i] + b.num_points)

        starting_indexes = tuple(indexes[:-1])
        ending_indexes = tuple(indexes[1:])

        index_map = {}
        for i, value in enumerate(starting_indexes):
            index_map[i] = slice(value, ending_indexes[i], None)
        return index_map

    @classmethod
    def merge(cls, sfms: Iterable[Self]) -> Self:
        sfm_key_sets = (set(sfm) for sfm in sfms)
        keys_in_common = set.intersection(*sfm_key_sets)

        # return if there are no keys/scalar fields in common
        if len(keys_in_common) == 0:
            return ScalarFieldManager(parent=None, fields={})
        new_sfm = ScalarFieldManager(parent=None)

        for common_key in keys_in_common:
            # Get the scalar field managers that have the key in common
            sfs: list[ScalarField] = [sfm[common_key] for sfm in sfms]

            # Check the names are the same. If not, take the most occurring name.
            # DECISION Name of scalar_field should always be lower case and match the key in _sfm dict
            sf_names: list[str] = [sf.name for sf in sfs]
            if len(set(sf_names)) != 1:
                logger.warning(f"While merging scalar field {common_key} different names were encountered.")
                name = max((counted_names := Counter(sf_names)), key=counted_names.get)
                logger.warning(f"Using the most occurring name for the scalar field: {name} out of ")
            else:
                name = sfs[0].name

            # Check if the original_dtype_states match. If not, do not use.
            if all([sfs[0] != sf for sf in sfs[1:]]):
                origin_dtype = sfs[0].origin_dtype
                data = np.concatenate([sf.arr for sf in sfs])
                sf = ScalarField(data, name=name, origin_dtype=origin_dtype)
                new_sfm.add_field(sf)
            else:
                logger.warning(
                    f"While merging scalar field {common_key} different list of previously performed "
                    f"operations were encountered. Merged scalar field will have an empty record!"
                )

        return new_sfm
