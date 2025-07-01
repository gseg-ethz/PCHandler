from __future__ import annotations

import logging
import weakref
from collections import Counter
from collections.abc import ItemsView, ValuesView, KeysView
from typing import TYPE_CHECKING, Iterable, Iterator, MutableMapping, Self, overload, Optional, Any

import numpy as np
import numpy.typing as npt
from pydantic import validate_call

if TYPE_CHECKING:
    from .core import PointCloudData

from ..base_types import (
    Array_Nx3_Float32_T,
    Array_Nx3_T,
    Array_Nx3_Uint8_T,
    IndexLike,
    VectorT,
    Vector_Float32_T,
    Vector_Uint8_T,
)
from ..constants import (
    DEFAULT_CONFIG,
    NORMAL_POTENTIAL_NAMES,
    RGB_ALL_POTENTIAL_NAMES,
    NORMALS_FIELD,
    RGB_FIELD,
    INTENSITY_FIELD,
    REFLECTANCE_FIELD
)
from .scalar_fields import (
    SF_T,
    LowerStr,
    NormalFields,
    RGBFields,
    ScalarField,
    AbstractScalarField,
    NormalisedInt16ScalarField,
    DtypeState
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

                elif isinstance(value, AbstractScalarField):
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

    def keys(self) -> KeysView[str]:
        return self.fields.keys()

    def values(self) -> ValuesView[SF_T]:
        return self.fields.values()

    def items(self) -> ItemsView[str, SF_T]:
        return self.fields.items()

    @overload
    def __getitem__(self, key: str) -> ScalarField | RGBFields | NormalFields: ...

    @overload
    def __getitem__(self, key: IndexLike) -> Self: ...

    @overload
    def __getitem__(self, key: LowerStr) -> ScalarField | RGBFields | NormalFields: ...

    def __getitem__(self, key: str | LowerStr | IndexLike) -> ScalarField | RGBFields | NormalFields | Self:

        if isinstance(key, str):
            return self.fields[key]

        return self.sample(key)

    def __setitem__(self, name: LowerStr, value: SF_T) -> None:
        origin_dtype = None

        if isinstance(value, AbstractScalarField):
            origin_dtype = value.origin_dtype
            value = value.arr

        if not isinstance(value, np.ndarray):
            value = value.arr

        if name == INTENSITY_FIELD:
            self.fields[name] = ScalarField(value, name=INTENSITY_FIELD, origin_dtype=origin_dtype)
            return None

        if name == REFLECTANCE_FIELD:
            self.fields[name] = ScalarField(value, name=REFLECTANCE_FIELD, origin_dtype=origin_dtype)
            return None

        if name in RGB_ALL_POTENTIAL_NAMES:
            return self._handle_rgb(name, value, origin_dtype=origin_dtype)

        if name in NORMAL_POTENTIAL_NAMES:
            return self._handle_normal(name, value, origin_dtype=origin_dtype)

        if self._parent is not None:
            if self.num_points != value.shape[0]:
                raise ValueError( f"Scalar field length does not equal #points: {self.num_points} != {value.shape[0]}")
        else:
            logger.warning('No parent object to compare length of scalar fields to corresponding coordinate set')

        if isinstance(value, np.ndarray):
            self.fields[name] = ScalarField(value, name=name, origin_dtype=origin_dtype)

        return None

    def __delitem__(self, key: str) -> None:
        del self.fields[key]

    # TODO Ensure mame of scalar_field should always be lower case and match the key in _sfm dict
    # TODO add test for lower case fields

    def add_field(self, sf_field: ScalarField | RGBFields | NormalFields) -> None:
        self[sf_field.name] = sf_field

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

    @rgb.setter
    def rgb(self, value: npt.NDArray[np.floating|np.uint8] | RGBFields) -> None:
        if not isinstance(value, (np.ndarray, RGBFields)):
            value: npt.NDArray[np.floating|np.uint8] = np.asarray(value)

        if isinstance(value, np.ndarray):
            value: RGBFields = RGBFields(value)
        self.add_field(value)

    @property
    def normals(self) -> NormalFields | None:
        return self.fields.get(NORMALS_FIELD, None)

    @normals.setter
    def normals(self, value: np.ndarray | NormalFields):
        if not isinstance(value, (np.ndarray, NormalFields)):
            value = np.asarray(value)

        if isinstance(value, np.ndarray):
            value = NormalFields(value)
        self.add_field(value)

    @property
    def intensity(self):
        return self.fields.get("intensity", None)

    @intensity.setter
    def intensity(self, value: np.ndarray | ScalarField):
        if isinstance(value, np.ndarray):
            value = ScalarField(value, name=INTENSITY_FIELD)
        self.add_field(value)

    @property
    def reflectance(self):
        return self.fields.get("reflectance", None)

    @reflectance.setter
    def reflectance(self, value: np.ndarray | ScalarField):
        if isinstance(value, np.ndarray):
            value = ScalarField(value, name=REFLECTANCE_FIELD)
        self.add_field(value)

    @validate_call(config=DEFAULT_CONFIG)
    def _handle_rgb(
            self,
            name: LowerStr,
            value: Vector_Uint8_T | Array_Nx3_Uint8_T,
            origin_dtype: Optional[DtypeState] = None) -> None:

        # Set the whole field
        if name in ("rgb", "rgba", "color", "colour", "colors", "colours"):
            self.fields[RGB_FIELD] = RGBFields(value[:, [0, 1, 2]], origin_dtype=origin_dtype)
            return

        elif name in ("bgr", "bgra"):
            self.fields[RGB_FIELD] = RGBFields(value[[2, 1, 0], :], origin_dtype=origin_dtype)
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
    def _handle_normal(
            self,
            name: LowerStr,
            value: Vector_Float32_T | Array_Nx3_Float32_T,
            origin_dtype: Optional[DtypeState] = None) -> None:
        # Set the whole field
        if name in ("nxnynz", "normals", "normal"):
            self.fields[NORMALS_FIELD] = NormalFields(arr=value[:, [0, 1, 2]], origin_dtype=origin_dtype)
            return

        elif name == "nznynx":
            self.fields[NORMALS_FIELD] = NormalFields(arr=value[:, [2, 1, 0]], origin_dtype=origin_dtype)
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

    def sample(self, mask: IndexLike, view=False) -> ScalarFieldManager:
        sampled = type(self)(fields={})

        for name, value in self.items():
            sampled[name] = value.sample(mask)

        return sampled

    def extract(self, mask: IndexLike) -> ScalarFieldManager:
        mask = self._parent().create_mask(mask)
        sample = self.sample(mask)
        self.reduce(~mask)
        return sample

    def reduce(self, mask: IndexLike) -> None:
        for name, value in self.items():
            self.fields[name] = value[mask]


    @classmethod
    def merge(cls, sfms: Iterable[Self]) -> Self:
        sfm_key_sets = (set(sfm) for sfm in sfms)
        keys_in_common = set.intersection(*sfm_key_sets)

        new_sfm = ScalarFieldManager(parent=None)

        if len(keys_in_common) == 0:
            return new_sfm

        # Get the scalar field managers that have the key in common
        for common_key in keys_in_common:
            sfs: list[ScalarField] = [sfm[common_key] for sfm in sfms]

            # Check the names are the same. If not, take the most occurring name.

            sf_names: list[str] = [sf.name for sf in sfs]
            if len(set(sf_names)) != 1:
                logger.warning(f"While merging scalar field {common_key} different names were encountered.")
                name = max((counted_names := Counter(sf_names)), key=counted_names.get)
                logger.warning(f"Using the most occurring name for the scalar field: {name} out of ")
            else:
                name = sfs[0].name

            # Check if the original_dtype_states match. If not, do not use.
            if all([sfs[0].origin_dtype.dtype == sf.origin_dtype.dtype for sf in sfs[1:]]):
                origin_dtype = sfs[0].origin_dtype
                data = np.concatenate([sf.arr for sf in sfs])
                sf = type(sfs[0])(data, name=name, origin_dtype=origin_dtype)
                new_sfm.add_field(sf)
            else:
                logger.warning(
                    f"While merging scalar field {common_key} different list of previously performed "
                    f"operations were encountered. Merged scalar field will have an empty record!"
                )

        return new_sfm
