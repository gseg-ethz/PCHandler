# pchandler - Toolbox for point-cloud handling, processing and analysis
#
# Copyright (c) 2025, Nicholas Meyer, Geosensors and Engineering Geodesy,
# Institute of Geodesy and Photogrammetry, ETH Zurich, Switzerland
# SPDX-License-Identifier: BSD-3-Clause
#
# Author: Nicholas Meyer (meyernic@ethz.ch)

from __future__ import annotations

import logging
import weakref
from collections import Counter
from collections.abc import ItemsView, KeysView, ValuesView
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Optional, Self, Sized, TypeAlias, cast, overload

import numpy as np
from GSEGUtils.base_arrays import BaseArray
from GSEGUtils.base_types import (
    Array_Nx3_Float32_T,
    Array_Nx3_Float_T,
    Array_Nx3_T,
    Array_Nx3_Uint8_T,
    Array_Uint8_T,
    IndexLike,
    Vector_Bool_T,
    Vector_Float32_T,
    Vector_Uint8_T,
    VectorT,
)
from GSEGUtils.validators import normalize_uint8

from pchandler.constants import INTENSITY_NAMES, NORMAL_NAMES, REFLECTANCE_NAMES, RGB_NAMES
from pchandler.scalar_fields.scalar_fields import (
    AbstractScalarField,
    DtypeState,
    LowerStr,
    NormalFields,
    RGBFields,
    ScalarField,
    SegmentationMap,
)

if TYPE_CHECKING:
    from pchandler import PointCloudData

__all__ = ["ScalarFieldManager"]

logger = logging.getLogger(__name__.split(".")[0])

#: Base scalar field TypeAlias
SF_T: TypeAlias = RGBFields | NormalFields | ScalarField | SegmentationMap

#: Scalar Field like TypeAlias, also supporting other arrays of similar shape
SFLikeT: TypeAlias = SF_T | VectorT | Array_Nx3_T

#: RGB like TypeAlias
RGBLikeT: TypeAlias = Array_Nx3_Uint8_T | Vector_Uint8_T | RGBFields

#: Normal like TypeAlias
NormalLikeT: TypeAlias = Array_Nx3_Float32_T | Vector_Float32_T | NormalFields

#: ScalarFieldManager like dict TypeAlias
SFMLikeT: TypeAlias = dict[str, SFLikeT]


class ScalarFieldManager:
    """Manages scalar fields associated with a point cloud.

    Contains mechanisms for adding, changing and removing scalar fields as well as validating lengths with respect to
    a parent point cloud object.

    Parameters
    ----------
    fields: dict[str, ScalarField|NormalFields|RGBFields]
    _parent: weakref.ReferenceType[PointCloudData] | None
    """

    _parent: Optional[weakref.ReferenceType[PointCloudData]]
    fields: dict[str, SF_T]

    def __init__(self, fields: Optional[SFMLikeT | Self] = None, *, parent: Optional[PointCloudData] = None) -> None:
        """Initialise the scalar field manager.

        Ensures all fields passed are converted to ScalarField objects.

        If a parent point cloud is provided, it is stored as a weakref and scalar field lengths will be validated
        against it.

        Parameters
        ----------
        fields: dict[str, ScalarField | NormalFields | RGBFields | np.ndarray] | ScalarFieldManager | None
        parent: PointCloudData | None
        """
        self._parent = weakref.ref(parent) if parent is not None else None

        if fields is None:
            self.fields = {}

        elif isinstance(fields, type(self)):
            self.fields = fields.fields

        elif isinstance(fields, dict):
            for key, value in fields.items():
                if isinstance(value, np.ndarray):
                    if key in RGB_NAMES.all:
                        value = RGBFields(value, name=key)
                    elif key in NORMAL_NAMES.all:
                        value = NormalFields(value, name=key)
                    else:
                        value = ScalarField(value, name=key)

                elif isinstance(value, AbstractScalarField):
                    value.name = key

                else:
                    raise TypeError(f"Type of input field is not of numpy array or ScalarField but {type(value)}")

                fields[key] = value

            self.fields = cast(dict[str, SF_T], fields)

        else:
            raise TypeError(f"Unknown fields type: {type(fields)}")

    def validate_lengths(self) -> None:
        """Check all scalar fields lengths match the number of points in the parent

        Returns
        -------

        """
        if self.parent:
            for field in self.values():
                if len(field) != len(self.parent):
                    raise ValueError(f"Scalar field '{field}' length does not match the number of points {len(field)}")
        else:
            logger.info("No parent point cloud to validate scalar field lengths against")

    def __len__(self) -> int:
        """Returns the number of scalar fields

        Returns
        -------
        int
        """
        return len(self.fields)

    def __iter__(self) -> Iterator[str]:
        """Returns an iterator for the scalar fields"""
        return iter(self.fields)

    def __contains__(self, key: object) -> bool:
        """Check if a scalar field exists

        Parameters
        ----------
        key: object

        Returns
        -------
        bool
        """
        return str(key).lower() in self.fields

    @overload
    def __getitem__(self, key: str) -> SF_T | None: ...

    @overload
    def __getitem__(self, key: IndexLike) -> Self: ...

    def __getitem__(self, key: str | LowerStr | IndexLike) -> Self | SF_T | None:
        """Get a scalar field by name or a sample of the scalar field based on a mask.

        Parameters
        ----------
        key: str | LowerStr | IndexLike

        Returns
        -------
        SF_T | ScalarFieldManager | None
        """
        if isinstance(key, str):
            if key in RGB_NAMES.all:
                return self._get_rgb(key)

            if key in NORMAL_NAMES.all:
                return self._get_normals(key)

            if key in INTENSITY_NAMES.all:
                return self.intensity

            if key in REFLECTANCE_NAMES.all:
                return self.reflectance

            return self.fields[key]

        return self.sample(key)

    def __setitem__(self, name: LowerStr, value: SF_T | VectorT | Array_Nx3_T | None) -> None:
        """Set the value of a scalar field.

        Set to None to delete an existing field.

        Parameters
        ----------
        name: LowerStr
        value: ScalarField | NormalFields | RGBFields | np.ndarray | None

        Returns
        -------

        """
        # Delete value if None is passed or ignore
        if value is None:
            if name in self:
                del self[name]
            return

        # Get DtypeState from ScalarFields
        origin_dtype = value.origin_dtype if isinstance(value, AbstractScalarField) else None

        if not isinstance(value, (BaseArray, np.ndarray)):
            logger.info(f"Unknown type for scalar field: {type(value)} - converting to numpy array")

        value = np.asarray(value)
        origin_dtype = DtypeState.generate(value) if origin_dtype is None else origin_dtype

        if self.num_points > 0 and self.num_points != value.shape[0]:
            raise ValueError(f"Scalar field length does not equal #points: {self.num_points} != {value.shape[0]}")

        if name in RGB_NAMES.all:
            self._set_rgb(name, value, origin_dtype=origin_dtype)

        elif name in NORMAL_NAMES.all:
            self._set_normals(name, value, origin_dtype=origin_dtype)

        elif name in INTENSITY_NAMES.all:
            self.fields[INTENSITY_NAMES.base] = ScalarField(value, name=INTENSITY_NAMES.base, origin_dtype=origin_dtype)

        elif name in REFLECTANCE_NAMES.all:
            self.fields[REFLECTANCE_NAMES.base] = ScalarField(
                value, name=REFLECTANCE_NAMES.base, origin_dtype=origin_dtype
            )

        else:
            self.fields[name] = ScalarField(value, name=name, origin_dtype=origin_dtype)

    def __delitem__(self, key: str) -> None:
        """Delete a scalar field

        Parameters
        ----------
        key: str

        Returns
        -------

        """
        del self.fields[key]

    def __getstate__(self):
        """Return the object's state.

        Returns
        -------

        """
        state = self.__dict__.copy()
        state["_parent"] = None
        return state

    def __setstate__(self, state: dict[str, Any] = None) -> None:
        """Set the object's state.

        Parameters
        ----------
        state: dict[str, Any]

        Returns
        -------

        """
        self.__dict__.update(state)

    def keys(self) -> KeysView[str]:
        """Return the scalar field names stored in the manager.

        Returns
        -------
        KeysView[str]
        """
        return self.fields.keys()

    def values(self) -> ValuesView[SF_T]:
        """Return the scalar field values stored in the manager.

        Returns
        -------
        ValuesView[SF_T]
        """
        return self.fields.values()

    def items(self) -> ItemsView[str, SF_T]:
        """Return the scalar field names and values stored in the manager.

        Returns
        -------
        ItemsView[str, SF_T]
        """
        return self.fields.items()

    @property
    def parent(self) -> PointCloudData | None:
        """Return the parent point cloud object.

        Returns
        -------
        PointCloudData | None
        """
        if self._parent is None:
            return None
        return self._parent()

    @parent.setter
    def parent(self, parent: PointCloudData) -> None:
        """Set the parent point cloud object with a weakref.

        Parameters
        ----------
        parent: PointCloudData

        Returns
        -------

        """
        if self._parent is not None and self._parent() is not parent:
            logger.warning(
                f"Parent already set as {self._parent()}. Will be overwritten by {parent}!",
                stack_info=True,
                stacklevel=1,
            )
        self._parent = weakref.ref(parent)
        self.validate_lengths()

    @property
    def shape(self) -> tuple[int, int]:
        """Return the shape of the scalar field manager.

        The shape will be in the form (num_points, num_fields).

        Returns
        -------
        tuple[int, int]
        """
        return self.num_points, len(self)

    @property
    def num_points(self) -> int:
        """Return the number of points in the parent point cloud. If np parent is set, return -1.

        Returns
        -------
        int
        """
        if self._parent is None:
            return -1

        return len(cast(Sized, self.parent))

    @property
    def rgb(self) -> RGBFields | None:
        """Get the RGB fields.

        Returns
        -------
        RGBFields | None
        """
        return self.fields.get(RGB_NAMES.base, None)

    @rgb.setter
    def rgb(self, value: Array_Nx3_Uint8_T | Array_Nx3_Float32_T | RGBFields | None) -> None:
        """Set the RGB fields.

        If set to None, the field will be deleted.

        Parameters
        ----------
        value: Array_Nx3_Uint8_T | Array_Nx3_Float32_T | RGBFields | None

        Returns
        -------

        """
        self[RGB_NAMES.base] = RGBFields(value) if value is not None else None

    @property
    def normals(self) -> NormalFields | None:
        """Get the normals fields.

        Returns
        -------
        NormalFields | None
        """
        return cast(NormalFields | None, self.fields.get(NORMAL_NAMES.base, None))

    @normals.setter
    def normals(self, value: Optional[Array_Nx3_Float_T | NormalFields]):
        """Set the normals fields.

        If set to None, the field will be deleted.

        Parameters
        ----------
        value: Array_Nx3_Float_T | NormalFields | None

        Returns
        -------

        """
        self[NORMAL_NAMES.base] = NormalFields(value) if value is not None else None

    @property
    def intensity(self) -> ScalarField | None:
        """Get the intensity field

        Returns
        -------
        ScalarField | None
        """
        return cast(ScalarField | None, self.fields.get(INTENSITY_NAMES.base, None))

    @intensity.setter
    def intensity(self, value: VectorT | ScalarField | None):
        """Set the intensity field.

        If set to None, the field will be deleted.

        Parameters
        ----------
        value: VectorT | ScalarField | None

        Returns
        -------

        """
        self[INTENSITY_NAMES.base] = ScalarField(value, name=INTENSITY_NAMES.base) if value is not None else None

    @property
    def reflectance(self) -> ScalarField | None:
        """Get the reflectance field.

        Returns
        -------
        ScalarField | None
        """
        return cast(ScalarField | None, self.fields.get(REFLECTANCE_NAMES.base, None))

    @reflectance.setter
    def reflectance(self, value: VectorT | ScalarField | None):
        """Set the reflectance field.

        If set to None, the field will be deleted.

        Parameters
        ----------
        value: VectorT | ScalarField | None

        Returns
        -------

        """
        self[REFLECTANCE_NAMES.base] = ScalarField(value, name=REFLECTANCE_NAMES.base) if value is not None else None

    def sample(self, mask: IndexLike) -> Self:
        """Sample the scalar fields based on a mask.

        Parameters
        ----------
        mask: IndexLike
            The mask corresponds to indexing the rows or points in the parent point cloud.

        Returns
        -------
        ScalarFieldManager
        """
        sampled = type(self)(fields={})

        for name, value in self.items():
            sampled[name] = value.sample(mask)

        return sampled

    def reduce(self, mask: IndexLike) -> None:
        """
        Reduces the entries in the fields based on the provided mask.

        Parameters
        ----------
        mask : IndexLike
            The mask corresponds to indexing the rows or points in the parent point cloud.

        Returns
        -------
        """
        for name, value in self.items():
            self.fields[name] = value[mask]

    def extract(self, mask: IndexLike) -> Self:
        """
        Extracts a subset of the current object based on the provided mask and returns a new instance.

        The current object is also reduced by removing the extracted points.

        Parameters
        ----------
        mask : IndexLike
            The mask corresponds to indexing the rows or points in the parent point cloud.

        Returns
        -------
        ScalarFieldManager
        """
        if len(self) == 0:
            return type(self)(fields={})

        parent: PointCloudData | None = self._parent() if self._parent is not None else None

        if parent:
            bool_mask: Vector_Bool_T = parent.create_mask(mask)
        else:
            bool_mask = list(self.values())[0].create_mask(mask)

        sample = self.sample(bool_mask)
        self.reduce(~bool_mask)
        return sample

    def add_field(self, sf_field: SF_T) -> None:
        """Adds a new scalar field to the manager.

        Parameters
        ----------
        sf_field : SF_T

        Returns
        -------
        """
        self[sf_field.name] = sf_field

    def remove_field(self, field_name: LowerStr) -> None:
        """Removes a scalar field from the manager.

        Parameters
        ----------
        field_name: LowerStr

        Returns
        -------

        """
        del self.fields[field_name.lower()]

    def create_field(self, name: str, data: VectorT | Array_Nx3_T) -> None:
        """Create a scalar field from name and array

        Supports Array_Nx3_T for the creation of RGB or Normals fields.

        Parameters
        ----------
        name: str
        data: VectorT | Array_Nx3_T

        Returns
        -------

        """
        sf = ScalarField(data, name=name)
        self.add_field(sf)

    def _get_rgb(self, name: LowerStr) -> SF_T | None:
        """Handler for getting the RGB values based on the different RGB name conmventions.

        Parameters
        ----------
        name: LowerStr

        Returns
        -------

        """
        if self.rgb is None:
            return None

        elif name in RGB_NAMES.names:
            return self.rgb

        elif name in RGB_NAMES.scalars:
            index = RGB_NAMES.get_position(name)
            value = ScalarField(self.rgb.arr[:, index], name=name, origin_dtype=self.rgb.origin_dtype)
            return value / value.max() if name in RGB_NAMES.float else value

        elif name is RGB_NAMES.reverse:
            return self.rgb[:, [2, 1, 0]]

        else:
            raise KeyError(f"Unknown key made it into _handle_rgb : {name}")

    def _get_normals(self, name: LowerStr) -> SF_T | None:
        """Handler for getting the normal values based on the different normal name conmventions

        Parameters
        ----------
        name: LowerStr

        Returns
        -------

        """
        if self.normals is None:
            return None

        elif name in NORMAL_NAMES.names:
            return self.normals

        elif name in NORMAL_NAMES.scalars:
            index = NORMAL_NAMES.get_position(name)
            return ScalarField(self.normals.arr[:, index], name=name, origin_dtype=self.normals.origin_dtype)

        elif name is NORMAL_NAMES.reverse:
            return self.normals[:, [2, 1, 0]]

        else:
            raise KeyError(f"Unknown key made it into normals : {name}")

    def _set_rgb(self, name: LowerStr, value: RGBLikeT, origin_dtype: DtypeState | None = None) -> None:
        """Handler for setting the RGB values based on the different RGB name conmventions.

        Parameters
        ----------
        name: LowerStr
        value: RGBLikeT
        origin_dtype: DtypeState | None

        Returns
        -------

        """
        if name in RGB_NAMES.names:
            self.fields[RGB_NAMES.base] = RGBFields(value, origin_dtype=cast(DtypeState, origin_dtype))
            return

        elif name is RGB_NAMES.reverse:
            value = value[:, [2, 1, 0]]
            self.fields[RGB_NAMES.base] = RGBFields(value, origin_dtype=cast(DtypeState, origin_dtype))
            return

        if name in RGB_NAMES.float:
            value = cast(Array_Uint8_T, normalize_uint8(value))

        # TODO update initialize to receive and origin_dtype in case it's being defined by a sequence of vectors
        if name in RGB_NAMES.scalars:
            if self.rgb is None:
                self.rgb = RGBFields.initialize(self.num_points)

            index = RGB_NAMES.get_position(name)
            self.rgb.arr[:, index] = Vector_Uint8_T(value)  # Perform validation as it's being assigned directly

        else:
            raise KeyError(f"Unknown key made it into _handle_rgb : {name}")

    def _set_normals(self, name: LowerStr, value: NormalLikeT, origin_dtype: DtypeState | None = None) -> None:
        """Handler for setting the normal values based on the different normal name conmventions.

        Parameters
        ----------
        name: LowerStr
        value: NormalLikeT
        origin_dtype: DtypeState | None

        Returns
        -------

        """
        if name in NORMAL_NAMES.names:
            self.fields[NORMAL_NAMES.base] = NormalFields(value, origin_dtype=cast(DtypeState, origin_dtype))

        elif name in NORMAL_NAMES.scalars:
            if self.normals is None:
                self.normals = NormalFields.initialize(self.num_points)

            index = NORMAL_NAMES.get_position(name)
            self.normals.arr[:, index] = value

        elif name is NORMAL_NAMES.reverse:
            value = value[:, [2, 1, 0]]
            self.fields[NORMAL_NAMES.base] = NormalFields(value, origin_dtype=cast(DtypeState, origin_dtype))

        else:
            raise KeyError(f"Unknown key made it into normals : {name}")

    @classmethod
    def merge(cls, scalar_field_managers: Iterable[Self]) -> Self:
        """Merges a list of scalar field managers into one.

        Parameters
        ----------
        scalar_field_managers: Iterable[ScalarFieldManager]]

        Returns
        -------

        """

        sfm_key_sets = (set(sfm) for sfm in scalar_field_managers)

        if len(list(scalar_field_managers)) == 0:
            raise ValueError("Cannot merge empty list of scalar field managers.")

        keys_in_common = set.intersection(*sfm_key_sets)

        new_sfm = cls(parent=None)

        if len(keys_in_common) == 0:
            return new_sfm

        # Get the scalar field managers that have the key in common
        for common_key in keys_in_common:
            sfs: list[SF_T] = cast(list[SF_T], [sfm[common_key] for sfm in scalar_field_managers])

            # Check the names are the same. If not, take the most occurring name.
            sf_names: list[str] = [sf.name for sf in sfs]
            if len(set(sf_names)) != 1:
                logger.warning(f"While merging scalar field {common_key} different names were encountered.")
                name = max((counted_names := Counter(sf_names)), key=lambda x: counted_names.get(x, 0))
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

    def as_struct_array(self) -> np.ndarray:
        dtype = []
        data = []

        for name, value in self.fields.items():
            data.append(value)

            dtype.append((name, value.dtype, value.shape))

        array = np.empty(1, dtype=dtype)
        for name, value in self.fields.items():
            array[value.name] = value.arr
        return array
