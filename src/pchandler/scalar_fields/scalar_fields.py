# pchandler - Toolbox for point-cloud handling, processing and analysis
#
# Copyright (c) 2025, Nicholas Meyer, Geosensors and Engineering Geodesy,
# Institute of Geodesy and Photogrammetry, ETH Zurich, Switzerland
# SPDX-License-Identifier: BSD-3-Clause
#
# Author: Nicholas Meyer (meyernic@ethz.ch)

from __future__ import annotations

import logging
from typing import (
    Annotated,
    Any,
    NamedTuple,
    NotRequired,
    Optional,
    Self,
    TypeAlias,
    TypedDict,
    Unpack,
)

import numpy as np
import numpy.typing as npt
from GSEGUtils.base_arrays import ArrayNx3, BaseVector, FixedLengthArray
from GSEGUtils.base_types import (
    Array_Float32_T,
    Array_Nx3_Float32_T,
    Array_Nx3_Float_T,
    Array_Nx3_T,
    Array_Nx3_Uint8_T,
    Array_Uint8_T,
    ArrayT,
    LowerStr,
    SfNameT,
    Vector_Bool_T,
    Vector_Float_T,
    Vector_Float32_T,
    Vector_Int16_T,
    Vector_Uint8_T,
    Vector_Uint16_T,
    VectorT,
)
from GSEGUtils.validators import normalize_int16, normalize_min_max, normalize_uint8
from pydantic import BeforeValidator, field_validator, model_validator

from pchandler.constants import NORMAL_NAMES, RGB_NAMES

__all__ = ['ScalarField', 'RGBFields', 'NormalFields', 'SegmentationMap', 'ScalarFieldUint8', 'ScalarFieldBoolean',
           'ScalarFieldFloat32', 'NormalisedInt16ScalarField', 'AbstractScalarField', 'DtypeState']

logger = logging.getLogger(__name__.split(".")[0])


class DtypeState(NamedTuple):
    """Contains the original dtype and limits of an array used to create a ScalarField array.

    Parameters
    ----------
    dtype: np.dtype
    lower: npt.NDArray[np.number] | float | int
    upper: npt.NDArray[np.number] | float | int
    """
    dtype: npt.DTypeLike
    lower: npt.NDArray[np.number] | float | int
    upper: npt.NDArray[np.number] | float | int

    @classmethod
    def generate(cls, array: ArrayT | ArrayNx3 | BaseVector) -> DtypeState:
        """Generates a DtypeState object from an array.

        Parameters
        ----------
        array: ArrayT | ArrayNx3 | BaseVector

        Returns
        -------
        DtypeState
        """
        if not hasattr(array, "dtype"):
            raise TypeError(f"Array does not have dtype attribute: {array}")
        return DtypeState(dtype=array.dtype, lower=array.min(), upper=array.max())

    @staticmethod
    def validate(obj: DtypeState) -> None:
        """Ensure that the lower value is less than the upper value.

        Parameters
        ----------
        obj: DtypeState

        Returns
        -------

        """
        if (obj is not None) and (obj.lower >= obj.upper):
            raise ValueError(f"lower must be less than upper. {obj=}")


SfOrigDtT: TypeAlias = Optional[DtypeState]


class _ScalarKwargT(TypedDict, total=False):
    name: str
    origin_dtype: NotRequired[DtypeState]


class AbstractScalarField(FixedLengthArray):
    """Abstract scalar field class containing validation and conversion to origianl dtype methods

    Parameters
    ----------
    arr: VectorT | Array_Nx3_T
    name: SfNameT
    origin_dtype: SfOrigDtT

    """
    name: LowerStr
    origin_dtype: DtypeState

    def __init__(self, arr: VectorT | Array_Nx3_T | Self, name: SfNameT = None, origin_dtype: SfOrigDtT = None):
        """Initialise a scalar field.

        Parameters
        ----------
        arr: VectorT | Array_Nx3_T | Self
        name: SfNameT, optional
        origin_dtype: SfOrigDtT, optional
        """
        kwargs: dict[str, Any] = {"name": name, "origin_dtype": origin_dtype}
        super().__init__(arr, **kwargs)

    # noinspection PyNestedDecorators
    @model_validator(mode="before")
    @classmethod
    def _validate_model_before(cls, data: Any) -> Any:
        """Extract the array data and original dtype when initializing the scalar field.

        Parameters
        ----------
        data: Any

        Returns
        -------
        Any
        """
        if data["name"] is None:
            # Extract name from field if it exists, otherwise get default if available
            if hasattr(data["arr"], "name"):
                data["name"] = data["arr"].name
            else:
                data["name"] = cls.model_fields["name"].default

        # Get the origin_dtype if it exists
        if data["origin_dtype"] is None:
            if hasattr(data["arr"], "origin_dtype"):
                data["origin_dtype"] = data["arr"].origin_dtype
            else:
                data["origin_dtype"] = DtypeState.generate(data["arr"])

        return data

    def get_original_data(self) -> Array_Nx3_T | BaseVector:
        """Return the original data stored in the scalar field.

        Performs the inverse normalization from the DtypeState stored with the scalar field.

        Returns
        -------
        Array_Nx3_T | BaseVector
        """
        current_dtype_state = DtypeState.generate(self.arr)

        if current_dtype_state == self.origin_dtype:
            return self.arr.copy()

        return normalize_min_max(
            array=self.arr.copy(),
            lower=float(self.origin_dtype.lower),
            upper=float(self.origin_dtype.upper),
            target_dtype=self.origin_dtype.dtype,
        )


class ScalarField(BaseVector, AbstractScalarField):
    def __init__(self, arr: VectorT | Self, name: SfNameT = None, origin_dtype: SfOrigDtT = None):
        """Scalar Field class to support individual fields represented by a vector (1D array) shape

        Parameters
        ----------
        arr: VectorT | Array_Nx3_T | Self
        name: SfNameT, optional
        origin_dtype: SfOrigDtT, optional
        """
        kwargs: dict[str, Any] = {"name": name, "origin_dtype": origin_dtype}
        super().__init__(arr, **kwargs)


class ScalarFieldTriplet(ArrayNx3, AbstractScalarField):
    """Scalar Field Triplet class to support RGB and Normal fields.

    Parameters
    ----------
    arr: Array_Nx3_T
    """
    def __init__(self, arr: Array_Nx3_T | Self, name: SfNameT = None, origin_dtype: SfOrigDtT = None):
        kwargs: dict[str, Any] = {"name": name, "origin_dtype": origin_dtype}
        super().__init__(arr, **kwargs)

    @classmethod
    def initialize(cls, size: int, value: Array_Nx3_Uint8_T | None = None, name: str = "") -> Self:
        dtype = cls.model_fields["arr"].annotation.__dict__["__args__"][1]
        if value is None:
            value = np.zeros((size, 3), dtype=dtype)
        return cls(value, name=name)


class RGBFields(ScalarFieldTriplet):
    """RGB / Color Field

     Represented by a scalar field triplet of Uint8 values.

    Parameters
    ----------
    arr: Array_Nx3_Uint8_T
    """
    arr: Array_Nx3_Uint8_T
    name: str = RGB_NAMES.base

    def __init__(self, arr: Array_Nx3_Uint8_T | Array_Nx3_Float_T | Self, **kwargs: Unpack[_ScalarKwargT]):
        """Initialise an RGB field.

        Will convert any floating point values to uint8.

        Parameters
        ----------
        arr: Array_Nx3_Uint8_T | Array_Nx3_Float_T | Self
        kwargs: dict[str, Any]
        """
        super().__init__(arr, **kwargs)

    # TODO should we still do this for floating point values? Should there be a check if they're in the range of 0-255?
    # noinspection PyNestedDecorators
    @field_validator("arr", mode="before")
    @classmethod
    def _normalise_to_uint8(cls, data: npt.NDArray[Any]) -> Array_Uint8_T:
        """Automatically convert the input array to uint8

        Parameters
        ----------
        data: npt.NDArray[Any]

        Returns
        -------
        Array_Uint8_T
        """
        return normalize_uint8(data)

    @field_validator("name", mode="before")
    @classmethod
    def _override_name(cls, value: Any) -> str:
        """Set the scalar field name to the constant RGB_NAMES.base"""
        return RGB_NAMES.base

    @property
    def red(self) -> Vector_Uint8_T:
        """Return the red component of the RGB values.

        Returns
        -------
        Vector_Uint8_T
        """
        return self.arr[:, 0]

    @property
    def r(self) -> Vector_Uint8_T:
        """Return the red component of the RGB values.

        Returns
        -------
        Vector_Uint8_T
        """
        return self.red

    @property
    def green(self) -> Vector_Uint8_T:
        """Return the green component of the RGB values.

        Returns
        -------
        Vector_Uint8_T
        """
        return self.arr[:, 1]

    @property
    def g(self) -> Vector_Uint8_T:
        """Return the green component of the RGB values.

        Returns
        -------
        Vector_Uint8_T
        """
        return self.green

    @property
    def blue(self) -> Vector_Uint8_T:
        """Return the blue component of the RGB values.

        Returns
        -------
        Vector_Uint8_T
        """
        return self.arr[:, 2]

    @property
    def b(self) -> Vector_Uint8_T:
        """Return the blue component of the RGB values.

        Returns
        -------
        Vector_Uint8_T
        """
        return self.blue

    def as_normalised_float32(self) -> Array_Nx3_Float32_T:
        """Return the RGB values as a normalized float32 array in the range of [0,1].

        Returns
        -------
        Array_Nx3_Float32_T
        """
        return normalize_min_max(self.arr, 0, 1, np.float32)


class NormalFields(ScalarFieldTriplet):
    """
    Normal Vector Field represented by a scalar field triplet.

    Parameters
    ----------
    arr: Array_Nx3_Float32_T
    """
    arr: Array_Nx3_Float32_T
    name: str = NORMAL_NAMES.base

    def __init__(self, arr: Array_Nx3_Float_T | Self, **kwargs: Unpack[_ScalarKwargT]):
        """Initialize a normal vector field.

        Values will be normalized to unit vectors and converted to float32.

        Parameters
        ----------
        arr: Array_Nx3_Float_T | Self
        kwargs
        """
        super().__init__(arr, **kwargs)


    # noinspection PyNestedDecorators
    @field_validator("arr", mode="before")
    @classmethod
    def _ensure_unit_vector(cls, array: Array_Nx3_Float_T) -> Array_Nx3_Float32_T :
        """Converts the input array to a set of unit vectors.

        Parameters
        ----------
        array: Array_Nx3_Float_T

        Returns
        -------
        Array_Nx3_Float32_T
        """
        if not (np.issubdtype(array.dtype, np.floating) or np.issubdtype(array.dtype, np.signedinteger)):
            raise TypeError("Dtype of normals array must be of type floating or signed integer}")

        result: Array_Nx3_Float32_T = np.asarray(array, dtype=np.float32)
        base_vectors: Array_Nx3_Float32_T = np.linalg.norm(result, axis=1).reshape(-1, 1)
        if np.allclose(base_vectors, 1):
            return result

        return (result / base_vectors).astype(np.float32)

    @field_validator("name", mode="before")
    @classmethod
    def _override_name(cls, value: Any) -> str:
        """Force the scalar field name to the constant"""
        return NORMAL_NAMES.base

    @property
    def nx(self) -> Vector_Float32_T:
        """Return the X component of the normal vector."""
        return self.arr[:, 0]

    @property
    def ny(self) -> Vector_Float32_T:
        """Return the Y component of the normal vector."""
        return self.arr[:, 1]

    @property
    def nz(self) -> Vector_Float32_T:
        """Return the Z component of the normal vector."""
        return self.arr[:, 2]

    # TODO need to sort out the parameters in this method signature as name is not required
    @classmethod
    def initialize(cls, length: int, value: Array_Nx3_Float_T | Vector_Float_T | None = None, name: str = "") -> Self:
        """Initialize an array for the normals field

        When passing in a set of values, the

        Parameters
        ----------
        length: int
        value: Array_Nx3_Float32_T | None, optional
            Existing set of values to initialize the array with
        name: str, optional
            in case of initializing from an array, an additional field of

        Returns
        -------

        """
        dtype = cls.model_fields["arr"].annotation.__dict__["__args__"][1]
        if value is None:
            value = np.zeros((length, 3), dtype=dtype)
            value[:, 2] = 1
        return cls(value, name=name)


class SegmentationMap(ScalarField):
    """
    Segmentation map scalar field class. Supports uint8 and uint16 dtypes for indexing different point cloud segments.

    Parameters
    ----------
    arr: Vector_Uint8_T | Vector_Uint16_T
    """
    arr: Vector_Uint8_T | Vector_Uint16_T

    def __init__(self, arr: Vector_Uint8_T | Vector_Uint16_T | Self, **kwargs: Unpack[_ScalarKwargT]):
        super().__init__(arr, **kwargs)

    @classmethod
    def initialize(cls, name: LowerStr, pt_cloud_sizes: list[int]) -> Self:
        """Initialize a segmentation map scalar field based on the number of points in each point cloud.

        Parameters
        ----------
        name: LowerStr
        pt_cloud_sizes: list[int]

        Returns
        -------
        SegmentationMap
        """
        vector_length = sum(pt_cloud_sizes)

        if len(pt_cloud_sizes) <= 2**8 - 1:
            arr = np.zeros(vector_length, dtype=np.uint8)

        elif len(pt_cloud_sizes) <= 2**16 - 1:
            arr = np.zeros(vector_length, dtype=np.uint16)

        else:
            raise ValueError(f"Segmentation map for more than {2 ** 16} classes {len(pt_cloud_sizes)} not supported.")

        return cls(arr, name=name)


class ScalarFieldUint8(ScalarField):
    """
    Scalar field that only supports uint8 dtype.
    """
    arr: Vector_Uint8_T

    def __init__(self, arr: Vector_Uint8_T | Self, **kwargs: Unpack[_ScalarKwargT]):
        """

        Parameters
        ----------
        arr: Vector_Uint8_T | Self
        kwargs: Unpack[_ScalarKwargT]
        """
        super().__init__(arr, **kwargs)


class ScalarFieldBoolean(ScalarField):
    """
    Scalar field that only supports boolean arrays.
    """
    arr: Vector_Bool_T

    def __init__(self, arr: Vector_Bool_T | Self, **kwargs: Unpack[_ScalarKwargT]):
        """

        Parameters
        ----------
        arr: Vector_Bool_T | Self
        kwargs: Unpack[_ScalarKwargT]
        """
        super().__init__(arr, **kwargs)


class ScalarFieldFloat32(ScalarField):
    """
    Scalar field class which supports only Float32 dtypes
    """
    arr: Vector_Float32_T

    def __init__(self, arr: Vector_Float32_T | Self, **kwargs: Unpack[_ScalarKwargT]):
        """

        Parameters
        ----------
        arr: Vector_Float32_T | Self
        kwargs: Unpack[_ScalarKwargT]
        """
        super().__init__(arr, **kwargs)


class NormalisedInt16ScalarField(ScalarField):
    """
    Scalar field class that automatically normalizes the input to Int16 dtype
    """

    arr: Annotated[Vector_Int16_T, BeforeValidator(normalize_int16)]

    def __init__(self, arr: VectorT | Self, **kwargs: Unpack[_ScalarKwargT]):
        """

        Parameters
        ----------
        arr: VectorT | Self
        kwargs: Unpack[_ScalarKwargT]
        """
        super().__init__(arr, **kwargs)

    def to_uint8(self) -> ScalarFieldUint8:
        """Returns a converted copy of the scalar field with the dtype set to uint8.

        Returns
        -------
        ScalarFieldUint8
        """
        return ScalarFieldUint8(normalize_uint8(self.arr), name=self.name, origin_dtype=self.origin_dtype)
