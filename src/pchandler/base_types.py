from __future__ import annotations

from typing import Annotated, Union, Sequence

import numpy as np
import numpy.typing as npt
from numpydantic import NDArray, Shape  # type: ignore
from numpydantic.dtype import (
    Bool,
    Float,
    Float32,
    Int8,
    Int16,
    Int32,
    Integer,
    UInt8,
    UInt16,
    Float64,
) # type: ignore
from pydantic import BeforeValidator

from shapely import Polygon

from pchandler.validators import (
    extract_array,
    validate_n_by_2_transposed,
    validate_n_by_3_transposed,
    validate_transposed_vector,
)

IndexLike = Union[
    int, slice, npt.NDArray[np.bool_], npt.NDArray[np.integer], list[int], tuple[int, ...], tuple[slice, ...]
]

ArrayValidator = BeforeValidator(extract_array)
TransposedVector = BeforeValidator(validate_transposed_vector)
TransposedNx2 = BeforeValidator(validate_n_by_2_transposed)
TransposedNx3 = BeforeValidator(validate_n_by_3_transposed)

ArrayDtypes = (Integer, Float, Bool)
ArrayT = NDArray[Shape["*, ..."], ArrayDtypes]
Array_NxM_T = NDArray[Shape["*, *"], ArrayDtypes]  # Intensity/depth image
Array_NxM_3_T = NDArray[Shape["*, *, 3"], ArrayDtypes]  # RGB image
Array_Nx2_T = Annotated[NDArray[Shape["*, 2"], ArrayDtypes], TransposedNx2]  # Image coordinates
Array_Nx3_T = Annotated[NDArray[Shape["*, 3"], ArrayDtypes], TransposedNx3]  # 3D Coordinates / normals
Array_Nx3_Float32_T = Annotated[NDArray[Shape["*, 3"], Float32], TransposedNx3]  # Optimised coordinates
Array_Nx3_Uint8_T = Annotated[NDArray[Shape["*, 3"], UInt8], TransposedNx3]  # RGB
Array_3x3_T = NDArray[Shape["4, 4"], ArrayDtypes]  # Rotation Matrix
Array_4x4_T = NDArray[Shape["4, 4"], ArrayDtypes]  # Affine Transformation
VectorT = Annotated[NDArray[Shape["*"], ArrayDtypes], TransposedVector]
Vector_Int32_T = Annotated[NDArray[Shape["*"], Int32], TransposedVector]
Vector_Int16_T = Annotated[NDArray[Shape["*"], Int16], TransposedVector]
Vector_Int8_T = Annotated[NDArray[Shape["*"], Int8], TransposedVector]
Vector_Uint16_T = Annotated[NDArray[Shape["*"], UInt16], TransposedVector]  # Intensity Values
Vector_Uint8_T = Annotated[NDArray[Shape["*"], UInt8], TransposedVector]  # Single RGB field
Vector_Float32_T = Annotated[NDArray[Shape["*"], Float32], TransposedVector]  # Normal vector field
Vector_Bool_T = Annotated[NDArray[Shape["*"], Bool], TransposedVector]  # Mask or boolean vector
Vector_2_T = Annotated[NDArray[Shape["2"], ArrayDtypes], TransposedVector]  # Image coordinate
Vector_3_T = Annotated[NDArray[Shape["3"], ArrayDtypes], TransposedVector]  # 3D coordinate
Vector_Float64_T = Annotated[NDArray[Shape["3"], Float64], TransposedVector]

ValidatedPolygonT = Annotated[Sequence | npt.NDArray[np.floating | np.integer] | Polygon, BeforeValidator(lambda x: Polygon(x))]
