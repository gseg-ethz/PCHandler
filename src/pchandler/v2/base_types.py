from __future__ import annotations

from typing import Union, Annotated

import numpy as np

from numpydantic import NDArray, Shape
from numpydantic.dtype import Integer, Float, Bool, UInt8, Float32, UInt16, Int8, Int16, Int32
from pydantic import BeforeValidator
from validators import extract_array, validate_transposed_vector, validate_n_by_2_transposed, validate_n_by_3_transposed

IndexLike = Union[int, slice, np.ndarray[np.bool_], np.ndarray[np.integer], list[int], tuple[int, ...], tuple[slice, ...]]

ArrayValidator = BeforeValidator(extract_array)
TransposedVector = BeforeValidator(validate_transposed_vector)
TransposedNx2 = BeforeValidator(validate_n_by_2_transposed)
TransposedNx3 = BeforeValidator(validate_n_by_3_transposed)

ArrayDtypes = (Integer, Float, Bool)
ArrayT = Annotated[NDArray[Shape['*, ...'], ArrayDtypes], ArrayValidator]
Array_NxM_T = Annotated[NDArray[Shape['*, *'], ArrayDtypes], ArrayValidator]                # Intensity/depth image
Array_NxM_3_T = Annotated[NDArray[Shape['*, *, 3'], ArrayDtypes], ArrayValidator]           # RGB image
Array_Nx2_T = Annotated[NDArray[Shape['*, 2'], ArrayDtypes], TransposedNx2, ArrayValidator] # Image coordinates
Array_Nx3_T = Annotated[NDArray[Shape['*, 3'], ArrayDtypes], TransposedNx3, ArrayValidator] # 3D Coordinates / normals
Array_Nx3_float32_T = Annotated[NDArray[Shape['*, 3'], Float32],TransposedNx3,  ArrayValidator] # Optimised coordinates
Array_Nx3_uint8_T = Annotated[NDArray[Shape['*, 3'], UInt8], TransposedNx3, ArrayValidator] # RGB
Array_3x3_T = Annotated[NDArray[Shape['4, 4'], ArrayDtypes], ArrayValidator]                # Rotation Matrix
Array_4x4_T = Annotated[NDArray[Shape['4, 4'], ArrayDtypes], ArrayValidator]                # Affine Transformation
VectorT = Annotated[NDArray[Shape['*'], ArrayDtypes], TransposedVector, ArrayValidator]
VectorT_Int32 = Annotated[NDArray[Shape['*'], Int32], TransposedVector, ArrayValidator]
VectorT_Int16 = Annotated[NDArray[Shape['*'], Int16], TransposedVector, ArrayValidator]
VectorT_Int8 = Annotated[NDArray[Shape['*'], Int8], TransposedVector, ArrayValidator]
VectorT_Uint16 = Annotated[NDArray[Shape['*'], UInt16], TransposedVector, ArrayValidator]   # Intensity Values
VectorT_Uint8 = Annotated[NDArray[Shape['*'], UInt8], TransposedVector, ArrayValidator]     # Single RGB field
VectorT_Float32 = Annotated[NDArray[Shape['*'], Float32], TransposedVector, ArrayValidator] # Normal vector field
VectorT_Bool = Annotated[NDArray[Shape['*'], Bool], TransposedVector, ArrayValidator]       # Mask or boolean vector
Vector_2_T = Annotated[NDArray[Shape['2'], ArrayDtypes], ArrayValidator]                    # Image coordinate
Vector_3_T = Annotated[NDArray[Shape['3'], ArrayDtypes], ArrayValidator]                    # 3D coordinate

