from __future__ import annotations

from typing import Annotated, Union, Sequence, Any, Optional

import numpy as np
import numpy.typing as npt
from numpydantic import NDArray, Shape  # type: ignore
from numpydantic.dtype import (Bool, Float, Float32, Int8, Int16, Int32, Integer, UInt8, UInt16, Float64) # type: ignore
from pydantic import BeforeValidator
from shapely import Polygon


ValidatedPolygonT = Annotated[
    Sequence | npt.NDArray[np.floating | np.integer] | Polygon, BeforeValidator(lambda x: Polygon(x))
]

ArrayDtypes = (Integer, Float, Bool)
IndexDtypes = (Integer, Bool)

ArrayT = NDArray[Shape["*, ..."], ArrayDtypes]          # Arrays of any shape but support integers, floats and booleans
Array_NxM_T = NDArray[Shape["*, *"], ArrayDtypes]       # Intensity/depth image
Array_NxM_3_T = NDArray[Shape["*, *, 3"], ArrayDtypes]  # RGB image
Array_Nx2_T = NDArray[Shape["*, 2"], ArrayDtypes]       # Image coordinates
Array_Nx3_T = NDArray[Shape["*, 3"], ArrayDtypes]       # 3D Coordinates / normals
Array_Nx3_Float32_T = NDArray[Shape["*, 3"], Float32]   # Optimised coordinates
Array_Nx3_Uint8_T = NDArray[Shape["*, 3"], UInt8]       # RGB
Array_3x3_T = NDArray[Shape["4, 4"], ArrayDtypes]       # Rotation Matrix
Array_4x4_T = NDArray[Shape["4, 4"], ArrayDtypes]       # Affine Transformation
VectorT = NDArray[Shape["*"], ArrayDtypes]
Vector_Int32_T = NDArray[Shape["*"], Int32]
Vector_Int16_T = NDArray[Shape["*"], Int16]
Vector_Int8_T = NDArray[Shape["*"], Int8]
Vector_Uint16_T = NDArray[Shape["*"], UInt16]
Vector_Uint8_T = NDArray[Shape["*"], UInt8]             # Single RGB field
Vector_Float32_T = NDArray[Shape["*"], Float32]         # Normal vector field
Vector_Bool_T = NDArray[Shape["*"], Bool]               # Mask or boolean vector
Vector_2_T = NDArray[Shape["2"], ArrayDtypes]           # Image coordinate
Vector_3_T = NDArray[Shape["3"], ArrayDtypes]           # 3D coordinate
Vector_Float64_T = NDArray[Shape["3"], Float64]

# TODO find good naming for the index type and numpydantic types - Particularly this VectorIndex objects
Vector_IndexT = NDArray[Shape["*"], IndexDtypes]

IndexLike = Union[int, slice, npt.NDArray[np.bool_], npt.NDArray[np.integer], Sequence]
VectorIndexLike = Union[int, slice, Sequence, Vector_IndexT]

def make_ndarray_type(
        *dimensions: Optional[int | str],
        dtype: Optional[npt.DTypeLike] = None
) -> type[NDArray[Any, Any]]:
    """
    Helper function to generate the numpydantic type for a ndarray.

    Calling 'make_ndarray_type(None, 3, dtype=np.float32)' would return a numpydantic dtype corresponding to an array
    of shape (N, 3) with dtype = np.float32 and would provide pydantic validation on this
    """
    if len(dimensions) == 0:
        shape_list = ["*", "..."]
    else:
        shape_list = [str(x) if x is not None else "*" for x in dimensions]

    result : type[NDArray[Any, Any]] = NDArray[Shape[", ".join(shape_list)], dtype if dtype is not None else Any]

    return result