from __future__ import annotations

from typing import Annotated, Union, Sequence

import numpy as np
import numpy.typing as npt
from numpydantic import NDArray, Shape  # type: ignore
from numpydantic.dtype import (Bool, Float, Float32, Int8, Int16, Int32, Integer, UInt8, UInt16, Float64) # type: ignore
from pydantic import BeforeValidator
from shapely import Polygon

IndexLike = Union[
    int, slice, npt.NDArray[np.bool_], npt.NDArray[np.integer], list[int], tuple[int, ...], tuple[slice, ...]
]

ValidatedPolygonT = Annotated[
    Sequence | npt.NDArray[np.floating | np.integer] | Polygon, BeforeValidator(lambda x: Polygon(x))
]


ArrayDtypes = (Integer, Float, Bool)
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
