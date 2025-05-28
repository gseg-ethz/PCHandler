from typing import Literal, Union

import numpy as np
from numpy.typing import DTypeLike, NDArray

ScalarFieldNormalizeOp = tuple[Literal["normalize"], tuple[float, float]]
ScalarFieldDtypeConversion = tuple[Literal["dtype_conversion"], tuple[DTypeLike, DTypeLike]]

ScalarFieldOperations = ScalarFieldNormalizeOp | ScalarFieldDtypeConversion


IndexLike = Union[int, slice, NDArray[np.bool_], NDArray[np.int_], list[int], tuple[int, ...], tuple[slice, ...]]