from typing import Literal, Union, TYPE_CHECKING, NamedTuple

import numpy as np
from numpy.typing import DTypeLike, NDArray


ScalarFieldNormalizeOp = tuple[Literal["normalize"], tuple[float, float]]
ScalarFieldDtypeConversion = tuple[Literal["dtype_conversion"], tuple[DTypeLike, DTypeLike]]
ScalarFieldOperations = ScalarFieldNormalizeOp | ScalarFieldDtypeConversion

IndexLike = Union[int, slice, NDArray[np.bool_], NDArray[np.int_], list[int], tuple[int, ...], tuple[slice, ...]]


class OriginalFieldState(NamedTuple):
    dtype: DTypeLike
    upper: np.ndarray|float|int
    lower: np.ndarray|float|int