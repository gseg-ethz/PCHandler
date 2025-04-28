from numpy import ndarray
from typing import TypeAlias


NumOrArray: TypeAlias  = ndarray|float|int|list|tuple

NdarrayIndex: TypeAlias = ndarray|float|int|bool