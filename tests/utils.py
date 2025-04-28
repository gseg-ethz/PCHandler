from typing import Callable
from enum import Enum
import numpy as np


def numpy_in_dict_equality_check(a: dict, b: dict, np_check_func: Callable = np.all) -> bool:
    if a.keys() != b.keys():
        return False

    for key, value in a.items():
        if isinstance(value, np.ndarray):
            if not np_check_func(value == b[key]):
                return False
        else:
            if value != b[key]:
                return False

    return True