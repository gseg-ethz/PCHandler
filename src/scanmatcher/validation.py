from __future__ import annotations

import numpy as np

from src.scanmatcher.types import NumOrArray



def check_in_range(value: NumOrArray, target_min: float, target_max: float):
    value: np.ndarray = np.asarray(value)
    val_min: float|int = value.min()
    val_max: float|int = value.max()

    if (val_min < target_min) and (val_max > target_max):
        raise ValueError(f'Min and max values [{val_min},{val_max}] exceeds bounds [{target_min},{target_max}].')

    elif val_min < target_min:
        raise ValueError(f'Min value {val_min} exceeds lower limit {target_min}.')

    elif val_max > target_max:
        raise ValueError(f'Max value {val_max} exceeds upper limit {target_max}.')