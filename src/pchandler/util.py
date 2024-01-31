from enum import Enum
from typing import Optional

import numpy as np


class AngleUnit(Enum):
    RAD = "rad"
    DEGREE = "deg"
    GON = "gon"


def convert_angles(values: np.ndarray, source_unit: AngleUnit, target_unit: AngleUnit, out: Optional[np.ndarray] = None) -> np.ndarray:
    if source_unit == target_unit:
        if out is None:
            return values.copy()
        else:
            out = values
            return out

    match source_unit:
        case AngleUnit.RAD:
            match target_unit:
                case AngleUnit.DEGREE: return np.rad2deg(values, out=out)
                case AngleUnit.GON: return np.multiply(values, 200/np.pi, out=out)
        case AngleUnit.DEGREE:
            match target_unit:
                case AngleUnit.RAD: return np.deg2rad(values)
                case AngleUnit.GON: return np.multiply(values, 200/180, out=out)
        case AngleUnit.GON:
            match target_unit:
                case AngleUnit.RAD: return np.multiply(values, np.pi/200, out=out)
                case AngleUnit.DEGREE: return np.multiply(values, 180/200, out=out)

