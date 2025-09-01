# pchandler - Toolbox for point-cloud handling, processing and analysis
#
# Copyright (c) 2025, Nicholas Meyer, Geosensors and Engineering Geodesy,
# Institute of Geodesy and Photogrammetry, ETH Zurich, Switzerland
# SPDX-License-Identifier: BSD-3-Clause
#
# Author: Nicholas Meyer (meyernic@ethz.ch)

# Auto-generated stub for lazy exports
from typing import Any, Final, NoReturn

from .scalar_field_manager import SF_T, ScalarFieldManager
from .scalar_fields import NormalFields, RGBFields, ScalarField, ScalarFieldTriplet

__all__: Final[list[str]] = [
    "ScalarField",
    "ScalarFieldTriplet",
    "RGBFields",
    "NormalFields",
    "ScalarFieldManager",
    "SF_T",
]

def __getattr__(name: str) -> NoReturn: ...
