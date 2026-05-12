# pchandler - Toolbox for point-cloud handling, processing and analysis
#
# Copyright (c) 2025, Nicholas Meyer, Geosensors and Engineering Geodesy,
# Institute of Geodesy and Photogrammetry, ETH Zurich, Switzerland
# SPDX-License-Identifier: BSD-3-Clause
#
# Author: Nicholas Meyer (meyernic@ethz.ch)

# Auto-generated stub for lazy exports
from typing import Final, NoReturn

from .angle import Angle, AngleArray
from .fov import FoV, FoVTree

__all__: Final[list[str]] = ["Angle", "AngleArray", "FoV", "FoVTree"]

def __getattr__(name: str) -> NoReturn: ...
