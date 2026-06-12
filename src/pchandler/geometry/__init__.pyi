# pchandler – Toolbox for point-cloud handling, processing and analysis
#
# Copyright (c) 2022–2026 ETH Zurich
# Department of Civil, Environmental and Geomatic Engineering (D-BAUG)
# Institute of Geodesy and Photogrammetry
# Geosensors and Engineering Geodesy
#
# Authors:
#   Nicholas Meyer
#   Jon Allemand
#
# SPDX-License-Identifier: BSD-3-Clause

# pchandler - Toolbox for point-cloud handling, processing and analysis
#
# Copyright (c) 2025, Nicholas Meyer, Geosensors and Engineering Geodesy,
# Institute of Geodesy and Photogrammetry, ETH Zurich, Switzerland
# SPDX-License-Identifier: BSD-3-Clause
#
# Author: Nicholas Meyer (meyernic@ethz.ch)

# Auto-generated stub for lazy exports
from typing import Final, NoReturn

from . import coordinates as coordinates
from . import spherical as spherical
from . import splitter as splitter
from . import transforms as transforms
from . import util as util
from .optimal_shift import OptimizedShift, OptimizedShiftManager

__all__: Final[list[str]] = [
    "splitter",
    "util",
    "coordinates",
    "transforms",
    "spherical",
    "OptimizedShiftManager",
    "OptimizedShift",
]

def __getattr__(name: str) -> NoReturn: ...
