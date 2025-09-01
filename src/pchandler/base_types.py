# pchandler - Toolbox for point-cloud handling, processing and analysis
#
# Copyright (c) 2025, Nicholas Meyer, Geosensors and Engineering Geodesy,
# Institute of Geodesy and Photogrammetry, ETH Zurich, Switzerland
# SPDX-License-Identifier: BSD-3-Clause
#
# Author: Nicholas Meyer (meyernic@ethz.ch)

from __future__ import annotations

from typing import Annotated, Optional, TypedDict

import numpy.typing as npt
from pydantic import StringConstraints

LowerStr = Annotated[str, StringConstraints(strip_whitespace=True, to_lower=True)]
SfNameT = Optional[LowerStr]


class DtypeDict(TypedDict):
    names: list[LowerStr]
    formats: list[npt.DTypeLike]
