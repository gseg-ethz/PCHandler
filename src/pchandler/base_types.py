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

"""Project-local string and dtype type aliases shared across :mod:`pchandler`.

Provides :data:`LowerStr` (a lower-cased, stripped ``str`` for scalar-field
names), :data:`SfNameT` (the optional variant used as the canonical
``scalar_field`` name annotation), and :class:`DtypeDict` (the structured-array
schema accepted by NumPy ``dtype`` constructors).
"""

from __future__ import annotations

from typing import Annotated, Optional, TypedDict

import numpy.typing as npt
from pydantic import StringConstraints

LowerStr = Annotated[str, StringConstraints(strip_whitespace=True, to_lower=True)]
SfNameT = Optional[LowerStr]


class DtypeDict(TypedDict):
    """Structured-array dtype description compatible with :func:`numpy.dtype`.

    Attributes
    ----------
    names : list[LowerStr]
        Field names in declaration order.
    formats : list[npt.DTypeLike]
        Per-field dtype specifications, parallel to ``names``.
    """

    names: list[LowerStr]
    formats: list[npt.DTypeLike]
