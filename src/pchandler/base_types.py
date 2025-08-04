from __future__ import annotations

from typing import Annotated, Sequence

from numpy import floating, integer
from numpy.typing import NDArray
from pydantic import BeforeValidator
from shapely import Polygon

ValidatedPolygonT = Annotated[
    Sequence | NDArray[floating | integer] | Polygon, BeforeValidator(lambda x: Polygon(x))
]
