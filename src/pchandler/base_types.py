from __future__ import annotations

from typing import Annotated, Optional, TypedDict

import numpy.typing as npt
from pydantic import StringConstraints

LowerStr = Annotated[str, StringConstraints(strip_whitespace=True, to_lower=True)]
SfNameT = Optional[LowerStr]


class DtypeDict(TypedDict):
    names: list[LowerStr]
    formats: list[npt.DTypeLike]
