from typing import TypeAlias, Optional, Sequence

from GSEGUtils.base_types import VectorT, Array_Nx3_T, Array_Nx3_Uint8_T, Array_Nx3_Float32_T, Array_Nx3_Float_T, Vector_Uint8_T, Vector_Float32_T, ArrayT

from ..scalar_fields import ScalarFieldManager
from .core import RGBFields, NormalFields, ScalarField, ScalarFieldTriplet

SF_T: TypeAlias = RGBFields | NormalFields | ScalarField
SFLikeT: TypeAlias = SF_T | VectorT | Array_Nx3_T
RGBLikeT: TypeAlias = Array_Nx3_Uint8_T | Vector_Uint8_T | RGBFields
NormalLikeT: TypeAlias = Array_Nx3_Float32_T | Vector_Float32_T | NormalFields
SFMLikeT: TypeAlias = dict[str, SFLikeT]


RgbInputT = Optional[RGBFields | Array_Nx3_Float_T | Array_Nx3_Uint8_T]
NormalInputT = Optional[NormalFields | Array_Nx3_Float_T]
IntensityInputT = Optional[VectorT | ArrayT]
ReflectanceInputT = Optional[VectorT | ArrayT]
SFM_T = Optional[ScalarFieldManager | dict[str, ScalarField | ScalarFieldTriplet | Array_Nx3_T | VectorT | Sequence]]