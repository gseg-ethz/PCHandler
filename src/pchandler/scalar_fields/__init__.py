"""
Public API for ScalarField and ScalarFieldManager classes.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__all__ = []

_lazy_map = {
    "ScalarField": "scalar_fields",
    "ScalarFieldTriplet": "scalar_fields",
    "RGBFields": "scalar_fields",
    "NormalFields": "scalar_fields",
    "ScalarFieldManager": "scalar_field_manager",
    "SF_T": "scalar_field_manager",
}

__all__ = __all__ + list(_lazy_map)

if TYPE_CHECKING:
    from .scalar_field_manager import SF_T, ScalarFieldManager
    from .scalar_fields import NormalFields, RGBFields, ScalarField, ScalarFieldTriplet


def __getattr__(name: str):
    if name in _lazy_map:
        module = importlib.import_module(f"{__name__}.{_lazy_map[name]}")
        val = getattr(module, name)
    else:
        try:
            val = importlib.import_module(f"{__name__}.{name}")
        except ModuleNotFoundError:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = val
    return val


def __dir__():
    # so tab-completion / introspection shows the lazy names
    return sorted(set(__all__) | set(globals().keys()))
