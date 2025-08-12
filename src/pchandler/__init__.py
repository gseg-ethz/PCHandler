
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__author__ = "Nicholas Meyer"
__email__ = "meyernic@ethz.ch"

__all__ = [
    "data_io",
    "geometry",
    "filters",
    "scalar_fields",
    "base_types",
    "constants",
    "__author__",
    "__email__",
]

_lazy_map = {
    "PointCloudData": "core",
    "__version__": "_version",
    "version": "_version",
    "__version_tuple__": "_version",
    "version_tuple": "_version",
}

__all__ = __all__ + list(_lazy_map)

if TYPE_CHECKING:
    from . import base_types, constants, data_io, filters, geometry, scalar_fields
    from ._version import __version__, __version_tuple__, version, version_tuple
    from .core import PointCloudData


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
