"""
Public API for pchandler.geometry.
Re-exports key classes and functions from the submodules.
"""

from . import scalar_fields, splitter, util, scalar_field_manager
from .core import PointCloudData
# TODO update transforms when finished
# from .transforms import scale, translate
from .util import get_outline_polygon


__all__ = [
    "PointCloudData",
    "scalar_fields",
    "scalar_field_manager",
    "splitter",
    "transforms",
    "util",
    "get_outline_polygon",
]
