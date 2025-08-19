# Auto-generated stub for lazy exports
from typing import Any, Final, NoReturn

from .optimal_shift import OptimizedShift, OptimizedShiftManager

from . import coordinates as coordinates
from . import spherical as spherical
from . import splitter as splitter
from . import transforms as transforms
from . import util as util

__all__: Final[list[str]] = ['splitter', 'util', 'coordinates', 'transforms', 'spherical', 'OptimizedShiftManager', 'OptimizedShift']

def __getattr__(name: str) -> NoReturn: ...
