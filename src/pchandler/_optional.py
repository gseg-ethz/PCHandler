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

"""Optional-dependency capability probes for pchandler.

Private module — do not import from outside pchandler. External callers should
use the public re-exports: ``pchandler.filters.gpu.is_available()``,
``pchandler.filters.gpu.ensure_available()``.
"""

import logging
from typing import Optional

_HAS_OPEN3D: bool = False
_OPEN3D_ERROR: Optional[Exception] = None

try:
    import open3d  # noqa: F401
except (ImportError, RuntimeError) as e:
    _OPEN3D_ERROR = e
else:
    _HAS_OPEN3D = True

_HAS_PY4DGEO: bool = False
_PY4DGEO_ERROR: Optional[Exception] = None

try:
    import py4dgeo  # noqa: F401
except (ImportError, RuntimeError) as e:
    _PY4DGEO_ERROR = e
else:
    _HAS_PY4DGEO = True

_HAS_GPU: bool = False
_GPU_ERROR: Optional[Exception] = None

try:
    import cudf  # noqa: F401
    import cuspatial  # noqa: F401
    import geopandas  # noqa: F401
except (ImportError, RuntimeError) as e:
    _GPU_ERROR = e
else:
    _HAS_GPU = True

logger = logging.getLogger(__name__.split(".")[0])


def is_open3d_available() -> bool:
    """Check whether ``open3d`` is importable.

    Returns
    -------
    bool
        ``True`` if ``open3d`` imported cleanly, ``False`` otherwise.
    """
    return _HAS_OPEN3D


def ensure_open3d_available() -> None:
    """Raise ``ModuleNotFoundError`` if ``open3d`` is not available.

    Raises
    ------
    ModuleNotFoundError
        If ``open3d`` is not installed; chained from the original exception
        when one was captured.
    """
    if not _HAS_OPEN3D:
        msg = "Open3d is not installed. Install with `pip install pchandler[extra]`."
        if _OPEN3D_ERROR is not None:
            raise ModuleNotFoundError(msg) from _OPEN3D_ERROR
        raise ModuleNotFoundError(msg)


def is_py4dgeo_available() -> bool:
    """Check whether ``py4dgeo`` is importable.

    Returns
    -------
    bool
        ``True`` if ``py4dgeo`` imported cleanly, ``False`` otherwise.
    """
    return _HAS_PY4DGEO


def ensure_py4dgeo_available() -> None:
    """Raise ``ModuleNotFoundError`` if ``py4dgeo`` is not available.

    Raises
    ------
    ModuleNotFoundError
        If ``py4dgeo`` is not installed; chained from the original exception
        when one was captured.
    """
    if not _HAS_PY4DGEO:
        msg = "py4dgeo is not installed. Install with `pip install pchandler[extra]`."
        if _PY4DGEO_ERROR is not None:
            raise ModuleNotFoundError(msg) from _PY4DGEO_ERROR
        raise ModuleNotFoundError(msg)


def is_gpu_available() -> bool:
    """Check whether GPU support (cudf + cuspatial + geopandas) is importable.

    Returns
    -------
    bool
        ``True`` if the GPU dependencies imported cleanly, ``False`` otherwise.
    """
    return _HAS_GPU


def ensure_gpu_available() -> None:
    """Raise ``ImportError`` if GPU support is not available.

    Raises
    ------
    ImportError
        If GPU support is unavailable; chained from the original
        :class:`ImportError` or :class:`RuntimeError` when one was captured.
    """
    if not _HAS_GPU:
        msg = (
            "GPU support is not available. "
            "Install with `pip install pchandler[cuda11]` or `pip install pchandler[cuda12]`."
        )
        if _GPU_ERROR is not None:
            raise ImportError(msg) from _GPU_ERROR
        raise ImportError(msg)
