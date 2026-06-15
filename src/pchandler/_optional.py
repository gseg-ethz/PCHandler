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

Each optional dependency is probed lazily: at module load only
``importlib.util.find_spec`` is called (no side effects, no native-extension
initialisation).  The actual ``import`` is deferred to the first call of the
corresponding ``ensure_*_available()`` or ``is_*_available()`` function and
the result is memoised so the import occurs at most once per process.
"""

import importlib.util
import logging
from typing import Optional

logger = logging.getLogger(__name__.split(".")[0])

# ---------------------------------------------------------------------------
# Module-level availability flags (cheap spec-only probe — no actual import)
# ---------------------------------------------------------------------------

# True iff the package can be found on sys.path; does *not* import it.
_OPEN3D_FOUND: bool = importlib.util.find_spec("open3d") is not None
_PY4DGEO_FOUND: bool = importlib.util.find_spec("py4dgeo") is not None
_GPU_FOUND: bool = (
    importlib.util.find_spec("cudf") is not None
    and importlib.util.find_spec("cuspatial") is not None
    and importlib.util.find_spec("geopandas") is not None
)

# ---------------------------------------------------------------------------
# Per-dependency lazy-import state (populated on first ensure_*/is_* call)
# ---------------------------------------------------------------------------

_HAS_OPEN3D: Optional[bool] = None  # None → not yet probed
_OPEN3D_ERROR: Optional[Exception] = None

_HAS_PY4DGEO: Optional[bool] = None
_PY4DGEO_ERROR: Optional[Exception] = None

_HAS_GPU: Optional[bool] = None
_GPU_ERROR: Optional[Exception] = None


# ---------------------------------------------------------------------------
# Lazy probers — each imports the dependency once and memoises the outcome
# ---------------------------------------------------------------------------


def _probe_open3d() -> bool:
    """Import ``open3d`` exactly once and memoise the result."""
    global _HAS_OPEN3D, _OPEN3D_ERROR
    if _HAS_OPEN3D is None:
        try:
            import open3d  # noqa: F401

            _HAS_OPEN3D = True
        except (ImportError, RuntimeError) as e:
            _OPEN3D_ERROR = e
            _HAS_OPEN3D = False
    return bool(_HAS_OPEN3D)


def _probe_py4dgeo() -> bool:
    """Import ``py4dgeo`` exactly once and memoise the result."""
    global _HAS_PY4DGEO, _PY4DGEO_ERROR
    if _HAS_PY4DGEO is None:
        try:
            import py4dgeo  # noqa: F401

            _HAS_PY4DGEO = True
        except (ImportError, RuntimeError) as e:
            _PY4DGEO_ERROR = e
            _HAS_PY4DGEO = False
    return bool(_HAS_PY4DGEO)


def _probe_gpu() -> bool:
    """Import cudf/cuspatial/geopandas AND execute a smoke kernel; memoise.

    cudf 25.4 imports cleanly on no-GPU hosts (emits ``UserWarning: No NVIDIA GPU detected``).
    The ``numba_cuda.cudadrv.devices._DeviceList.__getitem__`` bare-``IndexError`` fires
    only at first kernel call, so the smoke probe surfaces it before ``.mask()``.

    Implements D-06: widened except tuple catches every documented failure mode of
    ``cudf.DataFrame({"x": [1]})`` on no-GPU hosts.
    """
    global _HAS_GPU, _GPU_ERROR
    if _HAS_GPU is None:
        try:
            import cudf
            import cuspatial  # noqa: F401
            import geopandas  # noqa: F401

            cudf.DataFrame({"x": [1]})  # Smoke probe — D-06 (surfaces numba_cuda IndexError before .mask())
            _HAS_GPU = True
        except (ImportError, RuntimeError, IndexError, Exception) as e:  # noqa: BLE001
            _GPU_ERROR = e
            _HAS_GPU = False
    return bool(_HAS_GPU)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_open3d_available() -> bool:
    """Check whether ``open3d`` is importable.

    Returns
    -------
    bool
        ``True`` if ``open3d`` imported cleanly, ``False`` otherwise.
    """
    if not _OPEN3D_FOUND:
        return False
    return _probe_open3d()


def ensure_open3d_available() -> None:
    """Raise ``ModuleNotFoundError`` if ``open3d`` is not available.

    Raises
    ------
    ModuleNotFoundError
        If ``open3d`` is not installed; chained from the original exception
        when one was captured.
    """
    if not _probe_open3d():
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
    if not _PY4DGEO_FOUND:
        return False
    return _probe_py4dgeo()


def ensure_py4dgeo_available() -> None:
    """Raise ``ModuleNotFoundError`` if ``py4dgeo`` is not available.

    Raises
    ------
    ModuleNotFoundError
        If ``py4dgeo`` is not installed; chained from the original exception
        when one was captured.
    """
    if not _probe_py4dgeo():
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
    if not _GPU_FOUND:
        return False
    return _probe_gpu()


def ensure_gpu_available() -> None:
    """Raise ``ImportError`` if GPU support is not available.

    Raises
    ------
    ImportError
        If GPU support is unavailable; chained from the original
        :class:`ImportError` or :class:`RuntimeError` when one was captured.
    """
    if not _probe_gpu():
        msg = (
            "GPU support is not available. "
            "Install with `pip install pchandler[cuda11]` or `pip install pchandler[cuda12]`."
        )
        if _GPU_ERROR is not None:
            raise ImportError(msg) from _GPU_ERROR
        raise ImportError(msg)
