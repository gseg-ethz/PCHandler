"""Tests for pchandler._optional capability probes + public re-exports."""

import importlib
import sys

import pytest


def test_is_open3d_available() -> None:
    """is_open3d_available() truth-tracks open3d importability."""
    pytest.importorskip("open3d")
    from pchandler._optional import is_open3d_available

    assert is_open3d_available() is True


def test_ensure_open3d_raises_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """ensure_open3d_available() raises ModuleNotFoundError with install hint when absent."""
    monkeypatch.setitem(sys.modules, "open3d", None)
    import pchandler._optional as _opt

    importlib.reload(_opt)
    try:
        with pytest.raises(ModuleNotFoundError, match="Open3d is not installed"):
            _opt.ensure_open3d_available()
    finally:
        # Restore module state for other tests
        monkeypatch.delitem(sys.modules, "open3d", raising=False)
        importlib.reload(_opt)


def test_py4dgeo_and_gpu_helpers_symmetric() -> None:
    """py4dgeo + GPU helpers have the same shape as open3d helpers."""
    from pchandler._optional import (
        ensure_gpu_available,
        ensure_py4dgeo_available,
        is_gpu_available,
        is_py4dgeo_available,
    )

    assert isinstance(is_py4dgeo_available(), bool)
    assert isinstance(is_gpu_available(), bool)
    # ensure_* raises only when the corresponding _HAS_* is False
    # — smoke test for callability:
    assert callable(ensure_py4dgeo_available)
    assert callable(ensure_gpu_available)


def test_gpu_public_api_regression() -> None:
    """pchandler.filters.gpu.is_available() + ensure_available() still callable.

    Plan 05-07 D-27 re-export preserves public surface.
    """
    from pchandler.filters import gpu

    assert callable(gpu.is_available)
    assert callable(gpu.ensure_available)


def test_lazy_outlier_import_no_open3d(monkeypatch: pytest.MonkeyPatch) -> None:
    """From pchandler.filters import CartesianOutlierFilter succeeds without open3d.

    Constructing the filter raises ModuleNotFoundError via ensure_open3d_available().
    """
    monkeypatch.setitem(sys.modules, "open3d", None)
    import pchandler._optional as _opt

    importlib.reload(_opt)
    # Re-import the filter so its module-level lazy refs pick up the reloaded _optional:
    if "pchandler.filters.outlier_filter" in sys.modules:
        importlib.reload(sys.modules["pchandler.filters.outlier_filter"])
    try:
        from pchandler.filters import CartesianOutlierFilter

        with pytest.raises(ModuleNotFoundError, match="Open3d is not installed"):
            CartesianOutlierFilter()
    finally:
        monkeypatch.delitem(sys.modules, "open3d", raising=False)
        importlib.reload(_opt)
