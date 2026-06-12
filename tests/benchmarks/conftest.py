"""Session-scoped synthetic fixtures for Phase 4 pchandler benchmarks (D-33).

Three reusable point-cloud fixtures shared across the Phase 4 PERF-01 / PERF-02
/ PERF-03 microbenchmarks (plans 04-01, 04-02, 04-03). All fixtures are
``scope="session"`` so that benchmarks across different test files do not pay
the cost of rebuilding the synthetic data — each fixture is constructed once
per pytest run.

Determinism is provided by ``np.random.default_rng(0)`` per fixture, so
benchmark numbers are reproducible across runs as long as the underlying
``PointCloudData`` constructor is deterministic.
"""

import numpy as np
import pytest

from pchandler import PointCloudData


@pytest.fixture(scope="session")
def pcd_1m_points() -> PointCloudData:
    """1M-point synthetic PCD for PERF-01 / PERF-02 / PERF-03 benchmarks."""
    rng = np.random.default_rng(0)
    xyz = rng.standard_normal((1_000_000, 3)).astype(np.float32) * 5.0
    return PointCloudData(xyz)


@pytest.fixture(scope="session")
def pcd_100k_points() -> PointCloudData:
    """100k-point synthetic PCD (smaller bench target)."""
    rng = np.random.default_rng(0)
    xyz = rng.standard_normal((100_000, 3)).astype(np.float32) * 5.0
    return PointCloudData(xyz)


@pytest.fixture(scope="session")
def pcd_with_intensity() -> PointCloudData:
    """PCD with intensity scalar field (PERF-03 / COUPLE-05 reuse)."""
    rng = np.random.default_rng(0)
    xyz = rng.standard_normal((100_000, 3)).astype(np.float32) * 5.0
    intensity = rng.random(100_000).astype(np.float64)  # in [0, 1]
    return PointCloudData(xyz, intensity=intensity)
