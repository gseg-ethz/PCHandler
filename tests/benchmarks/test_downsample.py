"""PERF-01 D-22 #2 microbenchmark — VoxelDownsample.sample on 1M-pt PCD.

Opt-in microbenchmark. Excluded from CI via ``pytest -m "not benchmark"``
per Phase 4 D-31. Run locally with ``pytest -m benchmark``.

Baseline (RESEARCH.md §Plan 04-01, lines 596-616, captured 2026-05-13):
  pre-fix  (np.unique(axis=0, return_inverse=True)) : 0.976 s
  post-fix (unique_rows_fast)                       : 0.248 s
  speedup  : ~3.93x

The fixture ``pcd_1m_points`` is session-scoped (D-33), defined in the
sibling :mod:`tests.benchmarks.conftest` module.
"""

import pytest

from pchandler.filters import VoxelDownsample


@pytest.mark.benchmark
def test_voxel_downsample_1m(benchmark, pcd_1m_points):
    """Benchmark VoxelDownsample.sample on a 1M-point synthetic PCD."""
    f = VoxelDownsample(voxel_size=0.1)
    result = benchmark(f.sample, pcd_1m_points)
    assert result.xyz.shape[0] > 0  # sanity, not perf assertion
