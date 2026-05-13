# pchandler - Toolbox for point-cloud handling, processing and analysis
#
# Copyright (c) 2025, Nicholas Meyer, Geosensors and Engineering Geodesy,
# Institute of Geodesy and Photogrammetry, ETH Zurich, Switzerland
# SPDX-License-Identifier: BSD-3-Clause
#
# Author: Nicholas Meyer (meyernic@ethz.ch)

"""PERF-02 microbenchmark — D-25 #5 + D-32 record.

Times ``get_outline_polygon`` on the 1M-pt session fixture (``pcd_1m_points``
from ``41_pchandler/tests/benchmarks/conftest.py``). With Phase 4 PERF-02's
auto-cap (D-24) the default-args call path trims to 100 000 points before
invoking ``alphashape.alphashape``, so the bench measures the bounded path —
the success criterion proof for ROADMAP SC-2.

This file is collected only under ``pytest -m benchmark`` (CI excludes it via
``-m "not benchmark"`` per Phase 4 D-31).
"""

import pytest

from pchandler.geometry.util import get_outline_polygon


@pytest.mark.benchmark
def test_get_outline_polygon_1m(benchmark, pcd_1m_points):
    """Benchmark default-args ``get_outline_polygon`` (auto-trim path) on 1M-pt PCD.

    The default ``nb_points=-1`` auto-caps the projection at
    ``_DEFAULT_OUTLINE_MAX_POINTS = 100_000``, so this bench measures the
    bounded-time path post-Phase-4 (PERF-02 D-24). Compare against the pre-fix
    baseline recorded in RESEARCH.md §"Plan 04-02" (100 k pts ~15.4 s on dev
    host; 1 M pts pre-fix would take minutes).
    """
    result = benchmark(get_outline_polygon, pcd_1m_points, "xy")
    # Sanity: returns a non-None shapely polygon.
    assert result is not None
