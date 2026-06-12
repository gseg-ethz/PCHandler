# pchandler - Toolbox for point-cloud handling, processing and analysis
#
# Copyright (c) 2025, Nicholas Meyer, Geosensors and Engineering Geodesy,
# Institute of Geodesy and Photogrammetry, ETH Zurich, Switzerland
# SPDX-License-Identifier: BSD-3-Clause
#
# Author: Nicholas Meyer (meyernic@ethz.ch)

"""PERF-03 D-29 #6 — splitter timing across tree sizes spanning `_SERIAL_THRESHOLD`.

Times :meth:`FoVTreePointCloudSplitter._direct_split` on the 1M-pt session
fixture (``pcd_1m_points`` from ``41_pchandler/tests/benchmarks/conftest.py``)
for tree sizes spanning the Phase 4 PERF-03 / D-28 threshold
(``_SERIAL_THRESHOLD = 16``). ``prefer="auto"`` switches from the inline
serial loop (sizes ≤ 16) to ``joblib.Parallel`` (sizes > 16) at the
boundary — the benchmark numbers feed the D-32 record in commit body
and ``04-03-SUMMARY.md``.

This file is collected only under ``pytest -m benchmark`` (CI excludes it
via ``-m "not benchmark"`` per Phase 4 D-31).
"""

import pytest

from pchandler.geometry.spherical import FoV, FoVTree
from pchandler.geometry.splitter import FoVTreePointCloudSplitter


def _build_n_fov_tree(n_fovs: int) -> FoVTree:
    """Build a 1xN-tile FoVTree containing exactly ``n_fovs`` leaves.

    Mirrors the helper in ``tests/geometry/test_splitter.py``; duplicated
    here so the benchmark file stays self-contained for ``pytest -m
    benchmark``.
    """
    new_fov = FoV(left=0.3, top=0.4, right=1.3, bottom=2.4)
    target_width = 1.0 / n_fovs + 1e-6
    tile_extent = FoV(left=0, right=target_width, top=0.0, bottom=2.5)
    tiles = new_fov.tile(tile_extent)
    tree = FoVTree.build_from_tiles(tiles)
    assert tree is not None
    return tree


@pytest.mark.benchmark
@pytest.mark.parametrize("n_fov", [8, 16, 17, 32])
def test_direct_split_by_tree_size(benchmark, pcd_1m_points, n_fov):
    """Time `_direct_split` under ``prefer="auto"`` for tree sizes spanning the threshold.

    Tree sizes were chosen to exercise both branches:
    - 8, 16 → serial (≤ `_SERIAL_THRESHOLD = 16`)
    - 17, 32 → parallel (> threshold)

    Reuses the deterministic ``pcd_1m_points`` session fixture so numbers
    are reproducible across runs.
    """
    tree = _build_n_fov_tree(n_fov)
    splitter = FoVTreePointCloudSplitter(tree, prefer="auto", n_jobs=-1)
    # Fresh PCD copy per round so `pcd.reduce(...)` in `_direct_split`
    # doesn't leak state into subsequent rounds.

    def _run():
        return splitter._direct_split(pcd_1m_points.copy(), tree)

    result = benchmark(_run)
    assert len(result) > 0


@pytest.mark.benchmark
@pytest.mark.parametrize("n_fov", [8, 16, 17, 32])
def test_iterative_split_by_tree_size(benchmark, pcd_1m_points, n_fov):
    """Companion bench for `_iterative_split` across the same tree sizes.

    The iterative path may differ from `_direct_split` because the
    per-level fan-out is gated on `len(tasks)` at each tree level (a
    deep but narrow tree could stay serial throughout even with many
    leaves overall).
    """
    tree = _build_n_fov_tree(n_fov)
    splitter = FoVTreePointCloudSplitter(tree, prefer="auto", n_jobs=-1)

    def _run():
        return splitter._iterative_split(pcd_1m_points.copy(), tree)

    result = benchmark(_run)
    assert len(result) > 0
