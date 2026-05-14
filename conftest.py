"""Conditionally exclude optional-extras test modules when the extra is absent.

Phase 1 SC8 followup — gate the GPU test module when cudf/cuspatial aren't installed
(CI runners have no CUDA; cuda extras are optional and not part of `[dev]`).
DEP-04 complete — the open3d collect_ignore was removed; test modules now
self-skip via pytest.importorskip("open3d") at module top.
"""

# fmt: off
import importlib.util

collect_ignore = []
if importlib.util.find_spec("cudf") is None:
    # tests/filters/test_gpu.py instantiates PolygonFilterGPU / SphericalPolygonFilterGPU
    # which call pchandler.filters.gpu.ensure_available() and raise ImportError when
    # cudf/cuspatial are missing. The cuda11/cuda12 extras are optional and not
    # installed by `pip install .[dev]` in CI.
    collect_ignore += ["tests/filters/test_gpu.py"]
