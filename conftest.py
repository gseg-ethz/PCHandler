"""Conditionally exclude optional-extras test modules when the extra is absent.

Phase 0 D-A4 — gate the 2 open3d-eager test modules. Remove when DEP-04 lands.
Phase 1 SC8 followup — gate the GPU test module when cudf/cuspatial aren't installed
(CI runners have no CUDA; cuda extras are optional and not part of `[dev]`).
"""

# fmt: off
import importlib.util

collect_ignore = []
if importlib.util.find_spec("open3d") is None:
    collect_ignore += [
        "tests/test_pchandler_core.py",         # remove when DEP-04 fixes eager import in core.py
        "tests/filters/test_outlier_filter.py", # remove when DEP-04 fixes filters/outlier_filter.py:16  # noqa: E261
    ]
if importlib.util.find_spec("cudf") is None:
    # tests/filters/test_gpu.py instantiates PolygonFilterGPU / SphericalPolygonFilterGPU
    # which call pchandler.filters.gpu.ensure_available() and raise ImportError when
    # cudf/cuspatial are missing. The cuda11/cuda12 extras are optional and not
    # installed by `pip install .[dev]` in CI.
    collect_ignore += ["tests/filters/test_gpu.py"]
