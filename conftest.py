"""Phase 0 D-A4 — gate the 2 open3d-eager test modules. Remove when DEP-04 lands."""

# fmt: off
import importlib.util

collect_ignore = []
if importlib.util.find_spec("open3d") is None:
    collect_ignore += [
        "tests/test_pchandler_core.py",         # remove when DEP-04 fixes eager import in core.py
        "tests/filters/test_outlier_filter.py", # remove when DEP-04 fixes filters/outlier_filter.py:16  # noqa: E261
    ]
