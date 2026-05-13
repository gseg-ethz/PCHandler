# Static-check fixture for BUG-06 (Phase 3 D-09 / D-28).
#
# Passing a `Shifted` value where `Unshifted` is required (and vice versa)
# must be flagged by mypy --strict with the `[arg-type]` error code.
#
# This file is NOT a runtime pytest test — it is exercised via subprocess
# mypy in `test_static_check_optimal_shift.py`. Living under
# `static_check_fixtures/` (not `test_*.py`) keeps it out of pytest's
# default discovery glob.
import numpy as np

from pchandler.geometry.optimal_shift import Shifted, Unshifted


def expects_unshifted(x: Unshifted) -> None: ...
def expects_shifted(x: Shifted) -> None: ...


arr = np.zeros((10, 3), dtype=np.float64)
u: Unshifted = Unshifted(arr)
s: Shifted = Shifted(arr)

expects_unshifted(s)  # mypy must flag: arg-type
expects_shifted(u)  # mypy must flag: arg-type
