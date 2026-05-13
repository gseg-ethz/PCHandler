"""BUG-06 static-check assertion: mypy must catch Unshifted/Shifted misuse."""

import subprocess
import sys
from pathlib import Path

_FIXTURE = Path(__file__).parent / "static_check_fixtures" / "optimal_shift_misuse.py"


def test_optimal_shift_newtype_misuse_caught_by_mypy() -> None:
    """BUG-06: mypy --strict catches Shifted/Unshifted confusion at the call site."""
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "--strict", "--follow-imports=silent", str(_FIXTURE)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0, (
        f"mypy unexpectedly accepted the misuse fixture (stdout):\n{result.stdout}\n(stderr):\n{result.stderr}"
    )
    assert "[arg-type]" in result.stdout, (
        f"Expected '[arg-type]' error code in mypy output; got:\n{result.stdout}\n{result.stderr}"
    )
