"""BUG-06 static-check assertion: mypy must catch Unshifted/Shifted misuse."""

import subprocess
import sys
import tempfile
from pathlib import Path

_FIXTURE = Path(__file__).parent / "static_check_fixtures" / "optimal_shift_misuse.py"


def test_optimal_shift_newtype_misuse_caught_by_mypy() -> None:
    """BUG-06: mypy --strict catches Shifted/Unshifted confusion at the call site.

    Notes
    -----
    The project ``mypy.ini`` declares ``[mypy-tests.*] ignore_errors = true``
    (numpydantic strict-mode deferral, Plan 01-03b). Running mypy with the
    default config would silently swallow the intentional misuse here. We
    point mypy at an empty config file so its strict-mode flag and our
    ``--strict`` CLI override apply unobstructed.
    """
    # Empty mypy config — bypasses the project's [mypy-tests.*] ignore_errors
    # override (Plan 01-03b's numpydantic deferral) so this fixture's
    # intentional misuse actually surfaces.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False) as cfg:
        cfg.write("[mypy]\nstrict = true\n")
        cfg_path = cfg.name

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "mypy",
                "--strict",
                "--follow-imports=silent",
                "--config-file",
                cfg_path,
                str(_FIXTURE),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        Path(cfg_path).unlink(missing_ok=True)

    assert result.returncode != 0, (
        f"mypy unexpectedly accepted the misuse fixture (stdout):\n{result.stdout}\n(stderr):\n{result.stderr}"
    )
    assert "[arg-type]" in result.stdout, (
        f"Expected '[arg-type]' error code in mypy output; got:\n{result.stdout}\n{result.stderr}"
    )
