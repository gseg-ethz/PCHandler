"""TEST-06 pchandler-side: drift smoke test for ``__init__.pyi`` files.

Regenerates pchandler's ``__init__.pyi`` files in a tmp_path via the GSEGUtils
stub generator and asserts the exported symbol set matches the committed
stubs. Catches the canonical TEST-06 drift case: a `_lazy_map` edit without a
regenerator run.

Notes
-----
The committed ``__init__.pyi`` files are hand-customized (license headers +
``@overload`` discriminators per Phase 1 D-19) and intentionally diverge
from the generator's bare emission in formatting and discriminator depth.
This smoke test therefore compares **declared symbol sets** (the union of
``__all__`` entries + the eager-import re-exports), not raw bytes. The
generator's contract is "the symbols exported by ``__init__.py`` are
declared in ``__init__.pyi``"; anything beyond that is editorial.

Per RESEARCH §"Common Pitfalls" §5: the generator silently no-ops on
missing paths. This test asserts at least 3 ``.pyi`` files were actually
regenerated before comparing — a vacuous pass on a no-op is otherwise
possible.
"""

import ast
import shutil
import subprocess
import sys
from pathlib import Path


def _extract_declared_names(pyi_text: str) -> set[str]:
    """Return the set of symbol names a ``__init__.pyi`` declares.

    Combines (a) the list-literal assigned to ``__all__`` and (b) every
    name introduced by ``from .X import Y as Y`` / ``from . import X as X``
    statements. This is the generator's contract surface: anything an
    importer can `from pkg import ...` on.
    """
    tree = ast.parse(pyi_text)
    declared: set[str] = set()
    for node in ast.walk(tree):
        # ``__all__`` literal
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for tgt in targets:
                if isinstance(tgt, ast.Name) and tgt.id == "__all__":
                    value = node.value
                    if isinstance(value, ast.List):
                        for elt in value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                declared.add(elt.value)
        # ``from .X import Y as Y`` / ``from . import X as X`` — package-relative
        # re-exports only. Absolute imports (``from typing import Any``, etc.)
        # bring helper symbols into the stub's namespace but are NOT part of the
        # public re-export surface, so we skip them.
        if isinstance(node, ast.ImportFrom) and node.level >= 1:
            for alias in node.names:
                # Use the aliased name (or the original if no alias) — both count
                # as part of the public re-export surface.
                declared.add(alias.asname or alias.name)
    return declared


def test_stubs_drift_pchandler_real(tmp_path: Path) -> None:
    """Real pchandler/__init__.pyi files declare the same symbols as the regenerator.

    See module docstring for the symbol-set vs byte-equality rationale.
    """
    src = Path(__file__).resolve().parent.parent / "src" / "pchandler"
    work = tmp_path / "pchandler"
    shutil.copytree(src, work)

    # Strip existing .pyi so the regenerator writes fresh.
    for pyi in work.rglob("__init__.pyi"):
        pyi.unlink()

    result = subprocess.run(
        [sys.executable, "-m", "GSEGUtils.generate_init_stubs", str(work), "--walk", "--overwrite"],
        check=True,
        capture_output=True,
        text=True,
    )

    # Pitfall 5 (RESEARCH §"Common Pitfalls" §5): assert the generator emitted
    # files — without this guard a silent no-op produces a vacuous pass.
    regenerated_count = sum(1 for _ in work.rglob("__init__.pyi"))
    assert regenerated_count >= 3, (
        f"Generator silently no-op'd: expected >=3 __init__.pyi files regenerated under {work}, "
        f"found {regenerated_count}. Generator stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    # Symbol-set comparison for every committed .pyi vs its regenerated twin.
    committed_pyis = sorted(src.rglob("__init__.pyi"))
    assert committed_pyis, "No committed __init__.pyi files found under src/pchandler — invariant broken"

    for pyi in committed_pyis:
        rel = pyi.relative_to(src)
        regenerated_path = work / rel
        assert regenerated_path.exists(), (
            f"Regenerator did not emit {rel}. Generator stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

        committed_names = _extract_declared_names(pyi.read_text())
        regenerated_names = _extract_declared_names(regenerated_path.read_text())

        missing = regenerated_names - committed_names
        extra = committed_names - regenerated_names
        assert not missing and not extra, (
            f"\n=== Symbol drift in {rel} ===\n"
            f"Names present in regenerated but missing from committed (forgot to regenerate?): {sorted(missing)}\n"
            f"Names present in committed but missing from regenerated (stale committed stub?): {sorted(extra)}\n"
            f"--- committed ({pyi}) ---\n{pyi.read_text()}\n"
            f"--- regenerated ({regenerated_path}) ---\n{regenerated_path.read_text()}\n"
        )
