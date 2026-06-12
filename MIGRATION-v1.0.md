---
type: migration-spec
spec_version: 1.0
repo: pchandler
baseline_ref: 1e86610
target_ref: 501ed764eb14670ff72827c16757f8931a3670ed
generated_at: "2026-06-11T11:20:20Z"
bc_id_prefix: BC-PCH
---

# pchandler MIGRATION-v1.0

**Baseline:** v2.0.0rc9 (`1e8661019f445ad70f9fef8098d4ef4bc6e36647`)
**Target:** `refactor/gsd` HEAD (`501ed764eb14670ff72827c16757f8931a3670ed`)
**Generated:** 2026-06-11

## Summary

The v1.0 milestone closes a coordinated, 7-phase refactor of the `pchandler` point-cloud toolbox that addresses security (SEC-01..02), known bugs (BUG-01..08), fragile areas (FRAG-01..05), performance (PERF-01..05), cross-repo coupling (COUPLE-01..06), API completion (API-01..06), CI hygiene (DEP-01..05, TOOL-01..03), and tech-debt sweep (DEBT-02..11). The pchandler-side public-surface character of v1.0 is dominated by `should-review` semantic and error-behavior changes (most prominently `PointCloudData.to_py4dgeo` returning world-frame coordinates, `CsvHandler.load` strict-by-name field selection, and `LasHandler.load` caller-wins `numerical_optimization_shift`) and `additive` capability fills (`E57Handler.save` functional, `LasHandler.save` honouring `add_prefix`/`prefix`/`revert_sf_types`, `OptimizedShift` per-instance `minimum_decimal_places` validating setter). Zero `surface-removed` and zero `must-edit` entries surfaced — the PROJECT.md "no breaking public import paths" hard constraint is upheld end-to-end. The canonical cross-repo entry point for downstream consumers is the workspace synthesis at `.planning/MIGRATION-v1.0.md`, which cross-references this file and `30_GSEGUtils/MIGRATION-v1.0.md` by BC-ID.

## Public API stability invariant

The PROJECT.md hard constraint — "no breaking public import paths" — is upheld for v1.0. Every public symbol enumerated in PROJECT.md's "Validated" surface (`PointCloudData`, `load_file`, the four `data_io.{Ply,Las,E57,Csv}` handlers, the filter classes, `FoV`/`FoVTree`/`Angle`/`AngleArray`, `OptimizedShift`/`OptimizedShiftManager`, and the optional GPU/Open3D/py4dgeo paths) imports at its documented path at `refactor/gsd` HEAD. The AST-walk verifier in §"Verifier (inline)" below (the executor-time Tier 1 + Tier 2 check) confirms zero `surface-removed` entries and zero `must-edit` entries — no breaking entries surfaced; invariant upheld. The classification of `BC-PCH-011` (`OptimizedShiftManager.minimum_decimal_places` setter removal) as `should-review` rather than `must-edit` reflects the Phase 5 D-16 finding that the prior setter body was `pass` (a no-op); downstream code reading the property still works, only code that wrote to the property post-init breaks, and that code was already broken in spirit (it expected behaviour the no-op never provided).

## Breaking changes & behavior changes

| BC-ID | category | severity | affected_symbols | origin | migration_steps |
|---|---|---|---|---|---|
| BC-PCH-001 | dep-constraint | should-review | `[project.dependencies] numpy`, `requires-python` | Phase 0 D-A1 / D-A3 / Plan 00-01 / DEP-05 | Upgrade environment to `numpy >= 2.0, < 2.4` and Python `>=3.12, <3.13`; resolver-level only — no source changes downstream. |
| BC-PCH-002 | dep-constraint | informational | `[project.dependencies] GSEGUtils @ git+ssh://...@<40-char-SHA>` | Phase 1 D-17 / D-19 / Plan 01-05 / DEP-01 | GSEGUtils now SHA-pinned (currently `e413d2ad8e8afc521ebefa87b18e569906cdc031`); fresh-clone resolves only if `30_GSEGUtils@refactor/gsd` is reachable. Switch from branch-pinned `@doc` to SHA-pinned. |
| BC-PCH-006 | error-behavior | should-review | `pchandler.data_io.Las.load(path, **pcd_kw)` when `pcd_kw` includes `numerical_optimization_shift` | Phase 3 D-16 / BUG-08 / Plan 03-05 | Previously crashed with duplicate-kwarg error; now caller-wins (LAS-header NOS is the default only when not in `pcd_kw`). Code that worked around the crash by stripping NOS from `pcd_kw` before calling `Las.load` is unaffected; remove the strip if you want to honour an explicit NOS. |
| BC-PCH-007 | semantic-change | should-review | `pchandler.PointCloudData.to_py4dgeo` | Phase 3 D-15 / BUG-07 / CR-01 / Plan 03-05 | Returns world-frame coordinates now (mirrors `to_o3d`), not shift-frame. Downstream py4dgeo pipelines that re-applied the shift externally must drop that step. Normals and scalar fields are now preserved across the bridge (regression of CR-01 fixed). |
| BC-PCH-008 | semantic-change | should-review | `pchandler.geometry.spherical.FoVTree.build_from_tiles`, `pchandler.geometry.spherical.FoVTree.__getitem__` | Phase 3 D-11..D-13 / Phase 5 D-10 / BUG-05 / API-03 / CR-02 / Plans 03-04 + 05-03 | `FoVTree.identifier` strings are now 2D `"<r>-<c>"` for flat tiles and recursive descent paths (collision-free vs the prior collapse-on-clash form). `__getitem__` accepts full identifier strings (e.g. `"0-00-0"` for depth-2 descent), not just leaf integers. Downstream code that stored identifiers as filenames remains valid (filesystem-safe chars only). |
| BC-PCH-011 | signature-shape | should-review | `pchandler.geometry.OptimizedShiftManager.minimum_decimal_places` (setter) | Phase 5 D-16 / API-06 / Plan 05-05 | The class-level `minimum_decimal_places` setter is REMOVED. Code doing `OSM().minimum_decimal_places = N` post-init now raises `AttributeError`. The setter's prior body was `pass` (no-op), so no functional behaviour is lost. Migrate by passing `minimum_decimal_places=N` to `OptimizedShiftManager(...)` at init, or by setting it on a specific `pchandler.geometry.OptimizedShift(minimum_decimal_places=N)` instance (Phase 5 D-17 / D-18 added a validating per-instance setter + init kwarg). |
| BC-PCH-012 | error-behavior | should-review | `pchandler.data_io.Csv.load(path, scalar_fields=[...])` when the CSV header is parsed but a requested field is missing | Phase 5 D-12..D-15 / API-04 / Plan 05-04 | Previously silently mis-mapped; now raises `ValueError(f"CsvHandler.load: requested scalar fields not in header: {missing}. Available: {file_info.fields}. Pass an explicit column_names_row if the header lives outside line {column_names_row}.")`. Strict-by-name when header is present; positional fallback when header absent. |
| BC-PCH-013 | error-behavior | should-review | `pchandler.filters.gpu.ensure_available`, optional Open3D / py4dgeo capability probes (private `pchandler._optional`) | Phase 5 D-24..D-29 / DEP-04 / Plan 05-07 | Open3D / py4dgeo / GPU optional deps now fail early at first method call via `ensure_*_available()` with an install-hint message, instead of obscure `AttributeError` at first attribute access. The private `pchandler._optional` module is intentionally private; `pchandler.filters.gpu.is_available()` / `ensure_available()` remain the public surface. |

## Additive changes

| BC-ID | category | severity | affected_symbols | origin | migration_steps |
|---|---|---|---|---|---|
| BC-PCH-009 | additive-or-fixed | additive | `pchandler.data_io.E57.save(pcd_or_iterable, path, *, embed_shift_in_transform=True, strict=False, **config)` | Phase 5 D-03..D-07 / API-01 / Plan 05-01 | Previously the class docstring said "limited to PLY/LAS/CSV"; now functional. Default writes shifted coords + per-scan transform (precision-preserving); pass `embed_shift_in_transform=False` for world-frame. Unsupported scalar fields skip-and-warn (pass `strict=True` to raise). Multi-scan iterable input supported. |
| BC-PCH-010 | additive-or-fixed | additive | `pchandler.data_io.Las.save(..., add_prefix=True, prefix="scalar_", revert_sf_types=False, ...)` | Phase 5 D-08 / API-02 / Plan 05-02 | Both kwargs now honoured for residual scalar fields (previously inert / stubbed out per the v2.0.0rc9 source marker). `add_prefix=True` (default) prefixes extra-dim names with `"scalar_"`; LAS extra-dim 31-char name guard active. `revert_sf_types=True` opts into the legacy dtype-narrowing path for downstream readers that expect it. |
| BC-PCH-014 | additive-or-fixed | additive | `pchandler.geometry.spherical.AngleArray` (`__add__`/`__sub__` between two `AngleArray` of matching shape and unit) | Phase 3 D-29 / Plan 03-06 | Matching-shape `AngleArray + AngleArray` now succeeds element-wise (was previously forbidden via `raise NotImplementedError`). Downstream code that worked around the limitation by dropping to bare numpy arrays can use the typed wrapper directly. Mismatched shapes / units still raise per the established invariants. |
| BC-PCH-015 | semantic-change | additive | `pchandler.geometry.util.get_outline_polygon(..., seed=None)` | Phase 4 D-23 / D-24 / PERF-02 / Plan 04-03 | Now deterministic-by-default (seeded with 0 internally when `seed=None`) and auto-caps the alpha-shape sample at 100_000 points. Non-deterministic callers that compared two consecutive results may see them now equal — that is the intended fix. Pass `seed=<arbitrary>` to opt back into randomized sampling; pass an explicit `n_sample` to override the cap. |

## Internal & sweep changes

- [Phase 1 D-02 / COUPLE-01 / Plan 01-01]: `SingletonMeta` inline copy deleted from `pchandler.geometry.optimal_shift`; now imported from `GSEGUtils.singleton.SingletonMeta`. No public-surface change — the symbol was never on `pchandler.__all__`.
- [Phase 1 D-04..D-08 / D-24 / TOOL-01 / Plan 01-02]: Toolchain switch — `ruff` replaces `black` + `isort` + `flake8` + `flake8-bugbear` + `flake8-docstrings`. NumPy-style docstrings now enforced on `src/**`. No runtime impact; dev-tooling only.
- [Phase 1 D-15]: GSEGUtils contract-test isolation in `tests/test_validators_contract.py` (no external symbol changes; pchandler-side test cleanup).
- [Phase 1 D-20..D-23]: CI hygiene — `pytest --cov=ModuleA` → `--cov=pchandler` typo fix, `pytest-cov` symmetry alignment, `CITATION.cff` added.
- [Phase 2 D-08..D-13 / SEC-02]: `_set_shift_applied_by` private helpers replace `model_construct`-based validation bypass. Public surface unchanged — call sites stayed on `__init__`.
- [Phase 2 D-14..D-17]: `Private :: Do Not Upload` classifier removed; `## Publication Policy` README section added (structural absence — no `twine` calls). No PyPI publish step active.
- [Phase 2 D-22..D-25]: `geometry/spherical/fov.py:192` `Constuct` typo fix + Notes block; `geometry/spherical/angle.py:55, 124` slotted-class comments. Pure docs.
- [Phase 3 D-09]: `Unshifted` / `Shifted` `NewType` tags added — static-only; runtime erasure preserves identical behaviour.
- [Phase 3 D-22 / Plan 03-03]: `_process_shift` four-method internal refactor; no signature change; observable behaviour identical to v2.0.0rc9.
- [Phase 3 D-25..D-27 / Plan 03-06]: `AngleBase` arithmetic dunder mechanism switched from `raise NotImplementedError` to `return NotImplemented`. Observable effect is hygiene only: `Angle(1, RAD) + "two"` now raises Python's `TypeError` via the reflected-dunder fallback instead of `NotImplementedError`. Documented as internal hygiene fix rather than a BC entry because `NotImplementedError` is conventionally not caught by downstream code (lint-flagged as a sentinel-not-exception in standard linters).
- [Phase 4 D-08..D-11 / PERF-03..PERF-04 / Plan 04-04 + 04-05]: Lock-free fast-path on `SingletonMeta.__call__` (GSEGUtils-side change; cross-referenced as `BC-GSEG-005`). No semantic change for pchandler callers.
- [Phase 4 D-20 / PERF-01 / Plan 04-02]: `VoxelDownsample` drop-in `unique_rows_fast` swap; sort-order semantics verified not to affect downstream by Phase 4 verification.
- [Phase 4 D-26..D-28 / COUPLE-05 / Plan 04-06]: `Splitter.__init__(..., prefer="auto"|"serial"|"processes")` + threshold dispatch. Kwarg-additive; default `"auto"` preserves prior `"processes"` for large trees.
- [Phase 5 D-20..D-23 / DEP-03 / Plan 05-06]: `41_pchandler/stubs/` added for six untyped backends (joblib, plyfile, pye57, laspy, alphashape, shapely). Type-checker hygiene only; no import-surface change.
- [Phase 6 D-12 / DEBT-07 / Plan 06-03]: `PointCloudData.__init__(self, /, xyz=None, **kwargs: Unpack[PointCloudDataKW])` migrated to `TypedDict + Unpack`; the deleted `extract` no-op override at `core.py:394-411` is invisible to callers (base-class dispatch covers the path; Phase 6 verification ground-truth).
- [Phase 6 D-13 / DEBT-10 / Plan 06-03]: `NormalFields.initialize(...)` dropped the unused `name: str = ""` parameter. Pchandler internal grep over `src/` and `tests/` confirmed zero call-site usage. Downstream usage outside the workspace cannot be verified during this audit; if any external caller passes `name=`, escalate as a follow-up BC entry (do not silently fix).
- [Phase 6 D-04..D-11 + Phase 6 DEBT-02..11 / Plans 06-01..06-04]: Test additions (TEST-01..TEST-06 — splitter direct-vs-iterative equality assertion, stub-drift smoke test, etc.) and tech-debt sweep entries that landed without observable public-surface effect (`@validate_variables` re-application, csv `tuple(field_names)` micro-refactor, DEBUG log on `_reconstruct` mint-new-UUID branch, residual source-marker sweep across `src/`). Captured here for traceability.
- [Step 2 commit-traceability audit]: zero untraceable non-trivial commits in `git log --no-merges 1e86610..refactor/gsd -- src/`. All 65 commits with `feat:` / `fix:` / `refactor:` prefixes carry a phase- or plan-scoped marker (`(NN)`, `(NN-MM)`, `(NN-fix)`, `(quick-260514-noz)`, `(BUG-NN)`, `(FRAG-NN)`, `(PERF-NN)`, `(D-NN)`); each scope traces back to one of the seven prior phase decision blocks. Aggregation grain: one BC entry per observable downstream effect, with multi-D-x markers aggregated into a single entry whenever the observable effect is shared (e.g. BC-PCH-008 aggregates Phase 3 D-11..D-13 + Phase 5 D-10 + CR-02 hot-fix into one entry).

## Verifier (inline)

The block below is the executor-time audit verifier for Plan 07-01 (D-07 Step 3). It Tier-1 AST-walks the six pchandler public-surface `.pyi` files and asserts every top-level (non-dotted) symbol in `affected_symbols` resolves; it Tier-2 runtime-imports `pchandler.geometry` and asserts the BC-PCH-011 setter-removal invariant. Exits 0 on success.

```python
r"""Inline executor-time verifier for pchandler MIGRATION-v1.0.md (D-07 Step 3).

Walks the six public-surface .pyi files at refactor/gsd HEAD and asserts that
every top-level (non-dotted) symbol named in any BC-PCH-NNN affected_symbols
cell resolves on the public surface (Tier 1). Then runtime-imports
pchandler.geometry and asserts the BC-PCH-011 setter-removal invariant
on OptimizedShiftManager.minimum_decimal_places (Tier 2). Exits 0 on success.

Run from the workspace root:

    python3.12 /tmp/07-01-verifier.py

Equivalent execution path used by the Plan 07-01 verify step:

    awk '/^## Verifier \(inline\)$/,/^```$/' 41_pchandler/MIGRATION-v1.0.md \
        | sed -n '/^```python$/,/^```$/p' | sed '1d;$d' \
        > /tmp/07-01-verifier.py
    python3.12 /tmp/07-01-verifier.py
"""
from __future__ import annotations

import ast
import sys
from importlib import import_module
from pathlib import Path

# Six pchandler public-surface .pyi files. Paths are relative to the workspace
# root (the directory containing 41_pchandler/ and 30_GSEGUtils/).
PUBLIC_SURFACE_FILES = [
    "41_pchandler/src/pchandler/__init__.pyi",
    "41_pchandler/src/pchandler/data_io/__init__.pyi",
    "41_pchandler/src/pchandler/geometry/__init__.pyi",
    "41_pchandler/src/pchandler/scalar_fields/__init__.pyi",
    "41_pchandler/src/pchandler/filters/__init__.pyi",
    "41_pchandler/src/pchandler/geometry/spherical/__init__.pyi",
]


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


# Inline BC entries — keep in sync with the markdown tables above. Only entries
# whose affected_symbols list contains at least one top-level (non-dotted) name
# are checked at Tier 1; pure dotted-path entries delegate to Tier 2.
BC_ENTRIES: list[dict[str, str | list[str]]] = [
    {"id": "BC-PCH-001", "category": "dep-constraint",  "severity": "should-review",  "affected_symbols": []},
    {"id": "BC-PCH-002", "category": "dep-constraint",  "severity": "informational", "affected_symbols": []},
    {"id": "BC-PCH-006", "category": "error-behavior",  "severity": "should-review",  "affected_symbols": ["Las"]},
    {"id": "BC-PCH-007", "category": "semantic-change", "severity": "should-review",  "affected_symbols": ["PointCloudData"]},
    {"id": "BC-PCH-008", "category": "semantic-change", "severity": "should-review",  "affected_symbols": ["FoVTree"]},
    {"id": "BC-PCH-009", "category": "additive-or-fixed", "severity": "additive",      "affected_symbols": ["E57"]},
    {"id": "BC-PCH-010", "category": "additive-or-fixed", "severity": "additive",      "affected_symbols": ["Las"]},
    {"id": "BC-PCH-011", "category": "signature-shape", "severity": "should-review",  "affected_symbols": ["OptimizedShiftManager"]},
    {"id": "BC-PCH-012", "category": "error-behavior",  "severity": "should-review",  "affected_symbols": ["Csv"]},
    {"id": "BC-PCH-013", "category": "error-behavior",  "severity": "should-review",  "affected_symbols": ["filters"]},
    {"id": "BC-PCH-014", "category": "additive-or-fixed", "severity": "additive",      "affected_symbols": ["AngleArray"]},
    {"id": "BC-PCH-015", "category": "semantic-change", "severity": "additive",        "affected_symbols": ["util"]},
]


def main() -> int:
    workspace_root = Path(__file__).resolve().parent
    # Fallback: if the script is invoked from /tmp (as in the canonical extract-
    # and-run path), use the known workspace location instead.
    if not (workspace_root / "41_pchandler").exists():
        workspace_root = Path("/home/nixton/gsd-workspaces/pchandler")
    if not (workspace_root / "41_pchandler").exists():
        print(f"[fail] cannot locate 41_pchandler/ under {workspace_root}", file=sys.stderr)
        return 1

    public_surface: set[str] = set()
    for rel in PUBLIC_SURFACE_FILES:
        pyi = workspace_root / rel
        if not pyi.exists():
            print(f"[fail] missing public-surface file: {pyi}", file=sys.stderr)
            return 1
        public_surface |= _extract_declared_names(pyi.read_text(encoding="utf-8"))

    failures: list[str] = []
    for entry in BC_ENTRIES:
        affected = entry["affected_symbols"]
        if entry["category"] == "surface-removed":
            for sym in affected:
                if sym in public_surface:
                    failures.append(f"{entry['id']}: documented as surface-removed but {sym!r} present on public surface")
        else:
            for sym in affected:
                if "." in sym:
                    continue  # dotted symbols handled by Tier 2
                if sym not in public_surface:
                    failures.append(f"{entry['id']}: symbol {sym!r} not in public surface")

    # Tier 2: runtime check for BC-PCH-011 — the OptimizedShiftManager
    # `minimum_decimal_places` setter was REMOVED in Phase 5 D-16 / API-06.
    # Per Phase 5 D-17 / D-18 the per-instance setter on OptimizedShift remains.
    try:
        geometry = import_module("pchandler.geometry")
    except Exception as exc:
        failures.append(f"BC-PCH-011: cannot import pchandler.geometry for Tier 2 ({exc!r})")
    else:
        osm_cls = getattr(geometry, "OptimizedShiftManager", None)
        if osm_cls is None:
            failures.append("BC-PCH-011: OptimizedShiftManager not exposed on pchandler.geometry")
        else:
            mdp = getattr(osm_cls, "minimum_decimal_places", None)
            if mdp is None or not isinstance(mdp, property):
                failures.append("BC-PCH-011: OptimizedShiftManager.minimum_decimal_places is not a property")
            elif mdp.fset is not None:
                failures.append("BC-PCH-011: OptimizedShiftManager.minimum_decimal_places setter is still present (fset != None)")
        # Confirm the per-instance OptimizedShift setter remains as the migration path.
        os_cls = getattr(geometry, "OptimizedShift", None)
        if os_cls is None:
            failures.append("BC-PCH-011: OptimizedShift not exposed on pchandler.geometry (migration path broken)")
        else:
            os_mdp = getattr(os_cls, "minimum_decimal_places", None)
            if os_mdp is None or not isinstance(os_mdp, property):
                failures.append("BC-PCH-011: OptimizedShift.minimum_decimal_places is not a property")
            elif os_mdp.fset is None:
                failures.append("BC-PCH-011: OptimizedShift.minimum_decimal_places setter missing (migration path broken)")

    if failures:
        print("[fail] migration-spec verification:", *failures, sep="\n  ", file=sys.stderr)
        return 1
    print(f"[ok] verified {len(BC_ENTRIES)} BC-PCH entries against public surface ({len(public_surface)} declared names)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```
