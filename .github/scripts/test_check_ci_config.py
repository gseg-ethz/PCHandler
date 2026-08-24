#!/usr/bin/env python3
# .github/scripts/test_check_ci_config.py
# Source: Phase 13 plan 13-03 Task 2 — unit tests pinning CI-12 assertions A1 through A7
# A8 added by Phase 17 plan 17-02 (CI-17, D-17-05, D-17-08 clause 1)
# A9 added by Phase 17 plan 17-05 (CI-17, D-17-10, D-17-15)
# A10 added by Phase 17 plan 17-07 round 3 (WINDOWS 30 -- the gpu.yml host break-out)
"""Unit tests for the CI-12 self-test's ten assertions.

Every assertion helper takes a root path, so each test writes a synthetic tree of
workflow, ruleset, composite-action and release-please files into a ``tmp_path``
fixture and asserts against that tree rather than against the repository. Three
tests deliberately run against the real repository root instead, because "passes
on today's tree" is itself a property worth pinning: A1, A2 and A5 all have a
current-tree true-negative that a synthetic fixture cannot demonstrate.

The A1 pair is the one that matters most. ``test_a1_passes_on_the_real_ci_yml``
and ``test_a1_fails_on_a_paths_filter_under_a_pull_request_trigger`` exist
together so that the day-one false positive described in
``check_ci_config``'s module docstring is demonstrably resolved by *narrowing*
the rule rather than by muting the check.

Run with::

    python -m pytest .github/scripts/test_check_ci_config.py -q

Note that ruff's ``per-file-ignores`` relax docstring rules under ``tests/**``
only, which does not cover ``.github/scripts/``, so every test here carries a
numpy-convention docstring by design rather than by accident.
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import check_ci_config as cc  # noqa: E402  — deliberately after the sys.path insertion above

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

DEFAULT_CONTEXTS = ("Lint (pre-commit)", "Tests (pytest)")
DEFAULT_ALLOWLIST = ("CHANGELOG.md", ".release-please-manifest.json", "docs/source/conf.py")

# A minimal workflow producing both default required contexts, with a clean
# trigger block. Individual tests override one aspect of it at a time.
CLEAN_CI_YML = """\
name: CI

on:
  pull_request:
  push:
    branches: [main, develop/gsd]

jobs:
  lint:
    name: Lint (pre-commit)
    runs-on: ubuntu-latest
    permissions:
      contents: read
    steps:
      - run: echo lint
  tests:
    name: Tests (pytest)
    runs-on: ubuntu-latest
    steps:
      - run: echo test
"""

# A minimal GPU workflow in the shape A8 inspects: ONE step body that installs,
# asserts the capability, and then runs the suite. The three moving parts are kept
# separate so a test can reorder or drop exactly one of them, which is the whole
# point of A8 — presence is not the property, POSITION is.
GPU_JOB_HEAD = """\
name: GPU Tests

on:
  pull_request:
    branches: [main]

jobs:
  gpu-tests:
    name: Tests (pytest, GPU)
    runs-on: [self-hosted, gpu]
    steps:
      - name: Run GPU tests in the RAPIDS container (podman)
        run: |
          podman run --rm "$GPU_IMAGE" bash -lc '
            set -euo pipefail
"""

GPU_INSTALL_LINE = "            pip install -e .[dev]\n"

GPU_ASSERT_BLOCK = (
    "            # POST-INSTALL GPU CAPABILITY ASSERT (CI-17, D-17-05)\n"
    '            python -c "if 1:\n'
    "                import sys\n"
    "                import pchandler._optional as opt\n"
    "                if not opt.is_gpu_available():\n"
    "                    sys.exit(1)\n"
    '                "\n'
)

GPU_SUITE_LINE = "            pytest tests/filters/test_gpu.py -v\n"

GPU_JOB_TAIL = "          '\n"

CLEAN_GPU_YML = GPU_JOB_HEAD + GPU_INSTALL_LINE + GPU_ASSERT_BLOCK + GPU_SUITE_LINE + GPU_JOB_TAIL

GPU_CONTEXTS = DEFAULT_CONTEXTS + ("Tests (pytest, GPU)",)

# A9's fixtures. The corridor is three artefacts that only mean anything together:
# a constraints file, an install that consumes it, and a digest artefact the job
# resolves. Each is varied ONE at a time below, because the whole point of A9 is
# that a present-but-unwired artefact is decoration.
GPU_CONSTRAINED_INSTALL_LINE = "            pip install -c .github/constraints/gpu.txt -e .[dev]\n"

CORRIDOR_GPU_YML = GPU_JOB_HEAD + GPU_CONSTRAINED_INSTALL_LINE + GPU_ASSERT_BLOCK + GPU_SUITE_LINE + GPU_JOB_TAIL

# A syntactically valid 64-hex digest that is deliberately NOT the shipped one, so
# a test can never accidentally assert against the real pin.
SYNTHETIC_DIGEST = "a1" * 32

# Declared HERE as a literal rather than read from `cc`, so that these fixtures do
# not inherit whatever the production constant happens to say. The two are pinned
# equal by `test_a9_passes_on_the_real_repository`, which is the direction that
# catches the production constant itself being wrong.
GPU_IMAGE_REPOSITORY = "ghcr.io/gseg-ethz/pchandler-gpu-runner"

DEFAULT_CONSTRAINTS = "# corridor\nnumpy >= 2.2, < 2.3\n"


def pinned_reference(digest: str = SYNTHETIC_DIGEST) -> str:
    """Return a full pinned image reference for the GHCR repository A9 pins.

    Parameters
    ----------
    digest
        Hex digest body, without the ``sha256:`` marker.

    Returns
    -------
    str
        The reference, in the shape ``<repository>@sha256:<digest>``.
    """
    return f"{GPU_IMAGE_REPOSITORY}@sha256:{digest}"


DEFAULT_DIGEST = pinned_reference() + "\n"


def build_corridor_tree(
    root: pathlib.Path,
    gpu_yml: str = CORRIDOR_GPU_YML,
    *,
    constraints: str | None = DEFAULT_CONSTRAINTS,
    digest: str | None = DEFAULT_DIGEST,
    contexts: tuple[str, ...] = GPU_CONTEXTS,
) -> pathlib.Path:
    """Write a synthetic tree carrying the GPU corridor's artefacts.

    Parameters
    ----------
    root
        Directory to populate; normally a ``tmp_path`` fixture.
    gpu_yml
        Text of the synthetic ``gpu.yml``.
    constraints
        Text of the constraints file, or ``None`` to omit the file entirely.
    digest
        Text of the digest artefact, or ``None`` to omit the file entirely. Pass
        the empty string to write an empty file.
    contexts
        Required status-check contexts written into the committed ruleset.

    Returns
    -------
    pathlib.Path
        The populated root, for chaining.
    """
    tree = build_tree(root, {"ci.yml": CLEAN_CI_YML, "gpu.yml": gpu_yml}, contexts=contexts)
    if constraints is not None:
        target = tree / cc.GPU_CONSTRAINTS_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(constraints)
    if digest is not None:
        target = tree / cc.GPU_DIGEST_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(digest)
    return tree


def build_tree(
    root: pathlib.Path,
    workflows: dict[str, str],
    *,
    contexts: tuple[str, ...] = DEFAULT_CONTEXTS,
    allowlist: tuple[str, ...] = DEFAULT_ALLOWLIST,
    changelog_path: str = "CHANGELOG.md",
    extra_files: tuple[str, ...] = ("docs/source/conf.py",),
) -> pathlib.Path:
    """Write a synthetic repository tree the assertion helpers can run against.

    Parameters
    ----------
    root
        Directory to populate; normally a ``tmp_path`` fixture.
    workflows
        Mapping of workflow filename to file text.
    contexts
        Required status-check contexts written into the committed ruleset.
    allowlist
        Paths written into the composite action's allowlist variable.
    changelog_path
        The release-please changelog path.
    extra_files
        The release-please extra files.

    Returns
    -------
    pathlib.Path
        The populated root, for chaining.
    """
    workflows_dir = root / cc.WORKFLOWS_DIR
    workflows_dir.mkdir(parents=True, exist_ok=True)
    for name, text in workflows.items():
        (workflows_dir / name).write_text(text)

    rulesets_dir = root / cc.RULESETS_DIR
    rulesets_dir.mkdir(parents=True, exist_ok=True)
    (rulesets_dir / "main.json").write_text(
        json.dumps(
            {
                "name": "protect-main",
                "rules": [
                    {
                        "type": "required_status_checks",
                        "parameters": {
                            "required_status_checks": [{"context": c, "integration_id": None} for c in contexts]
                        },
                    }
                ],
            }
        )
    )

    action_path = root / cc.COMPOSITE_ACTION_PATH
    action_path.parent.mkdir(parents=True, exist_ok=True)
    action_path.write_text(
        f"runs:\n  using: composite\n  steps:\n    - run: |\n        ALLOW='{' '.join(allowlist)}'\n"
    )

    (root / cc.RELEASE_PLEASE_CONFIG_PATH).write_text(
        json.dumps({"packages": {".": {"changelog-path": changelog_path, "extra-files": list(extra_files)}}})
    )
    return root


def build_gpu_tree(root: pathlib.Path, gpu_yml: str) -> pathlib.Path:
    """Write a synthetic tree whose committed ruleset requires the GPU context.

    Parameters
    ----------
    root
        Directory to populate; normally a ``tmp_path`` fixture.
    gpu_yml
        Text of the synthetic ``gpu.yml``.

    Returns
    -------
    pathlib.Path
        The populated root, for chaining.
    """
    return build_tree(root, {"ci.yml": CLEAN_CI_YML, "gpu.yml": gpu_yml}, contexts=GPU_CONTEXTS)


def with_trigger(trigger_block: str) -> str:
    """Return the clean CI workflow with its trigger block replaced.

    Parameters
    ----------
    trigger_block
        Replacement text for everything between the name line and the jobs key.

    Returns
    -------
    str
        The rewritten workflow text.
    """
    _, jobs = CLEAN_CI_YML.split("jobs:", 1)
    return f"name: CI\n\n{trigger_block}\njobs:{jobs}"


# --------------------------------------------------------------------------
# A1 — filter keys under the pull-request-side events
# --------------------------------------------------------------------------


def test_a1_passes_on_the_real_ci_yml() -> None:
    """A1 is clean on today's real workflow, whose push trigger already carries a branch filter."""
    assert cc.check_trigger_filters(REPO_ROOT) == []


def test_a1_fails_on_a_paths_filter_under_a_pull_request_trigger(tmp_path: pathlib.Path) -> None:
    """A paths filter under the pull-request trigger is flagged, naming file, key, event and contexts."""
    root = build_tree(tmp_path, {"ci.yml": with_trigger("on:\n  pull_request:\n    paths: ['src/**']\n")})
    violations = cc.check_trigger_filters(root)
    assert len(violations) == 1, violations
    message = violations[0]
    assert "ci.yml" in message
    assert "paths" in message
    assert "pull_request" in message
    assert "Lint (pre-commit)" in message and "Tests (pytest)" in message


def test_a1_ignores_a_branch_filter_under_the_push_trigger(tmp_path: pathlib.Path) -> None:
    """A push-side branch filter is not inspected — it cannot leave a pull-request check pending."""
    root = build_tree(tmp_path, {"ci.yml": with_trigger("on:\n  pull_request:\n  push:\n    branches: [main]\n")})
    assert cc.check_trigger_filters(root) == []
    assert "push" not in cc.A1_INSPECTED_EVENTS


def test_a1_skips_a_workflow_that_produces_no_required_context(tmp_path: pathlib.Path) -> None:
    """A workflow whose jobs produce no required context is not inspected at all."""
    unrelated = "name: Canary\n\non:\n  pull_request:\n    paths: ['src/**']\n\njobs:\n  canary:\n    steps: []\n"
    root = build_tree(tmp_path, {"ci.yml": CLEAN_CI_YML, "canary.yml": unrelated})
    assert cc.check_trigger_filters(root) == []


def test_a1_skips_a_reviewed_exception_and_every_entry_states_a_reason(tmp_path: pathlib.Path) -> None:
    """A file in the reviewed-exceptions table is skipped, and each table entry carries a stated reason."""
    filtered = with_trigger("on:\n  pull_request:\n    branches: [main]\n")
    root = build_tree(tmp_path, {"ci.yml": filtered})
    assert cc.check_trigger_filters(root) != []
    assert cc.check_trigger_filters(root, exceptions={"ci.yml": "reviewed for this test"}) == []
    for filename, reason in cc.A1_REVIEWED_EXCEPTIONS.items():
        assert len(reason) > 40, f"{filename} exception must state why, not merely exist"


def test_a1_reports_a_workflow_whose_trigger_block_cannot_be_resolved(tmp_path: pathlib.Path) -> None:
    """A required-check workflow with no resolvable trigger block is a violation, not a pass."""
    root = build_tree(tmp_path, {"ci.yml": "name: CI\n\njobs:" + CLEAN_CI_YML.split("jobs:", 1)[1]})
    violations = cc.check_trigger_filters(root)
    assert len(violations) == 1, violations
    assert "trigger" in violations[0].lower()


def test_a1_tolerates_a_bare_list_trigger_block(tmp_path: pathlib.Path) -> None:
    """A trigger block written as a bare list carries no filters and must not raise or be flagged."""
    root = build_tree(tmp_path, {"ci.yml": with_trigger("on: [push, pull_request]\n")})
    assert cc.check_trigger_filters(root) == []


def test_a1_tolerates_a_workflow_document_with_no_jobs_key(tmp_path: pathlib.Path) -> None:
    """A workflow document with no jobs key is handled without raising and without being flagged."""
    root = build_tree(tmp_path, {"ci.yml": CLEAN_CI_YML, "stub.yml": "name: Stub\n\non:\n  pull_request:\n"})
    assert cc.check_trigger_filters(root) == []


# --------------------------------------------------------------------------
# A2 — the base-repo-context pull-request trigger
# --------------------------------------------------------------------------


def test_a2_passes_on_the_real_workflows_directory() -> None:
    """The one excepted workflow still meets both asserted conditions on today's tree.

    This is not a vacuous pass: ``integrity.yml`` carries the banned trigger, so
    this call exercises the exception AND its two condition assertions against
    the real file rather than against a fixture.
    """
    assert cc.check_base_repo_context_trigger(REPO_ROOT) == []


def test_a2_fails_when_the_token_appears_only_inside_a_comment(tmp_path: pathlib.Path) -> None:
    """The scan is over raw text, so an occurrence in a comment is caught."""
    commented = f"name: CI\n\n# never use {cc.BASE_REPO_CONTEXT_TRIGGER} here\non:\n  pull_request:\n\njobs: {{}}\n"
    root = build_tree(tmp_path, {"ci.yml": commented})
    violations = cc.check_base_repo_context_trigger(root)
    assert len(violations) == 1, violations
    assert "ci.yml" in violations[0]


# --------------------------------------------------------------------------
# A2's one reviewed exception, and the two conditions asserted behind it
# --------------------------------------------------------------------------

# The real excepted workflow, read from the tree it ships in, so these fixtures
# drift with it rather than describing a file that no longer exists.
EXCEPTED_WORKFLOW = "integrity.yml"


def excepted_workflow_text() -> str:
    """Return the real excepted workflow's text.

    Returns
    -------
    str
        The contents of the repository's own excepted workflow file.
    """
    return (REPO_ROOT / cc.WORKFLOWS_DIR / EXCEPTED_WORKFLOW).read_text()


def test_a2_honours_the_exception_for_the_real_conforming_workflow(tmp_path: pathlib.Path) -> None:
    """The real excepted workflow passes A2: the trigger is allowed and both conditions hold."""
    root = build_tree(tmp_path, {"ci.yml": CLEAN_CI_YML, EXCEPTED_WORKFLOW: excepted_workflow_text()})
    assert cc.check_base_repo_context_trigger(root) == []


def test_a2_fails_when_the_excepted_workflow_grants_a_write_scope(tmp_path: pathlib.Path) -> None:
    """Widening the excepted workflow's permissions block fails condition (iii)."""
    widened = excepted_workflow_text().replace(
        "permissions:\n  contents: read\n", "permissions:\n  contents: read\n  pull-requests: write\n", 1
    )
    root = build_tree(tmp_path, {"ci.yml": CLEAN_CI_YML, EXCEPTED_WORKFLOW: widened})
    violations = cc.check_base_repo_context_trigger(root)
    assert len(violations) == 1, violations
    assert EXCEPTED_WORKFLOW in violations[0]
    assert "permissions" in violations[0]


def test_a2_fails_when_a_job_in_the_excepted_workflow_widens_the_permissions(tmp_path: pathlib.Path) -> None:
    """A job-level block REPLACES the workflow-level one, so it is asserted too.

    Without this, the workflow-level block could read clean while the job that
    actually runs held a write scope -- the whole exception defeated by one
    correctly-indented key.
    """
    widened = excepted_workflow_text().replace(
        "    timeout-minutes: 10\n",
        "    timeout-minutes: 10\n    permissions:\n      contents: write\n",
        1,
    )
    root = build_tree(tmp_path, {"ci.yml": CLEAN_CI_YML, EXCEPTED_WORKFLOW: widened})
    violations = cc.check_base_repo_context_trigger(root)
    assert len(violations) == 1, violations
    assert "job `integrity`" in violations[0]


def test_a2_fails_when_the_excepted_workflows_checkout_carries_a_ref_input(tmp_path: pathlib.Path) -> None:
    """A checkout with an explicit reference fails condition (i) -- base-ref checkout only."""
    with_ref = excepted_workflow_text().replace(
        "        uses: actions/checkout@93cb6efe18208431cddfb8368fd83d5badbf9bfd  # v5.0.1\n",
        "        uses: actions/checkout@93cb6efe18208431cddfb8368fd83d5badbf9bfd  # v5.0.1\n"
        "        with:\n          ref: ${{ github.event.pull_request.head.sha }}\n",
        1,
    )
    root = build_tree(tmp_path, {"ci.yml": CLEAN_CI_YML, EXCEPTED_WORKFLOW: with_ref})
    violations = cc.check_base_repo_context_trigger(root)
    assert len(violations) == 1, violations
    assert EXCEPTED_WORKFLOW in violations[0]
    assert cc.CHECKOUT_REF_INPUT in violations[0]


def test_a2_still_fails_on_the_banned_trigger_in_a_different_file(tmp_path: pathlib.Path) -> None:
    """The exception is keyed on a file path, not a global mute.

    The same trigger in any other workflow still fails the build, which is what
    distinguishes a named exception from having turned the assertion off.
    """
    offender = f"name: Other\n\non:\n  {cc.BASE_REPO_CONTEXT_TRIGGER}:\n    branches: [main]\n\njobs: {{}}\n"
    root = build_tree(
        tmp_path,
        {"ci.yml": CLEAN_CI_YML, EXCEPTED_WORKFLOW: excepted_workflow_text(), "other.yml": offender},
    )
    violations = cc.check_base_repo_context_trigger(root)
    assert len(violations) == 1, violations
    assert "other.yml" in violations[0]
    assert EXCEPTED_WORKFLOW not in violations[0]


def test_a2_exception_table_is_injectable_in_both_directions(tmp_path: pathlib.Path) -> None:
    """An empty table restores the blanket ban; a table naming a file excepts only that file."""
    root = build_tree(tmp_path, {"ci.yml": CLEAN_CI_YML, EXCEPTED_WORKFLOW: excepted_workflow_text()})
    assert cc.check_base_repo_context_trigger(root, exceptions={}) != []
    assert cc.check_base_repo_context_trigger(root, exceptions={EXCEPTED_WORKFLOW: "reviewed here"}) == []


def test_every_a2_exception_entry_states_its_reason_and_names_its_three_conditions() -> None:
    """Each entry says WHY, and says which conditions are asserted and which reviewed."""
    assert cc.A2_REVIEWED_EXCEPTIONS, "the table must never be silently emptied"
    for filename, reason in cc.A2_REVIEWED_EXCEPTIONS.items():
        assert len(reason) > 40, f"{filename} exception must state why, not merely exist"
        assert reason.count("ASSERTED") == 2, f"{filename} must name exactly the two asserted conditions"
        assert "REVIEWED" in reason, f"{filename} must say which condition is reviewed rather than enforced"


# --------------------------------------------------------------------------
# A3 — the pinned workflow name
# --------------------------------------------------------------------------


def test_a3_passes_when_the_name_matches_the_pinned_literal(tmp_path: pathlib.Path) -> None:
    """The pinned literal matches today's declared workflow name byte for byte."""
    assert cc.check_workflow_name_pin(build_tree(tmp_path, {"ci.yml": CLEAN_CI_YML})) == []
    assert cc.check_workflow_name_pin(REPO_ROOT) == []


def test_a3_fails_on_a_case_changed_name(tmp_path: pathlib.Path) -> None:
    """The comparison does not case-fold, so a case change is drift."""
    root = build_tree(tmp_path, {"ci.yml": CLEAN_CI_YML.replace("name: CI", "name: Ci", 1)})
    violations = cc.check_workflow_name_pin(root)
    assert violations and "Ci" in violations[0], violations


def test_a3_fails_on_a_whitespace_padded_name(tmp_path: pathlib.Path) -> None:
    """The comparison does not strip, so a padded name is drift."""
    root = build_tree(tmp_path, {"ci.yml": CLEAN_CI_YML.replace("name: CI", 'name: "CI "', 1)})
    assert cc.check_workflow_name_pin(root) != []


def test_a3_fails_when_a_workflow_run_trigger_names_an_undeclared_workflow(tmp_path: pathlib.Path) -> None:
    """A workflow-run trigger listing a name no workflow declares is a silent-breakage risk."""
    consumer = "name: Release\n\non:\n  workflow_run:\n    workflows: [Continuous Integration]\n\njobs: {}\n"
    root = build_tree(tmp_path, {"ci.yml": CLEAN_CI_YML, "release-please.yml": consumer})
    violations = cc.check_workflow_name_pin(root)
    assert violations and "Continuous Integration" in violations[0], violations


# --------------------------------------------------------------------------
# A4 — allowlist reconciliation against release-please's own configuration
# --------------------------------------------------------------------------


def test_a4_passes_when_the_allowlist_set_equals_the_release_configuration(tmp_path: pathlib.Path) -> None:
    """Changelog path plus extra files plus the manifest path is exactly the allowlist."""
    assert cc.check_fast_path_allowlist(build_tree(tmp_path, {"ci.yml": CLEAN_CI_YML})) == []
    assert cc.check_fast_path_allowlist(REPO_ROOT) == []


def test_a4_is_decided_by_set_equality_not_by_written_order(tmp_path: pathlib.Path) -> None:
    """Reordering the allowlist does not change the verdict."""
    root = build_tree(tmp_path, {"ci.yml": CLEAN_CI_YML}, allowlist=tuple(reversed(DEFAULT_ALLOWLIST)))
    assert cc.check_fast_path_allowlist(root) == []


def test_a4_fails_when_a_fourth_extra_file_is_added(tmp_path: pathlib.Path) -> None:
    """A fourth extra file fails the build with both sets printed and a named remedy."""
    root = build_tree(tmp_path, {"ci.yml": CLEAN_CI_YML}, extra_files=("docs/source/conf.py", "docs/source/api.rst"))
    violations = cc.check_fast_path_allowlist(root)
    assert len(violations) == 1, violations
    assert "docs/source/api.rst" in violations[0]
    assert "fast path" in violations[0]


# --------------------------------------------------------------------------
# A5 / A6 — job-level conditions on, and reachable from, a required context
# --------------------------------------------------------------------------


def test_a5_passes_on_the_real_ci_yml() -> None:
    """No job producing a required context carries a job-level condition today."""
    assert cc.check_job_level_conditions(REPO_ROOT) == []


def test_a5_fails_when_a_required_context_job_carries_a_job_level_condition(tmp_path: pathlib.Path) -> None:
    """A skipped job satisfies its required context, so a job-level condition on one is a violation."""
    conditioned = CLEAN_CI_YML.replace(
        "    name: Lint (pre-commit)\n", "    name: Lint (pre-commit)\n    if: false\n", 1
    )
    root = build_tree(tmp_path, {"ci.yml": conditioned})
    violations = cc.check_job_level_conditions(root)
    assert len(violations) == 1, violations
    assert "Lint (pre-commit)" in violations[0]


def test_a6_fails_when_a_required_context_job_depends_on_a_conditional_job(tmp_path: pathlib.Path) -> None:
    """A skipped dependency skips the dependent, which also counts as satisfying."""
    workflow = CLEAN_CI_YML.replace(
        "    name: Lint (pre-commit)\n", "    name: Lint (pre-commit)\n    if: false\n", 1
    ).replace("    name: Tests (pytest)\n", "    name: Tests (pytest)\n    needs: [lint]\n", 1)
    root = build_tree(tmp_path, {"ci.yml": workflow})
    violations = cc.check_conditional_dependencies(root)
    assert len(violations) == 1, violations
    assert "Tests (pytest)" in violations[0] and "lint" in violations[0]


def test_a6_passes_when_the_dependency_carries_no_condition(tmp_path: pathlib.Path) -> None:
    """Depending on an unconditional job is fine and must not be flagged."""
    workflow = CLEAN_CI_YML.replace("    name: Tests (pytest)\n", "    name: Tests (pytest)\n    needs: [lint]\n", 1)
    assert cc.check_conditional_dependencies(build_tree(tmp_path, {"ci.yml": workflow})) == []


# --------------------------------------------------------------------------
# A7 — least privilege on the lint job
# --------------------------------------------------------------------------


def test_a7_fails_when_the_lint_job_grants_a_write_scope(tmp_path: pathlib.Path) -> None:
    """Any write scope in the lint job's permissions block fails the build."""
    workflow = CLEAN_CI_YML.replace("      contents: read\n", "      contents: write\n      pull-requests: write\n", 1)
    root = build_tree(tmp_path, {"ci.yml": workflow})
    violations = cc.check_lint_least_privilege(root, pending={})
    assert len(violations) == 2, violations
    assert any("contents" in v for v in violations) and any("pull-requests" in v for v in violations)


def test_a7_passes_when_the_lint_job_is_narrowed_to_read(tmp_path: pathlib.Path) -> None:
    """A read-only permissions block on the lint job is clean."""
    assert cc.check_lint_least_privilege(build_tree(tmp_path, {"ci.yml": CLEAN_CI_YML}), pending={}) == []


def test_a7_fails_when_the_lint_job_declares_no_permissions_block(tmp_path: pathlib.Path) -> None:
    """An absent permissions block inherits the repository default and is a violation."""
    workflow = CLEAN_CI_YML.replace("    permissions:\n      contents: read\n", "", 1)
    violations = cc.check_lint_least_privilege(build_tree(tmp_path, {"ci.yml": workflow}), pending={})
    assert len(violations) == 1, violations
    assert "permissions" in violations[0]


def test_a7_pending_register_suppresses_a_scheduled_narrowing_and_names_its_removal(tmp_path: pathlib.Path) -> None:
    """A pending entry suppresses A7 for one context only, and every entry names the plan that removes it."""
    workflow = CLEAN_CI_YML.replace("      contents: read\n", "      contents: write\n", 1)
    root = build_tree(tmp_path, {"ci.yml": workflow})
    assert cc.check_lint_least_privilege(root, pending={cc.LINT_CONTEXT: "scheduled"}) == []
    assert cc.check_lint_least_privilege(root, pending={"Some Other Context": "scheduled"}) != []
    for context, reason in cc.A7_PENDING_NARROWINGS.items():
        assert "CI-16" in reason, context
        assert "13-04" in reason and "13-07" in reason, context


# --------------------------------------------------------------------------
# A8 — the GPU job's post-install capability assertion
# --------------------------------------------------------------------------


def test_a8_passes_on_the_real_gpu_yml() -> None:
    """The shipped `gpu.yml` asserts its capability after its own install and before the suite."""
    assert cc.check_gpu_post_install_capability_assert(REPO_ROOT) == []


def test_a8_fails_when_the_capability_assert_is_absent(tmp_path: pathlib.Path) -> None:
    """Reproduce the 2026-08-17 pre-state: install, then suite, with nothing in between."""
    without = GPU_JOB_HEAD + GPU_INSTALL_LINE + GPU_SUITE_LINE + GPU_JOB_TAIL
    violations = cc.check_gpu_post_install_capability_assert(build_gpu_tree(tmp_path, without))
    assert len(violations) == 1, violations
    assert cc.GPU_CAPABILITY_SENTINEL in violations[0]


def test_a8_fails_when_the_assert_precedes_the_install(tmp_path: pathlib.Path) -> None:
    """Position is the property: an assertion the install then invalidates witnesses nothing."""
    reordered = GPU_JOB_HEAD + GPU_ASSERT_BLOCK + GPU_INSTALL_LINE + GPU_SUITE_LINE + GPU_JOB_TAIL
    violations = cc.check_gpu_post_install_capability_assert(build_gpu_tree(tmp_path, reordered))
    assert len(violations) == 1, violations
    assert "BEFORE the" in violations[0]


def test_a8_fails_when_the_capability_guard_is_inverted(tmp_path: pathlib.Path) -> None:
    """A guard that fires on the healthy state is worse than no guard at all."""
    inverted = CLEAN_GPU_YML.replace("if not opt.is_gpu_available():", "if opt.is_gpu_available():", 1)
    violations = cc.check_gpu_post_install_capability_assert(build_gpu_tree(tmp_path, inverted))
    assert len(violations) == 1, violations
    assert "inverted" in violations[0]


def test_a8_fails_when_the_capability_step_is_neutered_by_continue_on_error(tmp_path: pathlib.Path) -> None:
    """`continue-on-error` turns a failed assertion back into a green required context."""
    neutered = CLEAN_GPU_YML.replace(
        "      - name: Run GPU tests in the RAPIDS container (podman)\n",
        "      - name: Run GPU tests in the RAPIDS container (podman)\n        continue-on-error: true\n",
        1,
    )
    violations = cc.check_gpu_post_install_capability_assert(build_gpu_tree(tmp_path, neutered))
    assert len(violations) == 1, violations
    assert "continue-on-error" in violations[0]


def test_a8_is_vacuous_when_no_committed_ruleset_requires_the_gpu_context(tmp_path: pathlib.Path) -> None:
    """A8 is inert where no committed payload requires the GPU context, and says so in both directions.

    A copy of `check_ci_config.py` in the sibling repository would meet no GPU
    context there, so a copy of this assertion must be a no-op. (Byte-identity
    across the two repositories is D-04's rule for `ruleset_lib.py`, not for this
    file, which has carried extra assertions since 17-02.)
    The same branch makes A8 inert under the audited `main.no-gpu.json` lab-outage
    override (D-17-04). Pinned in BOTH directions: the identical tree with the
    context required does report that nothing produces it.
    """
    without_gpu = build_tree(tmp_path / "sibling", {"ci.yml": CLEAN_CI_YML}, contexts=DEFAULT_CONTEXTS)
    assert cc.check_gpu_post_install_capability_assert(without_gpu) == []

    with_gpu = build_tree(tmp_path / "requiring", {"ci.yml": CLEAN_CI_YML}, contexts=GPU_CONTEXTS)
    violations = cc.check_gpu_post_install_capability_assert(with_gpu)
    assert len(violations) == 1, violations
    assert "nothing produces it" in violations[0]


# --------------------------------------------------------------------------
# A9 — the GPU corridor's artefacts: constrained install, relocated digest
# --------------------------------------------------------------------------


def test_a9_is_vacuous_when_no_committed_ruleset_requires_the_gpu_context(tmp_path: pathlib.Path) -> None:
    """A9 is inert where no committed payload requires the GPU context, pinned in BOTH directions.

    The sibling repository has no GPU stack, no constraints file and no digest
    artefact — so a copy of this assertion must be a no-op there rather than a
    demand for files that repository has no reason to carry. (Byte-identity is
    D-04's rule for `ruleset_lib.py`, not for this file.) The same branch makes A9 inert under
    the audited `main.no-gpu.json` lab-outage override (D-17-04).
    """
    sibling = build_tree(tmp_path / "sibling", {"ci.yml": CLEAN_CI_YML}, contexts=DEFAULT_CONTEXTS)
    assert cc.check_gpu_corridor_artifacts(sibling) == []

    requiring = build_tree(tmp_path / "requiring", {"ci.yml": CLEAN_CI_YML}, contexts=GPU_CONTEXTS)
    violations = cc.check_gpu_corridor_artifacts(requiring)
    assert violations, "the identical tree with the context required must NOT be clean"
    assert any("nothing produces it" in v for v in violations), violations


def test_a9_fails_when_the_container_install_is_unconstrained(tmp_path: pathlib.Path) -> None:
    """The seam is the install, not the file: an unconsumed constraints file constrains nothing.

    This reproduces the 2026-08-17 mechanism exactly. The container installs
    `.[dev]` and never `[cuda12]`, so neither the GPU extras' ceiling nor conda's
    numba reaches this resolve, and pip does not account for already-installed
    packages — numpy walked to 2.3.5 and numba refused it (D-17-10, spike 004).
    """
    tree = build_corridor_tree(tmp_path, CLEAN_GPU_YML)
    violations = cc.check_gpu_corridor_artifacts(tree)
    assert len(violations) == 1, violations
    assert cc.GPU_CONSTRAINTS_PATH in violations[0]
    assert "-c" in violations[0]


def test_a9_fails_when_the_constraints_file_is_missing_or_holds_no_requirement(
    tmp_path: pathlib.Path,
) -> None:
    """A wired `-c` pointing at nothing, or at comments only, is a corridor with no wall."""
    absent = cc.check_gpu_corridor_artifacts(build_corridor_tree(tmp_path / "absent", constraints=None))
    assert len(absent) == 1, absent
    assert "missing" in absent[0]

    empty = cc.check_gpu_corridor_artifacts(
        build_corridor_tree(tmp_path / "commented", constraints="# numpy >= 2.2, < 2.3\n")
    )
    assert len(empty) == 1, empty
    assert "no requirement" in empty[0]


def test_a9_fails_on_an_absent_empty_multiline_or_malformed_digest(tmp_path: pathlib.Path) -> None:
    """The pin must be one full-length digest or the job must not start at all.

    Four shapes, each a real failure mode of the `sed`-driven write-back in
    `gpu-image-refresh.yml`: no file, a truncated write, an appended second line,
    and a value that is not a full-length digest (a moving tag being the case
    that matters — T-17-19).
    """
    cases = {
        "absent": ("missing", None),
        "empty": ("", ""),
        "multiline": ("more than one", pinned_reference() + "\n" + pinned_reference() + "\n"),
        "tagged": ("full-length", f"{GPU_IMAGE_REPOSITORY}:cuda12-latest\n"),
        "short": ("full-length", f"{GPU_IMAGE_REPOSITORY}@sha256:a1b2c3\n"),
        # A prefix-valid value with a tampered SUFFIX. This case is why the check
        # uses `fullmatch` and not `match`: a mutation sweep found `match` survived
        # every other case here, and a suffix-extended reference is exactly what a
        # tampering attempt looks like (T-17-18).
        "suffixed": ("full-length", pinned_reference() + "-tampered\n"),
    }
    for name, (token, body) in cases.items():
        tree = build_corridor_tree(tmp_path / name, digest=body)
        violations = cc.check_gpu_corridor_artifacts(tree)
        assert len(violations) == 1, (name, violations)
        assert cc.GPU_DIGEST_PATH in violations[0], (name, violations)
        assert token in violations[0], (name, violations)


def test_a9_fails_on_an_inlined_digest_but_not_on_a_comment_mentioning_one(
    tmp_path: pathlib.Path,
) -> None:
    """A9 scans the PARSED job, so a header bullet describing the pin cannot fail the check it describes.

    Scanning raw text here would be the recurring trap of this phase: a search for
    a removed thing matches the prose documenting its removal. The parser strips
    comments by construction, which is the whole reason the scan is written this
    way.
    """
    inlined = CORRIDOR_GPU_YML.replace(
        '          podman run --rm "$GPU_IMAGE" bash -lc \'\n',
        f'          podman run --rm "{pinned_reference()}" bash -lc \'\n',
        1,
    )
    assert inlined != CORRIDOR_GPU_YML, "the inlining fixture did not apply"
    violations = cc.check_gpu_corridor_artifacts(build_corridor_tree(tmp_path / "inlined", inlined))
    assert len(violations) == 1, violations
    assert "inlined" in violations[0] or "inline" in violations[0], violations

    commented = CORRIDOR_GPU_YML.replace(
        "jobs:\n",
        f"# the pin lives in a versioned artefact and reads {pinned_reference()}\njobs:\n",
        1,
    )
    assert commented != CORRIDOR_GPU_YML, "the comment fixture did not apply"
    assert cc.check_gpu_corridor_artifacts(build_corridor_tree(tmp_path / "commented", commented)) == []


def test_a9_passes_on_the_real_repository() -> None:
    """The shipped tree carries the corridor: a constrained install and a relocated, well-formed pin."""
    assert cc.check_gpu_corridor_artifacts(REPO_ROOT) == []
    assert (REPO_ROOT / cc.GPU_CONSTRAINTS_PATH).is_file()
    assert (REPO_ROOT / cc.GPU_DIGEST_PATH).is_file()
    # The fixtures above build their references from a test-side literal on
    # purpose; this is the assertion that pins the production constant to it.
    assert cc.GPU_IMAGE_REPOSITORY == GPU_IMAGE_REPOSITORY


# --------------------------------------------------------------------------
# A10 — every `bash -c` payload reaches the shell as exactly one argument
# --------------------------------------------------------------------------

# The WINDOWS 30 shape, reduced to its essentials and kept as a LITERAL rather
# than read from git history: a container payload passed as one single-quoted
# argument, with one apostrophe in the prose. `BROKEN_PAYLOAD_YML` is the
# pre-state, `CLEAN_PAYLOAD_YML` the post-state, and they differ in exactly that
# apostrophe — which is what makes the pair a polarity test rather than two
# unrelated fixtures.
CLEAN_PAYLOAD_YML = """\
name: Payload

on:
  pull_request:

jobs:
  lint:
    name: Lint (pre-commit)
    runs-on: ubuntu-latest
    permissions:
      contents: read
    steps:
      - name: Run the payload in a container
        run: |
          podman run --rm image \\
            bash -lc '
              set -euo pipefail
              # The container numpy is held inside the corridor
              pip install -c constraints.txt -e .[dev]
            '
  tests:
    name: Tests (pytest)
    runs-on: ubuntu-latest
    steps:
      - run: echo test
"""

BROKEN_PAYLOAD_YML = CLEAN_PAYLOAD_YML.replace("# The container numpy", "# The container's numpy", 1)


def test_a10_passes_on_the_real_workflows_directory() -> None:
    """The shipped tree's two `bash -lc` payloads each reach the shell as one argument."""
    assert cc.check_bash_dash_c_quoting(REPO_ROOT) == []


def test_a10_fails_on_an_apostrophe_inside_a_single_quoted_payload(tmp_path: pathlib.Path) -> None:
    """Reproduce WINDOWS 30: one apostrophe in prose closes the payload and the rest runs on the host.

    TWO violations, and both are the truth rather than a duplicate. A single
    apostrophe leaves an ODD number of quote characters in the body, so the
    payload breaks out AND the closing quote of the invocation then opens a
    region nothing closes. The real pre-state (`5c4ab66`, `f3b16cf`) reports the
    same pair, and `bash` itself reports the second half as `unexpected EOF while
    looking for matching '`.
    """
    root = build_tree(tmp_path, {"ci.yml": CLEAN_CI_YML, "payload.yml": BROKEN_PAYLOAD_YML})
    violations = cc.check_bash_dash_c_quoting(root)
    assert len(violations) == 2, violations
    assert any("unterminated" in v for v in violations), violations
    broke = [v for v in violations if "breaks out of its own `bash -c` quoting" in v]
    assert len(broke) == 1, violations
    assert "s numpy is held inside the corridor" in broke[0]


def test_a10_passes_on_the_same_payload_without_the_apostrophe(tmp_path: pathlib.Path) -> None:
    """The polarity partner: the identical fixture minus the apostrophe is clean."""
    root = build_tree(tmp_path, {"ci.yml": CLEAN_CI_YML, "payload.yml": CLEAN_PAYLOAD_YML})
    assert cc.check_bash_dash_c_quoting(root) == []


def test_a10_reports_an_unparseable_payload_rather_than_skipping_it(tmp_path: pathlib.Path) -> None:
    """A check that cannot verify must not answer `safe`; an unquoted payload is a violation."""
    indirect = CLEAN_PAYLOAD_YML.replace(
        "            bash -lc '\n              set -euo pipefail\n"
        "              # The container numpy is held inside the corridor\n"
        "              pip install -c constraints.txt -e .[dev]\n            '\n",
        "            bash -lc $SCRIPT\n",
        1,
    )
    assert indirect != CLEAN_PAYLOAD_YML
    root = build_tree(tmp_path, {"ci.yml": CLEAN_CI_YML, "payload.yml": indirect})
    violations = cc.check_bash_dash_c_quoting(root)
    assert len(violations) == 1, violations
    assert "not a literal quoted string" in violations[0]


def test_a10_reports_an_unterminated_payload_quote(tmp_path: pathlib.Path) -> None:
    """An opening quote with no closer cannot be verified either, and says so by name."""
    unterminated = CLEAN_PAYLOAD_YML.replace("            '\n", "            \n", 1)
    assert unterminated != CLEAN_PAYLOAD_YML
    root = build_tree(tmp_path, {"ci.yml": CLEAN_CI_YML, "payload.yml": unterminated})
    violations = cc.check_bash_dash_c_quoting(root)
    assert violations, violations
    assert any("unterminated" in v for v in violations), violations


def test_a10_does_not_fire_on_a_double_quoted_payload_or_a_script_file(tmp_path: pathlib.Path) -> None:
    """No false positive on the two shapes that have no single-quote trap to spring."""
    double_quoted = CLEAN_PAYLOAD_YML.replace("bash -lc '", 'bash -lc "', 1).replace(
        "            '\n", '            "\n', 1
    )
    file_based = CLEAN_PAYLOAD_YML.replace(
        "            bash -lc '\n              set -euo pipefail\n"
        "              # The container numpy is held inside the corridor\n"
        "              pip install -c constraints.txt -e .[dev]\n            '\n",
        "            bash -l /work/.github/scripts/payload.sh\n",
        1,
    )
    assert file_based != CLEAN_PAYLOAD_YML
    for name, text in (("double", double_quoted), ("file", file_based)):
        root = build_tree(tmp_path / name, {"ci.yml": CLEAN_CI_YML, "payload.yml": text})
        assert cc.check_bash_dash_c_quoting(root) == [], name


def test_a10_ignores_a_bash_dash_c_that_is_only_mentioned_in_prose(tmp_path: pathlib.Path) -> None:
    """Parsed structure, not raw text: a `bash -lc` inside a comment is not an invocation."""
    mentioned = CLEAN_PAYLOAD_YML.replace(
        "      - name: Run the payload in a container\n        run: |\n",
        "      - name: Run the payload in a container\n        run: |\n"
        "          # we launch it with bash -lc 'like this' and it isn't an invocation\n",
        1,
    )
    assert mentioned != CLEAN_PAYLOAD_YML
    root = build_tree(tmp_path, {"ci.yml": CLEAN_CI_YML, "payload.yml": mentioned})
    assert cc.check_bash_dash_c_quoting(root) == []


def test_a10_is_not_gated_on_the_gpu_context(tmp_path: pathlib.Path) -> None:
    """Unlike A8 and A9, A10's class is generic — it must fire where no GPU context is required.

    A8 and A9 return early when no committed ruleset requires `Tests (pytest,
    GPU)`, because the corridor they describe belongs to this repository. Gating
    A10 the same way would make it vacuous in the sibling repository and in every
    workflow that is not the GPU one, which is the opposite of the point.
    """
    without_gpu = build_tree(
        tmp_path / "sibling", {"ci.yml": CLEAN_CI_YML, "payload.yml": BROKEN_PAYLOAD_YML}, contexts=DEFAULT_CONTEXTS
    )
    assert cc.check_gpu_post_install_capability_assert(without_gpu) == []
    assert cc.check_gpu_corridor_artifacts(without_gpu) == []
    generic = cc.check_bash_dash_c_quoting(without_gpu)
    assert any("breaks out of its own `bash -c` quoting" in v for v in generic), generic


def test_a10_is_reached_by_the_driver(tmp_path: pathlib.Path) -> None:
    """`run_all` carries A10, so the break-out reddens the build rather than only the helper."""
    root = build_tree(tmp_path, {"ci.yml": CLEAN_CI_YML, "payload.yml": BROKEN_PAYLOAD_YML})
    violations, _ = cc.run_all(root, pending={})
    assert any("breaks out of its own `bash -c` quoting" in v for v in violations), violations


# --------------------------------------------------------------------------
# Driver behaviour
# --------------------------------------------------------------------------


def test_violations_accumulate_rather_than_exiting_on_the_first(tmp_path: pathlib.Path) -> None:
    """A file with three problems produces three annotations and one exit."""
    broken = with_trigger("on:\n  pull_request:\n    paths: ['src/**']\n    branches: [main]\n").replace(
        "    name: Lint (pre-commit)\n", "    name: Lint (pre-commit)\n    if: false\n", 1
    )
    root = build_tree(tmp_path, {"ci.yml": broken})
    violations, _ = cc.run_all(root, pending={})
    assert len(violations) >= 3, violations


def test_run_all_is_clean_on_the_real_repository_and_counts_its_workflows() -> None:
    """The self-test passes on the tree it ships in, and reports how many workflows it inspected."""
    violations, inspected = cc.run_all(REPO_ROOT)
    assert violations == [], violations
    assert inspected == len(sorted((REPO_ROOT / cc.WORKFLOWS_DIR).glob("*.yml")))
