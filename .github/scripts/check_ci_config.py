#!/usr/bin/env python3
# .github/scripts/check_ci_config.py
# Source: Phase 13 CONTEXT.md D-05/D-08/D-11 + 13-RESEARCH.md R5/R6 — the CI-12 self-test
"""Fail the build on the CI-configuration regressions that would break the gate.

Invoked from the Lint (pre-commit) CI job as ``python
.github/scripts/check_ci_config.py``. It takes NO arguments, accumulates every
violation, emits each as a GitHub ``::error::`` annotation and exits 1 once;
exits 0 and prints one trailing OK line when the tree is clean.

Ten assertions, each named by the identifier later plans and the adversarial
suite refer to:

  A1  No filter key under a pull-request-side trigger on a workflow producing a
      required status-check context.
  A2  The base-repo-context pull-request trigger appears nowhere under the
      workflows directory, except in the ONE reviewed file named in
      ``A2_REVIEWED_EXCEPTIONS`` -- and there only while the two statically
      checkable conditions of that exception still hold.
  A3  The CI workflow's declared name still equals the pinned literal, and every
      workflow-run trigger names a workflow some file actually declares.
  A4  The fast-path allowlist still set-equals release-please's own changelog
      path, extra files and manifest path.
  A5  No job whose declared name is a required context carries a job-level
      condition.
  A6  No such job depends on a job that itself carries a job-level condition.
  A7  The lint job declares a permissions block granting no write scope.
  A8  The GPU job asserts its own GPU capability AFTER its dependency install
      and BEFORE the suite, in the same step body, with the sense not inverted
      and the failure not neutered.
  A9  The GPU corridor's two artefacts exist, are wired in, and are the only
      place the image pin lives: the container install carries ``-c`` pointing
      at the constraints file, that file yields a requirement, the digest
      artefact holds exactly one full-length pinned reference, and no concrete
      digest is inlined back into the GPU job.
  A10 Every ``bash -c`` / ``bash -lc`` invocation in every step body hands its
      script to the shell as exactly ONE argument: the quoting round-trips, and
      no bare word follows the closing quote. A body that breaks out of its own
      quotes runs on the runner host instead of inside the container.

**Why A1 inspects the pull-request-side events and NOT push.** A required status
check comes from the event that produced it. A workflow skipped by a filter never
reports, so the pull request waits on a check that will never arrive -- but a
*push*-side branch filter cannot strand a *pull-request* check, because the
pull-request run is a different run with a different trigger. Both repos'
``ci.yml`` already carries ``branches:`` under ``push:``, deliberately and
correctly, so including push in A1's inspected set would fail this assertion on
day one against a configuration nobody wants changed. That is a false positive,
and the resolution is to narrow the rule with the reason written down -- here,
and pinned in both directions by ``test_check_ci_config.py`` -- rather than to
mute the check. Stating the converse plainly for the same reason: a schedule is
not a filter and a dispatch cannot suppress a pull-request check, so neither
event is ever inspected either.

**What this script does NOT defend against, stated so it is not assumed.** A5
and A6 make a job-level condition on a required context a build failure. They do
not make such a condition un-suppressible: this script runs from inside the lint
job, from the pull request's own checkout, so a head-ref edit reaches it -- an
``if:`` on ``lint`` skips the very job that would have caught it, and a skipped
conclusion satisfies the gate.

What closes that circularity is ``.github/workflows/integrity.yml`` (Phase 13
D-19 / D-23, DESIGN-DECISIONS entry 46): a required-context job triggered on the
base-repository pull-request event, whose definition and checker both come from
the protected branch while its check run attaches to the pull request's head sha.
It calls :func:`check_job_level_conditions` and
:func:`check_conditional_dependencies` -- these functions, not a second copy of
them -- against a tree whose rulesets come from the base checkout and whose
workflows come from the head. A5 and A6 remain the fast build-time signal; the
un-skippable evaluation of the same two rules lives there.

**Superseded, and named so a reader does not act on it:** this paragraph
previously credited the ruleset ``workflows`` rule (Phase 13 D-15) with pinning
``ci.yml``'s definition to the protected branch. That rule requires GitHub
Enterprise Cloud and is org/enterprise-scoped, so it is unavailable to these
repositories at any payload shape -- proved empirically and recorded as
DESIGN-DECISIONS entry 45. Entry 46 replaces it.

Assertion helpers each take a root path so they are testable against synthetic
trees; the module-level driver below calls them against this repository. The
driver executes at import, matching ``check_publish_gate.py``'s straight-line
shape and keeping the invocation contract argument-free -- see plan 13-03's
deferred-review note for the consequence when the tree is dirty.
"""

import json
import pathlib
import re
import sys
from collections.abc import Iterator
from typing import Any

import ruleset_lib
import yaml  # available on ubuntu-latest runner by default

# Resolved from this file rather than from the process working directory, so the
# self-test always inspects the repository it ships in. The invocation contract
# is unchanged: still no arguments.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

WORKFLOWS_DIR = ".github/workflows"
RULESETS_DIR = ".github/rulesets"
COMPOSITE_ACTION_PATH = ".github/actions/classify-changes/action.yml"
SELF_PATH = ".github/scripts/check_ci_config.py"
RELEASE_PLEASE_CONFIG_PATH = "release-please-config.json"
# `release-type: simple` with no version file in either repo, so the manifest is
# the only member of the allowlist release-please does not name in its config.
RELEASE_PLEASE_MANIFEST_PATH = ".release-please-manifest.json"

# Sub-keys of a triggering event that suppress a run. Presence of any of these
# under an inspected event means some pushes/pull requests produce no run at all.
FILTER_KEYS: frozenset[str] = frozenset({"paths", "paths-ignore", "branches", "branches-ignore", "tags", "tags-ignore"})

# The events A1 inspects: the pull-request event and its base-repo-context
# sibling. See the module docstring for why `push`, `schedule` and
# `workflow_dispatch` are all deliberately absent.
A1_INSPECTED_EVENTS: tuple[str, ...] = ("pull_request", "pull_request_target")

# Deliberate, reviewed exceptions to A1, keyed by workflow filename. Each value
# states WHY, so removing one is a decision rather than an oversight.
A1_REVIEWED_EXCEPTIONS: dict[str, str] = {
    "gpu.yml": (
        "Phase 13 D-11 scopes `Tests (pytest, GPU)` to a required context on the protected branch "
        "ONLY, mirroring how D-06 scopes the docs context. A `branches:` filter under its "
        "pull-request trigger is therefore the intended scope rather than drift: the context is not "
        "required on any branch the filter excludes, so no check is left pending."
    ),
    "integrity.yml": (
        "Phase 13 D-23 scopes `Integrity (base-ref)` to a required context on the protected branch "
        "ONLY, exactly as D-11 scopes the GPU context and D-06 the docs one. A `branches:` filter "
        "under its trigger is therefore the intended scope rather than drift: the context is not "
        "required on any branch the filter excludes, so no check is left pending. The deferred "
        "`develop/gsd` extension is a recorded cost — a skip that reaches the integration branch is "
        "caught by the promotion pull request into `main`."
    ),
}

# The base-repo-context pull-request trigger. Never legitimate in either repo
# EXCEPT for the one file named in A2_REVIEWED_EXCEPTIONS below: it runs with the
# base repository's secrets against a fork's code, and one of these repos
# attaches a self-hosted lab runner.
BASE_REPO_CONTEXT_TRIGGER = "pull_request_target"

# The ONE reviewed exception to A2, keyed by workflow filename. CI-12 as amended
# by Phase 13 D-19 permits this trigger for a single file, and only while three
# conditions hold. Two of them are STATICALLY ASSERTED below and fail the build;
# the third is a property of what the steps do rather than of a key that can be
# looked up, so it is carried here as a reviewed reason. An excepted file is
# therefore never simply skipped -- a workflow that drifts out of the asserted
# conditions fails the build rather than quietly keeping its exemption.
#
# This file is byte-identical across both repos by design (D-04), so adding or
# removing an entry is a paired cross-repo change.
A2_REVIEWED_EXCEPTIONS: dict[str, str] = {
    "integrity.yml": (
        "Phase 13 D-19, recorded as DESIGN-DECISIONS entry 46 and licensed by CI-12 as amended by "
        "plan 13-14. This is the anti-skip mechanism itself: the ONLY control on this account that "
        "can catch a job-level condition added to a required-context job, because the build-time "
        "self-test lives inside the job such an edit skips. Its check run attaches to the pull "
        "request's HEAD sha while the runner sees the BASE commit, measured by the D-21 probe. "
        "THREE conditions make it safe. (i) BASE-REF CHECKOUT ONLY -- ASSERTED here: no checkout "
        "step in that workflow may carry a `ref:` input. (ii) IT NEVER CHECKS OUT OR EXECUTES "
        "HEAD-REF CODE, with reading the head's workflow YAML as DATA through the contents API "
        "explicitly permitted, because that is the entire point of the job and its parser and rules "
        "both come from the protected branch -- this condition is REVIEWED, not asserted, since it "
        "is a property of what the steps do rather than of a lookupable key. (iii) THE WORKFLOW "
        "GRANTS EXACTLY A READ CONTENT SCOPE -- ASSERTED here, at workflow level and at job level "
        "both, because a job-level block REPLACES rather than narrows the workflow-level one."
    ),
}

# The exact permissions mapping conditions (iii) allows. Compared as a whole
# mapping rather than key-by-key: "grants nothing beyond a read content scope"
# has to mean the block carries nothing else at all, or an added key slips
# through as long as the contents key still reads `read`.
READ_CONTENT_ONLY_PERMISSIONS: dict[str, str] = {"contents": "read"}

CHECKOUT_ACTION = "actions/checkout"
CHECKOUT_REF_INPUT = "ref"

CI_WORKFLOW_FILENAME = "ci.yml"
CI_WORKFLOW_NAME = "CI"

LINT_CONTEXT = "Lint (pre-commit)"

# A8's subject. `GPU_CONTEXT` is the check-run name `gpu.yml` declares AND the
# required-status-check context string `main.json` requires; A8 keys off exactly
# that pair, the way A5 and A7 do. `main.no-gpu.json` -- the audited lab-outage
# override (CI-15, D-17-04) -- deliberately does NOT require it, which is why A8
# is written to be vacuous when no committed payload requires the context at all.
GPU_CONTEXT = "Tests (pytest, GPU)"

# The token the capability block in `gpu.yml` carries so A8 can locate it by
# character index rather than by guessing at shell structure. Editing the token
# in either file without the other is what the paired real-tree test catches.
GPU_CAPABILITY_SENTINEL = "POST-INSTALL GPU CAPABILITY ASSERT"

# The guard whose SENSE A8 pins. `if not <alias>.is_gpu_available():` is the only
# accepted shape; an inverted or rewritten guard fails, because a capability
# assertion that fires on the healthy state is worse than none at all.
GPU_CAPABILITY_GUARD_PATTERN = re.compile(r"if\s+not\s+\w+\.is_gpu_available\(\)\s*:")

# A9's subjects. The corridor is three things that only mean anything together,
# which is why one assertion covers all of them.
#
# `GPU_CONSTRAINTS_PATH` is the seam where the container's numpy is actually
# decided. The container installs `.[dev]` and NEVER `[cuda12]`, so the GPU
# extras' ceiling never enters that resolve and conda's numba never enters it
# either -- and pip states outright that it "does not currently take into account
# all the packages that are installed". Pinning the image alone was MEASURED not
# to hold (D-17-10; .planning/spikes/004-numpy-floor-vs-gpu-stack/README.md).
GPU_CONSTRAINTS_PATH = ".github/constraints/gpu.txt"

# `GPU_DIGEST_PATH` sits OUTSIDE `.github/workflows/**` deliberately: that prefix
# is the one a GitHub App installation token structurally cannot write, which is
# why `gpu-image-refresh.yml`'s write-back has been rejected since 2026-07-01 and
# the workflow quarantined since 2026-07-30 (D-17-15, remedy (b); entry 45 for
# why widening the token is a decision rather than a fix).
GPU_DIGEST_PATH = ".github/digests/gpu-runner.txt"

GPU_IMAGE_REPOSITORY = "ghcr.io/gseg-ethz/pchandler-gpu-runner"

# The ONLY accepted spelling of the pin: the repository, an `@sha256:` marker and
# exactly 64 lowercase hex characters, anchored at both ends. A tag-shaped or
# truncated value is refused rather than resolved, because a moving tag behind a
# required context is the silent-substitution class this phase closes (T-17-19).
GPU_PINNED_REFERENCE_PATTERN = re.compile(re.escape(GPU_IMAGE_REPOSITORY) + r"@sha256:[0-9a-f]{64}")

# A concrete digest ANYWHERE in the parsed GPU job means a second source of truth
# has grown back. Matched against the LOADED document, never the raw file: the
# parser strips comments by construction, so the header bullet that describes the
# relocated pin cannot fail the check it describes. Searching raw text here is the
# recurring trap of this phase -- a search for a removed thing matches the prose
# documenting its removal.
GPU_CONCRETE_DIGEST_PATTERN = re.compile(r"@sha256:[0-9a-f]{64}")

# The flag that wires the constraints file into the install. A9 checks that the
# GPU job's single `pip install` body carries it AND names the constraints path;
# an unconsumed constraints file is decoration.
GPU_CONSTRAINTS_FLAG_PATTERN = re.compile(r"-c\s+" + re.escape(GPU_CONSTRAINTS_PATH))

# A7's pending-narrowing register, keyed by required context. An entry suppresses
# A7 for that one context and MUST state which requirement narrows it and which
# plan lands the narrowing. It is a narrowing with a stated reason, not a mute:
# the capability it suppresses is pinned in both directions by
# `test_check_ci_config.py`, which passes an empty register.
#
# EMPTY, and that is the intended steady state — A7 now evaluates every
# required-context lint job for real rather than skipping one. The single entry
# this register ever held covered CI-16's narrowing while the self-test ran one
# wave ahead of it (Phase 13 plan 13-03); it was deleted by plan 13-07 once BOTH
# repos' lint jobs were narrowed (GSEGUtils by 13-04, pchandler by 13-07), in one
# paired change. Adding an entry is likewise a paired cross-repo change: this
# file is kept in deliberate near-symmetry with the sibling repository's copy.
# NOT byte-identity, and NOT D-04: D-04 (Phase 13 CONTEXT) scopes byte-identity
# to `.github/scripts/ruleset_lib.py`, which IS identical in both repos. This
# file has been legitimately divergent since Phase 17 plan 17-02 -- the sibling
# carries A1 through A7, this copy carries A1 through A10, because A8 and A9
# describe a GPU corridor the sibling has no reason to own. The earlier "(D-04)"
# wording overstated a decision nobody made; corrected 2026-08-24 (17-07).
A7_PENDING_NARROWINGS: dict[str, str] = {}

# A10's subject. `bash -lc '<body>'` inside a `run:` step is how this repository
# launches a container payload (`podman run ... "$GPU_IMAGE" bash -lc '...'`),
# and the body is a SINGLE quoted argument. The pattern deliberately captures the
# option cluster so a `-c`-less invocation -- `bash -l /work/script.sh`, which is
# a file, not a quoted string -- is not treated as one of these.
#
# `lead` is consumed rather than looked behind, because Python requires
# fixed-width lookbehind and the alternatives here are not the same width; the
# caller adds `len(lead)` to recover the offset of the `bash` word itself.
BASH_DASH_C_PATTERN = re.compile(r"(?P<lead>^|[\s;|&(])(?:[\w./-]*/)?bash(?P<opts>(?:[ \t]+-[A-Za-z]+)+)\s+")

# What may legally follow the closing quote of a `bash -c` script argument: a
# command terminator, a redirection, a pipeline or list operator, a closing
# bracket, or a comment. Anything else is a WORD, and a word after the script is
# the signature of a quote that closed earlier than its author intended.
A10_LEGAL_AFTER_SCRIPT: frozenset[str] = frozenset("\n;|&<>)}#")


def load_workflows(root: pathlib.Path) -> list[tuple[pathlib.Path, dict[str, Any]]]:
    """Parse every workflow document under the workflows directory.

    Parameters
    ----------
    root
        Repository root to inspect.

    Returns
    -------
    list of tuple
        One ``(path, document)`` pair per workflow that parses to a mapping,
        in sorted filename order.
    """
    out: list[tuple[pathlib.Path, dict[str, Any]]] = []
    for path in sorted((root / WORKFLOWS_DIR).glob("*.yml")):
        with path.open() as handle:
            document = yaml.safe_load(handle)
        if isinstance(document, dict):
            out.append((path, document))
    return out


def committed_required_contexts(root: pathlib.Path) -> set[str]:
    """Return the union of required status-check contexts across the committed rulesets.

    Parameters
    ----------
    root
        Repository root to inspect.

    Returns
    -------
    set of str
        Every context string any committed payload requires.
    """
    contexts: set[str] = set()
    for path in sorted((root / RULESETS_DIR).glob("*.json")):
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            contexts.update(ruleset_lib.required_contexts(payload))
    return contexts


def _declared_jobs(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return the workflow's jobs mapping, defensively, as job id to job body."""
    jobs = document.get("jobs")
    if not isinstance(jobs, dict):
        return {}
    return {str(job_id): job for job_id, job in jobs.items() if isinstance(job, dict)}


def _job_display_name(job_id: str, job: dict[str, Any]) -> str:
    """Return the string GitHub reports as the check-run name for this job."""
    name = job.get("name")
    return name if isinstance(name, str) else job_id


def check_trigger_filters(
    root: pathlib.Path,
    exceptions: dict[str, str] | None = None,
) -> list[str]:
    """A1 -- flag filter keys under a pull-request-side trigger on a required-check workflow.

    Parameters
    ----------
    root
        Repository root to inspect.
    exceptions
        Reviewed-exceptions table keyed by workflow filename; defaults to
        :data:`A1_REVIEWED_EXCEPTIONS`.

    Returns
    -------
    list of str
        One entry per offending filter key.
    """
    table = A1_REVIEWED_EXCEPTIONS if exceptions is None else exceptions
    contexts = committed_required_contexts(root)
    violations: list[str] = []
    for path, document in load_workflows(root):
        produced = sorted(set(ruleset_lib.job_names(document)) & contexts)
        if not produced or path.name in table:
            continue
        block, form = ruleset_lib.trigger_block(document)
        if form is None or block is None:
            # A workflow whose triggers cannot be read has not been checked, and
            # an unchecked required-check workflow is a violation, not a pass.
            violations.append(
                f"{path.name}: no resolvable trigger block under either the quoted or the bare "
                f"`on:` key, on a workflow producing required context(s) {produced} — this "
                f"workflow cannot be checked, so it cannot be assumed clean."
            )
            continue
        if not isinstance(block, dict):
            continue  # a bare list trigger block carries no filter sub-keys
        for event in A1_INSPECTED_EVENTS:
            config = block.get(event)
            if not isinstance(config, dict):
                continue
            for key in sorted(FILTER_KEYS & set(config)):
                violations.append(
                    f"{path.name}: `{key}:` under `{event}:` on a workflow producing required "
                    f"context(s) {produced} — a filtered-out run never reports, so the pull request "
                    f"waits on that check forever and can never merge. Conditionality belongs on "
                    f"STEPS, not on the trigger."
                )
    return violations


def check_base_repo_context_trigger(
    root: pathlib.Path,
    exceptions: dict[str, str] | None = None,
) -> list[str]:
    """A2 -- flag the base-repo-context pull-request trigger, except in the one reviewed file.

    The scan is over raw file text rather than parsed keys, so an occurrence in a
    comment or nested under a reusable-workflow call is caught too. It is scoped
    to the workflows directory so this script's own search literal cannot match
    itself.

    **An excepted file is not simply skipped.** Before the exception is honoured,
    the two statically checkable conditions of :data:`A2_REVIEWED_EXCEPTIONS` are
    asserted and either failure is a violation:

      (i)   no checkout step anywhere in that workflow carries a ``ref:`` input --
            a checkout with an explicit reference is how the job would start
            looking at head code, and it is the single edit that would turn the
            exception into the privilege-escalation vector the general ban exists
            to stop;
      (iii) the workflow grants exactly a read content scope and nothing else, at
            workflow level AND in every job that declares its own block, because
            a job-level ``permissions:`` REPLACES the workflow-level one rather
            than narrowing it.

    The third condition -- that no head-supplied code is executed, with API reads
    of head workflow YAML permitted -- stays a *reviewed* condition stated in the
    exception's reason string, because it is a property of what the steps do
    rather than of a key that can be looked up. It is not enforced here and must
    not be read as though it were.

    Parameters
    ----------
    root
        Repository root to inspect.
    exceptions
        Reviewed-exceptions table keyed by workflow filename; defaults to
        :data:`A2_REVIEWED_EXCEPTIONS`. Injectable so the tests can pin both
        directions, mirroring :func:`check_trigger_filters`'s signature.

    Returns
    -------
    list of str
        One entry per offending workflow file, or per unmet condition on an
        excepted one.
    """
    table = A2_REVIEWED_EXCEPTIONS if exceptions is None else exceptions
    violations: list[str] = []
    for path in sorted((root / WORKFLOWS_DIR).glob("*.yml")):
        lines = [
            str(number)
            for number, line in enumerate(path.read_text().splitlines(), start=1)
            if BASE_REPO_CONTEXT_TRIGGER in line
        ]
        if not lines:
            continue
        if path.name in table:
            violations.extend(_check_exception_conditions(path))
            continue
        violations.append(
            f"{path.name}: `{BASE_REPO_CONTEXT_TRIGGER}` found on line(s) {', '.join(lines)} — "
            f"that trigger runs with the base repository's secrets against a fork's code, and "
            f"one of these repos attaches a self-hosted lab runner. It is never legitimate here. "
            f"Exactly one file is excepted, by path, in A2_REVIEWED_EXCEPTIONS, and this is not it."
        )
    return violations


def _check_exception_conditions(path: pathlib.Path) -> list[str]:
    """Assert conditions (i) and (iii) on the one workflow excepted from A2.

    Parameters
    ----------
    path
        The excepted workflow file.

    Returns
    -------
    list of str
        One entry per unmet condition; empty when the exception still holds.
    """
    try:
        with path.open() as handle:
            document = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as error:
        return [
            f"{path.name}: is excepted from A2 but could not be parsed — {error}. Its exception "
            f"conditions cannot be asserted, so the exception cannot be honoured."
        ]
    if not isinstance(document, dict):
        return [
            f"{path.name}: is excepted from A2 but does not parse to a mapping, so its exception "
            f"conditions cannot be asserted and the exception cannot be honoured."
        ]

    violations: list[str] = []

    # Condition (iii), workflow level. Compared as a whole mapping: an added key
    # must fail even while `contents: read` is still present.
    permissions = document.get("permissions")
    if permissions != READ_CONTENT_ONLY_PERMISSIONS:
        violations.append(
            f"{path.name}: is excepted from the `{BASE_REPO_CONTEXT_TRIGGER}` ban only while its "
            f"workflow-level `permissions:` block is exactly {READ_CONTENT_ONLY_PERMISSIONS}, and it "
            f"is {permissions!r} — condition (iii) of that exception. A base-repo-context job with "
            f"anything beyond a read content scope is the privilege escalation the general ban "
            f"exists to stop."
        )

    for job_id, job in _declared_jobs(document).items():
        # Condition (iii), job level. A job-level block REPLACES the
        # workflow-level one rather than narrowing it, so a job could otherwise
        # widen the scope while the workflow-level block still read clean.
        if "permissions" in job and job.get("permissions") != READ_CONTENT_ONLY_PERMISSIONS:
            violations.append(
                f"{path.name}: job `{job_id}` declares `permissions: {job.get('permissions')!r}`, "
                f"which is not exactly {READ_CONTENT_ONLY_PERMISSIONS} — condition (iii) of this "
                f"file's A2 exception. A job-level block REPLACES the workflow-level one, so this "
                f"widens the scope rather than narrowing it."
            )
        # Condition (i).
        steps = job.get("steps")
        for index, step in enumerate(steps if isinstance(steps, list) else []):
            if not isinstance(step, dict) or CHECKOUT_ACTION not in str(step.get("uses") or ""):
                continue
            inputs = step.get("with")
            if isinstance(inputs, dict) and CHECKOUT_REF_INPUT in inputs:
                violations.append(
                    f"{path.name}: job `{job_id}` step {index} checks out with a "
                    f"`{CHECKOUT_REF_INPUT}:` input — condition (i) of this file's A2 exception is "
                    f"base-ref checkout ONLY. A checkout carrying an explicit reference is how this "
                    f"job would start looking at head code, and it is the single edit that turns the "
                    f"exception into the vector the ban exists to stop."
                )
    return violations


def check_workflow_name_pin(root: pathlib.Path) -> list[str]:
    """A3 -- assert the CI workflow name is byte-exact and every workflow-run reference resolves.

    Parameters
    ----------
    root
        Repository root to inspect.

    Returns
    -------
    list of str
        One entry per name mismatch or unresolvable workflow-run reference.
    """
    workflows = load_workflows(root)
    violations: list[str] = []

    declared = {document.get("name") for _, document in workflows if isinstance(document.get("name"), str)}
    ci = [document for path, document in workflows if path.name == CI_WORKFLOW_FILENAME]
    if not ci:
        violations.append(f"{CI_WORKFLOW_FILENAME}: not found under {WORKFLOWS_DIR} — the pinned CI workflow is gone.")
    else:
        # Byte-exact on purpose: no case folding, no stripping. Any consumer
        # matching on this string matches it literally.
        actual = ci[0].get("name")
        if actual != CI_WORKFLOW_NAME:
            violations.append(
                f"{CI_WORKFLOW_FILENAME}: declared name is {actual!r}, not {CI_WORKFLOW_NAME!r} — the "
                f"run-counting rule in the workspace audit script "
                f"(.planning/scripts/audit_release_flow.py, CI_SUBTOTAL_WORKFLOW) keys on that exact "
                f"string, and any workflow keying on another workflow's name breaks silently."
            )

    for path, document in workflows:
        block, _ = ruleset_lib.trigger_block(document)
        if not isinstance(block, dict):
            continue
        config = block.get("workflow_run")
        if not isinstance(config, dict):
            continue
        for referenced in config.get("workflows") or []:
            if referenced not in declared:
                violations.append(
                    f"{path.name}: `workflow_run` lists workflow name {referenced!r}, which no "
                    f"workflow in this repository declares — this trigger will never fire, silently."
                )
    return violations


def _allowlist_from_composite(root: pathlib.Path) -> set[str] | None:
    """Extract the fast-path allowlist from the composite action's step body."""
    path = root / COMPOSITE_ACTION_PATH
    if not path.is_file():
        return None
    match = re.search(r"ALLOW='([^']*)'", path.read_text())
    if match is None:
        return None
    return set(match.group(1).split())


def check_fast_path_allowlist(root: pathlib.Path) -> list[str]:
    """A4 -- reconcile the fast-path allowlist against release-please's own configuration.

    Parameters
    ----------
    root
        Repository root to inspect.

    Returns
    -------
    list of str
        A single entry when the two sets differ, or when either side is unreadable.
    """
    allowlist = _allowlist_from_composite(root)
    if allowlist is None:
        return [
            f"{COMPOSITE_ACTION_PATH}: no `ALLOW='...'` assignment found — the fast-path allowlist "
            f"cannot be reconciled against {RELEASE_PLEASE_CONFIG_PATH}."
        ]

    config_path = root / RELEASE_PLEASE_CONFIG_PATH
    if not config_path.is_file():
        return [f"{RELEASE_PLEASE_CONFIG_PATH}: not found — the fast-path allowlist cannot be reconciled."]

    expected = {RELEASE_PLEASE_MANIFEST_PATH}
    for package in (json.loads(config_path.read_text()).get("packages") or {}).values():
        if not isinstance(package, dict):
            continue
        expected.add(str(package.get("changelog-path") or "CHANGELOG.md"))
        expected.update(str(extra) for extra in package.get("extra-files") or [])

    if allowlist == expected:
        return []
    return [
        f"{COMPOSITE_ACTION_PATH}: the fast-path allowlist {sorted(allowlist)} no longer set-equals "
        f"the release artifacts {sorted(expected)} derived from {RELEASE_PLEASE_CONFIG_PATH} "
        f"(changelog path + extra files) plus {RELEASE_PLEASE_MANIFEST_PATH}. A new extra file must "
        f"be added to the allowlist, or the fast path silently stops engaging on release pull "
        f"requests and the economy it exists to buy is lost without any failure."
    ]


def check_job_level_conditions(root: pathlib.Path) -> list[str]:
    """A5 -- flag a job-level condition on any job whose declared name is a required context.

    Parameters
    ----------
    root
        Repository root to inspect.

    Returns
    -------
    list of str
        One entry per offending job.
    """
    contexts = committed_required_contexts(root)
    violations: list[str] = []
    for path, document in load_workflows(root):
        for job_id, job in _declared_jobs(document).items():
            name = _job_display_name(job_id, job)
            if name in contexts and "if" in job:
                violations.append(
                    f"{path.name}: job `{job_id}` produces required context {name!r} and carries a "
                    f"job-level condition — a skipped job's `skipped` conclusion counts among the "
                    f"satisfying ones, so this greens the gate having executed nothing. "
                    f"Conditionality belongs on STEPS."
                )
    return violations


def check_conditional_dependencies(root: pathlib.Path) -> list[str]:
    """A6 -- flag a required-context job that depends on a job carrying a job-level condition.

    Parameters
    ----------
    root
        Repository root to inspect.

    Returns
    -------
    list of str
        One entry per offending dependency edge.
    """
    contexts = committed_required_contexts(root)
    violations: list[str] = []
    for path, document in load_workflows(root):
        jobs = _declared_jobs(document)
        for job_id, job in jobs.items():
            name = _job_display_name(job_id, job)
            if name not in contexts:
                continue
            needs = job.get("needs")
            needs = [needs] if isinstance(needs, str) else needs
            for dependency in needs or []:
                upstream = jobs.get(str(dependency))
                if isinstance(upstream, dict) and "if" in upstream:
                    violations.append(
                        f"{path.name}: job `{job_id}` produces required context {name!r} and depends "
                        f"on `{dependency}`, which carries a job-level condition — a skipped "
                        f"dependency skips the dependent, and that also counts as satisfying."
                    )
    return violations


def check_lint_least_privilege(
    root: pathlib.Path,
    pending: dict[str, str] | None = None,
) -> list[str]:
    """A7 -- assert the lint job declares a permissions block granting no write scope.

    Parameters
    ----------
    root
        Repository root to inspect.
    pending
        Pending-narrowing register keyed by required context; defaults to
        :data:`A7_PENDING_NARROWINGS`. A context listed here is exempt, and a
        listed context that is already narrowed emits a removal notice.

    Returns
    -------
    list of str
        One entry per write scope, or one entry when no permissions block exists.
    """
    register = A7_PENDING_NARROWINGS if pending is None else pending
    violations: list[str] = []
    found = False
    for path, document in load_workflows(root):
        for job_id, job in _declared_jobs(document).items():
            if _job_display_name(job_id, job) != LINT_CONTEXT:
                continue
            found = True
            granted = _write_scopes(job.get("permissions"))
            if granted is None:
                violations.append(
                    f"{path.name}: job `{job_id}` produces required context {LINT_CONTEXT!r} and "
                    f"declares no `permissions:` block, so it inherits the repository default "
                    f"instead of stating its least privilege explicitly (CI-16)."
                )
                continue
            if LINT_CONTEXT in register:
                if not granted:
                    print(
                        f"::notice::check_ci_config: A7's pending-narrowing entry for "
                        f"{LINT_CONTEXT!r} is no longer warranted — `{job_id}` in {path.name} grants "
                        f"no write scope. Delete the entry from A7_PENDING_NARROWINGS in "
                        f"{SELF_PATH}, in BOTH repos in the same paired change — this file is "
                        f"byte-identical across them by design (D-04)."
                    )
                continue
            for scope in granted:
                violations.append(
                    f"{path.name}: job `{job_id}` produces required context {LINT_CONTEXT!r} and "
                    f"grants `{scope}: write` — CI-16 narrows this job to read scopes only, and a "
                    f"write scope here is a token the pre-commit run does not need."
                )
    if not found:
        violations.append(
            f"No job under {WORKFLOWS_DIR} declares the name {LINT_CONTEXT!r}, yet it is a required "
            f"status-check context — nothing produces it, so the gate can never be satisfied."
        )
    return violations


def _gpu_capability_steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the job's run-steps whose body carries the A8 sentinel."""
    steps = job.get("steps")
    if not isinstance(steps, list):
        return []
    return [
        step
        for step in steps
        if isinstance(step, dict) and isinstance(step.get("run"), str) and GPU_CAPABILITY_SENTINEL in step["run"]
    ]


def _gpu_capability_step_violations(filename: str, job_id: str, step: dict[str, Any]) -> list[str]:
    """Check the sense, the teeth and above all the POSITION of one capability step.

    Split out of :func:`check_gpu_post_install_capability_assert` so each half stays
    under the complexity ceiling; the contract is entirely A8's, and every message
    names the file, the job, the requirement and what breaks when it is not held.
    """
    body = step["run"]
    violations: list[str] = []

    if "continue-on-error" in step:
        violations.append(
            f"{filename}: the capability step in job `{job_id}` declares `continue-on-error`, which "
            f"turns a failed capability assertion back into a green required context — the exact "
            f"fail-open D-17-05 exists to close. The only sanctioned escape for a degraded lab is "
            f"the committed `main.no-gpu.json` ruleset variant (D-17-04)."
        )
    if "|| true" in body:
        violations.append(
            f"{filename}: the capability step in job `{job_id}` contains `|| true`, which swallows "
            f"the non-zero exit the assertion exists to produce (D-17-04)."
        )
    if not GPU_CAPABILITY_GUARD_PATTERN.search(body):
        violations.append(
            f"{filename}: the capability step in job `{job_id}` does not match "
            f"{GPU_CAPABILITY_GUARD_PATTERN.pattern!r}. An absent or inverted guard either never "
            f"fires or fires on the healthy state; neither asserts the capability."
        )
    if "sys.exit(1)" not in body:
        violations.append(
            f"{filename}: the capability step in job `{job_id}` never calls `sys.exit(1)`, so a lost "
            f"GPU capability prints and continues instead of reddening the required context."
        )

    sentinel_at = body.index(GPU_CAPABILITY_SENTINEL)
    if "pip install" not in body:
        violations.append(
            f"{filename}: the capability step in job `{job_id}` runs no `pip install`. The assertion "
            f"is meaningless unless it FOLLOWS the install in the same container invocation — an "
            f"install in an earlier step is exactly the 2026-08-17 ordering that let a broken stack "
            f"through (D-17-05)."
        )
    elif sentinel_at < body.rindex("pip install"):
        violations.append(
            f"{filename}: {GPU_CAPABILITY_SENTINEL!r} in job `{job_id}` appears BEFORE the last "
            f"`pip install` in the same body. That is the pre-2026-08-17 ordering: the capability is "
            f"certified, then the install invalidates it, then the suite skips everything and exits "
            f"0 (CI-17, D-17-05)."
        )

    suite_at = body.find("pytest ")
    if suite_at == -1:
        violations.append(
            f"{filename}: the capability step in job `{job_id}` invokes no suite, so the assertion "
            f"gates nothing. It must sit in the same body as the run it protects."
        )
    elif sentinel_at > suite_at:
        violations.append(
            f"{filename}: {GPU_CAPABILITY_SENTINEL!r} in job `{job_id}` appears AFTER the suite "
            f"invocation. An assertion downstream of the run it is meant to gate cannot stop that "
            f"run from reporting success (CI-17, D-17-05)."
        )
    return violations


def check_gpu_post_install_capability_assert(root: pathlib.Path) -> list[str]:
    """A8 -- assert the GPU job checks its capability after its own install and before the suite.

    On 2026-08-17 the ``Tests (pytest, GPU)`` context reported ``success`` having
    executed nothing. The run log tells the whole story in four timestamps:
    ``12:33:31`` the pre-flight health check passed against the container's
    pristine numpy 2.0.2; ``12:33:47`` the job's own dependency install began
    uninstalling that numpy; ``12:33:54`` it finished with numpy 2.3.5, which
    ``numba`` refuses; ``12:33:57`` all three GPU tests skipped and pytest exited
    0. ``tests/filters/test_gpu.py`` carries a module-level ``skipif`` on
    ``is_gpu_available()``, so an inert GPU stack is indistinguishable from a
    passing suite. The assertion was never missing -- it simply ran before the
    step that invalidated it, and that ordering IS the defect (CI-17, D-17-05).

    A8 therefore checks position, not merely presence: the sentinel must sit
    after the LAST ``pip install`` and before the suite invocation, in the SAME
    step body, so that it runs inside the same container as both.

    **Vacuous where no GPU context is required, by design.** This file is kept
    in near-symmetry with the sibling repository's copy (see the module-level
    note on ``A7_PENDING_NARROWINGS``; byte-identity is D-04's rule for
    ``ruleset_lib.py``, not for this file), and the sibling repository's
    committed rulesets require no GPU context. A8 returns cleanly there rather
    than demanding a job that repository has no reason to declare. The same
    branch makes A8 inert under the audited ``main.no-gpu.json`` lab-outage
    override, which is the only sanctioned escape D-17-04 recognises.

    Parameters
    ----------
    root
        Repository root to inspect.

    Returns
    -------
    list of str
        One entry per violation; empty when the context is not required, or when
        the capability assertion is present, correctly ordered and un-neutered.
    """
    if GPU_CONTEXT not in committed_required_contexts(root):
        return []

    violations: list[str] = []
    found = False
    for path, document in load_workflows(root):
        for job_id, job in _declared_jobs(document).items():
            if _job_display_name(job_id, job) != GPU_CONTEXT:
                continue
            found = True
            carrying = _gpu_capability_steps(job)
            if not carrying:
                violations.append(
                    f"{path.name}: job `{job_id}` produces required context {GPU_CONTEXT!r} but no "
                    f"step carries {GPU_CAPABILITY_SENTINEL!r}. Without it a dependency install that "
                    f"breaks the GPU stack is followed by an all-skipped suite that exits 0, and the "
                    f"required context reports success having run nothing (CI-17, D-17-05)."
                )
            elif len(carrying) > 1:
                violations.append(
                    f"{path.name}: job `{job_id}` carries {GPU_CAPABILITY_SENTINEL!r} in "
                    f"{len(carrying)} steps. A8 decides ordering by character index within ONE body; "
                    f"two copies make that index ambiguous. Keep exactly one, in the step that also "
                    f"runs the install and the suite."
                )
            else:
                violations.extend(_gpu_capability_step_violations(path.name, job_id, carrying[0]))

    if not found:
        violations.append(
            f"No job under {WORKFLOWS_DIR} declares the name {GPU_CONTEXT!r}, yet it is a required "
            f"status-check context — nothing produces it, so the gate can never be satisfied."
        )
    return violations


def _string_values(node: Any) -> Iterator[str]:
    """Yield every string leaf of a parsed YAML node, depth-first.

    Used by A9 to scan the LOADED GPU job rather than the raw file, so that a
    comment mentioning the digest is invisible by construction.
    """
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _string_values(value)
    elif isinstance(node, list):
        for value in node:
            yield from _string_values(value)


def _gpu_constraints_violations(root: pathlib.Path) -> list[str]:
    """A9's constraints half -- the file exists and yields at least one requirement."""
    path = root / GPU_CONSTRAINTS_PATH
    if not path.is_file():
        return [
            f"{GPU_CONSTRAINTS_PATH} is missing, yet the GPU job's install points at it with `-c`. "
            f"pip fails on an unreadable constraints file, so this reddens the GPU context rather "
            f"than silently widening the corridor -- but the corridor is then undefined, and the "
            f"container's numpy is decided by whatever resolves (D-17-10)."
        ]
    requirements = [
        line.strip() for line in path.read_text().splitlines() if line.strip() and not line.strip().startswith("#")
    ]
    if not requirements:
        return [
            f"{GPU_CONSTRAINTS_PATH} carries no requirement line -- only comments or blanks. A "
            f"constraints file that constrains nothing is decoration: the container installs "
            f"`.[dev]` and never `[cuda12]`, so nothing else in that resolve holds numpy inside "
            f"the corridor (D-17-10, D-17-14)."
        ]
    return []


def _gpu_digest_violations(root: pathlib.Path) -> list[str]:
    """A9's digest half -- exactly one line, fully matching the pinned-reference pattern."""
    path = root / GPU_DIGEST_PATH
    if not path.is_file():
        return [
            f"{GPU_DIGEST_PATH} is missing, so the GPU job cannot resolve its image. The job fails "
            f"closed on this rather than falling back to a moving tag, which means the required "
            f"context goes red until the pin is restored (D-17-15, T-17-18)."
        ]
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(lines) != 1:
        return [
            f"{GPU_DIGEST_PATH} holds {len(lines)} non-empty lines; it must hold exactly one. "
            f"`gpu-image-refresh.yml` rewrites this file with `sed` and greps the written digest "
            f"back out, and both assume a single-line artefact -- more than one line makes the "
            f"rewrite ambiguous and the read non-deterministic."
        ]
    if not GPU_PINNED_REFERENCE_PATTERN.fullmatch(lines[0]):
        return [
            f"{GPU_DIGEST_PATH} does not hold a full-length pinned reference. Expected "
            f"{GPU_IMAGE_REPOSITORY}@sha256: followed by 64 lowercase hex characters; found "
            f"{lines[0]!r}. A tag-shaped or truncated value would let the image move under a "
            f"required context -- the silent substitution D-17-15 keeps closed (T-17-19)."
        ]
    return []


def _gpu_job_corridor_violations(filename: str, job_id: str, job: dict[str, Any]) -> list[str]:
    """A9's job half -- the install consumes the constraints file and no digest is inlined."""
    violations: list[str] = []
    bodies = [
        step["run"]
        for step in job.get("steps", [])
        if isinstance(step, dict) and isinstance(step.get("run"), str) and "pip install" in step["run"]
    ]
    if len(bodies) != 1:
        violations.append(
            f"{filename}: job `{job_id}` carries `pip install` in {len(bodies)} step bodies; A9 "
            f"expects exactly one, the container invocation that also runs the suite. Two installs "
            f"mean two resolves and only one of them is constrained."
        )
    elif not GPU_CONSTRAINTS_FLAG_PATTERN.search(bodies[0]):
        violations.append(
            f"{filename}: the container install in job `{job_id}` carries no `-c "
            f"{GPU_CONSTRAINTS_PATH}`. That flag is the ONLY seam where the corridor is enforced: "
            f"the install is `.[dev]`, never `[cuda12]`, so the GPU extras' ceiling and conda's "
            f"numba both sit outside this resolve, and pip does not account for already-installed "
            f"packages. Unconstrained, numpy walks past numba's wall and the suite skips behind a "
            f"green required context (CI-17, D-17-10)."
        )
    inlined = sorted({value for value in _string_values(job) if GPU_CONCRETE_DIGEST_PATTERN.search(value)})
    if inlined:
        violations.append(
            f"{filename}: job `{job_id}` inlines a concrete image digest in {len(inlined)} parsed "
            f"value(s). The pin lives in {GPU_DIGEST_PATH} and nowhere else -- a second copy here "
            f"is a second source of truth, and it is the copy `gpu-image-refresh.yml` cannot "
            f"rewrite, because a GitHub App token cannot write under {WORKFLOWS_DIR} (D-17-15)."
        )
    return violations


def check_gpu_corridor_artifacts(root: pathlib.Path) -> list[str]:
    """A9 -- assert the GPU corridor's artefacts exist, are wired in, and are the only pin.

    Three properties that are worthless apart. A constraints file nothing passes
    to pip constrains nothing. A digest artefact the job does not read pins
    nothing. And either of them is undone by a concrete digest left inline in the
    job, which becomes a second source of truth in the one place the refresh
    workflow's write-back cannot reach.

    **Why the constraints file rather than an image pin.** Measured, not assumed:
    the container installs ``.[dev]`` and never ``[cuda12]``, so conda's numba and
    the GPU extras' ceiling are both outside pip's resolve, and pip says so
    itself -- *"pip's dependency resolver does not currently take into account all
    the packages that are installed."* On 2026-08-17 that let numpy walk to 2.3.5,
    numba refused it, and the whole GPU suite skipped behind a green required
    context (D-17-10; ``.planning/spikes/004-numpy-floor-vs-gpu-stack/README.md``).

    **Why the digest sits outside the workflows directory.** GitHub refuses any
    GitHub App push touching ``.github/workflows/**`` unless the App holds
    ``workflows: write`` -- a permission with no key in a workflow's
    ``permissions:`` block. Relocating the pin unblocks a write-back broken since
    2026-07-01 with no new credential (D-17-15, remedy (b)).

    **The digest scan reads the PARSED job, never the raw file.** Comments are
    stripped by the parser, so the header bullet that documents the relocated pin
    cannot fail the check it describes.

    **Vacuous where no GPU context is required, by design** -- same branch as A8,
    for the same reason: a copy of this file in the sibling repository would meet
    no GPU stack there, so the assertion must be a no-op rather than a demand for
    artefacts that repository has no reason to carry. (Byte-identity across the
    two repositories is D-04's rule for ``ruleset_lib.py``; this file is
    deliberately divergent -- see the note on ``A7_PENDING_NARROWINGS``.)

    Parameters
    ----------
    root
        Repository root to inspect.

    Returns
    -------
    list of str
        One entry per violation; empty when the context is not required, or when
        the corridor is present, wired in and singly sourced.
    """
    if GPU_CONTEXT not in committed_required_contexts(root):
        return []

    violations: list[str] = [*_gpu_constraints_violations(root), *_gpu_digest_violations(root)]
    found = False
    for path, document in load_workflows(root):
        for job_id, job in _declared_jobs(document).items():
            if _job_display_name(job_id, job) != GPU_CONTEXT:
                continue
            found = True
            violations.extend(_gpu_job_corridor_violations(path.name, job_id, job))

    if not found:
        violations.append(
            f"No job under {WORKFLOWS_DIR} declares the name {GPU_CONTEXT!r}, yet it is a required "
            f"status-check context — nothing produces it, so the corridor this assertion checks has "
            f"no job to be wired into."
        )
    return violations


def _shell_quote_scan(body: str) -> tuple[dict[int, int], list[bool], int | None]:
    """Walk a shell script once, tracking quoting and comment state.

    A hand-written state machine rather than :mod:`shlex`, and the reason is
    stated so it is not "improved" later: ``shlex.split`` raises a single
    ``ValueError`` on an unbalanced quote and reports no position, and it
    tokenises the whole body -- which for a real ``run:`` script means pipelines,
    ``$(...)`` and here-documents it has no model for. A10 needs one specific
    thing instead: WHERE each top-level quote opens and closes, so it can ask
    what follows the one that carries a ``bash -c`` payload.

    Parameters
    ----------
    body
        The step's ``run:`` script, taken from the PARSED workflow document.

    Returns
    -------
    tuple
        ``(spans, masked, unterminated_at)``. ``spans`` maps the index of each
        quote that OPENS at the unquoted top level to the index of its matching
        closing quote. ``masked[i]`` is True where index ``i`` lies inside a
        quoted region or a comment, so a caller can reject a match that is not
        really an invocation. ``unterminated_at`` is the index of an opening
        quote that is never closed, or None; the walk stops there, so ``masked``
        beyond that index is not meaningful -- which is why an unterminated quote
        is itself reported as a violation rather than passed over.
    """
    spans: dict[int, int] = {}
    masked = [False] * len(body)
    index, end = 0, len(body)
    while index < end:
        char = body[index]
        if char == "\\":
            index += 2
            continue
        if char == "#" and (index == 0 or body[index - 1] in " \t\n;|&()"):
            stop = body.find("\n", index)
            stop = end if stop == -1 else stop
            for position in range(index, stop):
                masked[position] = True
            index = stop
            continue
        if char in "'\"":
            close = _matching_quote(body, index)
            if close is None:
                return spans, masked, index
            spans[index] = close
            for position in range(index, close + 1):
                masked[position] = True
            index = close + 1
            continue
        index += 1
    return spans, masked, None


def _matching_quote(body: str, opening: int) -> int | None:
    r"""Return the index closing the quote opened at ``opening``, or None if unterminated.

    Single quotes take the next ``'`` with no escape processing -- POSIX is
    explicit that a backslash is literal inside them, and that is exactly why a
    stray apostrophe in prose cannot be "escaped away" and closes the argument.
    Double quotes honour a backslash before ``$``, a backtick, ``"``, ``\\`` or a
    newline.

    Parameters
    ----------
    body
        The step's ``run:`` script.
    opening
        Index of the opening quote character.

    Returns
    -------
    int or None
        Index of the matching closing quote, or None when there is none.
    """
    quote = body[opening]
    if quote == "'":
        close = body.find("'", opening + 1)
        return None if close == -1 else close
    index, end = opening + 1, len(body)
    while index < end:
        if body[index] == "\\" and index + 1 < end and body[index + 1] in '$`"\\\n':
            index += 2
            continue
        if body[index] == '"':
            return index
        index += 1
    return None


def _bash_dash_c_violations(filename: str, job_id: str, step_name: str, body: str) -> list[str]:
    """Check every top-level ``bash -c`` invocation in one step body. A10's core.

    Parameters
    ----------
    filename
        Workflow filename, for the message.
    job_id
        Job identifier, for the message.
    step_name
        Step name (or its id, or a positional label), for the message.
    body
        The step's ``run:`` script.

    Returns
    -------
    list of str
        One entry per violation; empty when every invocation round-trips.
    """
    where = f"{filename}: step {step_name!r} in job `{job_id}`"
    if not BASH_DASH_C_PATTERN.search(body):
        return []

    spans, masked, unterminated_at = _shell_quote_scan(body)
    violations: list[str] = []
    if unterminated_at is not None:
        violations.append(
            f"{where} carries a `bash -c` payload AND an unterminated {body[unterminated_at]!r} quote "
            f"opened at character {unterminated_at}, line "
            f"{body.count(chr(10), 0, unterminated_at) + 1} of the step body. "
            f"A10 cannot verify the payload reaches the shell as one argument, and a check that cannot "
            f"verify must not answer `safe` (WINDOWS 30)."
        )

    for match in BASH_DASH_C_PATTERN.finditer(body):
        if "c" not in match.group("opts"):
            continue
        word_at = match.start() + len(match.group("lead"))
        if word_at < len(masked) and masked[word_at]:
            continue  # quoted or commented mention, not an invocation
        violations.extend(_one_invocation_violations(where, body, spans, match.end()))
    return violations


def _one_invocation_violations(where: str, body: str, spans: dict[int, int], script_at: int) -> list[str]:
    """Check the single script argument of one ``bash -c`` invocation.

    Parameters
    ----------
    where
        Pre-rendered "file: step in job" prefix for the message.
    body
        The step's ``run:`` script.
    spans
        Top-level quote spans from :func:`_shell_quote_scan`.
    script_at
        Index at which the script argument is expected to begin.

    Returns
    -------
    list of str
        One entry per violation for this invocation.
    """
    line = body.count("\n", 0, min(script_at, len(body))) + 1
    if script_at >= len(body) or body[script_at] not in "'\"":
        found = body[script_at : script_at + 24] if script_at < len(body) else "<end of body>"
        return [
            f"{where} invokes `bash -c` at line {line} of the step body with a script argument that is "
            f"not a literal quoted string; it reads {found!r}. A10 verifies quoting by round-trip, so it cannot "
            f"verify this shape -- and an unverifiable invocation is a violation, not a skip. Write the "
            f"payload as one quoted argument, or move it to a script file and call that file instead."
        ]
    close = spans.get(script_at)
    if close is None:
        return [
            f"{where} invokes `bash -c` at line {line} of the step body with a {body[script_at]!r}-quoted "
            f"script that is never closed, so the payload does not round-trip its own quoting (WINDOWS 30)."
        ]

    index = close + 1
    while index < len(body) and (body[index] in " \t" or body[index : index + 2] == "\\\n"):
        index += 2 if body[index] == "\\" else 1
    if index >= len(body) or body[index] in A10_LEGAL_AFTER_SCRIPT:
        return []

    leaked = body[index:].split("\n", 1)[0]
    broke_at = body.count("\n", 0, close) + 1
    return [
        f"{where} breaks out of its own `bash -c` quoting: the {body[script_at]!r} opened at line "
        f"{line} of the step body is closed by a {body[script_at]!r} at line {broke_at} of that body, "
        f"and the bare word(s) "
        f"{leaked.strip()!r} then follow the script argument. Everything after that quote is handed "
        f"to the OUTER shell -- on this repository that means it runs on the RUNNER HOST instead of "
        f"inside the container, behind a required status check a satisfying host would report green. "
        f"An apostrophe in prose is enough; that is exactly how WINDOWS 30 happened (commit 5c4ab66, "
        f"run 32705493768). Remove the apostrophe, or restructure the payload."
    ]


def check_bash_dash_c_quoting(root: pathlib.Path) -> list[str]:
    """A10 -- assert every ``bash -c`` payload reaches the shell as exactly one argument.

    On 2026-08-21 commit ``5c4ab66`` added comment prose to ``gpu.yml``'s
    container payload. One apostrophe -- ``container's`` -- closed the single
    quote that payload lives in. ``podman`` then received twenty-three arguments
    instead of sixteen, the container got four lines and exited 0, and the
    dependency install, the CI-17 capability assert and the entire GPU suite ran
    on the RUNNER HOST. On that Debian host they failed loudly; on a host whose
    environment happened to satisfy the imports the same break-out yields a green
    ``Tests (pytest, GPU)`` required context for a suite that never entered the
    container. A10 makes the CLASS a build failure rather than the instance
    (WINDOWS 30).

    **Parsed structure, never a raw-text grep.** The bodies come from the loaded
    workflow documents, and the quoting is decided by a shell-quoting walk
    (:func:`_shell_quote_scan`), so a ``bash -lc`` mentioned inside a comment or
    inside another quoted string is not mistaken for an invocation. This phase
    has been bitten repeatedly by text searches -- including one where a search
    for a removed thing matched the prose documenting its removal.

    **Deliberately NOT gated on the GPU context.** A8 and A9 return early where no
    committed ruleset requires ``Tests (pytest, GPU)``, because the corridor they
    describe is this repository's. A10's class is generic: any workflow, any job,
    any step. Gating it on the GPU context would make it vacuous everywhere it is
    not already unnecessary, which is the opposite of the point.

    **What it deliberately refuses, with the reason.** ``bash -c script name
    args...`` -- the POSIX form where words after the script become ``$0``,
    ``$1``... -- is indistinguishable by inspection from a quote break-out, and no
    workflow in either repository uses it. A10 rejects it. The remedy if one is
    ever wanted is a script file, which A10 does not inspect at all because a file
    has no quoting to break.

    Parameters
    ----------
    root
        Repository root to inspect.

    Returns
    -------
    list of str
        One entry per violation; empty when every payload round-trips.
    """
    violations: list[str] = []
    for path, document in load_workflows(root):
        for job_id, job in _declared_jobs(document).items():
            steps = job.get("steps")
            if not isinstance(steps, list):
                continue
            for position, step in enumerate(steps):
                if not isinstance(step, dict) or not isinstance(step.get("run"), str):
                    continue
                label = step.get("name") or step.get("id") or f"<step {position}>"
                violations.extend(_bash_dash_c_violations(path.name, job_id, str(label), step["run"]))
    return violations


def _write_scopes(permissions: Any) -> list[str] | None:
    """Return the permission keys granted at write, or None when no block is declared."""
    if permissions is None:
        return None
    if isinstance(permissions, str):
        # The shorthand forms: `permissions: read-all` / `write-all`.
        return [] if permissions == "read-all" else [permissions]
    if not isinstance(permissions, dict):
        return None
    return [str(key) for key, value in permissions.items() if str(value) == "write"]


def run_all(
    root: pathlib.Path,
    exceptions: dict[str, str] | None = None,
    pending: dict[str, str] | None = None,
) -> tuple[list[str], int]:
    """Run every assertion against a tree and return the violations plus the workflow count.

    Parameters
    ----------
    root
        Repository root to inspect.
    exceptions
        A1's reviewed-exceptions table; defaults to the module constant.
    pending
        A7's pending-narrowing register; defaults to the module constant.

    Returns
    -------
    tuple
        The accumulated violations and the number of workflows inspected.
    """
    violations = [
        *check_trigger_filters(root, exceptions),
        *check_base_repo_context_trigger(root),  # its own table; see A2_REVIEWED_EXCEPTIONS
        *check_workflow_name_pin(root),
        *check_fast_path_allowlist(root),
        *check_job_level_conditions(root),
        *check_conditional_dependencies(root),
        *check_lint_least_privilege(root, pending),
        *check_gpu_post_install_capability_assert(root),
        *check_gpu_corridor_artifacts(root),
        *check_bash_dash_c_quoting(root),
    ]
    return violations, len(load_workflows(root))


# The driver runs under `python .github/scripts/check_ci_config.py`, which is the
# ONLY invocation contract this file has and is unchanged by the guard below.
#
# The guard exists because this module is now IMPORTED as a library, by
# `assert_no_skip.py` (which calls A5 and A6 directly rather than reimplementing
# them) and by `test_check_ci_config.py`. Without it, importing this module runs
# the whole self-test against THIS repository as an import side effect — and on a
# tree that failed any assertion it would `sys.exit(1)` inside the importer,
# before the importer's own rules had run. For `assert_no_skip.py` that is a
# wrong-verdict bug in a branch-protection gate: the integrity context would go
# red reporting the BASE tree's self-test findings, having never looked at the
# pull request at all. The straight-line shape plan 13-03 chose is preserved;
# only the import side effect is removed.
if __name__ == "__main__":
    findings, inspected_count = run_all(REPO_ROOT)
    if findings:
        for finding in findings:
            print(f"::error::{finding}")
        sys.exit(1)
    print(f"check_ci_config: OK — assertions A1 through A10 clean across {inspected_count} workflow(s)")
