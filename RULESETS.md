# Branch Rulesets — PCHandler

**Repository:** gseg-ethz/PCHandler
**Protected branches:** `main`, `develop/gsd`
**Configured:** TBD — configured in Phase 11 (PCHandler goes public; rulesets unavailable on free-plan private repos)
**Ruleset snapshots:** `.github/rulesets/main.json`, `.github/rulesets/develop.json`

---

## Bypass List

**The bypass list is EMPTY on both rulesets.**

No actor — not the release-please bot, not repository admins — is on any bypass list. The
"Include administrators" bypass is intentionally not in use, meaning admins are subject to the
same rules as all other contributors. CI is a hard gate for everyone.

**Why no bypass is needed for release-please (CONTEXT D-02/D-03):**

1. release-please force-pushes only to its scratch branch
   `release-please--branches--main`, which is **not** a targeted ref (`refs/heads/main` and
   `refs/heads/develop/gsd` are the exact targets). The scratch branch is unprotected and
   works without any bypass.
2. release-please pushes moving tags (`refs/tags/v2`, `refs/tags/v2.0`). Branch rulesets do
   **not** cover `refs/tags/*`, so no bypass is required for tags either. (A separate tag
   ruleset would need the GitHub App as an `Integration`-type bypass actor with
   `Administration: write` scope — that is out of scope here and tracked as deferred.)

**Consequence:** No entity can merge a red-CI pull request — CI is a hard gate for everyone
including admins (CONTEXT D-02).

---

## Approval Policy

**PR required before merging; `required_approving_review_count = 0`.**

Direct pushes to `main` or `develop/gsd` are blocked. All changes must arrive via a pull
request with CI passing.

**Why 0 required approvals (CONTEXT D-05/D-06):**

The repository is driven mostly solo (owner + agent + CI) with Tomislav as an occasional
second human reviewer. GitHub forbids self-approval, so any `>= 1` approval requirement with
an empty bypass list (above) would **deadlock** solo-authored PRs — no one could approve your
own PR, and there is no bypass actor to unblock it. Setting 0 approvals keeps CI as the sole
hard gate while avoiding the self-approval deadlock.

The following flags are explicitly **OFF** per CONTEXT D-07:

- `require_last_push_approval`: false
- `require_code_owner_review`: false
- `dismiss_stale_reviews_on_push`: false (explicitly off per original PROT-01 intent)

**Future tightening:** Moving to 1 required approval + admin-on-bypass (for PRs) is the clean
next step when Tomislav becomes a routine reviewer (CONTEXT D-08).

---

## Required Status Checks

Check-run context strings are sourced verbatim from the `name:` fields in
`.github/workflows/ci.yml`. **If any CI job is renamed, the ruleset context string must be
updated in lockstep with the job rename.**

| Branch | Required Check |
|--------|---------------|
| `main` | `Lint (pre-commit)` |
| `main` | `Tests (pytest)` |
| `develop/gsd` | `Lint (pre-commit)` |
| `develop/gsd` | `Tests (pytest)` |

**GPU check sentinel decision — option (b): `Tests (pytest, GPU)` REMOVED from required checks.**

PCHandler's `gpu-tests` job has `if: github.event_name == 'push' || github.event_name == 'workflow_dispatch'`.
GitHub **skips** the job on `pull_request` events and **treats skipped as passing** for required
status checks [VERIFIED: emmer.dev + GitHub community forum]. This means including
`Tests (pytest, GPU)` as a required check on `protect-main` would silently allow every PR to
merge without the GPU suite running — the check appears as "skipped (passing)" rather than
"blocked."

Option (b) was chosen (lowest friction for a solo-maintained repo with an unreliable
self-hosted runner): remove `Tests (pytest, GPU)` from required PR checks entirely.
GPU enforcement happens **post-merge on push to `main`** — the push to `main` itself must pass
rulesets (which require a PR), and the GPU job runs on that push. A broken GPU merge is caught
on the post-merge CI run, before any release-please tag is created.

This decision is recorded in 11-CONTEXT.md (Phase 11 GPU sentinel decision, option b).

**`strict_required_status_checks_policy` is OFF (CONTEXT D-11):** The "require branches to be
up to date before merging" enforcement is disabled. On low-traffic repos its benefit is
marginal.

---

## Other Rules

Both rulesets (`main` and `develop/gsd`) include:

| Rule | Effect |
|------|--------|
| `non_fast_forward` | Force-pushes to the protected branch are blocked |
| `deletion` | The protected branch cannot be deleted |
| `required_linear_history` | Merge commits that create a non-linear history are blocked; only squash or rebase merges are permitted |

---

## Deferred

The following items are explicitly deferred per 10-CONTEXT.md D-15:

- **Rules-as-code automation:** A `workflow_dispatch` bootstrap workflow that applies the
  committed JSON via `gh api PUT /repos/.../rulesets`, an admin-capable apply token (reuse
  release-please App with `Administration: write`, or a fine-grained PAT), and a read-only
  drift-detection watchdog. Deferred because with one admin and two repos, drift risk is low;
  re-addable later with no rework via the GitHub UI's ruleset JSON export.

- **Tag protection:** A tag ruleset on `refs/tags/v*`. Branch rulesets do not cover tags. If
  ever added, release-please's moving-tag push would need the GitHub App as an `Integration`
  bypass actor with `Administration: write` scope (CONTEXT D-04).

- **Tighter review policy (C2/C3):** 1 required approval + admin-on-bypass once Tomislav
  becomes a routine reviewer (CONTEXT D-08).

---

## Verification

After PCHandler goes public and the rulesets are applied via the GitHub UI (Plan 11-07),
confirm that the live state matches this committed snapshot:

```bash
gh api /repos/gseg-ethz/PCHandler/rulesets
```

Expected response: an array containing two ruleset objects with `"name": "protect-main"` and
`"name": "protect-develop-gsd"`, both with `"enforcement": "active"`.

To inspect a specific ruleset in detail (replace `{id}` with the ruleset ID from the list):

```bash
gh api /repos/gseg-ethz/PCHandler/rulesets/{id}
```

Cross-check `bypass_actors`, `conditions.ref_name.include`, and the
`required_status_checks.required_status_checks` array against the values in
`.github/rulesets/main.json` and `.github/rulesets/develop.json`.

**UI gotcha (documented in 10-03-SUMMARY D-2):** Enter bare branch names (`main`, `develop/gsd`)
in the "Include by pattern" field — GitHub prepends `refs/heads/` automatically. Typing the
full `refs/heads/main` produces `refs/heads/refs/heads/main` (non-matching, silently wrong).
