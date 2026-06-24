# Release Process — PCHandler

**Repository:** gseg-ethz/PCHandler
**First published:** TBD (fill in after first production publish)
**Publish workflow:** `.github/workflows/publish-pypi.yml`

---

## Trusted Publisher Binding

This package publishes to PyPI via OIDC trusted publishing — no long-lived API token is stored
in CI secrets. The GitHub Actions workflow receives a short-lived OIDC token (5-minute
lifetime) from GitHub's token endpoint, which PyPI validates against the registered trusted
publisher before accepting the upload.

### Claim fields

| Field | Value |
|-------|-------|
| Owner (GitHub) | gseg-ethz |
| Repository | PCHandler |
| Workflow filename | publish-pypi.yml |
| Environment name | pypi |

These four fields are an exact-match claim. PyPI rejects the upload if any field differs by
even a single character (case-sensitive). See Pitfall 7 in `11-RESEARCH.md`.

---

## What NOT to Rename

Renaming any of the following breaks the trusted publisher claim and requires deleting and
re-creating the publisher record on PyPI (`pypi.org/manage/account/publishing/`):

- **Repository name** (`PCHandler` → any other name)
- **Workflow filename** (`publish-pypi.yml` → any other name)
- **GitHub Environment name** (`pypi` → any other name)

The PyPI *project* name (`pchandler`) is also immutable once published (PEP 763).

---

## Production Gate

The `pypi` GitHub Environment requires a human reviewer before the publish job can proceed
(D-05). This gate is configured with:

- **Required reviewer:** repository owner (nixton-meyer)
- **Prevent self-review:** ON — the release author cannot approve their own publish
- **Prevent admin bypass:** ON — even administrators cannot skip the review gate

The combination of D-04 (no publisher registered pre-clearance) and D-05 (environment gate
post-clearance) provides layered defense: the OIDC token is never minted without both a
registered trusted publisher claim AND a logged human approval.

The `testpypi` environment (used by `publish-testpypi.yml`) does NOT require a reviewer —
it is a dry-run path with no production impact.

---

## Rollback Procedure

PyPI does not allow deleting or replacing a published version (PEP 763 version immutability).
Once a version is uploaded, that version number is permanently burned for the project name.

To deprecate a broken release:

1. Navigate to `pypi.org/manage/project/pchandler/releases/`.
2. Select the broken version.
3. Click **Yank release**.

Yanked versions are hidden from install resolvers (`pip install pchandler` skips yanked
versions by default) but remain downloadable by users who pin the exact version. Yanking is
reversible; deletion is not possible.

**Version planning:** D-11 specifies cutting fresh patches (`v2.0.1`, not `v2.0.0`) for the
first production publish, so a botched first run costs only a `.1` patch version rather than
the headline release number.

---

## Verifying a Release

Download a wheel from a GitHub Release and verify its Sigstore provenance attestation:

```bash
gh attestation verify pchandler-2.0.1-py3-none-any.whl --repo gseg-ethz/PCHandler
```

Replace the wheel filename with the version you downloaded. The attestation is produced
automatically by `pypa/gh-action-pypi-publish@v1.14.0` during the publish job (PEP 740).

See [README.md](README.md) § Verifying Releases for the condensed one-liner reference.
