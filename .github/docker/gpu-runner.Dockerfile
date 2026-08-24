# Purpose: a WARM BASE for the ci.yml `gpu-tests` job — RAPIDS GPU stack (conda) + git/ssh.
# pchandler + GSEGUtils are installed FRESH from the checked-out commit at JOB time
# (ci.yml gpu-tests "Install pchandler editable" step), where the GSEGUtils deploy key and the
# exact source-under-test exist. The image deliberately does NOT bake pchandler in.
#
# Redesigned 2026-06-15 (09-02 live commissioning). The original Task-1 line
# `COPY . ; RUN pip install -e .[cuda12,dev]` was both broken AND redundant:
#   * setuptools-scm could not version pchandler at build time (no git/tags in the base);
#   * GSEGUtils is a git+ssh dep with no deploy key available during the image build;
#   * the [cuda12] PyPI wheels (cudf-cu12 ...) duplicate/conflict with the base's conda RAPIDS;
#   * and it duplicated the gpu-tests job's own editable install of the current source.
# A CI image should carry the slow, stable stack and let each run test the exact commit fresh.
#
# ==============================================================================================
# BASE CHANGE 2026-08-24 (Phase 17, ROUND 4) — metapackage base -> SLIM base, RAPIDS STILL 25.04
# ==============================================================================================
# WAS:  FROM rapidsai/base:25.04-cuda12.8-py3.12   (ubuntu 24.04 / CUDA 12.8.0 / py3.12)
# NOW:  FROM rapidsai/miniforge-cuda:25.08-cuda12.8.0-base-ubuntu24.04-py3.12
#
# SAME OS, SAME CUDA, SAME PYTHON, SAME RAPIDS VERSION. Read from the two image configs, not
# assumed: both carry `org.opencontainers.image.ref.name=ubuntu` / `version=24.04`,
# `CUDA_VERSION=12.8.0`, `PYTHON_VERSION=3.12` and conda at `/opt/conda` already on PATH. What
# changes is ONLY the image VARIANT: `rapidsai/base` ships the `rapids=25.4` METAPACKAGE as the
# environment root spec; `rapidsai/miniforge-cuda` ships conda + CUDA and nothing else. The
# RAPIDS packages this image needs are installed BY NAME below, still at 25.04.
#
# NOTE ON THE TAG: `rapidsai/miniforge-cuda` has NO `25.04-` prefixed tag (queried read-only
# against the registry tag list: 1280 tags, zero starting `25.04`, oldest versioned prefix is
# `25.08`). The prefix is that image's own BUILD CYCLE and carries no RAPIDS packages, so it does
# not set the RAPIDS version — the specs below do. `25.08-` is preferred over the unversioned
# rolling `cuda12.8.0-base-ubuntu24.04-py3.12` because the rolling tag is re-pushed. Index digest
# at the time of writing: sha256:75aa2e766ef90c9024f503f0672e9c1dc24122822e01d8777a35782374058dab
#
# WHY, in one sentence: three consecutive build failures were all caused by SUBTRACTING from the
# metapackage and letting the conda solver re-plan the environment. This inverts that — build UP
# from a base that never had the capping packages, so there is no numba ceiling to fight.
#
# The full three-round record, with verbatim solver transcripts, is at
# `.planning/phases/17-gpu-corridor-and-0-6-0-adoption/17-06-SOLVER-OUTPUT.md`. Condensed:
#   R1 (run 32493219857) — plain `conda install numba=0.61.2`: LibMambaUnsatisfiableError.
#       `cuxfilter 25.04` requires `numba >=0.59.1,<0.61.0a0` and is a leaf of `rapids=25.4`.
#       cudf and cupy were NOT implicated — this was never D-17-17's escalation.
#   R2 (run 32498609639) — `conda remove rapids cuxfilter dask-cuda`: libmamba RE-SOLVES from the
#       surviving history specs rather than excising names, and `rapids` is this environment's
#       ROOT EXPLICIT SPEC, so 350 packages went, cudf/cuspatial/geopandas among them. Caught by
#       the cascade assertion. It also proved the positive: with the cap lifted, conda resolves
#       numba 0.61.2 + cupy 13.6.0 natively in 12.35 s, no channel widening — D-17-17 vindicated.
#   R3 (run 32700116989) — `conda remove --force-remove`: surgical (3 packages, 12 s, the three
#       GPU imports survived) but it left `cugraph`/`raft-dask` requiring an absent
#       `dask-cuda 25.04.*`; the next solve repaired that dangling edge by DOWNGRADING 103
#       packages and re-basing the whole RAPIDS stack 25.04 -> 24.12, where `dask-cuda` declares
#       `numba >=0.57` with no ceiling. Caught by the DROP CHECK clause added that round — every
#       other clause passed. A `print` beside a passing assert is the shape of a silent green,
#       which is why the assertion layer below asserts EVERY version this image must carry.
#
# EXPLICIT MAINTAINER OVERRIDE. The standing instruction in `17-06-PLAN.md` was "do not touch the
# base tag — that is D-17-18's `newer-rapids-base` fallback". That prohibition was written to
# block a RAPIDS VERSION change. This is not one: RAPIDS stays at 25.04 and only the image
# variant moves, from metapackage-bearing to slim. The maintainer authorised this specific change
# with that distinction stated. `newer-rapids-base` is separately DEAD BY MEASUREMENT: the
# `rapidsai` channel has no `cuspatial` above 25.04.00 at all, so a newer RAPIDS base would ship
# no matching cuspatial.
FROM rapidsai/miniforge-cuda:25.08-cuda12.8.0-base-ubuntu24.04-py3.12

USER root

# git           — setuptools-scm versions the editable pchandler install at job time.
# openssh-client — clones the GSEGUtils git+ssh dependency at job time
#                  (simplifies to https/PyPI once the repos go public at milestone close).
# Built on rootful GitHub-hosted `ubuntu-latest` (D-04a — the lab box only pulls + runs, never
# builds), so standard apt works. (A rootless build on the box would fail apt's _apt-sandbox and
# openssh's sgid postinst for lack of a /etc/subuid range — not relevant to the CI build path.)
RUN apt-get update \
    && apt-get install -y --no-install-recommends git openssh-client \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------------------------
# THE GPU STACK, INSTALLED BY NAME (Phase 17 — D-17-16, D-17-17; ROADMAP success criterion 3).
#
# `pchandler.filters.gpu` imports exactly three things: cudf, cuspatial, geopandas
# (`src/pchandler/filters/gpu.py`). Those, plus the corridor pins, plus what the gpu-tests job
# itself needs, are the whole of this environment. Nothing else is installed, and in particular
# `cuxfilter` and `dask-cuda` — the only two packages reachable from `rapids=25.4` that cap numba
# below 0.61 — are never present to cap anything. That is the entire premise of this layer, and
# the assertion layer below fails the build if either of them turns up anyway.
#
# WHY numba IS PINNED AT ALL. numba is the SOLE blocker for the numpy floor GSEGUtils 0.6.0
# requires, and it is a HARD RUNTIME GUARD rather than a declared cap:
# `numba/__init__.py::_ensure_critical_deps` raises
#     ImportError: Numba needs NumPy 2.0 or less.
# at `import cudf` — i.e. BEFORE the `cudf.DataFrame({"x": [1]})` smoke probe D-06 was designed
# around ever runs. numba 0.60.0 (what an unpinned solve would take) has a numpy <= 2.0 ceiling,
# so the moment the gpu-tests job installs `.[dev]` under `.github/constraints/gpu.txt`
# (`numpy >= 2.2, < 2.3`), `is_gpu_available()` returns False, all three GPU tests skip, pytest
# exits 0 and the required context `Tests (pytest, GPU)` reports green having tested nothing.
# numba 0.61.2 declares `numpy<2.3,>=1.24` and is accepted by cudf 25.4 (`numba<0.62.0a0`).
# 0.62/0.63 are NOT options — cudf 25.4 refuses them. numpy 2.2.6 + numba 0.61.2 + cupy +
# cudf/cuspatial 25.4 was measured end to end through a real `cuspatial.point_in_polygon` kernel
# on an RTX 3090 Ti: `.planning/spikes/004-numpy-floor-vs-gpu-stack/README.md`, row 4.
#
# WHY cupy IS BOUNDED `>=13.5,<14`: that is what criterion 3 and D-17-17 ask for — NOT the 14.1.1
# the spike's pip resolver happened to pick. cupy 14 is a major bump validated on exactly one
# kernel. R2 measured that `<14` solves natively, at 13.6.0, above the `>= 13.5` floor.
#
# WHY THE numpy IN THIS IMAGE IS 2.0.x AND THAT IS CORRECT — read this before "fixing" it.
# The conda build of `cudf 25.4.0` declares `numpy >=1.23,<2.1` (read from the rapidsai channel's
# own release metadata for build `cuda12_py312_250409_6bc42063`, the only linux-64/py312/cuda12
# build there is). The PyPI wheel `cudf-cu12 25.4.0` declares `numpy<3.0a0,>=1.23` — a MATERIAL
# METADATA DIVERGENCE between the two distribution channels for the same version, and the second
# such divergence this phase has found (the first was cuml/cuxfilter's numba cap, which exists on
# conda and differs on PyPI). Consequences, both deliberate:
#   * conda CANNOT put numpy 2.2.x in this image while cudf 25.04 is installed. Any assertion
#     demanding `numpy >= 2.2` here is unsatisfiable by construction, not by accident.
#   * It does not need to. The corridor's numpy arrives AT JOB TIME from
#     `.github/constraints/gpu.txt` via `pip install -c ... -e .[dev]` (D-17-10, D-17-14, wired in
#     17-05) — the one seam where numpy is decided. R3 measured the same thing on the old base:
#     "numpy was never moved by this install." Spike 004 row 4 then measured the lifted state
#     working end to end on real hardware, so conda's `<2.1` is a conservative packaging bound,
#     not a runtime wall like numba's.
# The assertion below therefore pins numpy to exactly `2.0.x` — the band cudf and numba 0.61.2
# jointly permit — so that a drift in EITHER direction fails the build loudly.
#
# WHY `python=3.12` IS PINNED EXPLICITLY: a NEW risk the slim base introduces. `rapidsai/base`
# carried `python 3.12.*` in its own pinned specs (visible in R1's solver output as
# "pin on python 3.12.*"); miniforge-cuda does not. Without this spec a solve is free to take a
# py311 build of cudf and move the interpreter under the whole image.
#
# WHY `pip` IS NAMED: the gpu-tests job runs `pip install -c .github/constraints/gpu.txt -e .[dev]`
# INSIDE this container. On the metapackage base pip arrived with the RAPIDS stack; on a slim base
# it must be asked for. Its absence would not surface until 17-07, on lab hardware.
#
# `cuda-version>=12.0,<=12.8` is RAPIDS' own documented form for a 25.04 / CUDA 12.8 install — a
# range rather than `=12.8`, which leaves the solver room while keeping the conda-side CUDA at or
# below the image's own `CUDA_VERSION=12.8.0`.
#
# CHANNELS ARE NAMED EXPLICITLY HERE, and that is not a widening. `rapidsai/base` shipped a
# `.condarc` configuring rapidsai/pytorch/conda-forge/nvidia; miniforge configures conda-forge
# only, and cudf/cuspatial live on the `rapidsai` channel. `-c rapidsai -c conda-forge -c nvidia`
# is the channel set RAPIDS documents for its own installs and the same set (minus pytorch, which
# nothing here needs) the old base configured for itself.
ENV PATH=/opt/conda/bin:${PATH}

# One line, deliberately: `17-06-PLAN.md` Task 1s gate matches the specs against the conda
# install with a single-line regex (`conda install[^\n]*numba[=\s]*0\.61\.2`), so splitting the
# spec list across continuations would break a check that is not wrong.
RUN conda install -y -c rapidsai -c conda-forge -c nvidia "python=3.12" "cuda-version>=12.0,<=12.8" "cudf>=25.4.0,<25.5.0a0" "cuspatial>=25.4.0,<25.5.0a0" "geopandas" "pip" "numba=0.61.2" "cupy>=13.5,<14" \
    && conda clean -afy

# ---------------------------------------------------------------------------------------------
# 17-07 AMENDMENT (2026-08-24) — THE INSTALL SEAM. Two coupled defects, both MEASURED on the very
# first lab dispatch of this image: run 32705493768, job 97365584143, on gseg-pc105.
#
# WHAT HAPPENED. "Pre-flight GPU health check" PASSED on this image. cudf, cuspatial and geopandas
# imported, a real numba kernel ran on the RTX 3090 Ti, and "cuspatial smoke OK" printed at
# 09:17:04. Half a second later the next step died BEFORE pytest ever started:
#
#     error: externally-managed-environment
#     x This environment is externally managed
#     |-> To install Python packages system-wide, try apt install python3-xyz ...
#         See /usr/share/doc/python3.13/README.venv for more information.
#
# READ THE LAST LINE BEFORE CONCLUDING ANYTHING. It names python3.13, and the whole message is
# Debian/Ubuntu PEP 668 text. THIS environment is conda python 3.12 — the same log shows
# /opt/conda/lib/python3.12/site-packages/numba_cuda/... one second earlier. So the pip that ran
# was NOT this environment pip. It was the DISTRO pip, backed by the system interpreter, which
# has no cudf, no cuspatial and no geopandas in it and never will.
#
# WHY, mechanically. gpu.yml runs its container body as `podman run --user 0 ... bash -lc`, i.e.
# a ROOT LOGIN SHELL. Debian /etc/profile ASSIGNS PATH outright for uid 0 —
# PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" — discarding the
# ENV PATH=/opt/conda/bin:... this Dockerfile sets. `python` still resolved to conda ONLY because
# Ubuntu ships no /usr/bin/python for it to shadow; /usr/bin/pip DOES exist, so `pip` and
# `python` came from two different interpreters. The metapackage base masked this; the slim base
# does not. It is an unforeseen consequence of the 2026-08-24 base VARIANT change, not of the
# corridor.
#
# THE FIX IS AT THE SOURCE, and both halves are needed. (a) A login shell must resolve BOTH `pip`
# and `python` to this environment. (b) If this environment carries a PEP 668 marker, the marker
# is wrong for the image role — a CI runner image exists to have the project under test installed
# into it — so it is cleared. Maintainer decision, 2026-08-24.
#
# `--break-system-packages` in gpu.yml was CONSIDERED AND REJECTED. It would have silenced the
# marker while leaving the install pointed at the wrong interpreter — turning a loud failure into
# a silent one, and carrying the workaround in CI config instead of fixing the image.
#
# THE DISTRO PYTHON KEEPS ITS OWN MARKER, DELIBERATELY. Nothing should ever be installed there,
# and leaving it in place is what makes the login-shell assertion below load-bearing rather than
# decorative: if PATH ever regresses, pip hits that marker again and the BUILD dies here, on a
# GPU-less hosted runner, instead of on the lab host after a human approval click.
# ---------------------------------------------------------------------------------------------

# (a) PATH. Sorted last in /etc/profile.d so it is the final word on PATH in any login shell,
# whatever the base image own conda hooks do before it.
RUN set -eu; \
    printf '%s\n' \
      '# Phase 17 / plan 17-07, 2026-08-24. Debian /etc/profile ASSIGNS PATH for uid 0, which' \
      '# discards the ENV PATH this image sets. gpu.yml runs its container body under bash -lc' \
      '# as root, so without this file `pip` resolves to the DISTRO pip (system python3.13)' \
      '# while `python` resolves to conda python 3.12 — two different interpreters. Measured in' \
      '# run 32705493768. Do not remove without reading the block above it in the Dockerfile.' \
      'export PATH="/opt/conda/bin:$PATH"' \
      > /etc/profile.d/zzz-conda-first.sh; \
    chmod 0644 /etc/profile.d/zzz-conda-first.sh; \
    cat /etc/profile.d/zzz-conda-first.sh

# (b) PEP 668. Sweep the CONDA PREFIX ONLY. The diagnostic listing is printed first and covers the
# whole filesystem, so the log records what was there whether or not anything was removed — a
# silent no-op and a clean environment must not look identical.
RUN set -eu; \
    echo "PEP 668 markers present anywhere in the image BEFORE the sweep:"; \
    find / -xdev -name EXTERNALLY-MANAGED -type f -print 2>/dev/null || true; \
    echo "PEP 668 markers removed from the conda prefix:"; \
    find /opt/conda -name EXTERNALLY-MANAGED -type f -print -delete; \
    echo "PEP 668 sweep complete (conda prefix only; the distro python keeps its marker by design)"

# BUILD-TIME ASSERTION THAT THE PIN TOOK (threat T-17-24). Read the resolved versions from
# conda's OWN metadata, NOT by importing the packages: this image is built on a GPU-less
# `ubuntu-latest` runner, where importing a CUDA library fails for an entirely different reason
# with the same symptom. A mismatch fails this layer and therefore the whole build, so the
# registry cannot hold an image that does not carry the pinned stack — "the image carries numba
# 0.61.2" becomes a property the build enforces instead of a claim in a comment. conda-forge
# splits cupy into `cupy` (metapackage) and `cupy-core`; either name satisfies the check.
#
# EVERY VERSION THIS IMAGE MUST CARRY IS ASSERTED, NOT PRINTED. R3 came within one clause of
# pushing a silently re-based RAPIDS 24.12 stack whose only witness was `cudf=24.12.00` inside a
# `print`. The printed line at the end is a RECORD for the build log; it guards nothing. The
# absence checks for `cuxfilter`/`dask-cuda`/`rapids` are what make this base change falsifiable:
# if the slim premise is wrong and a capping branch reappears, this build dies here rather than
# shipping an image that quietly resolved numba somewhere else.
RUN conda list --json > /tmp/conda-pins.json \
    && python -c 'import json, pathlib, re, sysconfig; \
pkgs = {p["name"]: p["version"] for p in json.load(open("/tmp/conda-pins.json"))}; \
mm = lambda n: tuple(int(x) for x in re.findall(r"[0-9]+", pkgs.get(n) or "")[:2]); \
assert mm("python") == (3, 12), "PIN CHECK FAILED: python resolved to %r, expected 3.12.x — the slim base carries no python pin of its own, so an unpinned solve can move the interpreter" % (pkgs.get("python"),); \
assert pkgs.get("numba") == "0.61.2", "PIN CHECK FAILED: numba resolved to %r, expected exactly 0.61.2" % (pkgs.get("numba"),); \
assert mm("numpy") == (2, 0), "PIN CHECK FAILED: numpy resolved to %r, expected 2.0.x — conda cudf 25.4.0 declares numpy>=1.23,<2.1 and numba 0.61.2 declares numpy>=1.24,<2.3; the corridors 2.2.x arrives at JOB time from .github/constraints/gpu.txt, not here" % (pkgs.get("numpy"),); \
cp = pkgs.get("cupy") or pkgs.get("cupy-core"); \
m = re.match(r"^13\.([0-9]+)([^0-9]|$)", cp or ""); \
assert m is not None and int(m.group(1)) >= 5, "PIN CHECK FAILED: cupy resolved to %r, expected 13.x with minor >= 5 (D-17-17 refuses cupy 14 as the default)" % (cp,); \
assert mm("cudf") == (25, 4), "PIN CHECK FAILED: cudf resolved to %r, expected 25.04.x — a stack re-base is exactly what R3 nearly shipped" % (pkgs.get("cudf"),); \
assert mm("cuspatial") == (25, 4), "PIN CHECK FAILED: cuspatial resolved to %r, expected 25.04.x" % (pkgs.get("cuspatial"),); \
assert mm("numba-cuda") == (0, 4), "PIN CHECK FAILED: numba-cuda resolved to %r, expected 0.4.x — R3 saw it fall to 0.0.17.1 as the visible edge of a silent 25.04 -> 24.12 re-base" % (pkgs.get("numba-cuda"),); \
assert "geopandas" in pkgs, "PIN CHECK FAILED: geopandas absent — pchandler.filters.gpu imports it"; \
assert "pip" in pkgs, "PIN CHECK FAILED: pip absent — the gpu-tests job needs it to install .[dev] under .github/constraints/gpu.txt inside this image"; \
em = sorted(str(q) for q in list(pathlib.Path(sysconfig.get_path("stdlib")).glob("EXTERNALLY-MANAGED")) + list(pathlib.Path("/opt/conda/lib").glob("python*/EXTERNALLY-MANAGED"))); \
assert not em, "PEP 668 CHECK FAILED: %r is still present in the conda prefix, so pip install -e .[dev] inside this image would be refused by PEP 668 exactly as it was in run 32705493768 — the 17-07 sweep did not take" % (em,); \
back = [n for n in ("cuxfilter", "dask-cuda", "rapids") if n in pkgs]; \
assert not back, "SLIM CHECK FAILED: %r present in an image built from the slim base, so the numba<0.61 capping branch is back and the premise of the 2026-08-24 base change is wrong" % (back,); \
print("PIN CHECK OK: python=%s numba=%s cupy=%s numba-cuda=%s numpy=%s cudf=%s cuspatial=%s geopandas=%s pip=%s" % (pkgs.get("python"), pkgs.get("numba"), cp, pkgs.get("numba-cuda"), pkgs.get("numpy"), pkgs.get("cudf"), pkgs.get("cuspatial"), pkgs.get("geopandas"), pkgs.get("pip"))); \
print("SLIM CHECK OK: rapids/cuxfilter/dask-cuda all absent; cugraph=%s cuml=%s" % (pkgs.get("cugraph"), pkgs.get("cuml"))); \
print("PEP 668 CHECK OK: no EXTERNALLY-MANAGED marker in %s" % (sysconfig.get_path("stdlib"),))' \
    && rm -f /tmp/conda-pins.json

# BUILD-TIME ASSERTION ON THE INSTALL SEAM (17-07). Everything above is read from conda metadata;
# this layer asserts the property the gpu-tests job actually depends on, in the SAME KIND OF SHELL
# the job uses — `bash -l`, a LOGIN shell, as root. It proves three things at once and fails the
# build if any of them is false:
#   1. a login shell resolves `python` inside /opt/conda;
#   2. a login shell resolves `pip`    inside /opt/conda  <- the defect that killed run 32705493768;
#   3. that pip can actually install a package into this environment, i.e. PEP 668 does not refuse
#      it and the files land under /opt/conda.
# Clause 3 is a REAL INSTALL of a throwaway local package, not a version print: run 32705493768 had
# a green pre-flight, a working GPU, a correctly pinned image and a resolvable pip, and still could
# not install anything. Only an install proves an install.
#
# THE DIAGNOSTIC LINES ARE PRINTED BEFORE THE ASSERTIONS, ON PURPOSE. If this layer fails, the
# build log must already say what PATH was and where python and pip resolved, or the next reader
# is back to inferring a cause from an exit code — which is how this phase lost a lab approval.
RUN set -eu; \
    mkdir -p /tmp/pep668-probe/src/pep668probe; \
    printf '%s\n' 'OK = True' > /tmp/pep668-probe/src/pep668probe/__init__.py; \
    printf '%s\n' \
      '[build-system]' \
      'requires = ["setuptools>=68"]' \
      'build-backend = "setuptools.build_meta"' \
      '' \
      '[project]' \
      'name = "pep668probe"' \
      'version = "0.0.0"' \
      > /tmp/pep668-probe/pyproject.toml; \
    printf '%s\n' \
      'set -eu' \
      'echo "SEAM DIAGNOSTIC login-shell PATH: $PATH"' \
      'PY=$(command -v python || true)' \
      'PIP=$(command -v pip || true)' \
      'echo "SEAM DIAGNOSTIC login-shell python: ${PY:-NONE}"' \
      'echo "SEAM DIAGNOSTIC login-shell pip:    ${PIP:-NONE}"' \
      'pip --version || true' \
      'case "${PY:-NONE}" in' \
      '  /opt/conda/*) ;;' \
      '  *) echo "SEAM CHECK FAILED: a login shell resolves python to ${PY:-NONE}, outside /opt/conda"; exit 1 ;;' \
      'esac' \
      'case "${PIP:-NONE}" in' \
      '  /opt/conda/*) ;;' \
      '  *) echo "SEAM CHECK FAILED: a login shell resolves pip to ${PIP:-NONE}, outside /opt/conda. This is exactly the defect that failed run 32705493768: the distro pip refused the install under PEP 668 while python was conda 3.12."; exit 1 ;;' \
      'esac' \
      'pip install --no-deps --quiet /tmp/pep668-probe' \
      'LOC=$(python -c "import pep668probe; print(pep668probe.__file__)")' \
      'case "$LOC" in' \
      '  /opt/conda/*) ;;' \
      '  *) echo "SEAM CHECK FAILED: pip installed the probe to $LOC, outside /opt/conda"; exit 1 ;;' \
      'esac' \
      'pip uninstall -y -q pep668probe' \
      'echo "SEAM CHECK OK: login-shell python=${PY} pip=${PIP}; a real pip install landed at ${LOC}"' \
      > /tmp/seam-check.sh; \
    bash -l /tmp/seam-check.sh; \
    rm -rf /tmp/pep668-probe /tmp/seam-check.sh
# ---------------------------------------------------------------------------------------------

# NOTE: geopandas is now an EXPLICIT spec above rather than an inherited package — the slim base
# ships nothing, and `cuspatial` would pull it transitively (it declares `geopandas >=1.0.0`) but
# pchandler imports it directly, so it is named directly. Kept conda (not pip) to avoid mixing
# package managers in the conda env.

# The image carries NO pchandler/[dev] toolchain. The gpu-tests job installs `.[dev]` fresh
# against the checked-out source each run, so the toolchain and code always match the commit
# under test (no monthly-image drift).
