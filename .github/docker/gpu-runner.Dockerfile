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
# Base: RAPIDS 25.04 / CUDA 12.8 / Python 3.12. (D-03's `:25.04-cuda12.5` does not exist for
# stable 25.04 — Docker Hub has cuda11.8/12.0/12.8; 12.5 is 25.04a-alpha. Host driver
# 595.71.05 / CUDA 13.2 supports 12.8.) Provides cudf + cuspatial via conda — the GPU libs
# pchandler.filters.gpu imports (cudf, cuspatial, geopandas).
FROM rapidsai/base:25.04-cuda12.8-py3.12

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
# GPU DEPENDENCY CORRIDOR PIN (Phase 17 — D-17-16, D-17-17; ROADMAP success criterion 3).
#
# numba is the SOLE blocker for the numpy floor GSEGUtils 0.6.0 requires, and it is a HARD
# RUNTIME GUARD rather than a declared cap: `numba/__init__.py::_ensure_critical_deps` raises
#     ImportError: Numba needs NumPy 2.0 or less.
# at `import cudf` — i.e. BEFORE the `cudf.DataFrame({"x": [1]})` smoke probe D-06 was designed
# around ever runs. The RAPIDS base ships numba 0.60.0, whose ceiling is numpy <= 2.0. So the
# moment the gpu-tests job installs `.[dev]` under `.github/constraints/gpu.txt`
# (`numpy >= 2.2, < 2.3`), the container's `is_gpu_available()` returns False, all three GPU
# tests skip, pytest exits 0 and the required context `Tests (pytest, GPU)` reports green having
# tested nothing. Pinning numba here is what closes that.
#
# numba 0.61.2 declares `numpy<2.3,>=1.24` and is accepted by cudf 25.4 (`numba<0.62`).
# 0.62/0.63 are NOT options — cudf 25.4 refuses them. numpy 2.2.6 + numba 0.61.2 + cupy +
# cudf/cuspatial 25.4 was measured end to end through a real `cuspatial.point_in_polygon` kernel
# on an RTX 3090 Ti: `.planning/spikes/004-numpy-floor-vs-gpu-stack/README.md`, row 4. That
# spike installed PIP WHEELS in a venv, and its own § Name the Residual says so — "a venv is not
# the conda container ... whether conda will produce numba 0.61.2 with cudf 25.4 is untested,
# and it is the next question". This layer, and the assertion layer below it, are the answer.
#
# cupy is bounded `>=13.5,<14` because that is what criterion 3 and D-17-17 ask for — NOT the
# 14.1.1 the spike's resolver happened to pick. cupy 14 is a major bump validated on exactly one
# kernel, and refusing it as the default is deliberate. If `<14` cannot solve against cudf 25.4
# that is an escalation (D-17-17), not a range edit.
#
# Installed through CONDA, from the base image's OWN configured channels, honouring the standing
# instruction in this file not to mix package managers inside the conda env (D-17-16). Both
# packages are already present in the RAPIDS base — this PINS them, it does not introduce them.
ENV PATH=/opt/conda/bin:${PATH}
RUN conda install -y "numba=0.61.2" "cupy>=13.5,<14" \
    && conda clean -afy

# BUILD-TIME ASSERTION THAT THE PIN TOOK (threat T-17-24). Read the resolved versions from
# conda's OWN metadata, NOT by importing the packages: this image is built on a GPU-less
# `ubuntu-latest` runner, where importing a CUDA library fails for an entirely different reason
# with the same symptom. A mismatch fails this layer and therefore the whole build, so the
# registry cannot hold an image that does not carry the pinned stack — "the image carries numba
# 0.61.2" becomes a property the build enforces instead of a claim in a comment. conda-forge
# splits cupy into `cupy` (metapackage) and `cupy-core`; either name satisfies the check.
# The resolved versions are printed so the build log records what actually landed.
RUN conda list --json > /tmp/conda-pins.json \
    && python -c 'import json, re; \
pkgs = {p["name"]: p["version"] for p in json.load(open("/tmp/conda-pins.json"))}; \
nb = pkgs.get("numba"); \
cp = pkgs.get("cupy") or pkgs.get("cupy-core"); \
assert nb == "0.61.2", "PIN CHECK FAILED: numba resolved to %r, expected exactly 0.61.2" % (nb,); \
m = re.match(r"^13\.([0-9]+)([^0-9]|$)", cp or ""); \
assert m is not None and int(m.group(1)) >= 5, "PIN CHECK FAILED: cupy resolved to %r, expected 13.x with minor >= 5" % (cp,); \
print("PIN CHECK OK: numba=%s cupy=%s numba-cuda=%s numpy=%s cudf=%s cuspatial=%s" % (nb, cp, pkgs.get("numba-cuda"), pkgs.get("numpy"), pkgs.get("cudf"), pkgs.get("cuspatial")))' \
    && rm -f /tmp/conda-pins.json
# ---------------------------------------------------------------------------------------------

# NOTE: geopandas — pchandler.filters.gpu imports it. If the RAPIDS base does not already
# ship it (verified during 09-02 local build), uncomment the conda install below. Kept conda
# (not pip) to avoid mixing package managers in the conda env.
# RUN conda install -y -c conda-forge geopandas && conda clean -afy

# The image carries NO pchandler/[dev] toolchain. The gpu-tests job installs `.[dev]` fresh
# against the checked-out source each run, so the toolchain and code always match the commit
# under test (no monthly-image drift).
