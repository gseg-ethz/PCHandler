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
# packages are already present in the RAPIDS base, but "this PINS them, it does not introduce
# them" is only half true and round 2 measured which half: the base ships numba 0.60.0 (pinned
# UP to 0.61.2) and cupy 13.4.1 — which is BELOW the `>= 13.5` floor criterion 3 asks for, so the
# cupy half of this pin always had to MOVE a package, not merely ratify one.
ENV PATH=/opt/conda/bin:${PATH}

# ---- MEASURED 2026-08-21: the FIRST attempt at the pin below failed, and this layer is why ----
# Refresh run 32493219857 (job 96805513018) died after 66 s of solving with, verbatim:
#
#     LibMambaUnsatisfiableError: Encountered problems while solving:
#       - package cuxfilter-25.04.00-cuda12_py312_250409_gf64a6f4_0 requires
#         numba >=0.59.1,<0.61.0a0, but none of the providers can be installed
#
# The full 56-line transcript is on the record at
# `.planning/phases/17-gpu-corridor-and-0-6-0-adoption/17-06-SOLVER-OUTPUT.md`.
#
# THE BLOCKER IS NEITHER cudf NOR cupy. cudf 25.04 declares no numba constraint at all on conda,
# and `cupy>=13.5,<14` appears NOWHERE in the conflict tree — so this is not D-17-17's escalation,
# the range below is NOT widened, and widening it would not have helped. The blocker is
# `cuxfilter`, a RAPIDS dashboard / visualisation component that pchandler never imports
# (`src/pchandler/filters/gpu.py` imports cudf, cuspatial and geopandas, and nothing else). It is
# in this image only because the base installs the `rapids=25.4` METAPACKAGE, one leaf of which is
# `cuxfilter 25.04.*`.
#
# `dask-cuda 25.04` declares the SAME cap (`numba >=0.59.1,<0.61.0a0`) and is a SECOND leaf of the
# same metapackage. The solver never named it — LibMamba reports the first conflict it proves — so
# dropping cuxfilter alone would have bought a second, identical 7-minute failure. Read out of the
# `rapidsai` channel's own `linux-64/repodata.json`: across every 25.04 build exactly three
# packages cap numba below 0.61 (`cuxfilter`, `dask-cuda`, `cugraph-service-server`), and only the
# first two are reachable from `rapids`. `cugraph-dgl` / `cugraph-pyg` declare `numba >=0.57` with
# no ceiling, and are therefore not blockers.
#
# ---- MEASURED 2026-08-21, ROUND 2: a plain `conda remove` of that branch GUTS THE IMAGE ----
# Refresh run 32498609639 (job 96822856086) ran the drop as a plain `conda remove -y` and conda
# removed **350 packages**, verbatim from the transaction: cudf 25.4.0, cuspatial 25.04.00,
# geopandas 1.0.1, libcudf, libcuspatial, rmm, cuml, cuproj, cucim, cuvs, cugraph, raft-dask,
# pandas, pyarrow, shapely — plus numba 0.60.0, numpy 2.0.2 and cupy 13.4.1. `--prune` was NOT
# passed and made no difference.
#
# The prediction above was right about the DEPENDENCY GRAPH and wrong about the MECHANISM.
# `rapids=25.4` is this environment's ROOT EXPLICIT SPEC. Under the libmamba solver `conda remove`
# does not excise named packages — it RE-SOLVES the environment from the surviving history specs,
# so the governing rule is "what has no surviving explicit spec", not "who depends on what". With
# `rapids` gone from the spec set, everything whose only justification was `rapids` goes with it.
#
# The solve itself was never the problem. With the cap lifted, layer #10 solved in 12.35 s from
# this base's own channels with no `-c` and nothing widened, resolving numba 0.61.2 + cupy 13.6.0
# + numpy 2.2.6 — exactly the corridor spike 004 measured on an RTX 3090 Ti, and at `cupy < 14`,
# which vindicates D-17-17. The cascade assertion below is what stopped that green-looking build
# from pushing an image with the right numba and NO cudf.
#
# ---- ROUND 3, 2026-08-24: `--force-remove`, on an explicit one-attempt budget extension ----
# `conda remove --force-remove` skips dependency resolution entirely: it unlinks exactly the named
# packages and does not re-solve, so the 350-package prune cannot happen. That is the whole of the
# change from round 2. Two things about it are UNKNOWN and are what this attempt buys:
#   (a) whether dropping `rapids` from the environment's history/explicit specs is enough to stop
#       the next `conda install` reinstating `cuxfilter` (and with it the numba<0.61 cap). The
#       assertion layer below fails the build if either `cuxfilter` or `dask-cuda` comes back.
#   (b) whether an environment with force-removed packages is SOUND enough to trust on lab
#       hardware. It is not: conda's metadata will still claim `cugraph` / `raft-dask` are
#       installed while `dask-cuda` — which they require — is gone. pchandler imports only cudf,
#       cuspatial and geopandas, none of which reach the force-removed branch, so the corridor
#       this image exists to carry is unaffected; but this image is deliberately NOT a
#       general-purpose RAPIDS environment any more, and nothing here proves it is coherent
#       beyond those three imports. `17-07`'s real `3 passed` on `gseg-pc105` measures the three
#       imports and the kernel — it does not measure the rest of the environment.
#
# THIS REMAINS A MEASUREMENT, NOT AN ADOPTION. D-17-18's three fallbacks are still unchosen. The
# image is not permitted to be quietly wrong either way: the layer immediately after the drop
# fails the build if the force-remove reached pchandler's three GPU imports, and the layer after
# the pin fails it if numba, cupy, cudf, cuspatial or geopandas did not land exactly where they
# were meant to, or if the dropped branch came back.
RUN conda list --json > /tmp/pre-drop.json \
    && python -c 'import json; \
names = ["rapids", "cuxfilter", "dask-cuda"]; \
installed = {p["name"]: p["version"] for p in json.load(open("/tmp/pre-drop.json"))}; \
drop = [n for n in names if n in installed]; \
skip = [n for n in names if n not in installed]; \
print("DROP (installed, removing): " + (", ".join("%s=%s" % (n, installed[n]) for n in drop) or "-")); \
print("DROP (absent, nothing to do): " + (", ".join(skip) or "-")); \
open("/tmp/drop.txt", "w").write(" ".join(drop))' \
    && DROP="$(cat /tmp/drop.txt)" \
    && if [ -n "$DROP" ]; then conda remove -y --force-remove $DROP && conda clean -afy; \
       else echo "NOTHING DROPPED — the numba cap is somewhere else; the solve below will name it"; fi \
    && conda list --json > /tmp/post-drop.json \
    && python -c 'import json; \
pkgs = {p["name"]: p["version"] for p in json.load(open("/tmp/post-drop.json"))}; \
missing = [n for n in ("cudf", "cuspatial", "geopandas") if n not in pkgs]; \
assert not missing, "FORCE-REMOVE CASCADE CHECK FAILED: --force-remove still took %r with it — the drop itself is the cause, not the install that follows" % (missing,); \
print("POST-DROP OK: cudf=%s cuspatial=%s geopandas=%s | numba=%s cupy=%s numpy=%s | rapids=%s cuxfilter=%s dask-cuda=%s cugraph=%s" % (pkgs.get("cudf"), pkgs.get("cuspatial"), pkgs.get("geopandas"), pkgs.get("numba"), pkgs.get("cupy") or pkgs.get("cupy-core"), pkgs.get("numpy"), pkgs.get("rapids"), pkgs.get("cuxfilter"), pkgs.get("dask-cuda"), pkgs.get("cugraph")))' \
    && rm -f /tmp/pre-drop.json /tmp/post-drop.json /tmp/drop.txt

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
# EXTENDED 2026-08-21 alongside the drop layer above, because that layer introduces a new way
# for this image to be wrong: it now also fails the build if the cascade reached cudf,
# cuspatial or geopandas (the three `pchandler.filters.gpu` imports), and if `cuxfilter`
# survived — a build that kept the capping package proves nothing even with numba at 0.61.2.
RUN conda list --json > /tmp/conda-pins.json \
    && python -c 'import json, re; \
pkgs = {p["name"]: p["version"] for p in json.load(open("/tmp/conda-pins.json"))}; \
nb = pkgs.get("numba"); \
cp = pkgs.get("cupy") or pkgs.get("cupy-core"); \
assert nb == "0.61.2", "PIN CHECK FAILED: numba resolved to %r, expected exactly 0.61.2" % (nb,); \
m = re.match(r"^13\.([0-9]+)([^0-9]|$)", cp or ""); \
assert m is not None and int(m.group(1)) >= 5, "PIN CHECK FAILED: cupy resolved to %r, expected 13.x with minor >= 5" % (cp,); \
missing = [n for n in ("cudf", "cuspatial", "geopandas") if n not in pkgs]; \
assert not missing, "CASCADE CHECK FAILED: dropping the rapids/cuxfilter/dask-cuda branch took %r with it — pchandler.filters.gpu imports cudf, cuspatial and geopandas, so an image without them is useless whatever numba says" % (missing,); \
back = [n for n in ("cuxfilter", "dask-cuda") if n in pkgs]; \
assert not back, "DROP CHECK FAILED: %r came back on the install, so dropping rapids from the history specs did NOT stop the solver reinstating the numba<0.61 branch" % (back,); \
print("PIN CHECK OK: numba=%s cupy=%s numba-cuda=%s numpy=%s cudf=%s cuspatial=%s geopandas=%s" % (nb, cp, pkgs.get("numba-cuda"), pkgs.get("numpy"), pkgs.get("cudf"), pkgs.get("cuspatial"), pkgs.get("geopandas"))); \
print("DROP CHECK OK: cuxfilter absent; rapids=%s dask-cuda=%s cugraph=%s" % (pkgs.get("rapids"), pkgs.get("dask-cuda"), pkgs.get("cugraph")))' \
    && rm -f /tmp/conda-pins.json
# ---------------------------------------------------------------------------------------------

# NOTE: geopandas — pchandler.filters.gpu imports it. If the RAPIDS base does not already
# ship it (verified during 09-02 local build), uncomment the conda install below. Kept conda
# (not pip) to avoid mixing package managers in the conda env.
# RUN conda install -y -c conda-forge geopandas && conda clean -afy

# The image carries NO pchandler/[dev] toolchain. The gpu-tests job installs `.[dev]` fresh
# against the checked-out source each run, so the toolchain and code always match the commit
# under test (no monthly-image drift).
