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

# NOTE: geopandas — pchandler.filters.gpu imports it. If the RAPIDS base does not already
# ship it (verified during 09-02 local build), uncomment the conda install below. Kept conda
# (not pip) to avoid mixing package managers in the conda env.
# RUN conda install -y -c conda-forge geopandas && conda clean -afy

# The image carries NO pchandler/[dev] toolchain. The gpu-tests job installs `.[dev]` fresh
# against the checked-out source each run, so the toolchain and code always match the commit
# under test (no monthly-image drift).
