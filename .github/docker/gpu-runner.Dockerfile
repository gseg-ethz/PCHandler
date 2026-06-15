# Source: CONTEXT.md D-03 (amended) + RESEARCH.md §"Dockerfile shape"
# Base: RAPIDS 25.04 with CUDA 12.8 runtime + Python 3.12 (matches pchandler [cuda12] extras:
#   cudf-cu12==25.4.*, cuspatial-cu12==25.4.*, cuproj-cu12==25.4.*, cuml-cu12==25.4.*,
#   dask-cudf-cu12==25.4.*; the -cu12 wheels use CUDA Enhanced Compatibility across 12.x).
# DEVIATION 2026-06-15: D-03's locked `:25.04-cuda12.5-py3.12` does NOT exist on Docker Hub —
#   stable RAPIDS 25.04 publishes only cuda11.8 / cuda12.0 / cuda12.8 (12.5 is 25.04a-alpha only).
#   Chose cuda12.8 (newest CUDA-12 build for 25.04). Host driver 595.71.05 / CUDA 13.2 supports it.
# (see .planning/phases/09-self-hosted-docker-gpu-runner/09-WAVE-0-RUNNER-MANIFEST.md §Drift).
FROM rapidsai/base:25.04-cuda12.8-py3.12

# Repo source — buildx context is the repo root via gpu-image-refresh.yml's `context: .`.
COPY . /opt/pchandler
WORKDIR /opt/pchandler

# Editable install:
#   [cuda12] pulls the RAPIDS suite pinned to 25.4.*
#   [dev]    pulls ruff, mypy, pyright, pytest, pytest-cov, coverage, pytest-benchmark,
#            pytest-randomly
# Leading `-e .` is load-bearing: ci.yml's `gpu-tests` job runs an additional
# `pip install -e .[cuda12,dev]` inside the container at job time to pick up develop/gsd
# source changes between monthly image refreshes.
RUN pip install -e .[cuda12,dev]
