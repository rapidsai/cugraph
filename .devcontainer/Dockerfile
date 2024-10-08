# syntax=docker/dockerfile:1.5

ARG BASE
ARG PYTHON_PACKAGE_MANAGER=conda

FROM ${BASE} as pip-base

ENV DEFAULT_VIRTUAL_ENV=rapids

RUN apt update -y \
 && DEBIAN_FRONTEND=noninteractive apt install -y \
    libblas-dev liblapack-dev \
 && rm -rf /tmp/* /var/tmp/* /var/cache/apt/* /var/lib/apt/lists/*;

FROM ${BASE} as conda-base

ENV DEFAULT_CONDA_ENV=rapids

FROM ${PYTHON_PACKAGE_MANAGER}-base

ARG CUDA
ENV CUDAARCHS="RAPIDS"
ENV CUDA_VERSION="${CUDA_VERSION:-${CUDA}}"

ARG PYTHON_PACKAGE_MANAGER
ENV PYTHON_PACKAGE_MANAGER="${PYTHON_PACKAGE_MANAGER}"

ENV PYTHONSAFEPATH="1"
ENV PYTHONUNBUFFERED="1"
ENV PYTHONDONTWRITEBYTECODE="1"

ENV SCCACHE_REGION="us-east-2"
ENV SCCACHE_BUCKET="rapids-sccache-devs"
ENV VAULT_HOST="https://vault.ops.k8s.rapids.ai"
ENV HISTFILE="/home/coder/.cache/._bash_history"

# cugraph_pyg's setup.py needs this defined when building in a conda env
ENV CUDA_HOME="${CUDA_HOME:-/home/coder/.conda/envs/$DEFAULT_CONDA_ENV}"
