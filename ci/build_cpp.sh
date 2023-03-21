#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

set -euo pipefail

source rapids-env-update

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-logger "Begin cpp build"

LIBRAFT_CHANNEL=$(rapids-get-artifact ci/raft/pull-request/1333/c60c0cb/raft_conda_cpp_cuda11_$(arch).tar.gz)

rapids-mamba-retry mambabuild --channel "${LIBRAFT_CHANNEL}" conda/recipes/libcugraph

rapids-upload-conda-to-s3 cpp
