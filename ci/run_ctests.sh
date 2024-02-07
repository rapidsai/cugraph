#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

# Support customizing the ctests' install location
cd "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/gtests/libcugraph/"
ctest --output-on-failure "$@"

if [ -d "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/gtests/libcugraph_c/" ]; then
  # Support customizing the ctests' install location
  cd "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/gtests/libcugraph_c/"
  ctest --output-on-failure "$@"
fi
