#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Support customizing the ctests' install location
cd "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/gtests/libcugraph/"
ctest --output-on-failure --no-tests=error "$@"

if [ -d "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/gtests/libcugraph_c/" ]; then
  # Support customizing the ctests' install location
  cd "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/gtests/libcugraph_c/"
  ctest --output-on-failure --no-tests=error "$@"
fi
