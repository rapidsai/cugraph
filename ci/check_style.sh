#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

conda config --remove-key custom_multichannels
conda config --get
cat /opt/conda/.condarc

rapids-logger "Create checks conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-dependency-file-generator \
  --output conda \
  --file-key checks \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n checks
conda activate checks

# Run pre-commit checks
pre-commit run --hook-stage manual --all-files --show-diff-on-failure
