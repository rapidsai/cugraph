#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# try turning off the proxy cache
unset CONDA_CHANNEL_ALIAS

echo "--- trying to install 'dask-cudf' ---"
conda config --system --remove channels rapidsai-nightly
conda info
conda install \
  -vv \
  --yes \
  -c rapidsai \
  -c conda-forge \
  'dask-cudf=26.6.*'
echo "---"
