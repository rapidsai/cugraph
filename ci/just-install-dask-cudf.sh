#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# try turning off the proxy cache
unset CONDA_CHANNEL_ALIAS

# try to match 'conda info' from a passing (local) run to see if that matters
export CONDA_OVERRIDE_ARCHSPEC=aarch64
export CONDA_OVERRIDE_LINUX="6.5.0=0"

# clear the cache
conda clean --yes --index-cache
conda clean --yes --all

echo "--- looking for ANY local repodata ---"
find / -name '*repodata*' || true
echo "---"

# check the volumes GitHub mounts in
# -v "/home/runner/_work":"/__w"
# -v "/home/runner/externals":"/__e":ro
# -v "/home/runner/_work/_temp":"/__w/_temp"
# -v "/home/runner/_work/_actions":"/__w/_actions"
# -v "/home/runner/_work/_tool":"/__w/_tool"
# -v "/home/runner/_work/_temp/_github_home":"/github/home"
# -v "/home/runner/_work/_temp/_github_workflow":"/github/workflow"

echo "--- checking HOME (${HOME}) ---"
find "${HOME}" -name '*' || true
echo ""

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
