#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

# Support invoking run_cugraph_pyg_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/cugraph-pyg/cugraph_pyg

pytest --cache-clear --benchmark-disable "$@" .

# Test examples
for e in "$(pwd)"/examples/*.py; do
  rapids-logger "running example $e"
  (yes || true) | python $e
done
