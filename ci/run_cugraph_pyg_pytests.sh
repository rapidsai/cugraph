#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

# Support invoking run_cugraph_pyg_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/cugraph-pyg/cugraph_pyg

pytest --cache-clear --ignore=tests/mg "$@" .

# Test examples
for e in "$(pwd)"/examples/*; do
  rapids-logger "running example $e"
  python $e
done
