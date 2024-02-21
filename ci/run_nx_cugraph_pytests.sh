#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

# Support invoking run_nx_cugraph_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/nx-cugraph/nx_cugraph

pytest --capture=no --cache-clear --benchmark-disable "$@" tests
