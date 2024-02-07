#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

# Support invoking run_cugraph_benchmark_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../benchmarks

pytest --capture=no --benchmark-disable -m tiny "$@" cugraph/pytest-based/bench_algos.py
