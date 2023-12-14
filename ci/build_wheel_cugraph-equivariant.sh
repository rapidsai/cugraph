#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

./ci/build_wheel.sh cugraph-equivariant python/cugraph-equivariant
