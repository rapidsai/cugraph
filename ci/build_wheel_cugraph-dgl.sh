#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

./ci/build_wheel.sh cugraph-dgl python/cugraph-dgl
