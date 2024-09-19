#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

source ./ci/use_wheels_from_prs.sh

./ci/build_wheel.sh cugraph-dgl python/cugraph-dgl
