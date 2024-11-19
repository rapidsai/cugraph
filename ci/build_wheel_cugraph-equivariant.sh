#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

package_dir="python/cugraph-equivariant"

./ci/build_wheel.sh cugraph-equivariant ${package_dir}
./ci/validate_wheel.sh ${package_dir} dist
