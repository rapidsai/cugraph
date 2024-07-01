#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -eoxu pipefail

./ci/test_wheel.sh cugraph python/cugraph
