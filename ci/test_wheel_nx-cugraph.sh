#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -eoxu pipefail

# special case: package name needs to use underscores since the test script
# expects it to match the file name exactly. pip should allow - or _ when
# installing, so users should not need to worry about this.
./ci/test_wheel.sh nx_cugraph python/nx-cugraph
