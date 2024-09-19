#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -eoxu pipefail

source ./ci/use_wheels_from_prs.sh

./ci/test_wheel.sh pylibcugraph python/pylibcugraph
