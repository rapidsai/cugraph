#!/bin/bash
# Copyright (c) 2018-2022, NVIDIA CORPORATION.
########################
# cuGraph Style Tester #
########################

# Activate common conda env
PATH=/conda/bin:$PATH
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids

# Run pre-commit checks
pre-commit run --hook-stage manual --all-files --show-diff-on-failure