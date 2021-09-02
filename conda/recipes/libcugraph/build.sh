#!/usr/bin/env bash

# Copyright (c) 2019-2021, NVIDIA CORPORATION.

# This assumes the script is executed from the root of the repo directory

# NOTE: the "libcugraph" target also builds all single-gpu gtest binaries, and
# the "cpp-mgtests" target builds the multi-gpu gtest binaries (and installs
# some MG dependencies via cmake). The conda package does NOT include these test
# binaries or extra dependencies, but these are built here for use in CI runs.

./build.sh libcugraph cpp-mgtests -v --allgpuarch
