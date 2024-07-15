#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION.

./build.sh libcugraph libcugraph_etl cpp-mgtests -n -v --allgpuarch
cmake --install cpp/build
