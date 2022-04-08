#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION.

cmake --install cpp/build --component testing
cmake --install cpp/libcugraph_etl/build --component testing
