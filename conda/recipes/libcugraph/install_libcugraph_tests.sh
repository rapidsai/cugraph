#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

cmake --install cpp/build --component testing
cmake --install cpp/build --component testing_c
cmake --install cpp/build --component testing_mg
cmake --install cpp/libcugraph_etl/build --component testing
