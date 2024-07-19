#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

for component in testing testing_c testing_mg; do
    cp -r ${PREFIX}/tmp/install/libcugraph_components/* ${PREFIX}/
done
cp -r ${PREFIX}/tmp/install/libcugraph_etl_components/testing//* ${PREFIX}/
