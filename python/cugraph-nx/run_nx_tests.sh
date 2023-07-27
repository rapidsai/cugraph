#!/usr/bin/env bash
#
# Copyright (c) 2023, NVIDIA CORPORATION.
NETWORKX_GRAPH_CONVERT=cugraph NETWORKX_BACKEND_TEST_EXHAUSTIVE=True pytest --pyargs networkx "$@"
