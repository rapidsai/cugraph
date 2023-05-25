#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.
NETWORKX_GRAPH_CONVERT=cugraph pytest --pyargs networkx "$@"  # --cov --cov-report term-missing
