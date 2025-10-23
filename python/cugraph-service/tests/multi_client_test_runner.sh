#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# source this script (ie. do not run it) for easier job control from the shell
# FIXME: change this and/or cugraph_service so PYTHONPATH is not needed
PYTHONPATH=/Projects/cugraph/python/cugraph_service python client1_script.py &
sleep 1
PYTHONPATH=/Projects/cugraph/python/cugraph_service python client2_script.py &
PYTHONPATH=/Projects/cugraph/python/cugraph_service python client2_script.py &
PYTHONPATH=/Projects/cugraph/python/cugraph_service python client2_script.py &
PYTHONPATH=/Projects/cugraph/python/cugraph_service python client2_script.py &
PYTHONPATH=/Projects/cugraph/python/cugraph_service python client2_script.py &
PYTHONPATH=/Projects/cugraph/python/cugraph_service python client2_script.py &
PYTHONPATH=/Projects/cugraph/python/cugraph_service python client2_script.py
