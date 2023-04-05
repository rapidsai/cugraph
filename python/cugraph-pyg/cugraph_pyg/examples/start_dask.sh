#!/bin/bash

# Copyright (c) 2023, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

WORKER_RMM_POOL_SIZE=14G \
CUDA_VISIBLE_DEVICES=0,1 \
SCHEDULER_FILE=$(pwd)/scheduler.json \
../../../../mg_utils/run-dask-process.sh \
    scheduler workers \
    --tcp
