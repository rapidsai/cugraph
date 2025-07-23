# Copyright (c) 2025, NVIDIA CORPORATION.
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


import warnings

from pylibcugraph.comms import (
    cugraph_comms_init as DEPRECATED__cugraph_comms_init,
    cugraph_comms_shutdown as DEPRECATED__cugraph_comms_shutdown,
    cugraph_comms_create_unique_id as DEPRECATED__cugraph_comms_create_unique_id,
    cugraph_comms_get_raft_handle as DEPRECATED__cugraph_comms_get_raft_handle,
)


def cugraph_comms_init(*args, **kwargs):
    warnings.warn(
        "cugraph_comms_init has been moved to pylibcugraph.comms.cugraph_comms_init."
        "The original name will be removed in a future release.",
        FutureWarning,
    )
    return DEPRECATED__cugraph_comms_init(*args, **kwargs)


def cugraph_comms_shutdown(*args, **kwargs):
    warnings.warn(
        "cugraph_comms_shutdown has been moved to "
        "pylibcugraph.comms.cugraph_comms_shutdown."
        "The original name will be removed in a future release.",
        FutureWarning,
    )
    return DEPRECATED__cugraph_comms_shutdown(*args, **kwargs)


def cugraph_comms_create_unique_id(*args, **kwargs):
    warnings.warn(
        "cugraph_comms_create_unique_id has been moved to"
        " pylibcugraph.comms.cugraph_comms_create_unique_id."
        "The original name will be removed in a future release.",
        FutureWarning,
    )
    return DEPRECATED__cugraph_comms_create_unique_id(*args, **kwargs)


def cugraph_comms_get_raft_handle(*args, **kwargs):
    warnings.warn(
        "cugraph_comms_get_raft_handle has been moved to "
        "pylibcugraph.comms.cugraph_comms_get_raft_handle."
        "The original name will be removed in a future release.",
        FutureWarning,
    )
    return DEPRECATED__cugraph_comms_get_raft_handle(*args, **kwargs)
