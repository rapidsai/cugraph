# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


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
