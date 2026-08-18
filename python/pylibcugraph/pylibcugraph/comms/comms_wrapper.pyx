# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3


from pylibraft.common.handle cimport *
from pylibcugraph.comms.comms cimport init_subcomm as c_init_subcomm


def init_subcomms(handle, row_comm_size):
    """
    Initialize subcommunicators for multi-GPU communication.

    Parameters
    ----------
    handle : object
        Handle to device resources used by the underlying C++ algorithm call.
    row_comm_size : object
        Input argument `row_comm_size` passed to the backend algorithm.

    Returns
    -------
    object
            Algorithm result returned by the backend binding.
    """
    cdef size_t handle_size_t = <size_t>handle.getHandle()
    handle_ = <handle_t*>handle_size_t
    c_init_subcomm(handle_[0], row_comm_size)
