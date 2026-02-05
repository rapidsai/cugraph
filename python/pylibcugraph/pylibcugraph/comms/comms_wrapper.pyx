# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3


from pylibraft.common.handle cimport *
from pylibcugraph.comms.comms cimport init_subcomm as c_init_subcomm


def init_subcomms(handle, row_comm_size):
    cdef size_t handle_size_t = <size_t>handle.getHandle()
    handle_ = <handle_t*>handle_size_t
    c_init_subcomm(handle_[0], row_comm_size)
