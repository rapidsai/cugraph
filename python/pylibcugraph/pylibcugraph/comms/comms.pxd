# SPDX-FileCopyrightText: Copyright (c) 2020-2023, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from pylibraft.common.handle cimport *

cdef extern from "cugraph/partition_manager.hpp" namespace "cugraph::partition_manager":
   cdef void init_subcomm(handle_t &handle,
                          size_t row_comm_size)
