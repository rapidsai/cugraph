# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Have cython use python 3 syntax
# cython: language_level = 3

from pylibcugraph._cugraph_c.resource_handle cimport (
    cugraph_resource_handle_t,
)


cdef class ResourceHandle:
    cdef cugraph_resource_handle_t* c_resource_handle_ptr
