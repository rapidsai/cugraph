# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Have cython use python 3 syntax
# cython: language_level = 3

from pylibcugraph._cugraph_c.error cimport (
    cugraph_error_code_t,
    cugraph_error_t,
)
from pylibcugraph._cugraph_c.lookup_src_dst cimport (
    cugraph_lookup_container_t,
)
from pylibcugraph.resource_handle cimport (
    ResourceHandle,
)
from pylibcugraph.graphs cimport (
    _GPUGraph,
)

cdef class EdgeIdLookupTable:
    cdef ResourceHandle handle,
    cdef _GPUGraph graph,
    cdef cugraph_lookup_container_t* lookup_container_c_ptr
