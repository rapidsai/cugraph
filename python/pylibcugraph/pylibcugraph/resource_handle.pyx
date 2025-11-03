# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Have cython use python 3 syntax
# cython: language_level = 3

from pylibcugraph._cugraph_c.resource_handle cimport (
    cugraph_create_resource_handle,
    cugraph_free_resource_handle,
)
#from cugraph.dask.traversal cimport mg_bfs as c_bfs
from pylibcugraph cimport resource_handle as c_resource_handle


cdef class ResourceHandle:
    """
    RAII-stye resource handle class to manage individual create/free calls and
    the corresponding pointer to a cugraph_resource_handle_t
    """
    def __cinit__(self, handle=None):
        cdef void* handle_ptr = NULL
        cdef size_t handle_size_t
        if handle is not None:
            # FIXME: rather than assume a RAFT handle here, consider something
            # like a factory function in cugraph (which already has a RAFT
            # dependency and makes RAFT assumptions) that takes a RAFT handle
            # and constructs/returns a ResourceHandle
            handle_size_t = <size_t>handle
            handle_ptr = <void*>handle_size_t

        self.c_resource_handle_ptr = cugraph_create_resource_handle(handle_ptr)
        # FIXME: check for error

    def __dealloc__(self):
        # FIXME: free only if handle is a valid pointer
        cugraph_free_resource_handle(self.c_resource_handle_ptr)
