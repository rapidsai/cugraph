# SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Have cython use python 3 syntax
# cython: language_level = 3

cdef extern from "cugraph_c/resource_handle.h":

    ctypedef struct cugraph_resource_handle_t:
        pass

    # FIXME: the void* raft_handle arg will change in a future release
    cdef cugraph_resource_handle_t* \
        cugraph_create_resource_handle(
	    void* raft_handle
	)

    cdef int \
        cugraph_resource_handle_get_rank(
	    const cugraph_resource_handle_t* handle
	)

    cdef void \
        cugraph_free_resource_handle(
            cugraph_resource_handle_t* p_handle
        )
