# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Have cython use python 3 syntax
# cython: language_level = 3

from pylibcugraph._cugraph_c.graph cimport (
    cugraph_graph_t,
)
from pylibcugraph._cugraph_c.types cimport (
    cugraph_data_type_id_t)


# Base class allowing functions to accept either SGGraph or MGGraph
# This is not visible in python
cdef class _GPUGraph:
    cdef cugraph_data_type_id_t vertex_type
    cdef cugraph_graph_t* c_graph_ptr

    cdef bint has_edge_weights(self)
    cdef bint has_edge_ids(self)
    cdef bint has_edge_types(self)
    cdef bint has_edge_start_times(self)
    cdef bint has_edge_end_times(self)

cdef class SGGraph(_GPUGraph):
    pass

cdef class MGGraph(_GPUGraph):
    pass
