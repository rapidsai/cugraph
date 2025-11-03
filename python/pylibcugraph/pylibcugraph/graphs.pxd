# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Have cython use python 3 syntax
# cython: language_level = 3

from pylibcugraph._cugraph_c.graph cimport (
    cugraph_graph_t,
    cugraph_type_erased_device_array_view_t,
)
from pylibcugraph._cugraph_c.types cimport (
    cugraph_data_type_id_t)


# Base class allowing functions to accept either SGGraph or MGGraph
# This is not visible in python
cdef class _GPUGraph:
    cdef cugraph_data_type_id_t vertex_type
    cdef cugraph_graph_t* c_graph_ptr
    cdef cugraph_type_erased_device_array_view_t* edge_id_view_ptr
    cdef cugraph_type_erased_device_array_view_t** edge_id_view_ptr_ptr
    cdef cugraph_type_erased_device_array_view_t* weights_view_ptr
    cdef cugraph_type_erased_device_array_view_t** weights_view_ptr_ptr

cdef class SGGraph(_GPUGraph):
    pass

cdef class MGGraph(_GPUGraph):
    pass
