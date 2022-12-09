# Copyright (c) 2022, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Have cython use python 3 syntax
# cython: language_level = 3

from pylibcugraph._cugraph_c.error cimport (
    cugraph_error_code_t,
    cugraph_error_t,
)

from pylibcugraph._cugraph_c.resource_handle cimport (
    cugraph_resource_handle_t,
    bool_t,
)

from pylibcugraph._cugraph_c.similarity_algorithms cimport (
    cugraph_similarity_result_t
)


from pylibcugraph._cugraph_c.graph cimport cugraph_graph_t

from pylibcugraph._cugraph_c.array cimport (
    cugraph_type_erased_device_array_view_t
)

cdef extern from "cugraph_c/graph_functions.h":
    #"""
    #ctypedef struct cugraph_similarity_result_t:
    #    pass
    #"""
    ctypedef struct cugraph_vertex_pairs_t:
        pass
    

from pylibcugraph._cugraph_c.error cimport (
    cugraph_error_code_t,
    cugraph_error_t,
)
from pylibcugraph._cugraph_c.array cimport (
    cugraph_type_erased_device_array_view_t,
)
from pylibcugraph._cugraph_c.graph cimport (
    cugraph_graph_t,
)


cdef extern from "cugraph_c/graph_functions.h":
    ###########################################################################
    # vertex_pairs
    ctypedef struct cugraph_vertex_pairs_t:
        pass
    
    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_vertex_pairs_get_first(
            cugraph_vertex_pairs_t* vertex_pairs
        )
    
    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_vertex_pairs_get_second(
            cugraph_vertex_pairs_t* vertex_pairs
        )
    
    cdef void \
        cugraph_vertex_pairs_free(
            cugraph_vertex_pairs_t* vertex_pairs
        )
    
    cdef cugraph_error_code_t \
        cugraph_create_vertex_pairs(
            const cugraph_resource_handle_t* handle,
            cugraph_graph_t* graph,
            const cugraph_type_erased_device_array_view_t* first,
            const cugraph_type_erased_device_array_view_t* second,
            cugraph_vertex_pairs_t** vertex_pairs,
            cugraph_error_t** error
        )
    
    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_vertex_pairs_get_first(
            cugraph_vertex_pairs_t* vertex_pairs
        )
    
    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_vertex_pairs_get_second(
            cugraph_vertex_pairs_t* vertex_pairs
        )
    
    cdef void cugraph_vertex_pairs_free(
        cugraph_vertex_pairs_t* vertex_pairs
        )
    
    cdef cugraph_error_code_t cugraph_two_hop_neighbors(
        const cugraph_resource_handle_t* handle,
        const cugraph_graph_t* graph,
        const cugraph_type_erased_device_array_view_t* start_vertices,
        bool_t do_expensive_check,
        cugraph_vertex_pairs_t** result,
        cugraph_error_t** error)

    cdef cugraph_error_code_t \
        cugraph_two_hop_neighbors(
            const cugraph_resource_handle_t* handle,
            const cugraph_graph_t* graph,
            const cugraph_type_erased_device_array_view_t* start_vertices,
            cugraph_vertex_pairs_t** result,
            cugraph_error_t** error
        )
    
    ###########################################################################
    # induced_subgraph
    ctypedef struct cugraph_induced_subgraph_result_t:
        pass

    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_induced_subgraph_get_sources(
            cugraph_induced_subgraph_result_t* induced_subgraph
        )
    
    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_induced_subgraph_get_destinations(
            cugraph_induced_subgraph_result_t* induced_subgraph
        )
    
    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_induced_subgraph_get_edge_weights(
            cugraph_induced_subgraph_result_t* induced_subgraph
        )
    
    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_induced_subgraph_get_subgraph_offsets(
            cugraph_induced_subgraph_result_t* induced_subgraph
        )
    
    cdef void \
        cugraph_induced_subgraph_result_free(
            cugraph_induced_subgraph_result_t* induced_subgraph
        )
    
    cdef cugraph_error_code_t \
        cugraph_extract_induced_subgraph(
            const cugraph_resource_handle_t* handle,
            cugraph_graph_t* graph,
            const cugraph_type_erased_device_array_view_t* subgraph_offsets,
            const cugraph_type_erased_device_array_view_t* subgraph_vertices,
            bool_t do_expensive_check,
            cugraph_induced_subgraph_result_t** result,
            cugraph_error_t** error
        )
