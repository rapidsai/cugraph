# Copyright (c) 2019, NVIDIA CORPORATION.
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

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf._lib.cudf cimport *

ctypedef fused VT:
    cython.int
    
ctypedef fused WT:
    cython.float
    cython.double

cdef extern from "cugraph.h" namespace "cugraph":

    cppclass edge_list [VT, WT]:
        VT *src_indices
        VT *dest_indices
        WT *edge_data

    cppclass adj_list[VT, WT]:
        VT *offsets
        VT *indices
        WT *edge_data
        void get_vertex_identifiers(VT *identifiers)
        void get_source_indices(VT *indices)

    ctypedef enum prop_type:
        PROP_UNDEF = 0
        PROP_FALSE
        PROP_TRUE

    struct Graph_properties:
        bool directed
        bool weighted
        bool multigraph
        bool bipartite
        bool tree
        prop_type has_negative_edges

    cppclass Graph[VT, WT]:
        edge_list[VT, WT] *edgeList
        adj_list[VT, WT] *adjList
        adj_list[VT, WT] *transposedAdjList
        Graph_properties *prop
        size_t v
        size_t e

    cdef void renumber_vertices[VT, WT](
        const VT *src,
        const VT *dst,
        VT *src_renumbered,
        VT *dst_renumbered,
        VT *numbering_map) except +

    cdef void edge_list_view[VT, WT]:
        Graph[VT, WT] *graph,
        size_t e,
        VT *source_indices,
        VT *destination_indices,
        WT *edge_data) except +
    cdef void add_edge_list[VT, WT](Graph *graph[VT, WT]) except +
    cdef void delete_edge_list[VT, WT](Graph *graph[VT, WT]) except +

    cdef void adj_list_view[VT, WT](
        Graph *graph[VT, WT],
        const size_t v, 
        const size_t e,
        VT *offsets,
        VT *indices,
        WT *edge_data) except +
    cdef void add_adj_list[VT, WT](Graph *graph[VT, WT]) except +
    cdef void delete_adj_list[VT, WT](Graph *graph[VT, WT]) except +

    cdef void transposed_adj_list_view[VT, WT](
        Graph[VT, WT] *graph
        const size_t v, 
        const size_t e,
        VT *offsets,
        VT *indices,
        WT *edge_data) except +
    cdef void add_transposed_adj_list[VT, WT](Graph *graph[VT, WT]) except +
    cdef void delete_transposed_adj_list[VT, WT](Graph *graph[VT, WT]) except +

    cdef void get_two_hop_neighbors[VT, WT](
        Graph* graph,
        VT *first,
        VT *second) except +

    cdef void degree[VT, WT](
        Graph *graph[VT, WT],
        VT *degree,
        int x) except +

    cdef void number_of_vertices[VT, WT](Graph *graph[VT, WT]) except +
