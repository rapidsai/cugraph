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


cdef extern from "cugraph.h" namespace "cugraph":

    struct edge_list:
        VT *src_indices
        gdf_column *dest_indices
        WT *edge_data

    struct adj_list:
        VT *offsets
        VT *indices
        WT *edge_data
        void get_vertex_identifiers(VT *identifiers)
        void get_source_indices(VT *indices)

    struct gdf_dynamic:
        void   *data

    ctypedef enum gdf_prop_type:
        PROP_UNDEF = 0
        PROP_FALSE
        PROP_TRUE

    struct Graph_properties:
        bool directed
        bool weighted
        bool multigraph
        bool bipartite
        bool tree
        gdf_prop_type has_negative_edges

    struct Graph:
        edge_list *edgeList
        adj_list *adjList
        adj_list *transposedAdjList
        gdf_dynamic  *dynAdjList
        Graph_properties *prop
        size_t numberOfVertices


    cdef void renumber_vertices(
        const VT *src,
        const VT *dst,
        VT *src_renumbered,
        VT *dst_renumbered,
        VT *numbering_map) except +

    cdef void edge_list_view(
        Graph *graph,
        const VT *source_indices,
        const VT *destination_indices,
        const WT *edge_data) except +
    cdef void add_edge_list(Graph *graph) except +
    cdef void delete_edge_list(Graph *graph) except +

    cdef void adj_list_view (
        Graph *graph,
        const VT *offsets,
        const VT *indices,
        const WT *edge_data) except +
    cdef void add_adj_list(Graph *graph) except +
    cdef void delete_adj_list(Graph *graph) except +

    cdef void transposed_adj_list_view (
        Graph *graph,
        const VT *offsets,
        const VT *indices,
        const WT *edge_data) except +
    cdef void add_transposed_adj_list(Graph *graph) except +
    cdef void delete_transposed_adj_list(Graph *graph) except +

    cdef void get_two_hop_neighbors(
        Graph* graph,
        VT *first,
        VT *second) except +

    cdef void degree(
        Graph *graph,
        VT *degree,
        int x) except +

    cdef void number_of_vertices(Graph *graph) except +
