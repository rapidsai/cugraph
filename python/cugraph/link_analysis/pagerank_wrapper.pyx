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

from cugraph.link_analysis.c_pagerank cimport *
from cugraph.structure.c_graph cimport *
from cugraph.utilities.column_utils cimport *
from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free
from cugraph.structure import graph_wrapper
import cudf
import cudf._lib as libcudf
import rmm
import numpy as np


def pagerank(input_graph, alpha=0.85, personalization=None, max_iter=100, tol=1.0e-5, nstart=None):
    """
    Call gdf_pagerank
    """

    cdef uintptr_t graph = graph_wrapper.allocate_cpp_graph()
    cdef gdf_graph * g = <gdf_graph*> graph

    if input_graph.transposedadjlist:
        graph_wrapper.add_transposed_adj_list(graph, input_graph.adjlist.offsets, input_graph.adjlist.indices, input_graph.adjlist.weights)
    else:
        if input_graph.edgelist.weights:
            graph_wrapper.add_edge_list(graph, input_graph.edgelist.edgelist_df['src'], input_graph.edgelist.edgelist_df['dst'], input_graph.edgelist.edgelist_df['weights'])    
        else:
            graph_wrapper.add_edge_list(graph, input_graph.edgelist.edgelist_df['src'], input_graph.edgelist.edgelist_df['dst'])
        err = gdf_add_transposed_adj_list(g)
        libcudf.cudf.check_gdf_error(err)
        offsets, indices, values = graph_wrapper.get_transposed_adj_list(graph)
        input_graph.transposedadjlist = input_graph.transposedAdjList(offsets, indices, values)

    # we should add get_number_of_vertices() to gdf_graph (and this should be
    # used instead of g.transposedAdjList.offsets.size - 1)
    num_verts = g.transposedAdjList.offsets.size - 1

    df = cudf.DataFrame()
    df['vertex'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    df['pagerank'] = cudf.Series(np.zeros(num_verts, dtype=np.float32))

    cdef bool has_guess = <bool> 0
    if nstart is not None:
        df['pagerank'][nstart['vertex']] = nstart['values']
        has_guess = <bool> 1

    cdef gdf_column c_identifier_col = get_gdf_column_view(df['vertex'])
    cdef gdf_column c_pagerank_col = get_gdf_column_view(df['pagerank'])
    cdef gdf_column c_pers_vtx
    cdef gdf_column c_pers_val

    err = g.transposedAdjList.get_vertex_identifiers(&c_identifier_col)
    libcudf.cudf.check_gdf_error(err)

    if personalization is None:
        err = gdf_pagerank(g, &c_pagerank_col, <gdf_column*> NULL, <gdf_column*> NULL,
                <float> alpha, <float> tol, <int> max_iter, has_guess)
    else:
        c_pers_vtx = get_gdf_column_view(personalization['vertex'])
        c_pers_val = get_gdf_column_view(personalization['values'])
        err = gdf_pagerank(g, &c_pagerank_col, &c_pers_vtx, &c_pers_val,
                <float> alpha, <float> tol, <int> max_iter, has_guess)

    libcudf.cudf.check_gdf_error(err)

    if input_graph.renumbered:
        df['vertex'] = input_graph.edgelist.renumber_map[df['vertex']]
    return df
