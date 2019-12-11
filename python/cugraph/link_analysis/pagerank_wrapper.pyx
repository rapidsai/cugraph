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

cimport cugraph.link_analysis.pagerank as c_pagerank
from cugraph.structure.graph cimport *
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
    Call pagerank
    """

    cdef uintptr_t graph = graph_wrapper.allocate_cpp_graph()
    cdef Graph * g = <Graph*> graph

    if input_graph.transposedadjlist:
        [offsets, indices] = graph_wrapper.datatype_cast([input_graph.transposedadjlist.offsets, input_graph.transposedadjlist.indices], [np.int32])
        [weights] = graph_wrapper.datatype_cast([input_graph.transposedadjlist.weights], [np.float32, np.float64])
        graph_wrapper.add_transposed_adj_list(graph, offsets, indices, weights)
    else:
        [src, dst] = graph_wrapper.datatype_cast([input_graph.edgelist.edgelist_df['src'], input_graph.edgelist.edgelist_df['dst']], [np.int32])
        if input_graph.edgelist.weights:
            [weights] = graph_wrapper.datatype_cast([input_graph.edgelist.edgelist_df['weights']], [np.float32, np.float64])
            graph_wrapper.add_edge_list(graph, src, dst, weights)    
        else:
            graph_wrapper.add_edge_list(graph, src, dst)
        add_transposed_adj_list(g)
        offsets, indices, values = graph_wrapper.get_transposed_adj_list(graph)
        input_graph.transposedadjlist = input_graph.transposedAdjList(offsets, indices, values)

    # we should add get_number_of_vertices() to Graph (and this should be
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

    g.transposedAdjList.get_vertex_identifiers(&c_identifier_col)
    
    if personalization is None:
        c_pagerank.pagerank(g, &c_pagerank_col, <gdf_column*> NULL, <gdf_column*> NULL,
                <float> alpha, <float> tol, <int> max_iter, has_guess)
    else:
        c_pers_vtx = get_gdf_column_view(personalization['vertex'])
        c_pers_val = get_gdf_column_view(personalization['values'])
        c_pagerank.pagerank(g, &c_pagerank_col, &c_pers_vtx, &c_pers_val,
                <float> alpha, <float> tol, <int> max_iter, has_guess)

    if input_graph.renumbered:
        df['vertex'] = input_graph.edgelist.renumber_map[df['vertex']]
    return df
