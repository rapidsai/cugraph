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
import numpy.ctypeslib as ctypeslib


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
        if len(nstart) != num_verts:
            raise ValueError('nstart must have initial guess for all vertices')
        if input_graph.renumbered is True:
            renumber_series = cudf.Series(input_graph.edgelist.renumber_map.index,
                                          index=input_graph.edgelist.renumber_map, dtype=np.int32)        
            vertex_renumbered = renumber_series.loc[nstart['vertex']]
            df['pagerank'][vertex_renumbered] = nstart['values']
        else:
            df['pagerank'][nstart['vertex']] = nstart['values']
        has_guess = <bool> 1

    #TODO FIX ME when graph class is upgraded to remove gdf_column
    cdef gdf_column c_identifier = get_gdf_column_view(df['vertex'])

    cdef uintptr_t c_pagerank_val = get_column_data_ptr(df['pagerank']._column)
    cdef uintptr_t c_pers_vtx = <uintptr_t>NULL
    cdef uintptr_t c_pers_val = <uintptr_t>NULL
    cdef sz = 0

    if personalization is not None:
        sz = personalization['vertex'].shape[0]
        personalization['vertex'] = personalization['vertex'].astype(np.int32)
        personalization['values'] = personalization['values'].astype(df['pagerank'].dtype)
        if input_graph.renumbered is True:
            renumber_df = cudf.DataFrame()
            renumber_df['map'] = input_graph.edgelist.renumber_map
            renumber_df['id'] = input_graph.edgelist.renumber_map.index.astype(np.int32)
            personalization_values = personalization.merge(renumber_df, left_on='vertex', right_on='map', how='left').drop('map')
            c_pers_vtx = get_column_data_ptr(personalization_values['id']._column)
            c_pers_val = get_column_data_ptr(personalization_values['values']._column)
        else:
            c_pers_vtx = get_column_data_ptr(personalization['vertex']._column)
            c_pers_val = get_column_data_ptr(personalization['values']._column)
    
    if (df['pagerank'].dtype == np.float32): 
        c_pagerank.pagerank[int, float](g, <float*> c_pagerank_val, sz, <int*> c_pers_vtx, <float*> c_pers_val,
                                     <float> alpha, <float> tol, <int> max_iter, has_guess)
    else: 
        c_pagerank.pagerank[int, double](g, <double*> c_pagerank_val, sz, <int*> c_pers_vtx, <double*> c_pers_val,
                            <float> alpha, <float> tol, <int> max_iter, has_guess)

    g.transposedAdjList.get_vertex_identifiers(&c_identifier)
    if input_graph.renumbered:
        df['vertex'] = input_graph.edgelist.renumber_map[df['vertex']]
    return df
