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

cimport cugraph.link_prediction.jaccard as c_jaccard
from cugraph.structure.graph cimport *
from cugraph.utilities.column_utils cimport *
from cugraph.structure import graph_wrapper
from cudf._lib.cudf cimport np_dtype_from_gdf_column
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free
from cython cimport floating

import cudf
import cudf._lib as libcudf
import rmm
import numpy as np


def jaccard(input_graph, vertex_pair=None):
    """
    Call jaccard_list
    """
    cdef uintptr_t graph = graph_wrapper.allocate_cpp_graph()
    cdef Graph * g = <Graph*> graph

    if input_graph.adjlist:
        [offsets, indices] = graph_wrapper.datatype_cast([input_graph.adjlist.offsets, input_graph.adjlist.indices], [np.int32])
        [weights] = graph_wrapper.datatype_cast([input_graph.adjlist.weights], [np.float32, np.float64])
        graph_wrapper.add_adj_list(graph, offsets, indices, weights)
    else:
        [src, dst] = graph_wrapper.datatype_cast([input_graph.edgelist.edgelist_df['src'], input_graph.edgelist.edgelist_df['dst']], [np.int32])
        if input_graph.edgelist.weights:
            [weights] = graph_wrapper.datatype_cast([input_graph.edgelist.edgelist_df['weights']], [np.float32, np.float64])
            graph_wrapper.add_edge_list(graph, src, dst, weights)
        else:
            graph_wrapper.add_edge_list(graph, src, dst)
        add_adj_list(g)
        offsets, indices, values = graph_wrapper.get_adj_list(graph)
        input_graph.adjlist = input_graph.AdjList(offsets, indices, values)

    cdef gdf_column c_result_col
    cdef gdf_column c_first_col
    cdef gdf_column c_second_col
    cdef gdf_column c_src_index_col

    if type(vertex_pair) == cudf.DataFrame:
        result_size = len(vertex_pair)
        result = cudf.Series(np.ones(result_size, dtype=np.float32))
        c_result_col = get_gdf_column_view(result)
        df = cudf.DataFrame()
        if input_graph.renumbered is True:
            renumber_df = cudf.DataFrame()
            renumber_df['map'] = input_graph.edgelist.renumber_map
            renumber_df['id'] = input_graph.edgelist.renumber_map.index.astype(np.int32)
            vp = vertex_pair.merge(renumber_df, left_on='first', right_on='map', how='left').drop('map').merge(renumber_df, left_on='second', right_on='map', how='left').drop('map')
            df['source'] = vp['first']
            df['destination'] = vp['second']
            c_first_col = get_gdf_column_view(vp['id_x'])
            c_second_col = get_gdf_column_view(vp['id_y'])
        else:
            first = vertex_pair[vertex_pair.columns[0]].astype(np.int32)
            second = vertex_pair[vertex_pair.columns[1]].astype(np.int32)
            df['source'] = first
            df['destination'] = second
            c_first_col = get_gdf_column_view(first)
            c_second_col = get_gdf_column_view(second)
        c_jaccard.jaccard_list(g,
                               <gdf_column*> NULL,
                               &c_first_col,
                               &c_second_col,
                               &c_result_col)
        df['jaccard_coeff'] = result
        return df

    else:
        # error check performed in jaccard.py
        assert vertex_pair is None
        # we should add get_number_of_edges() to Graph (and this should be
        # used instead of g.adjList.indices.size)
        num_edges = g.adjList.indices.size
        result = cudf.Series(np.ones(num_edges, dtype=np.float32), nan_as_null=False)
        c_result_col = get_gdf_column_view(result)

        c_jaccard.jaccard(g, <gdf_column*> NULL, &c_result_col)
        

        dest_data = rmm.device_array_from_ptr(<uintptr_t> g.adjList.indices.data,
                                            nelem=num_edges,
                                            dtype=np_dtype_from_gdf_column(g.adjList.indices))
        df = cudf.DataFrame()
        df['source'] = cudf.Series(np.zeros(num_edges, dtype=np_dtype_from_gdf_column(g.adjList.indices)))
        c_src_index_col = get_gdf_column_view(df['source'])
        g.adjList.get_source_indices(&c_src_index_col)
        
        df['destination'] = cudf.Series(dest_data)

        if input_graph.renumbered:
            print(df['source'])
            print(input_graph.edgelist.renumber_map)
            print(input_graph.edgelist.renumber_map[df['source']])
            df['source'] = input_graph.edgelist.renumber_map[df['source']].reset_index().drop('index')
            df['destination'] = input_graph.edgelist.renumber_map[df['destination']].reset_index().drop('index')

        df['jaccard_coeff'] = result

        return df
