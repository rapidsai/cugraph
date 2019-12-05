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

cimport cugraph.community.louvain as c_louvain
from cugraph.structure.graph cimport *
from cugraph.structure import graph_wrapper
from cugraph.utilities.column_utils cimport *
from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

import cudf
import cudf._lib as libcudf
import rmm
import numpy as np


def louvain(input_graph):
    """
    Call louvain
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

    # we should add get_number_of_vertices() to Graph (and this should be
    # used instead of g.adjList.offsets.size - 1)
    num_verts = g.adjList.offsets.size - 1

    df = cudf.DataFrame()
    df['vertex'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    cdef gdf_column c_index_col = get_gdf_column_view(df['vertex'])
    g.adjList.get_vertex_identifiers(&c_index_col)
    

    df['partition'] = cudf.Series(np.zeros(num_verts,dtype=np.int32))
    cdef gdf_column c_louvain_parts_col = get_gdf_column_view(df['partition'])

    cdef bool single_precision = False
    # this implementation is tied to cugraph.cu line 503
    # cudaDataType_t val_type = graph->adjList->edge_data?
    #     gdf_to_cudadtype(graph->adjList->edge_data): CUDA_R_32F;
    # this is tied to the low-level implementation detail of the lower level
    # function, and very vulnerable to low level changes. Better be
    # reimplemented, but we are planning to eventually remove nvgraph, so I may
    # leave as is right at this moment.
    if g.adjList.edge_data:
        if g.adjList.edge_data.dtype == GDF_FLOAT32:
            single_precision = True;
    else:
        single_precision = True;

    cdef float final_modularity_single_precision = 1.0
    cdef double final_modularity_double_precision = 1.0
    cdef int num_level = 0
    

    if single_precision:
        c_louvain.louvain(<Graph*>g,
                  <void*>&final_modularity_single_precision,
                  <void*>&num_level, &c_louvain_parts_col)
    else:
        c_louvain.louvain(<Graph*>g,
                  <void*>&final_modularity_double_precision,
                  <void*>&num_level, &c_louvain_parts_col)
    

    if input_graph.renumbered:
        df['vertex'] = input_graph.edgelist.renumber_map[df['vertex']]

    if single_precision:
        return df, <double>final_modularity_single_precision
    else:
        return df, final_modularity_double_precision
