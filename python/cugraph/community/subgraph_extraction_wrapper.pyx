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

from cugraph.community.subgraph_extraction cimport *
from cugraph.structure.graph cimport *
from cugraph.structure import graph_wrapper
from cugraph.utilities.column_utils cimport *
from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free
from libc.float cimport FLT_MAX_EXP

import cudf
import cudf._lib as libcudf
import rmm
import numpy as np


def subgraph(input_graph, vertices, subgraph):
    """
    Call extract_subgraph_vertex_nvgraph
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

    cdef uintptr_t rGraph = graph_wrapper.allocate_cpp_graph()
    cdef Graph* rg = <Graph*>rGraph
    cdef gdf_column vert_col = get_gdf_column_view(vertices)

    extract_subgraph_vertex_nvgraph(g, &vert_col, rg)

    if rg.edgeList is not NULL:
        df = cudf.DataFrame()
        df['src'], df['dst'], vals = graph_wrapper.get_edge_list(rGraph)
        if vals is not None:
            df['val'] = vals
            subgraph.from_cudf_edgelist(df, source='src', target='dst', edge_attr='val')
        else:
            subgraph.from_cudf_edgelist(df, source='src', target='dst')
        if input_graph.edgelist is not None:
            subgraph.renumbered = input_graph.renumbered
            subgraph.edgelist.renumber_map = input_graph.edgelist.renumber_map
    if rg.adjList is not NULL:
        off, ind, vals = graph_wrapper.get_adj_list(rGraph)
        subgraph.from_cudf_adjlist(off, ind, vals)
    if rg.transposedAdjList is not NULL:
        off, ind, vals = graph_wrapper.get_transposed_adj_list(rGraph)
        subgraph.transposedadjlist = subgraph.transposedAdjList(off, ind, vals)
