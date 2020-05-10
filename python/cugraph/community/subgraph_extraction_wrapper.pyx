# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

from cugraph.community.subgraph_extraction cimport extract_subgraph_vertex as c_extract_subgraph_vertex
from cugraph.structure.graph_new cimport *
from cugraph.structure import graph_new_wrapper
from cugraph.utilities.unrenumber import unrenumber
from libc.stdint cimport uintptr_t

import cudf
import rmm
import numpy as np


def subgraph(input_graph, vertices, subgraph):
    """
    Call extract_subgraph_vertex
    """
    src = None
    dst = None
    weights = None
    vertices_renumbered = None
    use_float = True

    if not input_graph.edgelist:
        input_graph.view_edge_list()

    [src, dst] = graph_new_wrapper.datatype_cast([input_graph.edgelist.edgelist_df['src'], input_graph.edgelist.edgelist_df['dst']], [np.int32])

    if input_graph.edgelist.weights:
        [weights] = graph_new_wrapper.datatype_cast([input_graph.edgelist.edgelist_df['weights']], [np.float32, np.float64])
        if weights.dtype == np.float64:
            use_float = False
        
    cdef GraphCOOView[int,int,float]  in_graph_float
    cdef GraphCOOView[int,int,double] in_graph_double
    cdef unique_ptr[GraphCOO[int,int,float]]  out_graph_float
    cdef unique_ptr[GraphCOO[int,int,double]] out_graph_double

    cdef uintptr_t c_src = src.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_dst = dst.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_weights = <uintptr_t> NULL

    if weights is not None:
        c_weights = weights.__cuda_array_interface__['data'][0]
    
    if input_graph.renumbered:
        renumber_series = cudf.Series(input_graph.edgelist.renumber_map.index,
                                      index=input_graph.edgelist.renumber_map, dtype=np.int32)
        vertices_renumbered = renumber_series.loc[vertices]
    else:
        vertices_renumbered = vertices

    cdef uintptr_t c_vertices = vertices_renumbered.__cuda_array_interface__['data'][0]

    num_verts = input_graph.number_of_vertices()
    num_edges = len(src)
    num_input_vertices = len(vertices)

    if use_float:
        in_graph_float = GraphCOOView[int,int,float](<int*>c_src, <int*>c_dst, <float*>c_weights, num_verts, num_edges);
        df = coo_to_df(move(c_extract_subgraph_vertex(in_graph_float, <int*>c_vertices, <int>num_input_vertices)));
    else:
        in_graph_double = GraphCOOView[int,int,double](<int*>c_src, <int*>c_dst, <double*>c_weights, num_verts, num_edges);
        df = coo_to_df(move(c_extract_subgraph_vertex(in_graph_double, <int*>c_vertices, <int>num_input_vertices)));

    # renumber vertices to match original input
    vertices_df = cudf.DataFrame()
    vertices_df['v'] = vertices_renumbered
    vertices_df = vertices_df.reset_index(drop=True).reset_index()

    df = df.merge(vertices_df, left_on='src', right_on='index', how='left').drop(['src', 'index']).rename({'v': 'src'})
    df = df.merge(vertices_df, left_on='dst', right_on='index', how='left').drop(['dst', 'index']).rename({'v': 'dst'})
    
    if input_graph.renumbered:
        df = unrenumber(input_graph.edgelist.renumber_map, df, 'src')
        df = unrenumber(input_graph.edgelist.renumber_map, df, 'dst')

    if weights is not None:
        subgraph.from_cudf_edgelist(df, source='src', destination='dst', edge_attr='weight')
    else:
        subgraph.from_cudf_edgelist(df, source='src', destination='dst')
