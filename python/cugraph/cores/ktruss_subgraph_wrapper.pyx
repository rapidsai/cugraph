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

from cugraph.cores.ktruss_subgraph cimport *
from cugraph.structure.graph_new cimport *
from cugraph.utilities.unrenumber import unrenumber
from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free
from libc.float cimport FLT_MAX_EXP

import cudf
import rmm
import numpy as np

def ktruss_subgraph_double(input_graph, k, use_weights, subgraph_truss):
    """
    Call ktruss
    """
    if not input_graph.edgelist:
        input_graph.view_edge_list()

    num_verts = input_graph.number_of_vertices()
    num_edges = len(input_graph.edgelist.edgelist_df)

    cdef uintptr_t c_src_indices = input_graph.edgelist.edgelist_df['src'].__cuda_array_interface__['data'][0]
    cdef uintptr_t c_dst_indices = input_graph.edgelist.edgelist_df['dst'].__cuda_array_interface__['data'][0]
    cdef uintptr_t c_weights = <uintptr_t> NULL

    if input_graph.edgelist.weights:
        c_weights = input_graph.edgelist.edgelist_df['weights'].__cuda_array_interface__['data'][0]

    cdef GraphCOOView[int,int,double] input_coo
    cdef GraphCOOView[int,int,double] output_coo

    input_coo = GraphCOOView[int,int,double](<int*>c_src_indices, <int*>c_dst_indices, <double*>c_weights, num_verts, num_edges)
    output_coo = GraphCOOView[int,int,double]()
    k_truss_subgraph(input_coo, k, output_coo);

    src_array = rmm.device_array_from_ptr(<uintptr_t> output_coo.src_indices,
            nelem=output_coo.number_of_edges,
            dtype=np.int32)

    dst_array = rmm.device_array_from_ptr(<uintptr_t> output_coo.dst_indices,
            nelem=output_coo.number_of_edges,
            dtype=np.int32)
    df = cudf.DataFrame()
    df['src'] = cudf.Series(src_array)
    df['dst'] = cudf.Series(dst_array)

    if input_graph.renumbered:
        unrenumber(input_graph.edgelist.renumber_map, df, 'src')
        unrenumber(input_graph.edgelist.renumber_map, df, 'dst')

    if input_graph.edgelist.weights and use_weights:
        weight_array = rmm.device_array_from_ptr(<uintptr_t> output_coo.edge_data,
                nelem=output_coo.number_of_edges,
                dtype=np.float)
        df['weights'] = cudf.Series(weight_array)
        subgraph_truss.from_cudf_edgelist(df, source='src', destination='dst', edge_attr='weights', renumber=False)
    else:
        subgraph_truss.from_cudf_edgelist(df, source='src', destination='dst', renumber=False)

def ktruss_subgraph_float(input_graph, k, use_weights, subgraph_truss):
    """
    Call ktruss
    """
    if not input_graph.edgelist:
        input_graph.view_edge_list()

    num_verts = input_graph.number_of_vertices()
    num_edges = len(input_graph.edgelist.edgelist_df)

    cdef uintptr_t c_src_indices = input_graph.edgelist.edgelist_df['src'].__cuda_array_interface__['data'][0]
    cdef uintptr_t c_dst_indices = input_graph.edgelist.edgelist_df['dst'].__cuda_array_interface__['data'][0]
    cdef uintptr_t c_weights = <uintptr_t> NULL

    if input_graph.edgelist.weights:
        c_weights = input_graph.edgelist.edgelist_df['weights'].__cuda_array_interface__['data'][0]

    cdef GraphCOOView[int,int,float] input_coo
    cdef GraphCOOView[int,int,float] output_coo

    input_coo = GraphCOOView[int,int,float](<int*>c_src_indices, <int*>c_dst_indices, <float*>c_weights, num_verts, num_edges)
    output_coo = GraphCOOView[int,int,float]()
    k_truss_subgraph(input_coo, k, output_coo);

    src_array = rmm.device_array_from_ptr(<uintptr_t> output_coo.src_indices,
            nelem=output_coo.number_of_edges,
            dtype=np.int32)

    dst_array = rmm.device_array_from_ptr(<uintptr_t> output_coo.dst_indices,
            nelem=output_coo.number_of_edges,
            dtype=np.int32)
    df = cudf.DataFrame()
    df['src'] = cudf.Series(src_array)
    df['dst'] = cudf.Series(dst_array)

    if input_graph.renumbered:
        unrenumber(input_graph.edgelist.renumber_map, df, 'src')
        unrenumber(input_graph.edgelist.renumber_map, df, 'dst')

    if input_graph.edgelist.weights and use_weights:
        weight_array = rmm.device_array_from_ptr(<uintptr_t> output_coo.edge_data,
                nelem=output_coo.number_of_edges,
                dtype=np.float32)
        df['weights'] = cudf.Series(weight_array)
        subgraph_truss.from_cudf_edgelist(df, source='src', destination='dst', edge_attr='weights', renumber=False)
    else:
        subgraph_truss.from_cudf_edgelist(df, source='src', destination='dst', renumber=False)

def ktruss_subgraph(input_graph, k, use_weights, subgraph_truss):
    if input_graph.edgelist.weights:
        if (input_graph.edgelist.edgelist_df['weights'].dtype == np.float32):
            ktruss_subgraph_float(input_graph, k, use_weights, subgraph_truss)
        else:
            ktruss_subgraph_double(input_graph, k, use_weights, subgraph_truss)
    else:
        ktruss_subgraph_float(input_graph, k, use_weights, subgraph_truss)
