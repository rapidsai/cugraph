# Copyright (c) 2020, NVIDIA CORPORATION.
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

from cugraph.traversal.traveling_salesman cimport traveling_salesman as c_traveling_salesman
from cugraph.structure import graph_primtypes_wrapper
from cugraph.structure.graph_primtypes cimport *
from libc.stdint cimport uintptr_t

import cudf
import numpy as np
import cupy as cp


def traveling_salesman(input_graph,
        pos_list=None,
        restarts=4096,
        distance="euclidean"):
    """
    Call traveling_salesman
    """

    if not input_graph.edgelist:
        input_graph.view_edge_list()

    [src_indices, dst_indices] = graph_primtypes_wrapper.datatype_cast(
            [input_graph.edgelist_df['src'], input_graph.edgelist_df['dst']],
            [np.int32])

    num_verts = input_graph.number_of_vertices()
    num_edges = len(input_graph.edgelist.edgelist_df['src'])

    if input_graph.edgelist.weights is not None:
        [weights] = graph_primtypes_wrapper.datatype_cast(
                [input_graph.edgelist.weights], [np.float32, np.float64])
    else:
        weights = cudf.Series(cp.full(num_edges, 1.0, dtype=np.float32))

    cdef uintptr_t x_pos = <uintptr_t>NULL
    cdef uintptr_t y_pos = <uintptr_t>NULL

    if pos_list is not None:
        if len(pos_list) != num_verts:
            raise ValueError('pos_list must have initial positions for all vertices')
        pos_list['x'] = pos_list['x'].astype(np.float32)
        pos_list['y'] = pos_list['y'].astype(np.float32)
        pos_list['x'][pos_list['vertex']] = pos_list['x']
        pos_list['y'][pos_list['vertex']] = pos_list['y']
        x_pos = pos_list['x'].__cuda_array_interface__['data'][0]
        y_pos = pos_list['y'].__cuda_array_interface__['data'][0]

    cdef unique_ptr[handle_t] handle_ptr
    handle_ptr.reset(new handle_t())
    handle_ = handle_ptr.get();
    cdef uintptr_t c_src_indices = src_indices.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_dst_indices = dst_indices.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_weights = weights.__cuda_array_interface__['data'][0]

    cdef GraphCOOView[int, int, float] graph_float
    cdef GraphCOOView[int, int, double] graph_double
    cdef float final_cost_float = 0.0
    cdef double final_cost_double = 0.0
    final_cost = None

    if graph_primtypes_wrapper.weight_type(input_graph) == np.float32:
        graph_float = GraphCOOView[int,int,float](<int*>c_src_indices,
                <int*>c_dst_indices, <float*>c_weights, num_verts,
                num_edges)
        final_cost_float = c_traveling_salesman(handle_[0],
                graph_float,
                <float*> x_pos,
                <float*> y_pos,
                <int> restarts)
        final_cost = final_cost_float
    else:
        graph_double = GraphCOOView[int,int,double](<int*>c_src_indices,
               <int*>c_dst_indices, <double*>c_weights, num_verts,
               num_edges)
        final_cost_double = c_traveling_salesman(handle_[0],
               graph_double,
               <float*> x_pos,
               <float*> y_pos,
               <int> restarts)
        final_cost = final_cost_double

    return final_cost
