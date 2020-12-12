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

from cugraph.traversal.traveling_salesman cimport force_atlas2 as c_force_atlas2
from cugraph.structure import graph_primtypes_wrapper
from cugraph.structure.graph_primtypes cimport *
from cugraph.structure import utils_wrapper
from libcpp cimport bool
from libc.stdint cimport uintptr_t

import cudf
import cudf._lib as libcudf
from numba import cuda
import numpy as np
import numpy.ctypeslib as ctypeslib


def traveling_salesman(input_graph,
                       pos_list=None,
                       restarts=4096,
                       weight=None,
                       distance="euclidean"):
    """
    Call traveling_salesman
    """

    if not input_graph.edgelist:
        input_graph.view_edge_list()

    num_verts = input_graph.number_of_vertices()
    num_edges = len(input_graph.edgelist.edgelist_df['src'])

    cdef GraphCOOView[int,int,float] graph_float

    cdef uintptr_t c_src_indices = input_graph.edgelist.edgelist_df['src'].__cuda_array_interface__['data'][0]
    cdef uintptr_t c_dst_indices = input_graph.edgelist.edgelist_df['dst'].__cuda_array_interface__['data'][0]
    cdef uintptr_t c_weights = <uintptr_t> NULL

    if input_graph.edgelist.weights:
        c_weights = input_graph.edgelist.edgelist_df['weights'].__cuda_array_interface__['data'][0]

    cdef uintptr_t x_start = <uintptr_t>NULL
    cdef uintptr_t y_start = <uintptr_t>NULL
    cdef uintptr_t pos_ptr = <uintptr_t>NULL

    if pos_list is not None:
        if len(pos_list) != num_verts:
            raise ValueError('pos_list must have initial positions for all vertices')
        pos_list['x'] = pos_list['x'].astype(np.float32)
        pos_list['y'] = pos_list['y'].astype(np.float32)
        pos_list['x'][pos_list['vertex']] = pos_list['x']
        pos_list['y'][pos_list['vertex']] = pos_list['y']
        x_start = pos_list['x'].__cuda_array_interface__['data'][0]
        y_start = pos_list['y'].__cuda_array_interface__['data'][0]


    graph_float = GraphCOOView[int,int,float](<int*>c_src_indices,
            <int*>c_dst_indices, <float*>c_weights, num_verts,
            num_edges)

    return df
