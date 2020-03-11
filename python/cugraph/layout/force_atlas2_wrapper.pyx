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

from cugraph.layout.force_atlas2 cimport *
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


def force_atlas2(input_graph,
                max_iter=1000,
                pos_list=None,
                gravity=1.0,
                scaling_ratio=1.0,
                barnes_hut_theta=0.5,
                edge_weight_influence=1.0,
                lin_log_mode=False,
                prevent_overlapping=False)

    """
    Call force_atlas2
    """

    if not input_graph.edgelist:
        input_graph.view_edge_list()
    
    num_verts = input_graph.number_of_vertices()
    num_edges = input_graph.number_of_edges()
    
    cdef GraphCOO[int,int,float] graph_float
    cdef uintptr_t c_src_indices = input_graph.edgelist.source.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_dst_indices = input_graph.edgelist.dest.__cuda_array_interface__['data'][0]
    graph_float = GraphCOO[int,int,float](<int*>c_src_indices, <int*>c_dst_indices, <float*>c_weights, num_verts, num_edges)
    graph_float.get_vertex_identifiers(<int*>c_identifier)
    
    df = cudf.DataFrame()
    df['vertex'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    df['x'] = cudf.Series(np.zeros(num_verts, dtype=np.float32))
    df['y'] = cudf.Series(np.zeros(num_verts, dtype=np.float32))

    cdef uintptr_t c_fa2_x_ptr = df['x'].__cuda_array_interface__['data'][0]
    cdef uintptr_t c_fa2_y_ptr = df['y'].__cuda_array_interface__['data'][0]
    if pos_list is not None:
        df['x'] = pos_list['x'].__cuda_array_interface__['data'][0]
        df['y'] = pos_list['y'].__cuda_array_interface__['data'][0]


    force_atlas2.force_atlas2(<Graph*>g,
                    <float*> c_fa2_x_ptr,
                    <float*> c_fa2_y_ptr,
                    <int>max_iter,
                    <float>gravity,
                    <float>scaling_ratio,
                    <float>edge_weight_influence,
                    <int>lin_log_mode,
                    <int>prevent_overlapping)

    return df

