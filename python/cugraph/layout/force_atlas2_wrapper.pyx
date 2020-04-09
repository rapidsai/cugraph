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

from cugraph.layout.force_atlas2 cimport force_atlas2 as c_force_atlas2
from cugraph.structure import graph_wrapper
from cugraph.structure.graph_new cimport *
from cugraph.utilities.column_utils cimport *
from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

import cudf
import cudf._lib as libcudf
import rmm
import numpy as np
import numpy.ctypeslib as ctypeslib


def force_atlas2(input_graph,
                 max_iter=1000,
                 pos_list=None,
                 outbound_attraction_distribution=True,
                 lin_log_mode=False,
                 prevent_overlapping=False,
                 edge_weight_influence=1.0,
                 jitter_tolerance=1.0,
                 barnes_hut_optimize=True,
                 barnes_hut_theta=0.5,
                 scaling_ratio=1.0,
                 strong_gravity_mode = False,
                 gravity=1.0):

    """
    Call force_atlas2
    """

    if not input_graph.edgelist:
        input_graph.view_edge_list()

    num_verts = input_graph.number_of_vertices()
    num_edges = input_graph.number_of_edges()

    cdef GraphCOO[int,int,float] graph_float
    cdef GraphCOO[int,int,double] graph_double

    df = cudf.DataFrame()
    df['vertex'] = cudf.Series(np.arange(num_verts, dtype=np.int32))
    df['x'] = cudf.Series(np.zeros(num_verts, dtype=np.float32))
    df['y'] = cudf.Series(np.zeros(num_verts, dtype=np.float32))

    cdef uintptr_t c_src_indices = input_graph.edgelist.edgelist_df['src'].__cuda_array_interface__['data'][0]
    cdef uintptr_t c_dst_indices = input_graph.edgelist.edgelist_df['dst'].__cuda_array_interface__['data'][0]
    cdef uintptr_t c_weights = <uintptr_t> NULL

    if input_graph.edgelist.weights:
        c_weights = input_graph.edgelist.edgelist_df['weights'].__cuda_array_interface__['data'][0]


    cdef uintptr_t x_pos = df['x'].__cuda_array_interface__['data'][0]
    cdef uintptr_t y_pos = df['y'].__cuda_array_interface__['data'][0]

    cdef uintptr_t x_start = <uintptr_t>NULL 
    cdef uintptr_t y_start = <uintptr_t>NULL 

    if pos_list is not None:
        df['vertex'] = pos_list['vertex']
        x_start = pos_list['x'].__cuda_array_interface__['data'][0]
        y_start = pos_list['y'].__cuda_array_interface__['data'][0]

    if input_graph.edgelist.weights \
        and input_graph.edgelist.edgelist_df['weights'].dtype == np.float64:
        graph_double = GraphCOO[int,int, double](<int*>c_src_indices,
                <int*>c_dst_indices, <double*>c_weights, num_verts, num_edges)

        c_force_atlas2[int, int, double](graph_double,
                    <float*>x_pos,
                    <float*>y_pos,
                    <int>max_iter,
                    <float*>x_start,
                    <float*>y_start,
                    <bool>outbound_attraction_distribution,
                    <bool>lin_log_mode,
                    <bool>prevent_overlapping,
                    <float>edge_weight_influence,
                    <float>jitter_tolerance,
                    <bool>barnes_hut_optimize,
                    <float>barnes_hut_theta,
                    <float>scaling_ratio,
                    <bool> strong_gravity_mode,
                    <float>gravity)
    else:
        graph_float = GraphCOO[int,int,float](<int*>c_src_indices, <int*>c_dst_indices, <float*>c_weights, num_verts, num_edges)
        c_force_atlas2[int, int, float](graph_float,
                    <float*>x_pos,
                    <float*>y_pos,
                    <int>max_iter,
                    <float*>x_start,
                    <float*>y_start,
                    <bool>outbound_attraction_distribution,
                    <bool>lin_log_mode,
                    <bool>prevent_overlapping,
                    <float>edge_weight_influence,
                    <float>jitter_tolerance,
                    <bool>barnes_hut_optimize,
                    <float>barnes_hut_theta,
                    <float>scaling_ratio,
                    <bool> strong_gravity_mode,
                    <float>gravity)

    return df

