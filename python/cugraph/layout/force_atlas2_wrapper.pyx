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

from cugraph.layout.force_atlas2 cimport force_atlas2 as c_force_atlas2
from cugraph.structure import graph_wrapper
from cugraph.structure.graph_new cimport *
from cugraph.structure import utils_wrapper
from cugraph.utilities.column_utils cimport *
from cugraph.utilities.unrenumber import unrenumber
from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

import cudf
import cudf._lib as libcudf
import rmm
import numpy as np
import numpy.ctypeslib as ctypeslib

cdef extern from "internals.hpp" namespace "cugraph::internals":
    cdef cppclass GraphBasedDimRedCallback


def force_atlas2(input_graph,
                 max_iter=500,
                 pos_list=None,
                 outbound_attraction_distribution=True,
                 lin_log_mode=False,
                 prevent_overlapping=False,
                 edge_weight_influence=1.0,
                 jitter_tolerance=1.0,
                 barnes_hut_optimize=True,
                 barnes_hut_theta=0.5,
                 scaling_ratio=1.0,
                 strong_gravity_mode=False,
                 gravity=1.0,
                 verbose=False,
                 callback=None):

    """
    Call force_atlas2
    """

    if not input_graph.edgelist:
        input_graph.view_edge_list()

    num_verts = input_graph.number_of_vertices()
    num_edges = len(input_graph.edgelist.edgelist_df['src'])

    cdef GraphCOOView[int,int,float] graph_float
    cdef GraphCOOView[int,int,double] graph_double

    df = cudf.DataFrame()
    df['vertex'] = cudf.Series(np.arange(num_verts, dtype=np.int32))

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
        if input_graph.renumbered is True:
            renumber_df = cudf.DataFrame()
            renumber_df['map'] = input_graph.edgelist.renumber_map
            renumber_df['id'] = input_graph.edgelist.renumber_map.index.astype(np.int32)
            start_pos = pos_list.merge(renumber_df, left_on='vertex', right_on='map', how='left').drop('map')
            # Remap pos and vertices
            df['vertex'][start_pos['id']] = start_pos['vertex']
            start_pos['x'][start_pos['id']] = start_pos['x']
            start_pos['y'][start_pos['id']] = start_pos['y']
            x_start = start_pos['x'].__cuda_array_interface__['data'][0]
            y_start = start_pos['y'].__cuda_array_interface__['data'][0]
        else:
            df['vertex'] = pos_list['vertex']
            x_start = pos_list['x'].__cuda_array_interface__['data'][0]
            y_start = pos_list['y'].__cuda_array_interface__['data'][0]

    cdef uintptr_t callback_ptr = 0
    if callback:
        callback_ptr = callback.get_native_callback()

    if input_graph.edgelist.weights \
            and input_graph.edgelist.edgelist_df['weights'].dtype == np.float64:

        pos = rmm.device_array(
                        (num_verts, 2),
                        order="F",
                        dtype=np.float64)

        pos_ptr = pos.device_ctypes_pointer.value

        graph_double = GraphCOOView[int,int, double](<int*>c_src_indices,
                <int*>c_dst_indices, <double*>c_weights, num_verts, num_edges)

        c_force_atlas2[int, int, double](graph_double,
                <float*>pos_ptr,
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
                <float>gravity,
                <bool> verbose,
                <GraphBasedDimRedCallback*>callback_ptr)

        pos_df = cudf.DataFrame.from_gpu_matrix(pos, columns=['x', 'y'])
        df['x'] = pos_df['x']
        df['y'] = pos_df['y']
    else:
        pos = rmm.device_array(
                (num_verts, 2),
                order="F",
                dtype=np.float32)

        pos_ptr = pos.device_ctypes_pointer.value

        graph_float = GraphCOOView[int,int,float](<int*>c_src_indices,
                <int*>c_dst_indices, <float*>c_weights, num_verts,
                num_edges)
        c_force_atlas2[int, int, float](graph_float,
                <float*>pos_ptr,
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
                <float>gravity,
                <bool> verbose,
                <GraphBasedDimRedCallback*>callback_ptr)

        pos_df = cudf.DataFrame.from_gpu_matrix(pos, columns=['x', 'y'])
        df['x'] = pos_df['x']
        df['y'] = pos_df['y']

    if pos_list is None and input_graph.renumbered:
        df = unrenumber(input_graph.edgelist.renumber_map, df, 'vertex')

    return df
