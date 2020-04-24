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

from cugraph.cores.k_core cimport k_core as c_k_core
from cugraph.structure.graph_new cimport *
from cugraph.structure import graph_new_wrapper
from cugraph.utilities.column_utils cimport *
from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free
from libc.float cimport FLT_MAX_EXP

import cudf
import rmm
import numpy as np
from rmm._lib.device_buffer cimport DeviceBuffer
from cudf.core.buffer import Buffer


#### FIXME:  Should return data frame instead of passing in k_core_graph...
####         Ripple down through implementation (algorithms.hpp, core_number.cu)

def weight_type(input_graph):
    weights_type = None
    if input_graph.edgelist.weights:
        weights_type = input_graph.edgelist.edgelist_df['weights'].dtype
    return weights_type


cdef (uintptr_t, uintptr_t, uintptr_t) graph_params(input_graph):
    if not input_graph.edgelist:
        input_graph.view_edge_list()

    [src, dst] = graph_new_wrapper.datatype_cast([input_graph.edgelist.edgelist_df['src'], input_graph.edgelist.edgelist_df['dst']], [np.int32])
    weights = None

    cdef uintptr_t c_src = src.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_dst = dst.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_weights = <uintptr_t>NULL

    if input_graph.edgelist.weights:
        [weights] = graph_new_wrapper.datatype_cast([input_graph.edgelist.edgelist_df['weights']], [np.float32, np.float64])
        c_weights = weights.__cuda_array_interface__['data'][0]
    return (c_src,c_dst,c_weights)


cdef (uintptr_t, uintptr_t) core_number_params(core_number):
    [core_number['vertex'], core_number['values']] = graph_new_wrapper.datatype_cast([core_number['vertex'], core_number['values']], [np.int32])
    cdef uintptr_t c_vertex = core_number['vertex'].__cuda_array_interface__['data'][0]
    cdef uintptr_t c_values = core_number['values'].__cuda_array_interface__['data'][0]
    return (c_vertex, c_values)


def k_core_float(input_graph, k, core_number):
    c_src, c_dst, c_weights = graph_params(input_graph)
    c_vertex, c_values = core_number_params(core_number)

    num_verts = input_graph.number_of_vertices()
    num_edges = input_graph.number_of_edges()
    cdef GraphCOOView[int,int,float] in_graph
    in_graph = GraphCOOView[int,int,float](<int*>c_src, <int*>c_dst, <float*>c_weights, num_verts, num_edges)
    cdef unique_ptr[GraphCOO[int,int,float]] out_graph = move(c_k_core[int,int,float](in_graph, k, <int*>c_vertex, <int*>c_values, len(core_number)))
    cdef GraphCOOContents[int,int,float] contents = move(out_graph.get()[0].release())
    src = DeviceBuffer.c_from_unique_ptr(move(contents.src_indices))
    dst = DeviceBuffer.c_from_unique_ptr(move(contents.dst_indices))
    wgt = DeviceBuffer.c_from_unique_ptr(move(contents.edge_data))
    src = Buffer(src)
    dst = Buffer(dst)

    df = cudf.DataFrame()
    df['src'] = cudf.core.column.build_column(data=src, dtype="int32", size=contents.number_of_edges)
    df['dst'] = cudf.core.column.build_column(data=dst, dtype="int32", size=contents.number_of_edges)
    if weight_type(input_graph) == np.float32:
        wgt = Buffer(wgt)
        df['weight'] = cudf.core.column.build_column(data=wgt, dtype="float32", size=contents.number_of_edges)

    print('DEBUG_MESSAGE k_core_wrapper.pyx:92 number of edges', contents.number_of_edges)
    print('DEBUG_MESSAGE k_core_wrapper.pyx:93 number of df edges', len(df))
    
    return df


def k_core_double(input_graph, k, core_number):
    c_src, c_dst, c_weights = graph_params(input_graph)
    c_vertex, c_values = core_number_params(core_number)

    num_verts = input_graph.number_of_vertices()
    num_edges = input_graph.number_of_edges()
    cdef GraphCOOView[int,int,double] in_graph
    in_graph = GraphCOOView[int,int,double](<int*>c_src, <int*>c_dst, <double*>c_weights, num_verts, num_edges)
    cdef unique_ptr[GraphCOO[int,int,double]] out_graph = move(c_k_core[int,int,double](in_graph, k, <int*>c_vertex, <int*>c_values, len(core_number)))
    cdef GraphCOOContents[int,int,double] contents = move(out_graph.get()[0].release())
    src = DeviceBuffer.c_from_unique_ptr(move(contents.src_indices))
    dst = DeviceBuffer.c_from_unique_ptr(move(contents.dst_indices))
    wgt = DeviceBuffer.c_from_unique_ptr(move(contents.edge_data))
    src = Buffer(src)
    dst = Buffer(dst)

    df = cudf.DataFrame()
    df['src'] = cudf.core.column.build_column(data=src, dtype="int32", size=contents.number_of_edges)
    df['dst'] = cudf.core.column.build_column(data=dst, dtype="int32", size=contents.number_of_edges)
    if weight_type(input_graph) == np.float64:
        wgt = Buffer(wgt)
        df['weight'] = cudf.core.column.build_column(data=wgt, dtype="float64", size=contents.number_of_edges)
    
    return df


def k_core(input_graph, k, core_number):
    """
    Call k_core
    """

    if weight_type(input_graph) == np.float64:
        return k_core_double(input_graph, k, core_number)
    else:
        return k_core_float(input_graph, k, core_number)
