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


cdef (uintptr_t, uintptr_t, uintptr_t, int, int) graph_params(input_graph):
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

    num_verts = input_graph.number_of_vertices()
    num_edges = len(input_graph.edgelist.edgelist_df)
    return (c_src,c_dst,c_weights,num_verts,num_edges)


cdef (uintptr_t, uintptr_t) core_number_params(core_number):
    [core_number['vertex'], core_number['values']] = graph_new_wrapper.datatype_cast([core_number['vertex'], core_number['values']], [np.int32])
    cdef uintptr_t c_vertex = core_number['vertex'].__cuda_array_interface__['data'][0]
    cdef uintptr_t c_values = core_number['values'].__cuda_array_interface__['data'][0]
    return (c_vertex, c_values)


cdef GraphCOOViewType get_graph_view(input_graph, GraphCOOViewType* dummy=NULL):
    c_src, c_dst, c_weights, num_verts, num_edges = graph_params(input_graph)
    cdef GraphCOOViewType in_graph
    if GraphCOOViewType is GraphCOOViewFloat:
        in_graph = GraphCOOViewFloat(<int*>c_src, <int*>c_dst, <float*>c_weights, num_verts, num_edges)
    elif GraphCOOViewType is GraphCOOViewDouble:
        in_graph = GraphCOOViewDouble(<int*>c_src, <int*>c_dst, <double*>c_weights, num_verts, num_edges)
    return in_graph


cdef coo_to_df(GraphCOOType graph):
    contents = move(graph.get()[0].release())
    src = DeviceBuffer.c_from_unique_ptr(move(contents.src_indices))
    dst = DeviceBuffer.c_from_unique_ptr(move(contents.dst_indices))
    wgt = DeviceBuffer.c_from_unique_ptr(move(contents.edge_data))
    src = Buffer(src)
    dst = Buffer(dst)
    wgt = Buffer(wgt)

    src = cudf.Series(data=src, dtype="int32")
    dst = cudf.Series(data=dst, dtype="int32")

    df = cudf.DataFrame()
    df['src'] = src
    df['dst'] = dst
    if wgt.nbytes != 0:
        if GraphCOOType is GraphCOOFloat:
            wgt = cudf.Series(data=wgt, dtype="float32")
        elif GraphCOOType is GraphCOODouble:
            wgt = cudf.Series(data=wgt, dtype="float64")
        df['weight'] = wgt
    return df


def k_core_float(input_graph, k, core_number):
    c_vertex, c_values = core_number_params(core_number)
    cdef GraphCOOViewFloat in_graph = get_graph_view[GraphCOOViewFloat](input_graph)
    return coo_to_df(move(c_k_core[int,int,float](in_graph, k, <int*>c_vertex, <int*>c_values, len(core_number))))


def k_core_double(input_graph, k, core_number):
    c_vertex, c_values = core_number_params(core_number)
    cdef GraphCOOViewDouble in_graph = get_graph_view[GraphCOOViewDouble](input_graph)
    return coo_to_df(move(c_k_core[int,int,double](in_graph, k, <int*>c_vertex, <int*>c_values, len(core_number))))


def k_core(input_graph, k, core_number):
    """
    Call k_core
    """

    if weight_type(input_graph) == np.float64:
        return k_core_double(input_graph, k, core_number)
    else:
        return k_core_float(input_graph, k, core_number)
