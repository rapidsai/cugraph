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


#### FIXME:  Should return data frame instead of passing in k_core_graph...
####         Ripple down through implementation (algorithms.hpp, core_number.cu)

def k_core(input_graph, k, core_number):
    """
    Call k_core
    """
    if not input_graph.edgelist:
        input_graph.view_edge_list()

    [src, dst] = graph_new_wrapper.datatype_cast([input_graph.edgelist.edgelist_df['src'], input_graph.edgelist.edgelist_df['dst']], [np.int32])
    weights = None
    weights_type = np.float32

    num_verts = input_graph.number_of_vertices()
    num_edges = len(src)

    [core_number['vertex'], core_number['values']] = graph_new_wrapper.datatype_cast([core_number['vertex'], core_number['values']], [np.int32])

    cdef uintptr_t c_src = src.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_dst = dst.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_vertex = core_number['vertex'].__cuda_array_interface__['data'][0]
    cdef uintptr_t c_values = core_number['values'].__cuda_array_interface__['data'][0]
    cdef uintptr_t c_weights = <uintptr_t>NULL

    if input_graph.edgelist.weights:
        [weights] = graph_new_wrapper.datatype_cast([input_graph.edgelist.edgelist_df['weights']], [np.float32, np.float64])
        weight_type = weights.dtype
        c_weights = weights.__cuda_array_interface__['data'][0]

    cdef GraphCOO[int,int,float] in_graph_float
    cdef GraphCOO[int,int,float] out_graph_float
    cdef GraphCOO[int,int,double] in_graph_double
    cdef GraphCOO[int,int,double] out_graph_double

    df = cudf.DataFrame()
    
    if weights_type == np.float32:
        in_graph_float = GraphCOO[int,int,float](<int*>c_src, <int*>c_dst, <float*>c_weights, num_verts, num_edges)
        c_k_core[int,int,float](in_graph_float, k, <int*> c_vertex, <int*> c_values, len(core_number), out_graph_float)

        tmp = rmm.device_array_from_ptr(<uintptr_t>out_graph_float.src_indices,
                                        nelem=out_graph_float.number_of_edges,
                                        dtype=np.int32)
        df['src'] = cudf.Series(tmp)

        tmp = rmm.device_array_from_ptr(<uintptr_t>out_graph_float.dst_indices,
                                        nelem=out_graph_float.number_of_edges,
                                        dtype=np.int32)
        df['dst'] = cudf.Series(tmp)

        if weights is not None:
            tmp = rmm.device_array_from_ptr(<uintptr_t>out_graph_float.edge_data,
                                            nelem=out_graph_float.number_of_edges,
                                            dtype=np.int32)
            df['weight'] = tmp
    else:
        in_graph_double = GraphCOO[int,int,double](<int*>c_src, <int*>c_dst, <double*>c_weights, num_verts, num_edges)
        c_k_core[int,int,double](in_graph_double, k, <int*> &c_vertex, <int*> &c_values, len(core_number), out_graph_double)

        tmp = rmm.device_array_from_ptr(<uintptr_t>out_graph_double.src_indices,
                                        nelem=out_graph_double.number_of_edges,
                                        dtype=np.int32)
        df['src'] = cudf.Series(tmp)

        tmp = rmm.device_array_from_ptr(<uintptr_t>out_graph_double.dst_indices,
                                        nelem=out_graph_double.number_of_edges,
                                        dtype=np.int32)
        df['dst'] = cudf.Series(tmp)

        if weights is not None:
            tmp = rmm.device_array_from_ptr(<uintptr_t>out_graph_double.edge_data,
                                            nelem=out_graph_double.number_of_edges,
                                            dtype=np.int32)
            df['weight'] = tmp
        

    return df
