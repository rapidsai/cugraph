# Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
from cugraph.structure.graph_primtypes cimport *
from cugraph.structure import graph_primtypes_wrapper
from libc.stdint cimport uintptr_t
import numpy as np


#### FIXME:  Should return data frame instead of passing in k_core_graph...
####         Ripple down through implementation (algorithms.hpp, core_number.cu)

cdef (uintptr_t, uintptr_t) core_number_params(core_number):
    [core_number['vertex'], core_number['values']] = graph_primtypes_wrapper.datatype_cast([core_number['vertex'], core_number['values']], [np.int32])
    cdef uintptr_t c_vertex = core_number['vertex'].__cuda_array_interface__['data'][0]
    cdef uintptr_t c_values = core_number['values'].__cuda_array_interface__['data'][0]
    return (c_vertex, c_values)


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
    [input_graph.edgelist.edgelist_df['src'],
     input_graph.edgelist.edgelist_df['dst']] = graph_primtypes_wrapper.datatype_cast([input_graph.edgelist.edgelist_df['src'],
                                                                                       input_graph.edgelist.edgelist_df['dst']],
                                                                                      [np.int32])
    if graph_primtypes_wrapper.weight_type(input_graph) == np.float64:
        return k_core_double(input_graph, k, core_number)
    else:
        return k_core_float(input_graph, k, core_number)
