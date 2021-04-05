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

from cugraph.community.ktruss_subgraph cimport *
from cugraph.structure.graph_primtypes cimport *
from cugraph.structure import graph_primtypes_wrapper
import numpy as np


def ktruss_subgraph_float(input_graph, k, use_weights):
    cdef GraphCOOViewFloat in_graph = get_graph_view[GraphCOOViewFloat](input_graph, use_weights)
    return coo_to_df(move(k_truss_subgraph[int,int,float](in_graph, k)))


def ktruss_subgraph_double(input_graph, k, use_weights):
    cdef GraphCOOViewDouble in_graph = get_graph_view[GraphCOOViewDouble](input_graph, use_weights)
    return coo_to_df(move(k_truss_subgraph[int,int,double](in_graph, k)))


def ktruss_subgraph(input_graph, k, use_weights):
    [input_graph.edgelist.edgelist_df['src'],
     input_graph.edgelist.edgelist_df['dst']] = graph_primtypes_wrapper.datatype_cast([input_graph.edgelist.edgelist_df['src'],
                                                                                       input_graph.edgelist.edgelist_df['dst']],
                                                                                      [np.int32])
    if graph_primtypes_wrapper.weight_type(input_graph) == np.float64 and use_weights:
        return ktruss_subgraph_double(input_graph, k, use_weights)
    else:
        return ktruss_subgraph_float(input_graph, k, use_weights)
