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

from cugraph.community.ktruss_subgraph cimport *
from cugraph.structure.graph_new cimport *
from cugraph.structure import graph_new_wrapper
from cugraph.utilities.unrenumber import unrenumber
from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free
from libc.float cimport FLT_MAX_EXP

import cudf
import rmm
import numpy as np


def ktruss_subgraph_float(input_graph, k, use_weights):
    cdef GraphCOOViewFloat in_graph = get_graph_view[GraphCOOViewFloat](input_graph, use_weights)
    return coo_to_df(move(k_truss_subgraph[int,int,float](in_graph, k)))


def ktruss_subgraph_double(input_graph, k, use_weights):
    cdef GraphCOOViewDouble in_graph = get_graph_view[GraphCOOViewDouble](input_graph, use_weights)
    return coo_to_df(move(k_truss_subgraph[int,int,double](in_graph, k)))


def ktruss_subgraph(input_graph, k, use_weights):
    if graph_new_wrapper.weight_type(input_graph) == np.float64 and use_weights:
        return ktruss_subgraph_double(input_graph, k, use_weights)
    else:
        return ktruss_subgraph_float(input_graph, k, use_weights)
