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

from cugraph.cores.c_k_core cimport *
from cugraph.structure.c_graph cimport *
from cugraph.utilities.column_utils cimport *
from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free
from libc.float cimport FLT_MAX_EXP

import cudf
import cudf._lib as libcudf
import rmm
import numpy as np


def k_core(graph_ptr, k_core_graph_ptr, k, core_number):
    """
    Call k_core
    """
    cdef uintptr_t graph = graph_ptr
    cdef Graph* g = <Graph*>graph

    cdef uintptr_t rGraph = k_core_graph_ptr
    cdef Graph* rg = <Graph*>rGraph

    cdef gdf_column c_vertex = get_gdf_column_view(core_number['vertex'])
    cdef gdf_column c_values = get_gdf_column_view(core_number['values'])
    k_core(g, k, &c_vertex, &c_values, rg)

    
