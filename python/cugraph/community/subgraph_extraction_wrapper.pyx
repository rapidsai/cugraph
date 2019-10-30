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

from cugraph.community.c_subgraph_extraction cimport *
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


def subgraph(graph_ptr, vertices, subgraph_ptr):
    """
    Call gdf_extract_subgraph_vertex_nvgraph
    """

    cdef uintptr_t graph = graph_ptr
    cdef gdf_graph * g = < gdf_graph *> graph

    cdef uintptr_t rGraph = subgraph_ptr
    cdef gdf_graph* rg = <gdf_graph*>rGraph
    cdef gdf_column vert_col = get_gdf_column_view(vertices)

    err = gdf_add_adj_list(g)
    libcudf.cudf.check_gdf_error(err)

    err = gdf_extract_subgraph_vertex_nvgraph(g, &vert_col, rg)
    libcudf.cudf.check_gdf_error(err)
