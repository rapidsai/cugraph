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

from cugraph.traversal.c_bfs cimport *
from cugraph.structure.c_graph cimport *
from cugraph.utilities.column_utils cimport *
from libcpp cimport bool
from libc.stdint cimport uintptr_t

import cudf
import cudf._lib as libcudf
import numpy as np


def bfs(graph_ptr, start, directed=True):
    """
    Call cugraph::bfs
    """

    cdef uintptr_t graph = graph_ptr
    cdef Graph* g = <Graph*>graph

    err = cugraph::add_adj_list(g)
    

    # we should add get_number_of_vertices() to Graph (and this should be
    # used instead of g.adjList.offsets.size - 1)
    num_verts = g.adjList.offsets.size - 1

    if not 0 <= start < num_verts:
        raise ValueError("Starting vertex should be between 0 to number of vertices")

    df = cudf.DataFrame()
    df['vertex'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    df['distance'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    df['predecessor'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    cdef gdf_column c_vertex_col = get_gdf_column_view(df['vertex'])
    cdef gdf_column c_distance_col = get_gdf_column_view(df['distance'])
    cdef gdf_column c_predecessor_col = get_gdf_column_view(df['predecessor'])

    err = g.adjList.get_vertex_identifiers(&c_vertex_col)
    

    cugraph::bfs(g, &c_distance_col, &c_predecessor_col, <int>start, <bool>directed)
    

    return df
