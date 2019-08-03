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

from cugraph.traversal.c_sssp cimport *
from cugraph.structure.c_graph cimport *
from cugraph.utilities.column_utils cimport *
from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free
from libc.float cimport FLT_MAX_EXP

from cugraph.utilities.column_utils import gdf_to_np_dtype_tmp
import cudf
from librmm_cffi import librmm as rmm
import numpy as np


def sssp(graph_ptr, source):
    """
    Call gdf_sssp_nvgraph
    """
    cdef uintptr_t graph = graph_ptr
    cdef gdf_graph* g = <gdf_graph*>graph

    err = gdf_add_adj_list(g)
    cudf.bindings.cudf_cpp.check_gdf_error(err)

    # we should add get_number_of_vertices() to gdf_graph (and this should be
    # used instead of g.adjList.offsets.size - 1)
    num_verts = g.adjList.offsets.size - 1

    data_type = np.float32
    if g.adjList.edge_data:
        data_type = gdf_to_np_dtype_tmp[g.adjList.edge_data.dtype]

    df = cudf.DataFrame()
    df['vertex'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    cdef gdf_column c_identifier_col = get_gdf_column_view(df['vertex'])
    df['distance'] = cudf.Series(np.zeros(num_verts, dtype=data_type))
    cdef gdf_column c_distance_col = get_gdf_column_view(df['distance'])
    df['predecessor'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    cdef gdf_column c_predecessors_col = get_gdf_column_view(df['predecessor'])

    err = g.adjList.get_vertex_identifiers(&c_identifier_col)
    cudf.bindings.cudf_cpp.check_gdf_error(err)

    err = gdf_sssp(g, &c_distance_col, &c_predecessors_col, <int>source)
    cudf.bindings.cudf_cpp.check_gdf_error(err)

    return df
