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

from cugraph.cores.c_core_number cimport *
from cugraph.structure.c_graph cimport *
from cugraph.utilities.column_utils cimport *
from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free
from libc.float cimport FLT_MAX_EXP

import cudf
import cudf._lib as libcudf
from librmm_cffi import librmm as rmm
import numpy as np


def core_number(graph_ptr):
    """
    Call gdf_core_number
    """
    cdef uintptr_t graph = graph_ptr
    cdef gdf_graph* g = <gdf_graph*>graph

    err = gdf_add_adj_list(g)
    libcudf.cudf.check_gdf_error(err)

    # we should add get_number_of_vertices() to gdf_graph (and this should be
    # used instead of g.adjList.offsets.size - 1)
    num_verts = g.adjList.offsets.size - 1

    df = cudf.DataFrame()
    df['vertex'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    cdef gdf_column c_identifier_col = get_gdf_column_view(df['vertex'])
    df['core_number'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    cdef gdf_column c_core_number_col = get_gdf_column_view(df['core_number'])

    err = g.adjList.get_vertex_identifiers(&c_identifier_col)
    libcudf.cudf.check_gdf_error(err)

    err = gdf_core_number(g, &c_core_number_col)

    libcudf.cudf.check_gdf_error(err)

    return df

