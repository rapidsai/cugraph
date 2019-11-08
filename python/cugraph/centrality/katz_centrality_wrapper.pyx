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

from cugraph.centrality.c_katz_centrality cimport *
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


def katz_centrality(graph_ptr, alpha=0.1, max_iter=100, tol=1.0e-5, nstart=None, normalized=True):
    """
    Call cugraph::katz_centrality
    """
    cdef uintptr_t graph = graph_ptr
    cdef cugraph::Graph* g = <cugraph::Graph*>graph

    err = cugraph::add_adj_list(g)
    

    # we should add get_number_of_vertices() to cugraph::Graph (and this should be
    # used instead of g.adjList.offsets.size - 1)
    num_verts = g.adjList.offsets.size - 1

    df = cudf.DataFrame()
    df['vertex'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    cdef gdf_column c_identifier_col = get_gdf_column_view(df['vertex'])
    df['katz_centrality'] = cudf.Series(np.zeros(num_verts, dtype=np.float64))
    cdef gdf_column c_katz_centrality_col = get_gdf_column_view(df['katz_centrality'])

    cdef bool has_guess = <bool> 0
    if nstart is not None:
        cudf.bindings.copying.apply_scatter([nstart['values']._column],
                                            nstart['vertex']._column._data.mem,
                                            [df['katz_centrality']._column])
        has_guess = <bool> 1

    err = g.adjList.get_vertex_identifiers(&c_identifier_col)
    

    cugraph::katz_centrality(g, &c_katz_centrality_col, alpha, max_iter, tol, has_guess, normalized)

    

    return df
