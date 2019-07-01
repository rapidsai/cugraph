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

from cugraph.link_analysis.c_pagerank cimport *
from cugraph.structure.c_graph cimport *
from cugraph.utilities.column_utils cimport *
from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

import cudf
from librmm_cffi import librmm as rmm
import numpy as np


def pagerank(graph_ptr,alpha=0.85, personalization=None, max_iter=100, tol=1.0e-5, nstart=None):
    """
    Call gdf_pagerank
    """

    cdef uintptr_t graph = graph_ptr
    cdef gdf_graph* g = <gdf_graph*>graph

    err = gdf_add_transposed_adj_list(g)
    cudf.bindings.cudf_cpp.check_gdf_error(err)

    # we should add get_number_of_vertices() to gdf_graph (and this should be
    # used instead of g.transposedAdjList.offsets.size - 1)
    num_verts = g.transposedAdjList.offsets.size - 1

    df = cudf.DataFrame()  
    df['vertex'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    cdef gdf_column c_identifier_col = get_gdf_column_view(df['vertex']) 
    df['pagerank'] = cudf.Series(np.zeros(num_verts, dtype=np.float32))
    cdef gdf_column c_pagerank_col = get_gdf_column_view(df['pagerank'])    

    cdef bool has_guess = <bool> 0
    if nstart is not None:
        cudf.bindings.copying.apply_scatter([nstart['values']._column],
                                            nstart['vertex']._column._data.mem,
                                            [df['pagerank']._column])
        has_guess = <bool> 1

    cdef gdf_column c_pers_vtx
    cdef gdf_column c_pers_val

    err = g.transposedAdjList.get_vertex_identifiers(&c_identifier_col)
    cudf.bindings.cudf_cpp.check_gdf_error(err)
    if personalization is None:
        err = gdf_pagerank(g, &c_pagerank_col, <gdf_column*> NULL, <gdf_column*> NULL,
                <float> alpha, <float> tol, <int> max_iter, has_guess)
    else:
        c_pers_vtx = get_gdf_column_view(personalization['vertex'])
        c_pers_val = get_gdf_column_view(personalization['values'])
        err = gdf_pagerank(g, &c_pagerank_col, &c_pers_vtx, &c_pers_val,
                <float> alpha, <float> tol, <int> max_iter, has_guess)

    cudf.bindings.cudf_cpp.check_gdf_error(err)

    return df
