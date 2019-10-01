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

from cugraph.link_prediction.c_jaccard cimport *
from cugraph.structure.c_graph cimport *
from cugraph.utilities.column_utils cimport *
from cudf._lib.cudf cimport np_dtype_from_gdf_column
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

import cudf
import cudf._lib as libcudf
import rmm
import numpy as np
from numpy.core.numeric import result_type


def jaccard_w(graph_ptr, weights, first=None, second=None):
    """
    call gdf_jaccard_list
    """

    cdef uintptr_t graph = graph_ptr
    cdef gdf_graph * g = <gdf_graph*> graph

    err = gdf_add_adj_list(g)
    libcudf.cudf.check_gdf_error(err)

    cdef gdf_column c_result_col
    cdef gdf_column c_weight_col
    cdef gdf_column c_first_col
    cdef gdf_column c_second_col
    cdef gdf_column c_index_col

    if type(first) == cudf.Series and type(second) == cudf.Series:
        result_size = len(first)
        result = cudf.Series(np.ones(result_size, dtype=np.float32))
        c_result_col = get_gdf_column_view(result)
        c_weight_col = get_gdf_column_view(weights)
        c_first_col = get_gdf_column_view(first)
        c_second_col = get_gdf_column_view(second)
        err = gdf_jaccard_list(g,
                               &c_weight_col,
                               &c_first_col,
                               &c_second_col,
                               &c_result_col)
        libcudf.cudf.check_gdf_error(err)
        df = cudf.DataFrame()
        df['source'] = first
        df['destination'] = second
        df['jaccard_coeff'] = result
        return df

    else:
        # error check performed in jaccard.py
        assert first is None and second is None
        # we should add get_number_of_edges() to gdf_graph (and this should be
        # used instead of g.adjList.indices.size)
        num_edges = g.adjList.indices.size
        result = cudf.Series(np.ones(num_edges, dtype=np.float32))
        c_result_col = get_gdf_column_view(result)
        c_weight_col = get_gdf_column_view(weights)

        err = gdf_jaccard(g, &c_weight_col, &c_result_col)
        libcudf.cudf.check_gdf_error(err)

        dest_data = rmm.device_array_from_ptr(<uintptr_t> g.adjList.indices.data,
                                            nelem=num_edges,
                                            dtype=np_dtype_from_gdf_column(g.adjList.indices))
        df = cudf.DataFrame()
        df['source'] = cudf.Series(np.zeros(num_edges, dtype=np_dtype_from_gdf_column(g.adjList.indices)))
        c_index_col = get_gdf_column_view(df['source'])
        err = g.adjList.get_source_indices(&c_index_col);
        libcudf.cudf.check_gdf_error(err)
        df['destination'] = cudf.Series(dest_data)
        df['jaccard_coeff'] = result

        return df
