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

from cugraph.components.c_connectivity cimport *
from cugraph.structure.c_graph cimport *
from cugraph.utilities.column_utils cimport *
from cudf._lib.utils cimport table_from_dataframe
from libc.stdint cimport uintptr_t

import cudf
import cudf._lib as libcudf
import numpy as np

def weakly_connected_components(graph_ptr):
    """
    Call gdf_connected_components
    """

    cdef uintptr_t graph = graph_ptr
    cdef Graph* g = <Graph*>graph

    err = cugraph::add_adj_list(<Graph*> graph)
    

    # we should add get_number_of_vertices() to Graph (and this should be
    # used instead of g.adjList.offsets.size - 1)
    num_verts = g.adjList.offsets.size - 1

    df = cudf.DataFrame()
    df['labels'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    df['vertices'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    
    cdef cudf_table* tbl = table_from_dataframe(df)

    cdef cugraph_cc_t connect_type=CUGRAPH_WEAK
    cugraph::connected_components(g, <cugraph_cc_t>connect_type, tbl)
    

    del tbl

    return df


def strongly_connected_components(graph_ptr):
    """
    Call gdf_connected_components
    """

    cdef uintptr_t graph = graph_ptr
    cdef Graph* g = <Graph*>graph

    err = cugraph::add_adj_list(<Graph*> graph)
    

    # we should add get_number_of_vertices() to Graph (and this should be
    # used instead of g.adjList.offsets.size - 1)
    num_verts = g.adjList.offsets.size - 1

    df = cudf.DataFrame()
    df['labels'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    df['vertices'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    
    cdef cudf_table* tbl = table_from_dataframe(df)

    cdef cugraph_cc_t connect_type=CUGRAPH_STRONG
    cugraph::connected_components(g, <cugraph_cc_t>connect_type, tbl)
    

    del tbl

    return df
