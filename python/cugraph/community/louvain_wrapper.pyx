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

from cugraph.community.c_louvain cimport *
from cugraph.structure.c_graph cimport *
from cugraph.utilities.column_utils cimport *
from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

import cudf
import cudf._lib as libcudf
import rmm
import numpy as np


def louvain(graph_ptr):
    """
    Call gdf_louvain
    """

    cdef uintptr_t graph = graph_ptr
    cdef Graph* g = <Graph*>graph

    err = cugraph::add_adj_list(g)
    

    # we should add get_number_of_vertices() to Graph (and this should be
    # used instead of g.adjList.offsets.size - 1)
    num_verts = g.adjList.offsets.size - 1

    df = cudf.DataFrame()
    df['vertex'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    cdef gdf_column c_index_col = get_gdf_column_view(df['vertex'])
    err = g.adjList.get_vertex_identifiers(&c_index_col)
    

    df['partition'] = cudf.Series(np.zeros(num_verts,dtype=np.int32))
    cdef gdf_column c_louvain_parts_col = get_gdf_column_view(df['partition'])

    cdef bool single_precision = False
    # this implementation is tied to cugraph.cu line 503
    # cudaDataType_t val_type = graph->adjList->edge_data?
    #     gdf_to_cudadtype(graph->adjList->edge_data): CUDA_R_32F;
    # this is tied to the low-level implementation detail of the lower level
    # function, and very vulnerable to low level changes. Better be
    # reimplemented, but we are planning to eventually remove nvgraph, so I may
    # leave as is right at this moment.
    if g.adjList.edge_data:
        if g.adjList.edge_data.dtype == GDF_FLOAT32:
            single_precision = True;
    else:
        single_precision = True;

    cdef float final_modularity_single_precision = 1.0
    cdef double final_modularity_double_precision = 1.0
    cdef int num_level = 0
    

    if single_precision:
        cugraph::louvain(<Graph*>g,
                          <void*>&final_modularity_single_precision,
                          <void*>&num_level, &c_louvain_parts_col)
    else:
        cugraph::louvain(<Graph*>g,
                          <void*>&final_modularity_double_precision,
                          <void*>&num_level, &c_louvain_parts_col)
    

    if single_precision:
        return df, <double>final_modularity_single_precision
    else:
        return df, final_modularity_double_precision
