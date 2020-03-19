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

#cimport cugraph.traversal.bfs as c_bfs
"""
from  cugraph.traversal.bfs cimport bfs as c_bfs
from cugraph.structure.graph_new cimport *
from cugraph.structure import graph_wrapper
from cugraph.utilities.column_utils cimport *
from libcpp cimport bool
from libc.stdint cimport uintptr_t

import cudf
import cudf._lib as libcudf
import numpy as np
"""

"""
cimport cugraph.traversal.bfs as c_bfs

from cugraph.structure import graph_wrapper
from cugraph.structure.graph_new cimport *
from cugraph.utilities.column_utils cimport *
from cugraph.utilities.unrenumber import unrenumber
from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

import cudf
import cudf._lib as libcudf
import rmm
import numpy as np
import numpy.ctypeslib as ctypeslib
"""
cimport cugraph.traversal.bfs as c_bfs
from cugraph.structure.graph_new cimport *
from cugraph.structure import graph_wrapper
from cugraph.utilities.column_utils cimport *
from cudf._lib.cudf cimport np_dtype_from_gdf_column
from cugraph.utilities.unrenumber import unrenumber
from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free
from libc.float cimport FLT_MAX_EXP

import cudf
import cudf._lib as libcudf
import rmm
import numpy as np



def bfs(input_graph, start, directed=True):
    """
    Call bfs
    """
    cdef GraphCSR[int, int, float] graph_float

    cdef uintptr_t c_weights = <uintptr_t> NULL
    cdef uintptr_t c_offsets = <uintptr_t> NULL
    cdef uintptr_t c_indices = <uintptr_t> NULL

    cdef uintptr_t c_vertex_col = <uintptr_t> NULL
    cdef uintptr_t c_distance_col = <uintptr_t> NULL
    cdef uintptr_t c_predecessors_col = <uintptr_t> NULL


    if input_graph.adjlist is None:
        input_graph.view_adj_list()

    [offsets, indices] = graph_wrapper.datatype_cast([input_graph.adjlist.offsets, input_graph.adjlist.indices], [np.int32])
    # But it should not be used with weighted graphs
    [weights] = graph_wrapper.datatype_cast([input_graph.adjlist.weights], [np.int32])
    c_offsets = offsets.__cuda_array_interface__['data'][0]
    c_indices = indices.__cuda_array_interface__['data'][0]
    c_weights = weights.__cuda_array_interface__['data'][0]

    # we should add get_number_of_vertices() to Graph (and this should be
    # used instead of g.adjList.offsets.size - 1)
    num_verts = input_graph.number_of_vertices()
    num_edges = len(indices)

    if input_graph.renumbered is True:
        start = input_graph.edgelist.renumber_map[input_graph.edgelist.renumber_map==start].index[0]
    if not 0 <= start < num_verts:
        raise ValueError("Starting vertex should be between 0 to number of vertices")

    df = cudf.DataFrame()
    df['vertex'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    c_vertex_col = df['vertex'].__cuda_array_interface__['data'][0]

    df['distance'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    c_distance_col = df['distance'].__cuda_array_interface__['data'][0]

    df['predecessor'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    c_predecessor_col = df['predecessor'].__cuda_array_interface__['data'][0]

    # TODO: Either we set [int, int, float] or we add an explicit [int, int, int] in graph.cu
    graph_float = GraphCSR[int, int, float](<int*>c_offsets,
                                        <int*>c_indices,
                                        <float*>NULL,
                                        num_verts,
                                        num_edges)
    c_bfs.bfs[int, int, float](graph_float,
                             <int*>c_distance_col,
                             <int*>c_predecessors_col,
                             <int>start,
                             directed)
    graph_float.get_vertex_identifiers(<int*>c_vertex_col)
    if input_graph.renumbered:
        df = unrenumber(input_graph.edgelist.renumber_map, df, 'vertex')
    """

    if input_graph.renumbered:
        if isinstance(input_graph.edgelist.renumber_map, cudf.DataFrame):
            n_cols = len(input_graph.edgelist.renumber_map.columns) - 1
            unrenumbered_df_ = df.merge(input_graph.edgelist.renumber_map, left_on='vertex', right_on='id', how='left').drop(['id', 'vertex'])
            unrenumbered_df = unrenumbered_df_.merge(input_graph.edgelist.renumber_map, left_on='predecessor', right_on='id', how='left').drop(['id', 'predecessor'])
            unrenumbered_df.columns = ['distance']+['vertex_'+str(i) for i in range(n_cols)]+['predecessor_'+str(i) for i in range(n_cols)]
            cols = unrenumbered_df.columns.to_list()
            df = unrenumbered_df[cols[1:n_cols+1] + [cols[0]] + cols[n_cols:]]
        else:
            df['vertex'] = input_graph.edgelist.renumber_map[df['vertex']]
            df['predecessor'][df['predecessor']>-1] = input_graph.edgelist.renumber_map[df['predecessor'][df['predecessor']>-1]]

    """
    return df