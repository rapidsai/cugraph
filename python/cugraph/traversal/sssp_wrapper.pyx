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

cimport cugraph.traversal.sssp as c_sssp
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


def sssp(input_graph, source):
    """
    Call sssp_nvgraph
    """
    cdef GraphCSR[int, int, float]  graph_float     # For weighted float graph (SSSP) and Unweighted (BFS)
    cdef GraphCSR[int, int, double] graph_double    # For weighted double graph (SSSP)

    cdef uintptr_t c_weights = <uintptr_t> NULL     #
    cdef uintptr_t c_offsets = <uintptr_t> NULL     #
    cdef uintptr_t c_indices = <uintptr_t> NULL     #

    cdef uintptr_t c_vertex_col = <uintptr_t> NULL      #
    cdef uintptr_t c_distance_col = <uintptr_t> NULL    #
    cdef uintptr_t c_predecessor_col = <uintptr_t> NULL #

    offset = None
    indices = None

    # SSSP is expecting the graph to be in CSR format
    if not input_graph.adjlist:
        input_graph.view_adj_list()

    # Obtain CSR offsets and indices from the input_graph.adjlist
    # Offsets and indices data type can only be int (signed, 32-bit)
    [offsets, indices] = graph_wrapper.datatype_cast([input_graph.adjlist.offsets, input_graph.adjlist.indices], [np.int32])
    c_offsets = offsets.__cuda_array_interface__['data'][0]
    c_indices = indices.__cuda_array_interface__['data'][0]

    num_verts = input_graph.number_of_vertices()
    num_edges = len(indices)

    if input_graph.renumbered is True:
        source = input_graph.edgelist.renumber_map[input_graph.edgelist.renumber_map == source].index[0]
    if not 0 <= source < num_verts:
        raise ValueError("Starting vertex should be between 0 to number of vertices")


    # Generation of the result
    df = cudf.DataFrame()

    df['vertex'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    c_vertex_col = df['vertex'].__cuda_array_interface__['data'][0]


    df['predecessor'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    c_distance_col = df['predecessor'].__cuda_array_interface__['data'][0]

    if input_graph.edgelist.weights:
        [weights] = graph_wrapper.datatype_cast([input_graph.edgelist.edgelist_df['weights']], [np.float32, np.float64])
        c_weights = weights.__cuda_array_interface__['data'][0]

        df['distance'] = cudf.Series(np.zeros(num_verts, dtype=weights.dtype))
        c_distance_col = df['distance'].__cuda_array_interface__['data'][0]

        if (df['distance'].dtype == np.float32):
            graph_float = GraphCSR[int, int, float](<int*>c_offsets,
                                                     <int*>c_indices,
                                                     <float*>c_weights,
                                                     num_verts,
                                                     num_edges)
            c_sssp.sssp[int, int, float](graph_float, <float*>c_distance_col, <int*>c_predecessor_col, <int>source)
            graph_float.get_vertex_identifiers(<int*>c_vertex_col)
        else: # Should probably check for it
            graph_double = GraphCSR[int, int, double](<int*>c_offsets,
                                                      <int*>c_indices,
                                                      <double*>c_weights,
                                                      num_verts,
                                                      num_edges)
            c_sssp.sssp[int, int, double](graph_double, <double*>c_distance_col, <int*>c_predecessor_col, <int>source)
            graph_double.get_vertex_identifiers(<int*>c_vertex_col)
    else:
        df['distance'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
        c_distance_col = df['distance'].__cuda_array_interface__['data'][0]
        graph_float = GraphCSR[int, int, float](<int*>c_offsets,
                                                <int*>c_indices,
                                                <float*>NULL,
                                                num_verts,
                                                num_edges)
        c_bfs.bfs[int, int, float](graph_float, <int*>c_distance_col, <int*>c_predecessor_col, <int>source)

    if input_graph.renumbered:
        if isinstance(input_graph.edgelist.renumber_map, cudf.DataFrame):
            raise NotImplementedError
            n_cols = len(input_graph.edgelist.renumber_map.columns) - 1
            unrenumbered_df_ = df.merge(input_graph.edgelist.renumber_map, left_on='vertex', right_on='id', how='left').drop(['id', 'vertex'])
            unrenumbered_df = unrenumbered_df_.merge(input_graph.edgelist.renumber_map, left_on='predecessor', right_on='id', how='left').drop(['id', 'predecessor'])
            unrenumbered_df.columns = ['distance']+['vertex_'+str(i) for i in range(n_cols)]+['predecessor_'+str(i) for i in range(n_cols)]
            cols = unrenumbered_df.columns.to_list()
            df = unrenumbered_df[cols[1:n_cols+1] + [cols[0]] + cols[n_cols:]]
        else:
            df = unrenumber(input_graph.edgelist.renumber_map, df, 'vertex')
            #df['vertex'] = input_graph.edgelist.renumber_map[df['vertex']]
            #df['predecessor'][df['predecessor']>-1] = input_graph.edgelist.renumber_map[df['predecessor'][df['predecessor']>-1]]
    return df