# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

cimport cugraph.traversal.bfs as c_bfs
from cugraph.structure.graph_new cimport *
from cugraph.structure import graph_new_wrapper
from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.float cimport FLT_MAX_EXP

import cudf
import rmm
import numpy as np

def bfs(input_graph, start, directed=True,
        return_sp_counter=False):
    """
    Call bfs
    """
    # Step 1: Declare the different varibales
    cdef GraphCSRView[int, int, float]  graph_float     # For weighted float graph (SSSP) and Unweighted (BFS)
    cdef GraphCSRView[int, int, double] graph_double    # For weighted double graph (SSSP)

    # Pointers required for CSR Graph
    cdef uintptr_t c_offsets_ptr        = <uintptr_t> NULL # Pointer to the CSR offsets
    cdef uintptr_t c_indices_ptr        = <uintptr_t> NULL # Pointer to the CSR indices

    # Pointers for SSSP / BFS
    cdef uintptr_t c_identifier_ptr     = <uintptr_t> NULL # Pointer to the DataFrame 'vertex' Series
    cdef uintptr_t c_distance_ptr       = <uintptr_t> NULL # Pointer to the DataFrame 'distance' Series
    cdef uintptr_t c_predecessor_ptr    = <uintptr_t> NULL # Pointer to the DataFrame 'predecessor' Series
    cdef uintptr_t c_sp_counter_ptr     = <uintptr_t> NULL # Pointer to the DataFrame 'sp_counter' Series

    # Step 2: Verifiy input_graph has the expected format
    if input_graph.adjlist is None:
        input_graph.view_adj_list()

    cdef unique_ptr[handle_t] handle_ptr
    handle_ptr.reset(new handle_t())

    # Step 3: Extract CSR offsets, indices, weights are not expected
    #         - offsets: int (signed, 32-bit)
    #         - indices: int (signed, 32-bit)
    [offsets, indices] = graph_new_wrapper.datatype_cast([input_graph.adjlist.offsets, input_graph.adjlist.indices], [np.int32])
    c_offsets_ptr = offsets.__cuda_array_interface__['data'][0]
    c_indices_ptr = indices.__cuda_array_interface__['data'][0]

    # Step 4: Setup number of vertices and edges
    num_verts = input_graph.number_of_vertices()
    num_edges = input_graph.number_of_edges(directed_edges=True)

    # Step 5: Check if source index is valid
    if not 0 <= start < num_verts:
        raise ValueError("Starting vertex should be between 0 to number of vertices")

    # Step 6: Generate the cudf.DataFrame result
    #         Current implementation expects int (signed 32-bit) for distance
    df = cudf.DataFrame()
    df['vertex'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    df['distance'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    df['predecessor'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    if (return_sp_counter):
        df['sp_counter'] = cudf.Series(np.zeros(num_verts, dtype=np.double))

    # Step 7: Associate <uintptr_t> to cudf Series
    c_identifier_ptr = df['vertex'].__cuda_array_interface__['data'][0]
    c_distance_ptr = df['distance'].__cuda_array_interface__['data'][0]
    c_predecessor_ptr = df['predecessor'].__cuda_array_interface__['data'][0]
    if return_sp_counter:
        c_sp_counter_ptr = df['sp_counter'].__cuda_array_interface__['data'][0]

    # Step 8: Proceed to BFS
    # FIXME: [int, int, float] or may add an explicit [int, int, int] in graph.cu?
    graph_float = GraphCSRView[int, int, float](<int*> c_offsets_ptr,
                                            <int*> c_indices_ptr,
                                            <float*> NULL,
                                            num_verts,
                                            num_edges)
    graph_float.get_vertex_identifiers(<int*> c_identifier_ptr)
    # Different pathing wether shortest_path_counting is required or not
    c_bfs.bfs[int, int, float](handle_ptr.get()[0],
                               graph_float,
                               <int*> c_distance_ptr,
                               <int*> c_predecessor_ptr,
                               <double*> c_sp_counter_ptr,
                               <int> start,
                               directed)

    return df
