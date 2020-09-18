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

cimport cugraph.traversal.sssp as c_sssp
cimport cugraph.traversal.bfs as c_bfs
from cugraph.structure.graph_primtypes cimport *
from cugraph.structure import graph_primtypes_wrapper

from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.float cimport FLT_MAX_EXP

import cudf
import rmm
import numpy as np

def sssp(input_graph, source):
    """
    Call sssp
    """
    # Step 1: Declare the different variables
    cdef GraphCSRView[int, int, float]  graph_float     # For weighted float graph (SSSP) and Unweighted (BFS)
    cdef GraphCSRView[int, int, double] graph_double    # For weighted double graph (SSSP)

    # Pointers required for CSR Graph
    cdef uintptr_t c_offsets_ptr        = <uintptr_t> NULL # Pointer to the CSR offsets
    cdef uintptr_t c_indices_ptr        = <uintptr_t> NULL # Pointer to the CSR indices
    cdef uintptr_t c_weights_ptr        = <uintptr_t> NULL # Pointer to the CSR weights

    # Pointers for SSSP / BFS
    cdef uintptr_t c_identifier_ptr     = <uintptr_t> NULL # Pointer to the DataFrame 'vertex' Series
    cdef uintptr_t c_distance_ptr       = <uintptr_t> NULL # Pointer to the DataFrame 'distance' Series
    cdef uintptr_t c_predecessor_ptr    = <uintptr_t> NULL # Pointer to the DataFrame 'predecessor' Series

    cdef unique_ptr[handle_t] handle_ptr
    handle_ptr.reset(new handle_t())

    # Step 2: Verify that input_graph has the expected format
    #         the SSSP implementation expects CSR format
    if not input_graph.adjlist:
        input_graph.view_adj_list()

    # Step 3: Extract CSR offsets, indices and indices
    #         - offsets: int (signed, 32-bit)
    #         - indices: int (signed, 32-bit)
    #         - weights: float / double
    #         Extract data_type from weights (not None: float / double, None: signed int 32-bit)
    [offsets, indices] = graph_primtypes_wrapper.datatype_cast([input_graph.adjlist.offsets, input_graph.adjlist.indices], [np.int32])
    [weights] = graph_primtypes_wrapper.datatype_cast([input_graph.adjlist.weights], [np.float32, np.float64])
    c_offsets_ptr = offsets.__cuda_array_interface__['data'][0]
    c_indices_ptr = indices.__cuda_array_interface__['data'][0]

    data_type = np.int32
    if weights is not None:
        data_type = weights.dtype
        c_weights_ptr = weights.__cuda_array_interface__['data'][0]

    # Step 4: Setup number of vertices and number of edges
    num_verts = input_graph.number_of_vertices()
    num_edges = input_graph.number_of_edges(directed_edges=True)

    # Step 5: Check if source index is valid
    if not 0 <= source < num_verts:
        raise ValueError("Starting vertex should be between 0 to number of vertices")

    # Step 6: Generation of the result cudf.DataFrame
    #         Distances depends on data_type (c.f. Step 3)
    df = cudf.DataFrame()

    df['vertex'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    df['distance'] = cudf.Series(np.zeros(num_verts, dtype=data_type))
    df['predecessor'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))

    # Step 7: Associate <uintptr_t> to cudf Series
    c_identifier_ptr = df['vertex'].__cuda_array_interface__['data'][0]
    c_distance_ptr = df['distance'].__cuda_array_interface__['data'][0]
    c_predecessor_ptr = df['predecessor'].__cuda_array_interface__['data'][0]

    # Step 8: Dispatch to SSSP / BFS Based on weights
    #         - weights is not None: SSSP float or SSSP double
    #         - weights is None: BFS
    if weights is not None:
        if data_type == np.float32:
            graph_float = GraphCSRView[int, int, float](<int*> c_offsets_ptr,
                                                     <int*> c_indices_ptr,
                                                     <float*> c_weights_ptr,
                                                     num_verts,
                                                     num_edges)
            graph_float.get_vertex_identifiers(<int*> c_identifier_ptr)
            c_sssp.sssp[int, int, float](graph_float,
                                         <float*> c_distance_ptr,
                                         <int*> c_predecessor_ptr,
                                         <int> source)
        elif data_type == np.float64:
            graph_double = GraphCSRView[int, int, double](<int*> c_offsets_ptr,
                                                      <int*> c_indices_ptr,
                                                      <double*> c_weights_ptr,
                                                      num_verts,
                                                      num_edges)
            graph_double.get_vertex_identifiers(<int*> c_identifier_ptr)
            c_sssp.sssp[int, int, double](graph_double,
                                          <double*> c_distance_ptr,
                                          <int*> c_predecessor_ptr,
                                          <int> source)
        else: # This case should not happen
            raise NotImplementedError
    else:
        # FIXME: Something might be done here considering WT = float
        graph_float = GraphCSRView[int, int, float](<int*> c_offsets_ptr,
                                                <int*> c_indices_ptr,
                                                <float*> NULL,
                                                num_verts,
                                                num_edges)
        graph_float.get_vertex_identifiers(<int*> c_identifier_ptr)
        c_bfs.bfs[int, int, float](handle_ptr.get()[0],
                                   graph_float,
                                   <int*> c_distance_ptr,
                                   <int*> c_predecessor_ptr,
                                   <double*> NULL,
                                   <int> source)

    return df
