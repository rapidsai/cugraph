# Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
from cugraph.structure.graph_utilities cimport *
from cugraph.structure import graph_primtypes_wrapper
from libcpp cimport bool
from libc.stdint cimport uintptr_t
import cudf
import numpy as np

def bfs(input_graph, start, depth_limit, direction_optimizing=False):
    """
    Call bfs
    """
    # Step 1: Declare the different varibales
    cdef graph_container_t graph_container

    numberTypeMap = {np.dtype("int32") : <int>numberTypeEnum.int32Type,
                     np.dtype("int64") : <int>numberTypeEnum.int64Type,
                     np.dtype("float32") : <int>numberTypeEnum.floatType,
                     np.dtype("double") : <int>numberTypeEnum.doubleType}

    weight_t = np.dtype("float32")
    [src, dst] = graph_primtypes_wrapper.datatype_cast([input_graph.edgelist.edgelist_df['src'], input_graph.edgelist.edgelist_df['dst']], [np.int32])
    weights = None

    # Pointers for SSSP / BFS
    cdef uintptr_t c_identifier_ptr     = <uintptr_t> NULL # Pointer to the DataFrame 'vertex' Series
    cdef uintptr_t c_distance_ptr       = <uintptr_t> NULL # Pointer to the DataFrame 'distance' Series
    cdef uintptr_t c_predecessor_ptr    = <uintptr_t> NULL # Pointer to the DataFrame 'predecessor' Series
    if depth_limit is None:
        depth_limit = c_bfs.INT_MAX

    # Step 2: Verifiy input_graph has the expected format

    cdef unique_ptr[handle_t] handle_ptr
    handle_ptr.reset(new handle_t())
    handle_ = handle_ptr.get();

    # Step 3: Setup number of vertices and edges
    num_verts = input_graph.number_of_vertices()
    num_edges = input_graph.number_of_edges(directed_edges=True)

    # Step 4: Extract COO
    cdef uintptr_t c_src_vertices = src.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_dst_vertices = dst.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_edge_weights = <uintptr_t>NULL

    # Step 5: Check if source index is valid
    # FIXME: Updates to multi-seed BFS support disabled this check. Re-enable ASAP.
    #if not 0 <= start < num_verts:
    #    raise ValueError("Starting vertex should be between 0 to number of vertices")

    # Step 6: Generate the cudf.DataFrame result
    #         Current implementation expects int (signed 32-bit) for distance
    df = cudf.DataFrame()
    df['vertex'] = cudf.Series(np.arange(num_verts), dtype=np.int32)
    df['distance'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    df['predecessor'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))

    # Step 7: Associate <uintptr_t> to cudf Series
    c_identifier_ptr = df['vertex'].__cuda_array_interface__['data'][0]
    c_distance_ptr = df['distance'].__cuda_array_interface__['data'][0]
    c_predecessor_ptr = df['predecessor'].__cuda_array_interface__['data'][0]
    cdef uintptr_t c_start_ptr = start.__cuda_array_interface__['data'][0]

    is_symmetric = not input_graph.is_directed()

    # Step 8: Proceed to BFS
    populate_graph_container(graph_container,
                             handle_[0],
                             <void*>c_src_vertices, <void*>c_dst_vertices, <void*>c_edge_weights,
                             <void*>NULL,
                             <void*>NULL,
                             0,
                             <numberTypeEnum>(<int>(numberTypeEnum.int32Type)),
                             <numberTypeEnum>(<int>(numberTypeEnum.int32Type)),
                             <numberTypeEnum>(<int>(numberTypeMap[weight_t])),
                             num_edges,
                             num_verts, num_edges,
                             False,
                             is_symmetric,
                             False,
                             False)

    # Different pathing wether shortest_path_counting is required or not
    c_bfs.call_bfs[int, float](handle_ptr.get()[0],
                               graph_container,
                               <int*> c_identifier_ptr,
                               <int*> c_distance_ptr,
                               <int*> c_predecessor_ptr,
                               <int> depth_limit,
                               <int*> c_start_ptr,
                               len(start),
                               direction_optimizing)

    return df
