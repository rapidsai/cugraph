#
# Copyright (c) 2020, NVIDIA CORPORATION.
#
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
#

from cugraph.structure.utils_wrapper import *
from cugraph.mg.traversal cimport mg_bfs as c_bfs
import cudf
from cugraph.structure.graph_new cimport *
import cugraph.structure.graph_new_wrapper as graph_new_wrapper
from libc.stdint cimport uintptr_t

def mg_bfs(input_df, local_data, rank, handle, start, return_distances=False):
    """
    Call pagerank
    """

    cdef size_t handle_size_t = <size_t>handle.getHandle()
    handle_ = <c_bfs.handle_t*>handle_size_t

    # Local COO information
    src = input_df['src']
    dst = input_df['dst']
    num_verts = local_data['verts'].sum()
    num_edges = local_data['edges'].sum()
    local_offset = local_data['offsets'][rank]
    src = src - local_offset
    num_local_verts = local_data['verts'][rank]
    num_local_edges = len(src)

    print("num_verts ", num_verts)

    # Convert to local CSR
    [src, dst] = graph_new_wrapper.datatype_cast([src, dst], [np.int32])
    _offsets, indices, weights = coo2csr(src, dst, None)
    offsets = _offsets[:num_local_verts + 1]
    del _offsets

    # Pointers required for CSR Graph
    cdef uintptr_t c_offsets_ptr = offsets.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_indices_ptr = indices.__cuda_array_interface__['data'][0]

    # Generate the cudf.DataFrame result
    df = cudf.DataFrame()
    df['vertex'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    df['predecessor'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    if (return_distances):
        df['distance'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))

    # Associate <uintptr_t> to cudf Series
    cdef uintptr_t c_identifier_ptr  = df['vertex'].__cuda_array_interface__['data'][0];
    cdef uintptr_t c_distance_ptr    = <uintptr_t> NULL # Pointer to the DataFrame 'distance' Series
    cdef uintptr_t c_predecessor_ptr = df['predecessor'].__cuda_array_interface__['data'][0];
    if (return_distances):
        c_distance_ptr = df['distance'].__cuda_array_interface__['data'][0]

    # Extract local data
    cdef uintptr_t c_local_verts = local_data['verts'].__array_interface__['data'][0]
    cdef uintptr_t c_local_edges = local_data['edges'].__array_interface__['data'][0]
    cdef uintptr_t c_local_offsets = local_data['offsets'].__array_interface__['data'][0]

    # BFS
    cdef GraphCSRView[int,int,float] graph
    graph= GraphCSRView[int, int, float](<int*> c_offsets_ptr,
                                         <int*> c_indices_ptr,
                                         <float*> NULL,
                                         num_verts,
                                         num_local_edges)
    graph.set_local_data(<int*>c_local_verts, <int*>c_local_edges, <int*>c_local_offsets)
    graph.set_handle(handle_)
    graph.get_vertex_identifiers(<int*>c_identifier_ptr)

    cdef bool direction = <bool> 1
    # MG BFS path assumes directed is true
    c_bfs.bfs[int, int, float](handle_[0],
                               graph,
                               <int*> c_distance_ptr,
                               <int*> c_predecessor_ptr,
                               <double*> NULL,
                               <int> start,
                               direction)

    return df
