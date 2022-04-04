#
# Copyright (c) 2020-2022, NVIDIA CORPORATION.
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
from cugraph.dask.traversal cimport mg_sssp as c_sssp
import cudf
from cugraph.structure.graph_utilities cimport *
import cugraph.structure.graph_primtypes_wrapper as graph_primtypes_wrapper
from libc.stdint cimport uintptr_t

def mg_sssp(input_df,
            src_col_name,
            dst_col_name,
            num_global_verts,
            num_global_edges,
            vertex_partition_offsets,
            rank,
            handle,
            segment_offsets,
            start):
    """
    Call sssp
    """

    cdef size_t handle_size_t = <size_t>handle.getHandle()
    handle_ = <c_sssp.handle_t*>handle_size_t

    # Local COO information
    src = input_df[src_col_name]
    dst = input_df[dst_col_name]
    vertex_t = src.dtype
    if num_global_edges > (2**31 - 1):
        edge_t = np.dtype("int64")
    else:
        edge_t = vertex_t
    if "value" in input_df.columns:
        weights = input_df['value']
        weight_t = weights.dtype
        is_weighted = True
    else:
        weights = None
        weight_t = np.dtype("float32")
        is_weighted = False

    # FIXME: Offsets and indices are currently hardcoded to int, but this may
    #        not be acceptable in the future.
    numberTypeMap = {np.dtype("int32") : <int>numberTypeEnum.int32Type,
                     np.dtype("int64") : <int>numberTypeEnum.int64Type,
                     np.dtype("float32") : <int>numberTypeEnum.floatType,
                     np.dtype("double") : <int>numberTypeEnum.doubleType}

    # FIXME: needs to be edge_t type not int
    cdef int num_local_edges = len(src)

    cdef uintptr_t c_src_vertices = src.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_dst_vertices = dst.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_edge_weights = <uintptr_t>NULL
    if weights is not None:
        c_edge_weights = weights.__cuda_array_interface__['data'][0]

    # FIXME: data is on device, move to host (to_pandas()), convert to np array and access pointer to pass to C
    vertex_partition_offsets_host = vertex_partition_offsets.values_host
    cdef uintptr_t c_vertex_partition_offsets = vertex_partition_offsets_host.__array_interface__['data'][0]

    cdef vector[int] v_segment_offsets_32
    cdef vector[long] v_segment_offsets_64
    cdef uintptr_t c_segment_offsets
    if (vertex_t == np.dtype("int32")):
        v_segment_offsets_32 = segment_offsets
        c_segment_offsets = <uintptr_t>v_segment_offsets_32.data()
    else:
        v_segment_offsets_64 = segment_offsets
        c_segment_offsets = <uintptr_t>v_segment_offsets_64.data()

    cdef graph_container_t graph_container

    populate_graph_container(graph_container,
                             handle_[0],
                             <void*>c_src_vertices, <void*>c_dst_vertices, <void*>c_edge_weights,
                             <void*>c_vertex_partition_offsets,
                             <void*>c_segment_offsets,
                             len(segment_offsets) - 1,
                             <numberTypeEnum>(<int>(numberTypeMap[vertex_t])),
                             <numberTypeEnum>(<int>(numberTypeMap[edge_t])),
                             <numberTypeEnum>(<int>(numberTypeMap[weight_t])),
                             num_local_edges,
                             num_global_verts, num_global_edges,
                             is_weighted,
                             False,
                             False, True)

    # Generate the cudf.DataFrame result
    df = cudf.DataFrame()
    df['vertex'] = cudf.Series(np.arange(vertex_partition_offsets.iloc[rank], vertex_partition_offsets.iloc[rank+1]), dtype=vertex_t)
    df['predecessor'] = cudf.Series(np.zeros(len(df['vertex']), dtype=vertex_t))
    df['distance'] = cudf.Series(np.zeros(len(df['vertex']), dtype=weight_t))

    # Associate <uintptr_t> to cudf Series
    cdef uintptr_t c_predecessor_ptr = df['predecessor'].__cuda_array_interface__['data'][0]
    cdef uintptr_t c_distance_ptr = df['distance'].__cuda_array_interface__['data'][0]

    # MG BFS path assumes directed is true
    if vertex_t == np.int32:
        if weight_t == np.float32:
            c_sssp.call_sssp[int, float](handle_[0],
                                         graph_container,
                                         <int*> NULL,
                                         <float*> c_distance_ptr,
                                         <int*> c_predecessor_ptr,
                                         <int> start)
        elif weight_t == np.float64:
            c_sssp.call_sssp[int, double](handle_[0],
                                          graph_container,
                                          <int*> NULL,
                                          <double*> c_distance_ptr,
                                          <int*> c_predecessor_ptr,
                                          <int> start)
    else:
        if weight_t == np.float32:
            c_sssp.call_sssp[long, float](handle_[0],
                                         graph_container,
                                         <long*> NULL,
                                         <float*> c_distance_ptr,
                                         <long*> c_predecessor_ptr,
                                         <long> start)
        elif weight_t == np.float64:
            c_sssp.call_sssp[long, double](handle_[0],
                                          graph_container,
                                          <long*> NULL,
                                          <double*> c_distance_ptr,
                                          <long*> c_predecessor_ptr,
                                          <long> start)
    return df
