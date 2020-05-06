# Copyright (c) 2020, NVIDIA CORPORATION.
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

from rmm._lib.device_buffer cimport DeviceBuffer
from cudf.core.buffer import Buffer
import cudf
import numpy as np
from libc.stdint cimport uintptr_t


cdef coo_to_df(GraphCOOPtrType graph):
    contents = move(graph.get()[0].release())
    src = DeviceBuffer.c_from_unique_ptr(move(contents.src_indices))
    dst = DeviceBuffer.c_from_unique_ptr(move(contents.dst_indices))
    wgt = DeviceBuffer.c_from_unique_ptr(move(contents.edge_data))
    src = Buffer(src)
    dst = Buffer(dst)
    wgt = Buffer(wgt)

    src = cudf.Series(data=src, dtype="int32")
    dst = cudf.Series(data=dst, dtype="int32")

    df = cudf.DataFrame()
    df['src'] = src
    df['dst'] = dst
    if wgt.nbytes != 0:
        if GraphCOOPtrType is GraphCOOPtrFloat:
            wgt = cudf.Series(data=wgt, dtype="float32")
        elif GraphCOOPtrType is GraphCOOPtrDouble:
            wgt = cudf.Series(data=wgt, dtype="float64")
        df['weight'] = wgt
    return df


cdef csr_to_series(GraphCSRPtrType graph):
    contents = move(graph.get()[0].release())
    offsets = DeviceBuffer.c_from_unique_ptr(move(contents.offsets))
    indices = DeviceBuffer.c_from_unique_ptr(move(contents.indices))
    weights = DeviceBuffer.c_from_unique_ptr(move(contents.edge_data))
    offsets = Buffer(offsets)
    indices = Buffer(indices)
    weights = Buffer(weights)

    csr_offsets = cudf.Series(data=offsets, dtype="int32")
    csr_indices = cudf.Series(data=indices, dtype="int32")

    csr_weights = None
    if weights.nbytes != 0:
        if GraphCSRPtrType is GraphCSRPtrFloat:
            csr_weights = cudf.Series(data=weights, dtype="float32")
        elif GraphCSRPtrType is GraphCSRPtrDouble:
            csr_weights = cudf.Series(data=weights, dtype="float64")
    return (csr_offsets, csr_indices, csr_weights)


cdef GraphCOOViewType get_graph_view(input_graph, GraphCOOViewType* dummy=NULL):
    if not input_graph.edgelist:
        input_graph.view_edge_list()

    weights = None

    cdef uintptr_t c_src = input_graph.edgelist.edgelist_df['src'].__cuda_array_interface__['data'][0]
    cdef uintptr_t c_dst = input_graph.edgelist.edgelist_df['dst'].__cuda_array_interface__['data'][0]
    cdef uintptr_t c_weights = <uintptr_t>NULL

    if input_graph.edgelist.weights:
        c_weights = input_graph.edgelist.edgelist_df['weights'].__cuda_array_interface__['data'][0]

    num_verts = input_graph.number_of_vertices()
    num_edges = len(input_graph.edgelist.edgelist_df)
    cdef GraphCOOViewType in_graph
    if GraphCOOViewType is GraphCOOViewFloat:
        in_graph = GraphCOOViewFloat(<int*>c_src, <int*>c_dst, <float*>c_weights, num_verts, num_edges)
    elif GraphCOOViewType is GraphCOOViewDouble:
        in_graph = GraphCOOViewDouble(<int*>c_src, <int*>c_dst, <double*>c_weights, num_verts, num_edges)
    return in_graph
