# Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

import numpy as np
from libc.stdint cimport uintptr_t
from libcpp.utility cimport move

from rmm._lib.device_buffer cimport DeviceBuffer
from cudf.core.buffer import as_buffer
import cudf


cdef move_device_buffer_to_column(
    unique_ptr[device_buffer] device_buffer_unique_ptr, dtype):
    """
    Transfers ownership of device_buffer_unique_ptr to a cuDF buffer which is
    used to construct a cudf column object, which is then returned. If the
    intermediate buffer is empty, the device_buffer_unique_ptr is still
    transfered but None is returned.
    """
    buff = DeviceBuffer.c_from_unique_ptr(move(device_buffer_unique_ptr))
    buff = as_buffer(buff)
    if buff.nbytes != 0:
        column = cudf.core.column.build_column(buff, dtype=dtype)
        return column
    return None


cdef move_device_buffer_to_series(
    unique_ptr[device_buffer] device_buffer_unique_ptr, dtype, series_name):
    """
    Transfers ownership of device_buffer_unique_ptr to a cuDF buffer which is
    used to construct a cudf.Series object with name series_name, which is then
    returned. If the intermediate buffer is empty, the device_buffer_unique_ptr
    is still transfered but None is returned.
    """
    column = move_device_buffer_to_column(move(device_buffer_unique_ptr), dtype)
    if column is not None:
        series = cudf.Series._from_data({series_name: column})
        return series
    return None


cdef coo_to_df(GraphCOOPtrType graph):
    # FIXME: this function assumes columns named "src" and "dst" and can only
    # be used for SG graphs due to that assumption.
    contents = move(graph.get()[0].release())
    src = move_device_buffer_to_column(move(contents.src_indices), "int32")
    dst = move_device_buffer_to_column(move(contents.dst_indices), "int32")

    if GraphCOOPtrType is GraphCOOPtrFloat:
        weight_type = "float32"
    elif GraphCOOPtrType is GraphCOOPtrDouble:
        weight_type = "float64"
    else:
        raise TypeError("Invalid GraphCOOPtrType")

    wgt = move_device_buffer_to_column(move(contents.edge_data), weight_type)

    df = cudf.DataFrame()
    df['src'] = src
    df['dst'] = dst
    if wgt is not None:
        df['weight'] = wgt

    return df


cdef csr_to_series(GraphCSRPtrType graph):
    contents = move(graph.get()[0].release())
    csr_offsets = move_device_buffer_to_series(move(contents.offsets),
                                               "int32", "csr_offsets")
    csr_indices = move_device_buffer_to_series(move(contents.indices),
                                               "int32", "csr_indices")

    if GraphCSRPtrType is GraphCSRPtrFloat:
        weight_type = "float32"
    elif GraphCSRPtrType is GraphCSRPtrDouble:
        weight_type = "float64"
    else:
        raise TypeError("Invalid GraphCSRPtrType")

    csr_weights = move_device_buffer_to_series(move(contents.edge_data),
                                               weight_type, "csr_weights")

    return (csr_offsets, csr_indices, csr_weights)


cdef GraphCSRViewType get_csr_graph_view(input_graph, bool weighted=True, GraphCSRViewType* dummy=NULL):
    if not input_graph.adjlist:
        input_graph.view_adj_list()

    cdef uintptr_t c_off = input_graph.adjlist.offsets.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_ind = input_graph.adjlist.indices.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_weights = <uintptr_t>NULL

    if input_graph.adjlist.weights is not None and weighted:
        c_weights = input_graph.adjlist.weights.__cuda_array_interface__['data'][0]

    num_verts = input_graph.number_of_vertices()
    num_edges = input_graph.number_of_edges(directed_edges=True)
    cdef GraphCSRViewType in_graph
    if GraphCSRViewType is GraphCSRViewFloat:
        in_graph = GraphCSRViewFloat(<int*>c_off, <int*>c_ind, <float*>c_weights, num_verts, num_edges)
    elif GraphCSRViewType is GraphCSRViewDouble:
        in_graph = GraphCSRViewDouble(<int*>c_off, <int*>c_ind, <double*>c_weights, num_verts, num_edges)
    return in_graph


cdef GraphCOOViewType get_coo_graph_view(input_graph, bool weighted=True, GraphCOOViewType* dummy=NULL):
    # FIXME: this function assumes columns named "src" and "dst" and can only
    # be used for SG graphs due to that assumption.
    if not input_graph.edgelist:
        input_graph.view_edge_list()

    num_edges = input_graph.number_of_edges(directed_edges=True)
    num_verts = input_graph.number_of_vertices()

    cdef uintptr_t c_src = input_graph.edgelist.edgelist_df['src'].__cuda_array_interface__['data'][0]
    cdef uintptr_t c_dst = input_graph.edgelist.edgelist_df['dst'].__cuda_array_interface__['data'][0]
    cdef uintptr_t c_weights = <uintptr_t>NULL

    # FIXME explicit check for None fails, different behavior than get_csr_graph_view
    if input_graph.edgelist.weights and weighted:
        c_weights = input_graph.edgelist.edgelist_df['weights'].__cuda_array_interface__['data'][0]

    cdef GraphCOOViewType in_graph
    if GraphCOOViewType is GraphCOOViewFloat:
        in_graph = GraphCOOViewFloat(<int*>c_src, <int*>c_dst, <float*>c_weights, num_verts, num_edges)
    elif GraphCOOViewType is GraphCOOViewDouble:
        in_graph = GraphCOOViewDouble(<int*>c_src, <int*>c_dst, <double*>c_weights, num_verts, num_edges)
    return in_graph


cdef GraphViewType get_graph_view(input_graph, bool weighted = True, GraphViewType* dummy=NULL):
    if GraphViewType is GraphCOOViewFloat:
        return get_coo_graph_view[GraphCOOViewFloat](input_graph, weighted, dummy)
    elif GraphViewType is GraphCOOViewDouble:
        return get_coo_graph_view[GraphCOOViewDouble](input_graph, weighted, dummy)
    elif GraphViewType is GraphCSRViewFloat:
        return get_csr_graph_view[GraphCSRViewFloat](input_graph, weighted, dummy)
    elif GraphViewType is GraphCSRViewDouble:
        return get_csr_graph_view[GraphCSRViewDouble](input_graph, weighted, dummy)
