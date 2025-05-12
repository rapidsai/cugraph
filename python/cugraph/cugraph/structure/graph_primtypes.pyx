# Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

from libc.stdint cimport uintptr_t
from libcpp.utility cimport move

from rmm.pylibrmm.device_buffer cimport DeviceBuffer
import pylibcudf
import cudf


cdef move_device_buffer_to_column(
    unique_ptr[device_buffer] device_buffer_unique_ptr,
    dtype,
    size_t itemsize,
):
    """
    Transfers ownership of device_buffer_unique_ptr to a cuDF buffer which is
    used to construct a cudf column object, which is then returned. If the
    intermediate buffer is empty, the device_buffer_unique_ptr is still
    transfered but None is returned.
    """
    cdef size_t buff_size = device_buffer_unique_ptr.get().size()
    cdef DeviceBuffer buff = DeviceBuffer.c_from_unique_ptr(move(device_buffer_unique_ptr))
    cdef size_t col_size = buff_size // itemsize
    result_column = pylibcudf.Column.from_rmm_buffer(
        buff,
        dtype,
        col_size,
        [],
    )
    if buff_size != 0:
        return result_column
    return None


cdef move_device_buffer_to_series(
    unique_ptr[device_buffer] device_buffer_unique_ptr,
    dtype,
    size_t itemsize,
    series_name
):
    """
    Transfers ownership of device_buffer_unique_ptr to a cuDF buffer which is
    used to construct a cudf.Series object with name series_name, which is then
    returned. If the intermediate buffer is empty, the device_buffer_unique_ptr
    is still transfered but None is returned.
    """
    column = move_device_buffer_to_column(
        move(device_buffer_unique_ptr),
        dtype,
        itemsize,
    )
    if column is not None:
        return cudf.Series.from_pylibcudf(column, metadata={"name": series_name})
    return None


cdef coo_to_df(GraphCOOPtrType graph):
    # FIXME: this function assumes columns named "src" and "dst" and can only
    # be used for SG graphs due to that assumption.
    contents = move(graph.get()[0].release())
    src = move_device_buffer_to_series(
        move(contents.src_indices),
        pylibcudf.DataType(pylibcudf.TypeId.INT32),
        4,
        None,
    )
    dst = move_device_buffer_to_series(
        move(contents.dst_indices),
        pylibcudf.DataType(pylibcudf.TypeId.INT32),
        4,
        None,
    )

    if GraphCOOPtrType is GraphCOOPtrFloat:
        weight_type = pylibcudf.DataType(pylibcudf.TypeId.FLOAT32)
        itemsize = 4
    elif GraphCOOPtrType is GraphCOOPtrDouble:
        weight_type = pylibcudf.DataType(pylibcudf.TypeId.FLOAT64)
        itemsize = 8
    else:
        raise TypeError("Invalid GraphCOOPtrType")

    wgt = move_device_buffer_to_series(
        move(contents.edge_data),
        weight_type,
        itemsize,
        None,
    )

    df = cudf.DataFrame()
    df['src'] = src
    df['dst'] = dst
    if wgt is not None:
        df['weight'] = wgt

    return df


cdef csr_to_series(GraphCSRPtrType graph):
    contents = move(graph.get()[0].release())
    csr_offsets = move_device_buffer_to_series(
        move(contents.offsets),
        pylibcudf.DataType(pylibcudf.TypeId.INT32),
        4,
        "csr_offsets"
    )
    csr_indices = move_device_buffer_to_series(
        move(contents.indices),
        pylibcudf.DataType(pylibcudf.TypeId.INT32),
        4,
        "csr_indices"
    )

    if GraphCSRPtrType is GraphCSRPtrFloat:
        weight_type = pylibcudf.DataType(pylibcudf.TypeId.FLOAT32)
        itemsize = 4
    elif GraphCSRPtrType is GraphCSRPtrDouble:
        weight_type = pylibcudf.DataType(pylibcudf.TypeId.FLOAT64)
        itemsize = 8
    else:
        raise TypeError("Invalid GraphCSRPtrType")

    csr_weights = move_device_buffer_to_series(
        move(contents.edge_data),
        weight_type,
        itemsize,
        "csr_weights"
    )

    return (csr_offsets, csr_indices, csr_weights)


cdef GraphCOOViewFloat get_coo_float_graph_view(input_graph, bool weighted=True):
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

    return GraphCOOViewFloat(<int*>c_src, <int*>c_dst, <float*>c_weights, num_verts, num_edges)


cdef GraphCOOViewDouble get_coo_double_graph_view(input_graph, bool weighted=True):
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

    return GraphCOOViewDouble(<int*>c_src, <int*>c_dst, <double*>c_weights, num_verts, num_edges)
