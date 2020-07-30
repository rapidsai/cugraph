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

from libc.stdint cimport uintptr_t
from cugraph.structure cimport utils as c_utils
from cugraph.structure.graph_new cimport *
from libc.stdint cimport uintptr_t

import cudf
import rmm
import numpy as np
from rmm._lib.device_buffer cimport DeviceBuffer
from cudf.core.buffer import Buffer
from cugraph.raft.dask.common.comms import worker_state


def weight_type(weights):
    weights_type = None
    if weights is not None:
        weights_type = weights.dtype
    return weights_type


def create_csr_float(source_col, dest_col, weights):
    num_verts = 0
    num_edges = len(source_col)

    cdef uintptr_t c_src = source_col.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_dst = dest_col.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_weights = <uintptr_t> NULL

    if weights is not None:
        c_weights = weights.__cuda_array_interface__['data'][0]

    cdef GraphCOOView[int,int,float] in_graph
    in_graph = GraphCOOView[int,int,float](<int*>c_src, <int*>c_dst, <float*>c_weights, num_verts, num_edges)
    return csr_to_series(move(c_utils.coo_to_csr[int,int,float](in_graph)))


def create_csr_double(source_col, dest_col, weights):
    num_verts = 0
    num_edges = len(source_col)

    cdef uintptr_t c_src = source_col.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_dst = dest_col.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_weights = <uintptr_t> NULL

    if weights is not None:
        c_weights = weights.__cuda_array_interface__['data'][0]

    cdef GraphCOOView[int,int,double] in_graph
    in_graph = GraphCOOView[int,int,double](<int*>c_src, <int*>c_dst, <double*>c_weights, num_verts, num_edges)
    return csr_to_series(move(c_utils.coo_to_csr[int,int,double](in_graph)))


def coo2csr(source_col, dest_col, weights=None):
    if len(source_col) != len(dest_col):
        raise Exception("source_col and dest_col should have the same number of elements")

    if source_col.dtype != dest_col.dtype:
        raise Exception("source_col and dest_col should be the same type")

    if source_col.dtype != np.int32:
        raise Exception("source_col and dest_col must be type np.int32")

    if weight_type(weights) == np.float64:
        return create_csr_double(source_col, dest_col, weights)
    else:
        return create_csr_float(source_col, dest_col, weights)


# FIXME: Does not support graph weights
# FIXME: Assumes that data is np.int32
def replicate_edgelist(input_data, session_id):
    cdef uintptr_t c_handle = <uintptr_t> NULL
    cdef uintptr_t c_src = <uintptr_t> NULL
    cdef uintptr_t c_dst = <uintptr_t> NULL

    result = None
    # 1. Get session information
    session_state = worker_state(session_id)
    number_of_workers = session_state["nworkers"]
    worker_idx = session_state["wid"]

    # 2. Get handle
    handle = session_state['handle']
    c_handle = <uintptr_t>handle.getHandle()

    #(placeholder, number_of_vertices, number_of_edges) = input_data
    _data, edgelist_size = input_data
    data = _data[0]
    has_data = type(data) is cudf.DataFrame
    src_identifiers = None
    dst_identifiers = None
    if has_data:
        src_identifiers = data['src']
        dst_identifiers = data['dst']
    else:
        src_identifiers = cudf.Series(np.zeros(edgelist_size), dtype=np.int32)
        dst_identifiers = cudf.Series(np.zeros(edgelist_size), dtype=np.int32)

    c_src =  src_identifiers.__cuda_array_interface__['data'][0]
    c_dst =  dst_identifiers.__cuda_array_interface__['data'][0]

    comms_bcast(c_handle, c_src, len(src_identifiers), src_identifiers.dtype)
    comms_bcast(c_handle, c_dst, len(dst_identifiers), dst_identifiers.dtype)

    if has_data:
        result = data
    else:
        result = cudf.DataFrame(data={"src": src_identifiers,
                                      "dst": dst_identifiers})
    return result


def replicate_cudf_series(input_data, session_id):
    cdef uintptr_t c_handle = <uintptr_t> NULL
    cdef uintptr_t c_result = <uintptr_t> NULL

    result = None

    session_state = worker_state(session_id)
    handle = session_state['handle']
    c_handle = <uintptr_t>handle.getHandle()

    (_data, size, dtype) = input_data

    data = _data[0]
    has_data = type(data) is cudf.Series
    if has_data:
        result = data
    else:
        result = cudf.Series(np.zeros(size), dtype=dtype)

    c_result = result.__cuda_array_interface__['data'][0]

    comms_bcast(c_handle, c_result, size, dtype)

    return result


cdef comms_bcast(uintptr_t handle,
                 uintptr_t value_ptr,
                 size_t count,
                 dtype):
    if dtype ==  np.int32:
        c_utils.comms_bcast((<handle_t*> handle)[0], <int*> value_ptr, count)
    elif dtype == np.float32:
        c_utils.comms_bcast((<handle_t*> handle)[0], <float*> value_ptr, count)
    elif dtype == np.float64:
        c_utils.comms_bcast((<handle_t*> handle)[0], <double*> value_ptr, count)
    else:
        raise TypeError("Unsupported broadcast type")