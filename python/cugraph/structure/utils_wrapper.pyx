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


# DBG
def _internal_replication_edgelist(input_data, session_id):
    from cugraph.raft.dask.common.comms import worker_state
    # start = time.perf_counter() DBG
    result = None
    # 1. Get session information
    session_state = worker_state(session_id)
    number_of_workers = session_state["nworkers"]
    worker_idx = session_state["wid"]

    # 2. Get handle
    handle = session_state['handle']

    # 3. Determine worker type
    #is_organizer = is_worker_organizer(worker_idx)

    #(placeholder, number_of_vertices, number_of_edges) = input_data
    _data, local_data, number_of_vertices, number_of_edges = input_data
    data = _data[0]
    print(data)

    src_identifiers = cudf.Series(np.zeros(number_of_edges), dtype=np.int32)
    dst_identifiers = cudf.Series(np.zeros(number_of_edges), dtype=np.int32)
    result = cudf.DataFrame(data={"src": src_identifiers,
                                  "dst": dst_identifiers})
    return result



cdef comms_bcast(uintptr_t handle,
                 uintptr_t value_ptr,
                 size_t count,
                 result_dtype):
    if result_dtype ==  np.int32:
        c_utils.comms_bcast((<handle_t*> handle)[0], <int*> value_ptr, count)
    elif result_dtype == np.float32:
        c_utils.comms_bcast((<handle_t*> handle)[0], <float*> value_ptr, count)
    elif result_dtype == np.float64:
        c_utils.comms_bcast((<handle_t*> handle)[0], <double*> value_ptr, count)
    else:
        raise TypeError("Unsupported broadcast type")