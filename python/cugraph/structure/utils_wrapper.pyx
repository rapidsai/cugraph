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
import dask.distributed as dd
from cugraph.dask.common.input_utils import get_mg_batch_data
import dask_cudf
import cugraph.comms.comms as Comms
import cugraph.dask.common.mg_utils as mg_utils


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

    if len(source_col) == 0:
        return cudf.Series(np.zeros(1, dtype=np.int32)), cudf.Series(np.zeros(1, dtype=np.int32)), weights

    if weight_type(weights) == np.float64:
        return create_csr_double(source_col, dest_col, weights)
    else:
        return create_csr_float(source_col, dest_col, weights)


def  replicate_cudf_dataframe(cudf_dataframe, client=None, comms=None):
    if type(cudf_dataframe) is not cudf.DataFrame:
        raise TypeError("Expected a cudf.Series to replicate")
    client = mg_utils.get_client() if client is None else client
    comms = Comms.get_comms() if comms is None else comms
    dask_cudf_df = dask_cudf.from_cudf(cudf_dataframe, npartitions=1)
    df_length = len(dask_cudf_df)

    _df_data =  get_mg_batch_data(dask_cudf_df)
    df_data =  mg_utils.prepare_worker_to_parts(_df_data, client)

    workers_to_futures = {worker: client.submit(_replicate_cudf_dataframe,
                          (data, cudf_dataframe.columns.values, cudf_dataframe.dtypes, df_length),
                          comms.sessionId,
                          workers=[worker]) for
                          (worker, data) in
                          df_data.worker_to_parts.items()}
    dd.wait(workers_to_futures)
    return workers_to_futures


def _replicate_cudf_dataframe(input_data, session_id):
    cdef uintptr_t c_handle = <uintptr_t> NULL
    cdef uintptr_t c_series = <uintptr_t> NULL

    result = None
    # 1. Get session information
    session_state = worker_state(session_id)

    # 2. Get handle
    handle = session_state['handle']
    c_handle = <uintptr_t>handle.getHandle()

    _data, columns, dtypes, df_length = input_data
    data = _data[0]
    has_data = type(data) is cudf.DataFrame

    series = None
    df_data = {}
    for idx, column in enumerate(columns):
        if has_data:
            series = data[column]
        else:
            dtype = dtypes[idx]
            series = cudf.Series(np.zeros(df_length), dtype=dtype)
            df_data[column] = series
        c_series =  series.__cuda_array_interface__['data'][0]
        comms_bcast(c_handle, c_series, df_length, series.dtype)

    if has_data:
        result = data
    else:
        result = cudf.DataFrame(data=df_data)
    return result


def  replicate_cudf_series(cudf_series, client=None, comms=None):
    if type(cudf_series) is not cudf.Series:
        raise TypeError("Expected a cudf.Series to replicate")
    client = mg_utils.get_client() if client is None else client
    comms = Comms.get_comms() if comms is None else comms
    dask_cudf_series =  dask_cudf.from_cudf(cudf_series,
                                            npartitions=1)
    series_length = len(dask_cudf_series)
    _series_data = get_mg_batch_data(dask_cudf_series)
    series_data = mg_utils.prepare_worker_to_parts(_series_data)

    dtype = cudf_series.dtype
    workers_to_futures = {worker:
                          client.submit(_replicate_cudf_series,
                                        (data, series_length, dtype),
                                        comms.sessionId,
                                         workers=[worker]) for
                           (worker, data) in
                           series_data.worker_to_parts.items()}
    dd.wait(workers_to_futures)
    return workers_to_futures


def _replicate_cudf_series(input_data, session_id):
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