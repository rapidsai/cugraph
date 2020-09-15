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
from libcpp.pair cimport pair

from cugraph.dask.community cimport louvain as c_louvain
from cugraph.structure.graph_primtypes cimport *

import cudf
import numpy as np


def louvain(input_df, local_data, rank, handle, max_level, resolution):
    """
    Call MG Louvain
    """
    # FIXME: This must be imported here to prevent a circular import
    from cugraph.structure import graph_primtypes_wrapper

    cdef size_t handle_size_t = <size_t>handle.getHandle()
    handle_ = <handle_t*>handle_size_t

    final_modularity = None

    # FIXME: much of this code is common to other algo wrappers, consider adding
    #        this to a shared utility as well (extracting pointers from
    #        dataframes, handling local_data, etc.)

    src = input_df['src']
    dst = input_df['dst']
    if "value" in input_df.columns:
        weights = input_df['value']
    else:
        weights = None

    num_verts = local_data['verts'].sum()
    num_edges = local_data['edges'].sum()

    local_offset = local_data['offsets'][rank]
    dst = dst - local_offset
    num_local_verts = local_data['verts'][rank]
    num_local_edges = len(src)

    cdef uintptr_t c_local_verts = local_data['verts'].__array_interface__['data'][0]
    cdef uintptr_t c_local_edges = local_data['edges'].__array_interface__['data'][0]
    cdef uintptr_t c_local_offsets = local_data['offsets'].__array_interface__['data'][0]

    [src, dst] = graph_primtypes_wrapper.datatype_cast([src, dst], [np.int32])
    if weights is not None:
        if weights.dtype in [np.float32, np.double]:
            [weights] = graph_primtypes_wrapper.datatype_cast([weights], [weights.dtype])
        else:
            raise TypeError(f"unsupported type {weights.dtype} for weights")

    _offsets, indices, weights = graph_primtypes_wrapper.coo2csr(dst, src, weights)
    offsets = _offsets[:num_local_verts + 1]
    del _offsets

    # Create the output dataframe
    df = cudf.DataFrame()
    df['vertex'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    df['partition'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))

    cdef uintptr_t c_offsets = offsets.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_indices = indices.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_weights = <uintptr_t>NULL
    if weights is not None:
        c_weights = weights.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_identifier = df['vertex'].__cuda_array_interface__['data'][0]
    cdef uintptr_t c_partition = df['partition'].__cuda_array_interface__['data'][0]

    cdef float final_modularity_float = 1.0
    cdef double final_modularity_double = 1.0
    cdef int num_level = 0

    cdef graph_container_t graph_container

    # FIXME: This dict should not be needed, instead update create_graph_t() to
    #        take weights.dtype directly
    # FIXME: Offsets and indices should also be void*, and have corresponding
    #        dtypes passed to create_graph_t()
    # FIXME: The excessive casting for the enum arg is needed to make cython
    #        understand how to pass the enum value (this is the same pattern
    #        used by cudf). This will not be needed with Cython 3.0
    weightTypeMap = {np.dtype("float32") : <int>weightTypeEnum.floatType,
                     np.dtype("double") : <int>weightTypeEnum.doubleType}

    graph_container = create_graph_t(handle_[0], <int*>c_offsets, <int*>c_indices,
            <void*>c_weights, <weightTypeEnum>(<int>(weightTypeMap[weights.dtype])),
            num_verts, num_local_edges,
            <int*>c_local_verts, <int*>c_local_edges, <int*>c_local_offsets,
            False, True)  # store_transposed, multi_gpu

    if weights.dtype == np.float32:
        final_modularity_float = c_louvain.call_louvain[float](
            handle_[0], graph_container, <int*>c_partition, max_level, resolution)
        final_modularity = final_modularity_float

    else:
        final_modularity_double = c_louvain.call_louvain[double](
            handle_[0], graph_container, <int*> c_partition, max_level, resolution)
        final_modularity = final_modularity_double

    return df, final_modularity
