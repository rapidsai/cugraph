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

from cugraph.dask.community cimport louvain as c_louvain
from cugraph.structure.graph_primtypes cimport *

import cudf
import numpy as np


def louvain(input_df,
            num_verts,
            num_edges,
            partition_row_size,
            partition_col_size,
            vertex_partition_offsets,
            rank,
            handle,
            max_level,
            resolution):
    """
    Call MG Louvain
    """
    # FIXME: This must be imported here to prevent a circular import
    from cugraph.structure import graph_primtypes_wrapper

    cdef size_t handle_size_t = <size_t>handle.getHandle()
    handle_ = <handle_t*>handle_size_t

    final_modularity = None

    # FIXME: much of this code is common to other algo wrappers, consider adding
    #        this to a shared utility as well

    src = input_df['src']
    dst = input_df['dst']
    if "value" in input_df.columns:
        weights = input_df['value']
    else:
        weights = None

    # COO
    cdef uintptr_t c_src_vertices = src.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_dst_vertices = dst.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_edge_weights = <uintptr_t>NULL
    if weights is not None:
        c_edge_weights = weights.__cuda_array_interface__['data'][0]

    # FIXME: data is on device, move to host (to_pandas()), convert to np array and access pointer to pass to C
    #cdef uintptr_t c_vertex_partition_offsets = vertex_partition_offsets.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_vertex_partition_offsets = vertex_partition_offsets.values_host.__array_interface__['data'][0]

    # FIXME: Offsets and indices are currently hardcoded to int, but this may
    #        not be acceptable in the future.
    weightTypeMap = {np.dtype("float32") : <int>numberTypeEnum.floatType,
                     np.dtype("double") : <int>numberTypeEnum.doubleType}

    cdef graph_container_t graph_container

    # FIXME: The excessive casting for the enum arg is needed to make cython
    #        understand how to pass the enum value (this is the same pattern
    #        used by cudf). This will not be needed with Cython 3.0
    populate_graph_container(graph_container,
                             handle_[0],
                             <void*>c_src_vertices, <void*>c_dst_vertices, <void*>c_edge_weights,
                             <void*>c_vertex_partition_offsets,
                             <numberTypeEnum>(<int>(numberTypeEnum.intType)),
                             <numberTypeEnum>(<int>(numberTypeEnum.intType)),
                             <numberTypeEnum>(<int>(weightTypeMap[weights.dtype])),
                             num_verts, num_edges,
                             partition_row_size, partition_col_size,
                             False, True)  # store_transposed, multi_gpu

    # Create the output dataframe
    df = cudf.DataFrame()
    df['vertex'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    df['partition'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))

    cdef uintptr_t c_identifiers = df['vertex'].__cuda_array_interface__['data'][0]
    cdef uintptr_t c_partition = df['partition'].__cuda_array_interface__['data'][0]

    if weights.dtype == np.float32:
        num_level, final_modularity_float = c_louvain.call_louvain[float](
            handle_[0], graph_container, <void*>c_identifiers, <void*>c_partition, max_level, resolution)
        final_modularity = final_modularity_float

    else:
        num_level, final_modularity_double = c_louvain.call_louvain[double](
            handle_[0], graph_container, <void*>c_identifiers, <void*>c_partition, max_level, resolution)
        final_modularity = final_modularity_double

    return df, final_modularity
