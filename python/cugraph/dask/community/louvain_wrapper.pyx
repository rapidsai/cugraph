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
    handle_ = <c_louvain.handle_t*>handle_size_t

    final_modularity = None

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
        if weights.dtype == np.float32:
            [weights] = graph_primtypes_wrapper.datatype_cast([weights], [np.float32])
        elif weights.dtype == np.double:
            [weights] = graph_primtypes_wrapper.datatype_cast([weights], [np.double])
        else:
            raise TypeError(f"unsupported type {weights.dtype} for weights")

        _offsets, indices, weights = graph_primtypes_wrapper.coo2csr(dst, src, weights)
    else:
        _offsets, indices, weights = graph_primtypes_wrapper.coo2csr(dst, src, None)

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

    cdef GraphCSRView[int,int,float] graph_float
    cdef GraphCSRView[int,int,double] graph_double

    cdef float final_modularity_float = 1.0
    cdef double final_modularity_double = 1.0
    cdef int num_level = 0

    if weights.dtype == np.float32:
        graph_float = GraphCSRView[int,int,float](<int*>c_offsets, <int*>c_indices,
                                                  <float*>c_weights, num_verts, num_local_edges)
        graph_float.set_local_data(<int*>c_local_verts, <int*>c_local_edges, <int*>c_local_offsets)
        graph_float.set_handle(handle_)
        num_level, final_modularity_float = \
            c_louvain.louvain[int,int,float](handle_[0], graph_float, <int*> c_partition, max_level, resolution)
        graph_float.get_vertex_identifiers(<int*>c_identifier)

        final_modularity = final_modularity_float

    else:
        graph_double = GraphCSRView[int,int,double](<int*>c_offsets, <int*>c_indices,
                                                    <double*>c_weights, num_verts, num_edges)
        graph_double.set_local_data(<int*>c_local_verts, <int*>c_local_edges, <int*>c_local_offsets)
        graph_double.set_handle(handle_)
        num_level, final_modularity_double = \
            c_louvain.louvain[int,int,double](handle_[0], graph_double, <int*> c_partition, max_level, resolution)
        graph_double.get_vertex_identifiers(<int*>c_identifier)

        final_modularity = final_modularity_double

    return df, final_modularity
