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


def louvain(input_df, local_data, wid, handle, max_level, resolution):
    """
    Call MG Louvain
    """

    cdef size_t handle_size_t = <size_t>handle.getHandle()
    handle_ = <c_louvain.handle_t*>handle_size_t

    # FIXME: view_adj_list() is not supported for a distributed graph but should
    # still be done?
    # if not input_df.adjlist:
    #     input_df.view_adj_list()

    weights = None
    final_modularity = None

    # FIXME: This must be imported here to prevent a circular import
    from cugraph.structure import graph_primtypes_wrapper

    [offsets, indices] = graph_primtypes_wrapper.datatype_cast([input_df.adjlist.offsets, input_df.adjlist.indices], [np.int32])

    num_verts = input_df.number_of_vertices()
    num_edges = input_df.number_of_edges(directed_edges=True)

    # FIXME: assuming adjlist is not present because of view_adj_list() FIXME above.
    #if input_df.adjlist.weights is not None:
    if input_df.adjlist and input_df.adjlist.weights is not None:
        [weights] = graph_primtypes_wrapper.datatype_cast([input_df.adjlist.weights], [np.float32, np.float64])
    else:
        weights = cudf.Series(np.full(num_edges, 1.0, dtype=np.float32))

    # Create the output dataframe
    df = cudf.DataFrame()
    df['vertex'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    df['partition'] = cudf.Series(np.zeros(num_verts,dtype=np.int32))

    cdef uintptr_t c_offsets = offsets.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_indices = indices.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_identifier = df['vertex'].__cuda_array_interface__['data'][0]
    cdef uintptr_t c_partition = df['partition'].__cuda_array_interface__['data'][0]
    cdef uintptr_t c_weights = weights.__cuda_array_interface__['data'][0]

    cdef GraphCSRView[int,int,float] graph_float
    cdef GraphCSRView[int,int,double] graph_double

    # FIXME: figure out parts
    cdef uintptr_t parts = 0

    cdef float final_modularity_float = 1.0
    cdef double final_modularity_double = 1.0
    cdef int num_level = 0

    cdef pair[int,float] resultpair_float
    cdef pair[int,double] resultpair_double

    if weights.dtype == np.float32:
        graph_float = GraphCSRView[int,int,float](<int*>c_offsets, <int*>c_indices,
                                                  <float*>c_weights, num_verts, num_edges)

        graph_float.get_vertex_identifiers(<int*>c_identifier)
        resultpair_float = c_louvain.louvain[int,int,float](handle_[0], graph_float, <int*>parts, max_level, resolution)

        final_modularity = resultpair_float.second

    else:
        graph_double = GraphCSRView[int,int,double](<int*>c_offsets, <int*>c_indices,
                                                    <double*>c_weights, num_verts, num_edges)

        graph_double.get_vertex_identifiers(<int*>c_identifier)
        resultpair_double = c_louvain.louvain[int,int,double](handle_[0], graph_double, <int*>parts, max_level, resolution)

        final_modularity = resultpair_double.second

    return df, final_modularity
