# Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

from cugraph.community.leiden cimport leiden as c_leiden
from cugraph.structure.graph_primtypes cimport *
from cugraph.structure import graph_primtypes_wrapper
from libc.stdint cimport uintptr_t

import cudf
import numpy as np


def leiden(input_graph, max_iter, resolution):
    """
    Call leiden
    """
    if not input_graph.adjlist:
        input_graph.view_adj_list()

    cdef unique_ptr[handle_t] handle_ptr
    handle_ptr.reset(new handle_t())

    weights = None
    final_modularity = None

    [offsets, indices] = graph_primtypes_wrapper.datatype_cast([input_graph.adjlist.offsets, input_graph.adjlist.indices], [np.int32])

    num_verts = input_graph.number_of_vertices()
    num_edges = input_graph.number_of_edges(directed_edges=True)

    if input_graph.adjlist.weights is not None:
        [weights] = graph_primtypes_wrapper.datatype_cast([input_graph.adjlist.weights], [np.float32, np.float64])
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

    cdef float final_modularity_float = 1.0
    cdef double final_modularity_double = 1.0
    cdef int num_level = 0

    if weights.dtype == np.float32:
        graph_float = GraphCSRView[int,int,float](<int*>c_offsets, <int*>c_indices,
                                                  <float*>c_weights, num_verts, num_edges)

        graph_float.get_vertex_identifiers(<int*>c_identifier)
        num_level, final_modularity_float = c_leiden(handle_ptr.get()[0],
                                                     graph_float,
                                                     <int*> c_partition,
                                                     <int> max_iter,
                                                     <float> resolution)

        final_modularity = final_modularity_float
    else:
        graph_double = GraphCSRView[int,int,double](<int*>c_offsets, <int*>c_indices,
                                                    <double*>c_weights, num_verts, num_edges)

        graph_double.get_vertex_identifiers(<int*>c_identifier)
        num_level, final_modularity_double = c_leiden(handle_ptr.get()[0],
                                                      graph_double,
                                                      <int*> c_partition,
                                                      <int> max_iter,
                                                      <double> resolution)
        final_modularity = final_modularity_double

    return df, final_modularity
