# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

from cugraph.centrality.katz_centrality cimport call_katz_centrality
from cugraph.structure.graph_primtypes cimport *
from cugraph.structure import graph_primtypes_wrapper
from libcpp cimport bool
from libc.stdint cimport uintptr_t

import cudf
import rmm
import numpy as np


def get_output_df(input_graph, nstart):
    num_verts = input_graph.number_of_vertices()
    df = cudf.DataFrame()
    df['vertex'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    df['katz_centrality'] = cudf.Series(np.zeros(num_verts, dtype=np.float64))

    if nstart is not None:
        if len(nstart) != num_verts:
            raise ValueError('nstart must have initial guess for all vertices')

        nstart['values'] = graph_primtypes_wrapper.datatype_cast([nstart['values']], [np.float64])
        df['katz_centrality'][nstart['vertex']] = nstart['values']

    return df


def katz_centrality(input_graph, alpha=None, max_iter=100, tol=1.0e-5, nstart=None, normalized=True):
    """
    Call katz_centrality
    """

    cdef unique_ptr[handle_t] handle_ptr
    handle_ptr.reset(new handle_t())
    handle_ = handle_ptr.get();

    df = get_output_df(input_graph, nstart)
    if nstart is not None:
        has_guess = True
    if alpha is None:
        alpha = 0
    beta = 0

    if not input_graph.adjlist:
        input_graph.view_adj_list()
    num_verts = input_graph.number_of_vertices()
    num_edges = input_graph.number_of_edges(directed_edges=True)
    [offsets, indices] = graph_primtypes_wrapper.datatype_cast([input_graph.adjlist.offsets, input_graph.adjlist.indices], [np.int32])
    [weights] = graph_primtypes_wrapper.datatype_cast([input_graph.adjlist.weights], [np.float32])

    cdef uintptr_t c_identifier = df['vertex'].__cuda_array_interface__['data'][0]
    cdef uintptr_t c_katz = df['katz_centrality'].__cuda_array_interface__['data'][0]
    cdef uintptr_t c_offsets = offsets.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_indices = indices.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_weights = <uintptr_t>NULL
    if weights is not None:
        c_weights = weights.__cuda_array_interface__['data'][0]
        weight_t = weights.dtype
    else:
        weight_t = np.dtype("float32")
    cdef uintptr_t c_local_verts = <uintptr_t> NULL
    cdef uintptr_t c_local_edges = <uintptr_t> NULL
    cdef uintptr_t c_local_offsets = <uintptr_t> NULL

    # FIXME: Offsets and indices are currently hardcoded to int, but this may
    #        not be acceptable in the future.
    numberTypeMap = {np.dtype("int32") : <int>numberTypeEnum.int32Type,
                     np.dtype("int64") : <int>numberTypeEnum.int64Type,
                     np.dtype("float32") : <int>numberTypeEnum.floatType,
                     np.dtype("double") : <int>numberTypeEnum.doubleType}

    cdef graph_container_t graph_container
    populate_graph_container_legacy(graph_container,
                                    <graphTypeEnum>(<int>(graphTypeEnum.LegacyCSR)),
                                    handle_[0],
                                    <void*>c_offsets, <void*>c_indices, <void*>c_weights,
                                    <numberTypeEnum>(<int>(numberTypeEnum.int32Type)),
                                    <numberTypeEnum>(<int>(numberTypeEnum.int32Type)),
                                    <numberTypeEnum>(<int>(numberTypeMap[weight_t])),
                                    num_verts, num_edges,
                                    <int*>c_local_verts, <int*>c_local_edges, <int*>c_local_offsets)

    call_katz_centrality[int,double](handle_[0], graph_container, <int*>c_identifier, <double*> c_katz, <double>alpha, <double>beta, <double>tol, <int>max_iter, has_guess, normalized)

    return df
