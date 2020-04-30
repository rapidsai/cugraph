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
from cugraph.structure.graph_new cimport *
from cugraph.structure cimport utils as c_utils

import cudf
import rmm
import numpy as np


def coo2csr(source_col, dest_col, weights=None):
    if len(source_col) != len(dest_col):
        raise Exception("source_col and dest_col should have the same number of elements")

    if source_col.dtype != dest_col.dtype:
        raise Exception("source_col and dest_col should be the same type")

    if source_col.dtype != np.int32:
        raise Exception("source_col and dest_col must be type np.int32")

    csr_weights = None
    num_edges = len(source_col)

    cdef uintptr_t c_src = source_col.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_dst = dest_col.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_weights = <uintptr_t> NULL
    cdef uintptr_t c_offsets = <uintptr_t> NULL
    cdef uintptr_t c_indices = <uintptr_t> NULL
    cdef uintptr_t c_csr_weights = <uintptr_t> NULL

    num_verts = 0

    cdef GraphCOOView[int,int,float] in_graph
    cdef unique_ptr[GraphCSR[int,int,float]] out_graph
    if weights is not None:
        c_weights = weights.__cuda_array_interface__['data'][0]

        if weights.dtype == np.float32:
            in_graph = GraphCOOView[int,int,float](<int*>c_src, <int*>c_dst, <float*>c_weights, num_verts, num_edges)
            out_graph = move(c_utils.coo_to_csr[int,int,float](in_graph))
            num_verts = c_utils.coo2csr_weighted[int, int, float](len(source_col),
                                                                  <const int*>c_src,
                                                                  <const int*>c_dst,
                                                                  <const float*>c_weights,
                                                                  <int**>&c_offsets,
                                                                  <int**>&c_indices,
                                                                  <float**>&c_csr_weights)

            csr_weights = cudf.Series(rmm.device_array_from_ptr(c_csr_weights,
                                                                nelem=num_edges,
                                                                dtype=np.float32))
        elif weights.dtype == np.float64:
            num_verts = c_utils.coo2csr_weighted[int, int, double](len(source_col),
                                                                   <const int*>c_src,
                                                                   <const int*>c_dst,
                                                                   <const double*>c_weights,
                                                                   <int**>&c_offsets,
                                                                   <int**>&c_indices,
                                                                   <double**>&c_csr_weights)

            csr_weights = cudf.Series(rmm.device_array_from_ptr(c_csr_weights,
                                                                nelem=num_edges,
                                                                dtype=np.float64))
    else:
        num_verts = c_utils.coo2csr[int, int](len(source_col),
                                              <const int*>c_src,
                                              <const int*>c_dst,
                                              <int**>&c_offsets,
                                              <int**>&c_indices)

        print("called coo2csr, num_verts = ", num_verts)
        print("c_offsets = ", c_offsets)
        print("c_indices = ", c_indices)

    offsets = rmm.device_array_from_ptr(c_offsets,
                                        nelem=num_verts+1,
                                        dtype=np.int32)
    indices = rmm.device_array_from_ptr(c_indices,
                                        nelem=num_edges,
                                        dtype=np.int32)

    return cudf.Series(offsets), cudf.Series(indices), csr_weights
