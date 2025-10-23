# SPDX-FileCopyrightText: Copyright (c) 2020-2021, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libc.stdint cimport uintptr_t
from cugraph.structure cimport utils as c_utils
from cugraph.structure.graph_primtypes cimport *
from libc.stdint cimport uintptr_t

import cudf
import numpy as np


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
