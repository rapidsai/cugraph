# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cugraph.utilities.path_retrieval cimport get_traversed_cost as c_get_traversed_cost
from cugraph.structure.graph_primtypes cimport *
from libc.stdint cimport uintptr_t
import cudf
import numpy as np


def get_traversed_cost(input_df, stop_vertex):
    """
    Call get_traversed_cost
    """
    num_verts = input_df.shape[0]
    vertex_t = input_df.vertex.dtype
    weight_t = input_df.weights.dtype

    df = cudf.DataFrame()
    df['vertex'] = input_df['vertex']
    df['info'] = cudf.Series(np.zeros(num_verts, dtype=weight_t))

    cdef unique_ptr[handle_t] handle_ptr
    handle_ptr.reset(new handle_t())
    handle_ = handle_ptr.get();

    cdef uintptr_t vertices = <uintptr_t>NULL
    cdef uintptr_t preds = <uintptr_t>NULL
    cdef uintptr_t out = <uintptr_t>NULL
    cdef uintptr_t info_weights = <uintptr_t>NULL

    vertices = input_df['vertex'].__cuda_array_interface__['data'][0]
    preds = input_df['predecessor'].__cuda_array_interface__['data'][0]
    info_weights = input_df['weights'].__cuda_array_interface__['data'][0]
    out = df['info'].__cuda_array_interface__['data'][0]

    if weight_t == np.float32:
        c_get_traversed_cost(handle_[0],
            <int *> vertices,
            <int *> preds,
            <float *> info_weights,
            <float *> out,
            <int> stop_vertex,
            <int> num_verts)
    elif weight_t == np.float64:
        c_get_traversed_cost(handle_[0],
            <int *> vertices,
            <int *> preds,
            <double *> info_weights,
            <double *> out,
            <int> stop_vertex,
            <int> num_verts)
    else:
        raise NotImplementedError

    return df
