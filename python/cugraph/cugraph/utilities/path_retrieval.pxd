# SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cugraph.structure.graph_primtypes cimport *

cdef extern from "cugraph/utilities/path_retrieval.hpp" namespace "cugraph":

    cdef void get_traversed_cost[vertex_t, weight_t](const handle_t &handle,
            const vertex_t *vertices,
            const vertex_t *preds,
            const weight_t *info_weights,
            weight_t *out,
            vertex_t stop_vertex,
            vertex_t num_vertices) except +
