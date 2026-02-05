# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Have cython use python 3 syntax
# cython: language_level = 3

from pylibcugraph._cugraph_c.types cimport (
    bool_t,
)
from pylibcugraph._cugraph_c.resource_handle cimport (
    cugraph_resource_handle_t,
)
from pylibcugraph._cugraph_c.error cimport (
    cugraph_error_code_t,
    cugraph_error_t,
)
from pylibcugraph._cugraph_c.graph cimport (
    cugraph_graph_t,
)

from pylibcugraph._cugraph_c.graph_functions cimport (
    cugraph_induced_subgraph_result_t,
)

cdef extern from "cugraph_c/tree_algorithms.h":
    ###########################################################################


    ###########################################################################
    # Minimum Spanning Tree
    cdef cugraph_error_code_t \
        cugraph_minimum_spanning_tree(
            const cugraph_resource_handle_t* handle,
            cugraph_graph_t* graph,
            bool_t do_expensive_check,
            cugraph_induced_subgraph_result_t** result,
            cugraph_error_t** error)
