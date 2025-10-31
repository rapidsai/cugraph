# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Have cython use python 3 syntax
# cython: language_level = 3

from pylibcugraph._cugraph_c.graph cimport (
     cugraph_graph_properties_t,
)


cdef class GraphProperties:
    cdef cugraph_graph_properties_t c_graph_properties
