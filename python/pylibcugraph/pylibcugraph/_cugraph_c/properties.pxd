# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Have cython use python 3 syntax
# cython: language_level = 3

cdef extern from "cugraph_c/properties.h":

    ctypedef struct cugraph_vertex_property_t:
        pass

    ctypedef struct cugraph_edge_property_t:
        pass

    ctypedef struct cugraph_vertex_property_view_t:
        pass

    ctypedef struct cugraph_edge_property_view_t:
        pass
