# SPDX-FileCopyrightText: Copyright (c) 2019-2021, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cugraph.structure.graph_primtypes cimport *


cdef extern from "cugraph/algorithms.hpp" namespace "cugraph":

    cdef unique_ptr[GraphCOO[VT,ET,WT]] minimum_spanning_tree[VT,ET,WT](const handle_t &handle,
        const GraphCSRView[VT,ET,WT] &graph) except +
