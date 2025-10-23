# SPDX-FileCopyrightText: Copyright (c) 2019-2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cugraph.structure.graph_primtypes cimport *
from cugraph.structure.graph_utilities cimport *


cdef extern from "cugraph/algorithms.hpp" namespace "cugraph":

    ctypedef enum cugraph_cc_t:
        CUGRAPH_STRONG "cugraph::cugraph_cc_t::CUGRAPH_STRONG"

    cdef void connected_components[VT,ET,WT](
        const GraphCSRView[VT,ET,WT] &graph,
        cugraph_cc_t connect_type,
        VT *labels) except +
