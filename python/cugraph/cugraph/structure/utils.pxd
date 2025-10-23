# SPDX-FileCopyrightText: Copyright (c) 2019-2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cugraph.structure.graph_primtypes cimport *
from libcpp.memory cimport unique_ptr


cdef extern from "cugraph/legacy/functions.hpp" namespace "cugraph":

    cdef unique_ptr[GraphCSR[VT,ET,WT]] coo_to_csr[VT,ET,WT](
            const GraphCOOView[VT,ET,WT] &graph) except +

    cdef void comms_bcast[value_t](
            const handle_t &handle,
            value_t *dst,
            size_t size) except +
