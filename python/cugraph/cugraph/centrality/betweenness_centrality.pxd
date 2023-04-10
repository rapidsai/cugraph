# Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

from cugraph.structure.graph_primtypes cimport *
from libcpp cimport bool

cdef extern from "cugraph/algorithms.hpp" namespace "cugraph":

    cdef void betweenness_centrality[VT, ET, WT, result_t](
        const handle_t &handle,
        const GraphCSRView[VT, ET, WT] &graph,
        result_t *result,
        bool normalized,
        bool endpoints,
        const WT *weight,
        VT k,
        const VT *vertices) except +

    cdef void edge_betweenness_centrality[VT, ET, WT, result_t](
        const handle_t &handle,
        const GraphCSRView[VT, ET, WT] &graph,
        result_t *result,
        bool normalized,
        const WT *weight,
        VT k,
        const VT *vertices) except +
