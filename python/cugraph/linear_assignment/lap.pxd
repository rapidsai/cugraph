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

cdef extern from "cugraph/algorithms.hpp" namespace "cugraph":

    cdef weight_t hungarian[vertex_t,edge_t,weight_t](
        const handle_t &handle,
        const GraphCOOView[vertex_t,edge_t,weight_t] &graph,
        vertex_t num_workers,
        const vertex_t *workers,
        vertex_t *assignments,
        weight_t epsilon) except +

    cdef weight_t hungarian[vertex_t,edge_t,weight_t](
        const handle_t &handle,
        const GraphCOOView[vertex_t,edge_t,weight_t] &graph,
        vertex_t num_workers,
        const vertex_t *workers,
        vertex_t *assignments) except +

cdef extern from "cugraph/algorithms.hpp":

    cdef weight_t dense_hungarian "cugraph::dense::hungarian" [vertex_t,weight_t](
        const handle_t &handle,
        const weight_t *costs,
        vertex_t num_rows,
        vertex_t num_columns,
        vertex_t *assignments,
        weight_t epsilon) except +

    cdef weight_t dense_hungarian "cugraph::dense::hungarian" [vertex_t,weight_t](
        const handle_t &handle,
        const weight_t *costs,
        vertex_t num_rows,
        vertex_t num_columns,
        vertex_t *assignments) except +
