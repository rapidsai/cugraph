#
# Copyright (c) 2020-2021, NVIDIA CORPORATION.
#
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
#

from cugraph.structure.graph_utilities cimport *
from libcpp cimport bool


cdef extern from "cugraph/utilities/cython.hpp" namespace "cugraph::cython":

    cdef void call_pagerank[vertex_t, weight_t](
        const handle_t &handle,
        const graph_container_t &g,
        vertex_t *identifiers,
        weight_t *pagerank,
        vertex_t size,
        vertex_t *personalization_subset,
        weight_t *personalization_values,
        double alpha,
        double tolerance,
        long long max_iter,
        bool has_guess) except +
