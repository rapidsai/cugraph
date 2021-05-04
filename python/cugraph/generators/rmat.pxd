# Copyright (c) 2021, NVIDIA CORPORATION.
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
from libcpp cimport bool
from cugraph.structure.graph_utilities cimport *

cdef extern from "experimental/graph_generator.hpp" namespace "cugraph::experimental":
    ctypedef enum generator_distribution_t:
        POWER_LAW "cugraph::experimental::generator_distribution_t::POWER_LAW"
        UNIFORM "cugraph::experimental::generator_distribution_t::UNIFORM"


cdef extern from "utilities/cython.hpp" namespace "cugraph::cython":
    cdef unique_ptr[graph_generator_t] call_generate_rmat_edgelist[vertex_t] (
        const handle_t &handle,
        size_t scale,
        size_t num_edges,
        double a,
        double b,
        double c,
        int seed,
        bool clip_and_flip,
        bool scramble_vertex_ids) except +

    cdef unique_ptr[graph_generator_t*] call_generate_rmat_edgelists[vertex_t] (
        const handle_t &handle,
        size_t n_edgelists,
        size_t min_scale,
        size_t max_scale,
        size_t edge_factor,
        generator_distribution_t size_distribution,
        generator_distribution_t edge_distribution,
        int seed,
        bool clip_and_flip,
        bool scramble_vertex_ids) except +