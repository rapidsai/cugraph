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

cdef extern from "cugraph/internals.hpp" namespace "cugraph::internals":
    cdef cppclass GraphBasedDimRedCallback

cdef extern from "cugraph/algorithms.hpp" namespace "cugraph":

    cdef void force_atlas2[vertex_t, edge_t, weight_t](
        const handle_t &handle,
        GraphCOOView[vertex_t, edge_t, weight_t] &graph,
        float *pos,
        const int max_iter,
        float *x_start,
        float *y_start,
        bool outbound_attraction_distribution,
        bool lin_log_mode,
        bool prevent_overlapping,
        const float edge_weight_influence,
        const float jitter_tolerance,
        bool barnes_hut_optimize,
        const float barnes_hut_theta,
        const float scaling_ratio,
        bool strong_gravity_mode,
        const float gravity,
        bool verbose,
        GraphBasedDimRedCallback *callback) except +
