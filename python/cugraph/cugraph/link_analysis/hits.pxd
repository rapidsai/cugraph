# Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

from cugraph.structure.graph_utilities cimport graph_container_t
from raft.common.handle cimport handle_t


cdef extern from "cugraph/utilities/cython.hpp" namespace "cugraph::cython":
    cdef void call_hits[vertex_t,weight_t](
        const handle_t &handle,
        const graph_container_t &g,
        weight_t *hubs,
        weight_t *authorities,
        int max_iter,
        weight_t tolerance,
        const weight_t *starting_value,
        bool normalized) except +
