# Copyright (c) 2019-2021, NVIDIA CORPORATION.
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


from libcpp.utility cimport pair
from cugraph.structure.graph_primtypes cimport *


cdef extern from "cugraph/algorithms.hpp" namespace "cugraph":

    cdef pair[size_t, weight_t] leiden[vertex_t,edge_t,weight_t](
        const handle_t &handle,
        const GraphCSRView[vertex_t,edge_t,weight_t] &graph,
        vertex_t *leiden_parts,
        size_t max_level,
        weight_t resolution) except +
