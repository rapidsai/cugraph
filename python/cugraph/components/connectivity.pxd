# Copyright (c) 2019, NVIDIA CORPORATION.
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

from cugraph.structure.graph cimport *
from cudf._lib.cudf cimport *


cdef extern from "cugraph.h" namespace "cugraph":

    cdef void connected_components(
        Graph *graph,
        cugraph_cc_t connect_type,
        cudf_table* table) except +

    ctypedef enum cugraph_cc_t:
        CUGRAPH_WEAK = 0,
        CUGRAPH_STRONG,
        NUM_CONNECTIVITY_TYPES
