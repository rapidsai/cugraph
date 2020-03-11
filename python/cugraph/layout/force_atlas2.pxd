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


cdef extern from "cugraph.h" namespace "cugraph":

    cdef void force_atlas2(
        Graph *graph,
        void *c_fa2_x_ptr,
        void *c_fa2_y_ptr,
        int max_iter,
        float gravity,
        float scaling_ratio,
        int edge_weight_influence,
        int lin_log_mode,
        int prevent_overlapping) except +
