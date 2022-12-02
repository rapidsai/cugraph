# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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


from pylibraft.common.handle cimport *
from libcpp cimport bool


cdef extern from "cugraph/utilities/cython.hpp" namespace "cugraph::cython":

    ctypedef enum numberTypeEnum:
        int32Type "cugraph::cython::numberTypeEnum::int32Type"
        int64Type "cugraph::cython::numberTypeEnum::int64Type"
        floatType "cugraph::cython::numberTypeEnum::floatType"
        doubleType "cugraph::cython::numberTypeEnum::doubleType"

    cdef cppclass graph_container_t:
       pass

    cdef void populate_graph_container(
        graph_container_t &graph_container,
        handle_t &handle,
        void *src_vertices,
        void *dst_vertices,
        void *weights,
        void *vertex_partition_offsets,
        void *segment_offsets,
        size_t num_segments,
        numberTypeEnum vertexType,
        numberTypeEnum edgeType,
        numberTypeEnum weightType,
        size_t num_local_edges,
        size_t num_global_vertices,
        size_t num_global_edges,
        bool is_weighted,
        bool is_symmetric,
        bool transposed,
        bool multi_gpu) except +
