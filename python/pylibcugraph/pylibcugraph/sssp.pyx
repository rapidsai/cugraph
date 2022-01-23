# Copyright (c) 2022, NVIDIA CORPORATION.
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

# Have cython use python 3 syntax
# cython: language_level = 3

from pylibcugraph._cugraph_c.cugraph_api cimport (
    bool_t,
    #data_type_id_t,
    cugraph_resource_handle_t,
)
from pylibcugraph._cugraph_c.error cimport (
    cugraph_error_code_t,
    cugraph_error_t,
)
from pylibcugraph._cugraph_c.graph cimport (
    cugraph_graph_t,
)
from pylibcugraph._cugraph_c.algorithms cimport (
    cugraph_paths_result_t,
    cugraph_sssp,
)

from pylibcugraph.resource_handle cimport (
    EXPERIMENTAL__ResourceHandle,
)
from pylibcugraph.graphs cimport (
    EXPERIMENTAL__Graph,
)
from pylibcugraph.utils cimport (
    assert_success,
)


def EXPERIMENTAL__sssp(EXPERIMENTAL__ResourceHandle resource_handle,
                       EXPERIMENTAL__Graph graph,
                       size_t source,
                       double cutoff,
                       bool_t compute_predecessors,
                       bool_t do_expensive_check):
    """
    """
    cdef cugraph_resource_handle_t* c_resource_handle_ptr = resource_handle.c_resource_handle_ptr
    cdef cugraph_graph_t* c_graph_ptr = graph.c_graph_ptr

    cdef cugraph_paths_result_t* result_ptr
    cdef cugraph_error_code_t error_code
    cdef cugraph_error_t* error_ptr

    error_code = cugraph_sssp(c_resource_handle_ptr,
                              c_graph_ptr,
                              source,
                              cutoff,
                              compute_predecessors,
                              do_expensive_check,
                              &result_ptr,
                              &error_ptr)

    assert_success(error_code, error_ptr, "cugraph_sssp")
