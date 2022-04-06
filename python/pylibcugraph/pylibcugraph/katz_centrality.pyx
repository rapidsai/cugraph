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

from libc.stdint cimport uintptr_t

from pylibcugraph._cugraph_c.resource_handle cimport (
    bool_t,
    data_type_id_t,
    cugraph_resource_handle_t,
)
from pylibcugraph._cugraph_c.error cimport (
    cugraph_error_code_t,
    cugraph_error_t,
)
from pylibcugraph._cugraph_c.array cimport (
    cugraph_type_erased_device_array_view_t,
)
from pylibcugraph._cugraph_c.graph cimport (
    cugraph_graph_t,
)
from pylibcugraph._cugraph_c.centrality_algorithms cimport (
    cugraph_centrality_result_t,
    cugraph_katz_centrality,
    cugraph_centrality_result_get_vertices,
    cugraph_centrality_result_get_values,
    cugraph_centrality_result_free,
)
from pylibcugraph.resource_handle cimport (
    EXPERIMENTAL__ResourceHandle,
)
from pylibcugraph.graphs cimport (
    _GPUGraph,
)
from pylibcugraph.utils cimport (
    assert_success,
    copy_to_cupy_array,
    assert_CAI_type,
    get_c_type_from_numpy_type,
)


def EXPERIMENTAL__katz_centrality():
    """
    Does katz centrality.
    """

    # cdef cugraph_resource_handle_t* c_resource_handle_ptr = \
    #     resource_handle.c_resource_handle_ptr
    # cdef cugraph_graph_t* c_graph_ptr = graph.c_graph_ptr

    # cdef cugraph_centrality_result_t* result_ptr
    # cdef cugraph_error_code_t error_code
    # cdef cugraph_error_t* error_ptr

    # error_code = cugraph_katz_centrality(c_resource_handle_ptr,
    #                                      c_graph_ptr,
    #                                      betas_ptr,
    #                                      alpha,
    #                                      beta,
    #                                      epsilon,
    #                                      max_iterations,
    #                                      do_expensive_check,
    #                                      &result_ptr,
    #                                      &error_ptr)
    return -1
