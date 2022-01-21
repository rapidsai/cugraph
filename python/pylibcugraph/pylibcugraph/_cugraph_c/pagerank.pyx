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

from pylibcugraph._cugraph_c._cugraph_api cimport (
    bool_t,
    #data_type_id_t,
)
from pylibcugraph._cugraph_c._error cimport (
    cugraph_error_code_t,
    cugraph_error_t,
)
from pylibcugraph._cugraph_c._array cimport (
    cugraph_type_erased_device_array_t,
    cugraph_type_erased_device_array_create,
    cugraph_type_erased_device_array_free,
)
from pylibcugraph._cugraph_c._graph cimport (
    cugraph_graph_t,
    cugraph_sg_graph_create,
    cugraph_graph_properties_t,
    cugraph_sg_graph_free,
)
from pylibcugraph._cugraph_c._algorithms cimport (
    cugraph_pagerank_result_t,
)

from pylibcugraph._cugraph_c.resource_handle cimport (
    EXPERIMENTAL__ResourceHandle,
)
from pylibcugraph._cugraph_c.graphs cimport (
    EXPERIMENTAL__Graph,
)
from pylibcugraph._cugraph_c.utils cimport (
    assert_success,
)


cdef cugraph_error_code_t \
    pagerank(EXPERIMENTAL__ResourceHandle resource_handle,
             EXPERIMENTAL__Graph graph,
             cugraph_type_erased_device_array_t* precomputed_vertex_out_weight_sums,
             double alpha,
             double epsilon,
             size_t max_iterations,
             bool_t has_initial_guess,
             bool_t do_expensive_check,
             cugraph_pagerank_result_t** result,
             cugraph_error_t** error
             ):
    """
    """
    pass
