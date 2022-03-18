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
    cugraph_type_erased_device_array_view_create,
    cugraph_type_erased_device_array_free,
    cugraph_type_erased_host_array_view_t,
)
from pylibcugraph._cugraph_c.graph cimport (
    cugraph_graph_t,
)
from pylibcugraph._cugraph_c.algorithms cimport (
    uniform_nbr_sample,
    cugraph_sample_result_t,
    cugraph_sample_result_get_sources,
    cugraph_sample_result_get_destinations,
    cugraph_sample_result_get_start_labels,
    cugraph_sample_result_get_index,
    cugraph_sample_result_get_counts,
    cugraph_sample_result_free,
)
from pylibcugraph.resource_handle cimport (
    EXPERIMENTAL__ResourceHandle,
)
from pylibcugraph.graphs cimport (
    _GPUGraph
)
from pylibcugraph.utils cimport (
    assert_success,
    copy_to_cupy_array,
    assert_CAI_type,
    get_c_type_from_numpy_type,
)

def EXPERIMENTAL__nbr_sampling(EXPERIMENTAL__ResourceHandle resource_handle,
                               _GPUGraph graph,
                               input_graph,
                               start_info_list,
                               h_fan_out,
                               bool_t without_replacement):
    """
    Does uniform neighborhood sampling.
    Parameters
    ----------
    input_graph: ???
    start_info_list: ???
    fanout_vals: ???
    with_replacement: ???

    Examples
    --------
    >>> import pylibcugraph, cupy, numpy

    """
    print("Hello from uniform_neighborhood_sampling.pyx!")
    
    cdef cugraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef cugraph_graph_t* c_graph_ptr = graph.c_graph_ptr

    cdef cugraph_sample_result_t* result_ptr
    cdef cugraph_error_code_t error_code
    cdef cugraph_error_t* error_ptr

    # FIXME: For now, just initialize the start_ptr,
    # start_labels_ptr, and fan_out_ptr. Fix later once
    # PR#2112 is merged
    cdef cugraph_type_erased_device_array_view_t* start_ptr
    cdef cugraph_type_erased_device_array_view_t* start_labels_ptr
    cdef cugraph_type_erased_host_array_view_t* fan_out_ptr

    error_code = uniform_nbr_sample(c_resource_handle_ptr,
                                    c_graph_ptr,
                                    start_ptr,
                                    start_labels_ptr,
                                    fan_out_ptr,
                                    without_replacement,
                                    &result_ptr,
                                    &error_ptr)
    assert_success(error_code, error_ptr, "uniform_nbr_sample")
    
    return 0

"""
def uniform_neighborhood_sampling_REAL(input_graph,
                                                start_info_list,
                                                h_fan_out,
                                                bool_t with_replacement):
    Test.
    
    # Remove below code once function signature is confirmed
    resource_handle = pylibcugraph.experimental.ResourceHandle()
    graph_props = pylibcugraph.experimental.GraphProperties(
        is_symmetric=False, is_multigraph=False)
    # Remove above code once function signature is confirmed
    cdef cugraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef cugraph_graph_t* c_graph_ptr = graph.c_graph_ptr
    cdef cugraph_error_code_t error_code
    cdef cugraph_error_t* error_ptr
    error_code = uniform_nbr_sample(c_resource_handle_ptr,
                                    c_graph_ptr,
                                    device_starts,
                                    device_ranks,
                                    num_starting_vs,
                                    h_fan_out,
                                    with_replacement)
    assert_success(error_code, error_ptr, "uniform_nbr_sample")
    return (src_vertices, dst_vertices, ranks, indices, rx_counts)
"""