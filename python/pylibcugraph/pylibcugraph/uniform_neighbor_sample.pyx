# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
    cugraph_resource_handle_t,
)
from pylibcugraph._cugraph_c.error cimport (
    cugraph_error_code_t,
    cugraph_error_t,
)
from pylibcugraph._cugraph_c.array cimport (
    cugraph_type_erased_device_array_view_t,
    cugraph_type_erased_device_array_view_create,
    cugraph_type_erased_device_array_view_free,
    cugraph_type_erased_host_array_view_t,
    cugraph_type_erased_host_array_view_create,
    cugraph_type_erased_host_array_view_free,
)
from pylibcugraph._cugraph_c.graph cimport (
    cugraph_graph_t,
)
from pylibcugraph._cugraph_c.algorithms cimport (
    cugraph_sample_result_t,
    cugraph_prior_sources_behavior_t,
    cugraph_sampling_options_t,
    cugraph_sampling_options_create,
    cugraph_sampling_options_free,
    cugraph_sampling_set_with_replacement,
    cugraph_sampling_set_return_hops,
    cugraph_sampling_set_prior_sources_behavior,
    cugraph_sampling_set_dedupe_sources,
    cugraph_sampling_set_renumber_results,

)
from pylibcugraph._cugraph_c.sampling_algorithms cimport (
    cugraph_uniform_neighbor_sample,

)
from pylibcugraph.resource_handle cimport (
    ResourceHandle,
)
from pylibcugraph.graphs cimport (
    _GPUGraph,
)
from pylibcugraph.utils cimport (
    assert_success,
    assert_CAI_type,
    assert_AI_type,
    get_c_type_from_numpy_type,
)
from pylibcugraph.internal_types.sampling_result cimport (
    SamplingResult,
)
from pylibcugraph._cugraph_c.random cimport (
    cugraph_rng_state_t
)
from pylibcugraph.random cimport (
    CuGraphRandomState
)

# TODO accept cupy/numpy random state in addition to raw seed.
def uniform_neighbor_sample(ResourceHandle resource_handle,
                            _GPUGraph input_graph,
                            start_list,
                            h_fan_out,
                            *,
                            bool_t with_replacement,
                            bool_t do_expensive_check,
                            bool_t with_edge_properties=<bool_t>False,
                            batch_id_list=None,
                            label_list=None,
                            label_to_output_comm_rank=None,
                            prior_sources_behavior=None,
                            bool_t deduplicate_sources=<bool_t>False,
                            bool_t return_hops=<bool_t>False,
                            bool_t renumber=<bool_t>False,
                            random_state=None):
    """
    Does neighborhood sampling, which samples nodes from a graph based on the
    current node's neighbors, with a corresponding fanout value at each hop.

    Parameters
    ----------
    resource_handle: ResourceHandle
        Handle to the underlying device and host resources needed for
        referencing data and running algorithms.

    input_graph : SGGraph or MGGraph
        The input graph, for either Single or Multi-GPU operations.

    start_list: device array type
        Device array containing the list of starting vertices for sampling.

    h_fan_out: numpy array type
        Device array containing the brancing out (fan-out) degrees per
        starting vertex for each hop level.

    with_replacement: bool
        If true, sampling procedure is done with replacement (the same vertex
        can be selected multiple times in the same step).

    do_expensive_check: bool
        If True, performs more extensive tests on the inputs to ensure
        validitity, at the expense of increased run time.
    
    with_edge_properties: bool
        If True, returns the edge properties of each edges along with the
        edges themselves.  Will result in an error if the provided graph
        does not have edge properties.
    
    batch_id_list: list[int32] (Optional)
        List of int32 batch ids that is returned with each edge.  Optional
        argument, defaults to NULL, returning nothing.
    
    label_list: list[int32] (Optional)
        List of unique int32 batch ids.  Required if also passing the
        label_to_output_comm_rank flag.  Default to NULL (does nothing)
    
    label_to_output_comm_rank: list[int32] (Optional)
        Maps the unique batch ids in label_list to the rank of the
        worker that should hold results for that batch id.
        Defaults to NULL (does nothing)
    
    prior_sources_behavior: str (Optional)
        Options are "carryover", and "exclude".
        Default will leave the source list as-is.
        Carryover will carry over sources from previous hops to the
        current hop.
        Exclude will exclude sources from previous hops from reappearing
        as sources in future hops.
    
    deduplicate_sources: bool (Optional)
        If True, will deduplicate the source list before sampling.
        Defaults to False.

    renumber: bool (Optional)
        If True, will renumber the sources and destinations on a
        per-batch basis and return the renumber map and batch offsets
        in additional to the standard returns.

    random_state: int (Optional)
        Random state to use when generating samples.  Optional argument,
        defaults to a hash of process id, time, and hostname.
        (See pylibcugraph.random.CuGraphRandomState)

    Returns
    -------
    A tuple of device arrays, where the first and second items in the tuple
    are device arrays containing the starting and ending vertices of each
    walk respectively, the third item in the tuple is a device array
    containing the start labels, and the fourth item in the tuple is a device
    array containing the indices for reconstructing paths.
    
    If renumber was set to True, then the fifth item in the tuple is a device
    array containing the renumber map, and the sixth item in the tuple is a
    device array containing the renumber map offsets (which delineate where
    the renumber map for each batch starts). 

    """
    cdef cugraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef cugraph_graph_t* c_graph_ptr = input_graph.c_graph_ptr

    assert_CAI_type(start_list, "start_list")
    assert_CAI_type(batch_id_list, "batch_id_list", True)
    assert_CAI_type(label_list, "label_list", True)
    assert_CAI_type(label_to_output_comm_rank, "label_to_output_comm_rank", True)
    assert_AI_type(h_fan_out, "h_fan_out")

    cdef cugraph_sample_result_t* result_ptr
    cdef cugraph_error_code_t error_code
    cdef cugraph_error_t* error_ptr

    cdef uintptr_t cai_start_ptr = \
        start_list.__cuda_array_interface__["data"][0]
    
    cdef uintptr_t cai_batch_id_ptr
    if batch_id_list is not None:
        cai_batch_id_ptr = \
            batch_id_list.__cuda_array_interface__['data'][0]
    
    cdef uintptr_t cai_label_list_ptr
    if label_list is not None:
        cai_label_list_ptr = \
            label_list.__cuda_array_interface__['data'][0]
    
    cdef uintptr_t cai_label_to_output_comm_rank_ptr
    if label_to_output_comm_rank is not None:
        cai_label_to_output_comm_rank_ptr = \
            label_to_output_comm_rank.__cuda_array_interface__['data'][0]
        
    cdef uintptr_t ai_fan_out_ptr = \
        h_fan_out.__array_interface__["data"][0]

    cdef cugraph_type_erased_device_array_view_t* start_ptr = \
        cugraph_type_erased_device_array_view_create(
            <void*>cai_start_ptr,
            len(start_list),
            get_c_type_from_numpy_type(start_list.dtype))

    cdef cugraph_type_erased_device_array_view_t* batch_id_ptr = <cugraph_type_erased_device_array_view_t*>NULL
    if batch_id_list is not None:
        batch_id_ptr = \
            cugraph_type_erased_device_array_view_create(
                <void*>cai_batch_id_ptr,
                len(batch_id_list),
                get_c_type_from_numpy_type(batch_id_list.dtype)
            )
 
    cdef cugraph_type_erased_device_array_view_t* label_list_ptr = <cugraph_type_erased_device_array_view_t*>NULL
    if label_list is not None:
        label_list_ptr = \
            cugraph_type_erased_device_array_view_create(
                <void*>cai_label_list_ptr,
                len(label_list),
                get_c_type_from_numpy_type(label_list.dtype)
            )
    
    cdef cugraph_type_erased_device_array_view_t* label_to_output_comm_rank_ptr = <cugraph_type_erased_device_array_view_t*>NULL
    if label_to_output_comm_rank is not None:
        label_to_output_comm_rank_ptr = \
            cugraph_type_erased_device_array_view_create(
                <void*>cai_label_to_output_comm_rank_ptr,
                len(label_to_output_comm_rank),
                get_c_type_from_numpy_type(label_to_output_comm_rank.dtype)
            )

    cdef cugraph_type_erased_host_array_view_t* fan_out_ptr = \
        cugraph_type_erased_host_array_view_create(
            <void*>ai_fan_out_ptr,
            len(h_fan_out),
            get_c_type_from_numpy_type(h_fan_out.dtype))

    
    cg_rng_state = CuGraphRandomState(resource_handle, random_state)

    cdef cugraph_rng_state_t* rng_state_ptr = \
        cg_rng_state.rng_state_ptr

    cdef cugraph_prior_sources_behavior_t prior_sources_behavior_e
    if prior_sources_behavior is None:
        prior_sources_behavior_e = cugraph_prior_sources_behavior_t.DEFAULT
    elif prior_sources_behavior == 'carryover':
        prior_sources_behavior_e = cugraph_prior_sources_behavior_t.CARRY_OVER
    elif prior_sources_behavior == 'exclude':
        prior_sources_behavior_e = cugraph_prior_sources_behavior_t.EXCLUDE
    else:
        raise ValueError(
            f'Invalid option {prior_sources_behavior}'
            ' for prior sources behavior'
        )

    cdef cugraph_sampling_options_t* sampling_options
    error_code = cugraph_sampling_options_create(&sampling_options, &error_ptr)
    assert_success(error_code, error_ptr, "cugraph_sampling_options_create")

    cugraph_sampling_set_with_replacement(sampling_options, with_replacement)
    cugraph_sampling_set_return_hops(sampling_options, return_hops)
    cugraph_sampling_set_dedupe_sources(sampling_options, deduplicate_sources)
    cugraph_sampling_set_prior_sources_behavior(sampling_options, prior_sources_behavior_e)
    cugraph_sampling_set_renumber_results(sampling_options, renumber)

    error_code = cugraph_uniform_neighbor_sample(
        c_resource_handle_ptr,
        c_graph_ptr,
        start_ptr,
        batch_id_ptr,
        label_list_ptr,
        label_to_output_comm_rank_ptr,
        fan_out_ptr,
        rng_state_ptr,
        sampling_options,
        do_expensive_check,
        &result_ptr,
        &error_ptr)
    assert_success(error_code, error_ptr, "cugraph_uniform_neighbor_sample")

    # Free the sampling options
    cugraph_sampling_options_free(sampling_options)

    # Free the two input arrays that are no longer needed.
    cugraph_type_erased_device_array_view_free(start_ptr)
    cugraph_type_erased_host_array_view_free(fan_out_ptr)
    if batch_id_list is not None:
        cugraph_type_erased_device_array_view_free(batch_id_ptr)

    # Have the SamplingResult instance assume ownership of the result data.
    result = SamplingResult()
    result.set_ptr(result_ptr)

    # Get cupy "views" of the individual arrays to return. These each increment
    # the refcount on the SamplingResult instance which will keep the data alive
    # until all references are removed and the GC runs.
    if with_edge_properties:
        cupy_sources = result.get_sources()
        cupy_destinations = result.get_destinations()
        cupy_edge_weights = result.get_edge_weights()
        cupy_edge_ids = result.get_edge_ids()
        cupy_edge_types = result.get_edge_types()
        cupy_batch_ids = result.get_batch_ids()
        cupy_offsets = result.get_offsets()
        cupy_hop_ids = result.get_hop_ids()

        if renumber:
            cupy_renumber_map = result.get_renumber_map()
            cupy_renumber_map_offsets = result.get_renumber_map_offsets()
            return (cupy_sources, cupy_destinations, cupy_edge_weights, cupy_edge_ids, cupy_edge_types, cupy_batch_ids, cupy_offsets, cupy_hop_ids, cupy_renumber_map, cupy_renumber_map_offsets)
        else:
            return (cupy_sources, cupy_destinations, cupy_edge_weights, cupy_edge_ids, cupy_edge_types, cupy_batch_ids, cupy_offsets, cupy_hop_ids)

    else:
        cupy_sources = result.get_sources()
        cupy_destinations = result.get_destinations()
        cupy_indices = result.get_indices()

        return (cupy_sources, cupy_destinations, cupy_indices)
