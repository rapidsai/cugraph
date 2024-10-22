# Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
    cugraph_compression_type_t,
    cugraph_sampling_options_t,
    cugraph_sampling_options_create,
    cugraph_sampling_options_free,
    cugraph_sampling_set_with_replacement,
    cugraph_sampling_set_return_hops,
    cugraph_sampling_set_prior_sources_behavior,
    cugraph_sampling_set_dedupe_sources,
    cugraph_sampling_set_renumber_results,
    cugraph_sampling_set_compress_per_hop,
    cugraph_sampling_set_compression_type,
    cugraph_sampling_set_retain_seeds,
)
from pylibcugraph._cugraph_c.sampling_algorithms cimport (
    cugraph_heterogeneous_uniform_neighbor_sample,
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
import warnings

# TODO accept cupy/numpy random state in addition to raw seed.
def heterogeneous_uniform_neighbor_sample(ResourceHandle resource_handle,
                                          _GPUGraph input_graph,
                                          start_vertex_list,
                                          start_vertex_offsets,
                                          h_fan_out,
                                          num_edge_types,
                                          bool_t with_replacement,
                                          bool_t do_expensive_check,
                                          with_edge_properties=False,
                                          prior_sources_behavior=None,
                                          deduplicate_sources=False,
                                          return_hops=False,
                                          renumber=False,
                                          retain_seeds=False,
                                          compression='COO',
                                          compress_per_hop=False,
                                          random_state=None,
                                          return_dict=False,):
    """
    # FIXME: Deprecate uniform_neighbor_sample
    Performs biased neighborhood sampling, which samples nodes from
    a graph based on the current node's neighbors, with a corresponding fan_out
    value at each hop. The edges are sampled uniformly.

    Parameters
    ----------
    resource_handle: ResourceHandle
        Handle to the underlying device and host resources needed for
        referencing data and running algorithms.

    input_graph : SGGraph or MGGraph
        The input graph, for either Single or Multi-GPU operations.

    start_vertex_list: device array type
        Device array containing the list of starting vertices for sampling.

    start_vertex_offsets: list[int] (Optional)
        Offsets of each label within the start vertex list.

    h_fan_out: tuple of numpy array type
        Device array containing the branching out (fan-out) degrees per
        starting vertex for each hop level

        The sampling method can use different fan_out values for each edge type
        which is not the case for homogeneous neighborhood sampling.

    num_edge_types: Number of edge types where a value of 1 translates to homogeneous neighbor
        sample whereas a value greater than 1 translates to heterogeneous neighbor
        sample.

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

    retain_seeds: bool (Optional)
        If True, will retain the original seeds (original source vertices)
        in the output even if they do not have outgoing neighbors.
        Defaults to False.

    compression: str (Optional)
        Options: COO (default), CSR, CSC, DCSR, DCSR
        Sets the compression format for the returned samples.

    compress_per_hop: bool (Optional)
        If False (default), will create a compressed edgelist for the
        entire batch.
        If True, will create a separate compressed edgelist per hop within
        a batch.

    random_state: int (Optional)
        Random state to use when generating samples.  Optional argument,
        defaults to a hash of process id, time, and hostname.
        (See pylibcugraph.random.CuGraphRandomState)

    return_dict: bool (Optional)
        Whether to return a dictionary instead of a tuple.
        Optional argument, defaults to False, returning a tuple.
        This argument will eventually be deprecated in favor
        of always returning a dictionary.

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
    cdef cugraph_resource_handle_t* c_resource_handle_ptr = (
        resource_handle.c_resource_handle_ptr
    )

    cdef cugraph_graph_t* c_graph_ptr = input_graph.c_graph_ptr
    cdef cugraph_type_erased_host_array_view_t* fan_out_ptr = <cugraph_type_erased_host_array_view_t*>NULL

    cdef bool_t c_deduplicate_sources = deduplicate_sources
    cdef bool_t c_return_hops = return_hops
    cdef bool_t c_renumber = renumber
    cdef bool_t c_compress_per_hop = compress_per_hop

    cdef cugraph_error_code_t error_code
    cdef cugraph_error_t* error_ptr
    cdef uintptr_t ai_fan_out_ptr

    # FIXME: refactor the way we are creating pointer. Can use a single helper function to create

    assert_CAI_type(start_vertex_list, "start_vertex_list")
    assert_CAI_type(start_vertex_offsets, "start_vertex_offsets", True)

    assert_AI_type(h_fan_out, "h_fan_out")

    ai_fan_out_ptr = \
        h_fan_out.__array_interface__["data"][0]

    fan_out_ptr = \
        cugraph_type_erased_host_array_view_create(
            <void*>ai_fan_out_ptr,
            len(h_fan_out),
            get_c_type_from_numpy_type(h_fan_out.dtype))



    cdef cugraph_sample_result_t* result_ptr

    cdef uintptr_t cai_start_ptr = \
        start_vertex_list.__cuda_array_interface__["data"][0]

    cdef uintptr_t cai_start_vertex_offsets_ptr
    if start_vertex_offsets is not None:
        cai_start_vertex_offsets_ptr = \
            start_vertex_offsets.__cuda_array_interface__['data'][0]


    cdef cugraph_type_erased_device_array_view_t* start_vertex_list_ptr = \
        cugraph_type_erased_device_array_view_create(
            <void*>cai_start_ptr,
            len(start_vertex_list),
            get_c_type_from_numpy_type(start_vertex_list.dtype))


    cdef cugraph_type_erased_device_array_view_t* start_vertex_offsets_ptr = <cugraph_type_erased_device_array_view_t*>NULL
    if start_vertex_offsets is not None:
        start_vertex_offsets_ptr = \
            cugraph_type_erased_device_array_view_create(
                <void*>cai_start_vertex_offsets_ptr,
                len(start_vertex_offsets),
                get_c_type_from_numpy_type(start_vertex_offsets.dtype)
            )

    cdef cugraph_type_erased_device_array_view_t* label_offsets_ptr = <cugraph_type_erased_device_array_view_t*>NULL
    if retain_seeds:
        if start_vertex_offsets is None:
            raise ValueError("Must provide label offsets if retain_seeds is True")

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

    cdef cugraph_compression_type_t compression_behavior_e
    if compression is None or compression == 'COO':
        compression_behavior_e = cugraph_compression_type_t.COO
    elif compression == 'CSR':
        compression_behavior_e = cugraph_compression_type_t.CSR
    elif compression == 'CSC':
        compression_behavior_e = cugraph_compression_type_t.CSC
    elif compression == 'DCSR':
        compression_behavior_e = cugraph_compression_type_t.DCSR
    elif compression == 'DCSC':
        compression_behavior_e = cugraph_compression_type_t.DCSC
    else:
        raise ValueError(
            f'Invalid option {compression}'
            ' for compression type'
        )

    cdef cugraph_sampling_options_t* sampling_options
    error_code = cugraph_sampling_options_create(&sampling_options, &error_ptr)
    assert_success(error_code, error_ptr, "cugraph_sampling_options_create")

    cugraph_sampling_set_with_replacement(sampling_options, with_replacement)
    cugraph_sampling_set_return_hops(sampling_options, c_return_hops)
    cugraph_sampling_set_dedupe_sources(sampling_options, c_deduplicate_sources)
    cugraph_sampling_set_prior_sources_behavior(sampling_options, prior_sources_behavior_e)
    cugraph_sampling_set_renumber_results(sampling_options, c_renumber)
    cugraph_sampling_set_compression_type(sampling_options, compression_behavior_e)
    cugraph_sampling_set_compress_per_hop(sampling_options, c_compress_per_hop)
    cugraph_sampling_set_retain_seeds(sampling_options, retain_seeds)

    error_code = cugraph_heterogeneous_uniform_neighbor_sample(
        c_resource_handle_ptr,
        rng_state_ptr,
        c_graph_ptr,
        start_vertex_list_ptr,
        start_vertex_offsets_ptr,
        fan_out_ptr,
        num_edge_types,
        sampling_options,
        do_expensive_check,
        &result_ptr,
        &error_ptr)
    assert_success(error_code, error_ptr, "cugraph_heterogeneous_uniform_neighbor_sample")

    # Free the sampling options
    cugraph_sampling_options_free(sampling_options)

    # Free the two input arrays that are no longer needed.
    cugraph_type_erased_device_array_view_free(start_vertex_list_ptr)
    cugraph_type_erased_host_array_view_free(fan_out_ptr)

    if start_vertex_offsets is not None:
        cugraph_type_erased_device_array_view_free(start_vertex_offsets_ptr)

    # Have the SamplingResult instance assume ownership of the result data.
    result = SamplingResult()
    result.set_ptr(result_ptr)

    # Get cupy "views" of the individual arrays to return. These each increment
    # the refcount on the SamplingResult instance which will keep the data alive
    # until all references are removed and the GC runs.
    # TODO Return everything that isn't null in release 23.12
    if with_edge_properties:
        cupy_majors = result.get_majors()
        cupy_major_offsets = result.get_major_offsets()
        cupy_minors = result.get_minors()
        cupy_edge_weights = result.get_edge_weights()
        cupy_edge_ids = result.get_edge_ids()
        cupy_edge_types = result.get_edge_types()
        cupy_batch_ids = result.get_batch_ids()
        cupy_label_hop_offsets = result.get_label_hop_offsets()

        if renumber:
            cupy_renumber_map = result.get_renumber_map()
            cupy_renumber_map_offsets = result.get_renumber_map_offsets()
            # TODO drop the placeholder for hop ids in release 23.12
            if return_dict:
                return {
                    'major_offsets': cupy_major_offsets,
                    'majors': cupy_majors,
                    'minors': cupy_minors,
                    'weight': cupy_edge_weights,
                    'edge_id': cupy_edge_ids,
                    'edge_type': cupy_edge_types,
                    'batch_id': cupy_batch_ids,
                    'label_hop_offsets': cupy_label_hop_offsets,
                    'hop_id': None,
                    'renumber_map': cupy_renumber_map,
                    'renumber_map_offsets': cupy_renumber_map_offsets
                }
            else:
                cupy_majors = cupy_major_offsets if cupy_majors is None else cupy_majors
                return (cupy_majors, cupy_minors, cupy_edge_weights, cupy_edge_ids, cupy_edge_types, cupy_batch_ids, cupy_label_hop_offsets, None, cupy_renumber_map, cupy_renumber_map_offsets)
        else:
            cupy_hop_ids = result.get_hop_ids() # FIXME remove this
            if return_dict:
                return {
                    'major_offsets': cupy_major_offsets,
                    'majors': cupy_majors,
                    'minors': cupy_minors,
                    'weight': cupy_edge_weights,
                    'edge_id': cupy_edge_ids,
                    'edge_type': cupy_edge_types,
                    'batch_id': cupy_batch_ids,
                    'label_hop_offsets': cupy_label_hop_offsets,
                    'hop_id': cupy_hop_ids,
                }
            else:
                cupy_majors = cupy_major_offsets if cupy_majors is None else cupy_majors
                return (cupy_majors, cupy_minors, cupy_edge_weights, cupy_edge_ids, cupy_edge_types, cupy_batch_ids, cupy_label_hop_offsets, cupy_hop_ids)

    else:
        # TODO this is deprecated, remove it in release 23.12
        warnings.warn(
            "Calling heterogeneous_uniform_neighbor_sample with the 'with_edge_properties' argument is deprecated."
            " Starting in release 23.12, this argument will be removed in favor of behaving like the "
            "with_edge_properties=True option, returning whatever properties are in the graph.",
            FutureWarning,
        )

        cupy_sources = result.get_sources()
        cupy_destinations = result.get_destinations()
        cupy_indices = result.get_indices()

        if return_dict:
            return {
                'sources': cupy_sources,
                'destinations': cupy_destinations,
                'indices': cupy_indices
            }
        else:
            return (cupy_sources, cupy_destinations, cupy_indices)
