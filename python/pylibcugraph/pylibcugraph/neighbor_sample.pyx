# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Have cython use python 3 syntax
# cython: language_level = 3

from libc.stdint cimport uintptr_t
from pylibcugraph._cugraph_c.types cimport (
    bool_t,
    SIZE_T
)
from pylibcugraph._cugraph_c.resource_handle cimport (
    cugraph_resource_handle_t,
)
from pylibcugraph._cugraph_c.properties cimport (
    cugraph_edge_property_view_t,
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
    cugraph_sampling_set_temporal_sampling_comparison,
    cugraph_temporal_sampling_comparison_t,
    cugraph_sampling_set_disjoint_sampling,
    cugraph_sampling_set_use_edge_weights_as_biases,
    cugraph_sampling_set_neighbor_selection,
    cugraph_neighbor_selection_t,
)
from pylibcugraph._cugraph_c.sampling_algorithms cimport (
    cugraph_neighbor_sample,
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


def neighbor_sample(ResourceHandle resource_handle,
                    _GPUGraph input_graph,
                    start_vertex_list,
                    h_fan_out,
                    *,
                    starting_vertex_start_times=None,
                    starting_vertex_end_times=None,
                    starting_vertex_label_offsets=None,
                    vertex_type_offsets=None,
                    num_edge_types=1,
                    bool_t with_replacement,
                    bool_t do_expensive_check,
                    prior_sources_behavior=None,
                    deduplicate_sources=False,
                    disjoint_sampling=False,
                    return_hops=False,
                    renumber=False,
                    retain_seeds=False,
                    compression='COO',
                    compress_per_hop=False,
                    random_state=None,
                    temporal_sampling_comparison=None,
                    use_edge_weights_as_biases=False,
                    neighbor_selection='random'):
    """
    Unified neighborhood sampling for homogeneous/heterogeneous and
    temporal/non-temporal graphs.

    Parameters
    ----------
    resource_handle: ResourceHandle
        Handle for device and host resources.

    input_graph : SGGraph or MGGraph
        The input graph.

    start_vertex_list: device array type
        Starting vertices for sampling.

    h_fan_out: numpy array type
        Fan-out degrees per hop (and per edge type when heterogeneous).

    starting_vertex_start_times: device array type (Optional)
        Per-seed lower time-window bounds. Requires temporal sampling.

    starting_vertex_end_times: device array type (Optional)
        Per-seed upper time-window bounds. Requires temporal sampling.

    starting_vertex_label_offsets: device array type (Optional)
        Offsets of each label within the start vertex list.

    vertex_type_offsets: device array type (Optional)
        Offsets for each vertex type; used for heterogeneous output processing.

    num_edge_types: int (Optional)
        Number of edge types. Use 1 for homogeneous sampling. Defaults to 1.

    with_replacement: bool
        If true, sampling is done with replacement.

    do_expensive_check: bool
        If True, performs more extensive input validation.

    prior_sources_behavior: str (Optional)
        Options are "carryover" and "exclude".

    deduplicate_sources: bool (Optional)
        If True, deduplicate sources before sampling. Defaults to False.

    disjoint_sampling: bool (Optional)
        If True, enables disjoint sampling. Required for temporal sampling.
        Defaults to False.

    return_hops: bool (Optional)
        If True, include hop information in the result. Defaults to False.

    renumber: bool (Optional)
        If True, renumber results on a per-batch basis. Defaults to False.

    retain_seeds: bool (Optional)
        If True, retain original seeds even without outgoing neighbors.
        Defaults to False.

    compression: str (Optional)
        Options: COO (default), CSR, CSC, DCSR, DCSC.

    compress_per_hop: bool (Optional)
        If True, create a compressed edgelist per hop. Defaults to False.

    random_state: int (Optional)
        Random state for sampling.

    temporal_sampling_comparison: str (Optional)
        If None (default), sampling is non-temporal. Otherwise one of
        'strictly_increasing', 'strictly_decreasing',
        'monotonically_increasing', 'monotonically_decreasing',
        or 'fixed_window'.

    use_edge_weights_as_biases: bool (Optional)
        If True and no separate edge-bias property is provided, use graph
        edge weights as biases. Explicit edge-bias arrays are not yet
        supported from pylibcugraph. Defaults to False (uniform).

    neighbor_selection: str (Optional)
        'random' (default) or 'last'. 'last' is not yet implemented.

    Returns
    -------
    dict
        Device arrays for the sampling result. Keys with null values are omitted.
    """
    cdef cugraph_resource_handle_t* c_resource_handle_ptr = (
        resource_handle.c_resource_handle_ptr
    )

    cdef cugraph_graph_t* c_graph_ptr = input_graph.c_graph_ptr
    cdef cugraph_type_erased_host_array_view_t* fan_out_ptr = (
        <cugraph_type_erased_host_array_view_t*>NULL
    )

    cdef bool_t c_deduplicate_sources = deduplicate_sources
    cdef bool_t c_return_hops = return_hops
    cdef bool_t c_renumber = renumber
    cdef bool_t c_compress_per_hop = compress_per_hop
    cdef bool_t c_use_edge_weights_as_biases = use_edge_weights_as_biases

    cdef cugraph_error_code_t error_code
    cdef cugraph_error_t* error_ptr
    cdef uintptr_t ai_fan_out_ptr

    assert_CAI_type(start_vertex_list, "start_vertex_list")
    assert_CAI_type(starting_vertex_start_times, "starting_vertex_start_times", True)
    assert_CAI_type(starting_vertex_end_times, "starting_vertex_end_times", True)
    assert_CAI_type(starting_vertex_label_offsets, "starting_vertex_label_offsets", True)
    assert_CAI_type(vertex_type_offsets, "vertex_type_offsets", True)
    assert_AI_type(h_fan_out, "h_fan_out")

    if starting_vertex_label_offsets is not None:
        last_elmnt_idx = len(starting_vertex_label_offsets) - 1
        if starting_vertex_label_offsets[last_elmnt_idx] != len(start_vertex_list):
            raise ValueError(
                "'starting_vertex_label_offsets' and 'start_vertex_list' must be proportional")

    if retain_seeds and starting_vertex_label_offsets is None:
        raise ValueError("Must provide label offsets if retain_seeds is True")

    if num_edge_types < 1:
        raise ValueError("num_edge_types must be at least 1")

    ai_fan_out_ptr = h_fan_out.__array_interface__["data"][0]
    fan_out_ptr = cugraph_type_erased_host_array_view_create(
        <void*>ai_fan_out_ptr,
        len(h_fan_out),
        get_c_type_from_numpy_type(h_fan_out.dtype))

    cdef cugraph_sample_result_t* result_ptr

    cdef uintptr_t cai_start_ptr = \
        start_vertex_list.__cuda_array_interface__["data"][0]

    cdef uintptr_t cai_starting_vertex_start_times_ptr
    if starting_vertex_start_times is not None:
        cai_starting_vertex_start_times_ptr = \
            starting_vertex_start_times.__cuda_array_interface__['data'][0]

    cdef uintptr_t cai_starting_vertex_end_times_ptr
    if starting_vertex_end_times is not None:
        cai_starting_vertex_end_times_ptr = \
            starting_vertex_end_times.__cuda_array_interface__['data'][0]

    cdef uintptr_t cai_starting_vertex_label_offsets_ptr
    if starting_vertex_label_offsets is not None:
        cai_starting_vertex_label_offsets_ptr = \
            starting_vertex_label_offsets.__cuda_array_interface__['data'][0]

    cdef uintptr_t cai_vertex_type_offsets_ptr
    if vertex_type_offsets is not None:
        cai_vertex_type_offsets_ptr = \
            vertex_type_offsets.__cuda_array_interface__['data'][0]

    cdef cugraph_type_erased_device_array_view_t* start_vertex_list_ptr = \
        cugraph_type_erased_device_array_view_create(
            <void*>cai_start_ptr,
            len(start_vertex_list),
            get_c_type_from_numpy_type(start_vertex_list.dtype))

    cdef cugraph_type_erased_device_array_view_t* starting_vertex_start_times_ptr = (
        <cugraph_type_erased_device_array_view_t*>NULL
    )
    if starting_vertex_start_times is not None:
        starting_vertex_start_times_ptr = \
            cugraph_type_erased_device_array_view_create(
                <void*>cai_starting_vertex_start_times_ptr,
                len(starting_vertex_start_times),
                get_c_type_from_numpy_type(starting_vertex_start_times.dtype)
            )

    cdef cugraph_type_erased_device_array_view_t* starting_vertex_end_times_ptr = (
        <cugraph_type_erased_device_array_view_t*>NULL
    )
    if starting_vertex_end_times is not None:
        starting_vertex_end_times_ptr = \
            cugraph_type_erased_device_array_view_create(
                <void*>cai_starting_vertex_end_times_ptr,
                len(starting_vertex_end_times),
                get_c_type_from_numpy_type(starting_vertex_end_times.dtype)
            )

    cdef cugraph_type_erased_device_array_view_t* starting_vertex_label_offsets_ptr = (
        <cugraph_type_erased_device_array_view_t*>NULL
    )
    if starting_vertex_label_offsets is not None:
        starting_vertex_label_offsets_ptr = \
            cugraph_type_erased_device_array_view_create(
                <void*>cai_starting_vertex_label_offsets_ptr,
                len(starting_vertex_label_offsets),
                SIZE_T
            )

    cdef cugraph_type_erased_device_array_view_t* vertex_type_offsets_ptr = (
        <cugraph_type_erased_device_array_view_t*>NULL
    )
    if vertex_type_offsets is not None:
        vertex_type_offsets_ptr = \
            cugraph_type_erased_device_array_view_create(
                <void*>cai_vertex_type_offsets_ptr,
                len(vertex_type_offsets),
                get_c_type_from_numpy_type(vertex_type_offsets.dtype)
            )

    cg_rng_state = CuGraphRandomState(resource_handle, random_state)
    cdef cugraph_rng_state_t* rng_state_ptr = cg_rng_state.rng_state_ptr

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

    cdef cugraph_neighbor_selection_t neighbor_selection_e
    if neighbor_selection is None or neighbor_selection == 'random':
        neighbor_selection_e = cugraph_neighbor_selection_t.CUGRAPH_NEIGHBOR_SELECTION_RANDOM
    elif neighbor_selection == 'last':
        neighbor_selection_e = cugraph_neighbor_selection_t.CUGRAPH_NEIGHBOR_SELECTION_LAST
    else:
        raise ValueError(f'Invalid option {neighbor_selection} for neighbor selection')

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
    cugraph_sampling_set_disjoint_sampling(sampling_options, disjoint_sampling)
    cugraph_sampling_set_use_edge_weights_as_biases(
        sampling_options, c_use_edge_weights_as_biases)
    cugraph_sampling_set_neighbor_selection(sampling_options, neighbor_selection_e)

    cdef cugraph_temporal_sampling_comparison_t temporal_sampling_comparison_e
    if temporal_sampling_comparison is not None:
        if temporal_sampling_comparison == 'strictly_increasing':
            temporal_sampling_comparison_e = (
                cugraph_temporal_sampling_comparison_t.STRICTLY_INCREASING)
        elif temporal_sampling_comparison == 'strictly_decreasing':
            temporal_sampling_comparison_e = (
                cugraph_temporal_sampling_comparison_t.STRICTLY_DECREASING)
        elif temporal_sampling_comparison == 'monotonically_increasing':
            temporal_sampling_comparison_e = (
                cugraph_temporal_sampling_comparison_t.MONOTONICALLY_INCREASING)
        elif temporal_sampling_comparison == 'monotonically_decreasing':
            temporal_sampling_comparison_e = (
                cugraph_temporal_sampling_comparison_t.MONOTONICALLY_DECREASING)
        elif temporal_sampling_comparison == 'fixed_window':
            temporal_sampling_comparison_e = (
                cugraph_temporal_sampling_comparison_t.FIXED_WINDOW)
        elif temporal_sampling_comparison == 'last':
            raise NotImplementedError(
                'The "last" temporal comparison type is currently unsupported.')
        else:
            raise ValueError(
                f'Invalid option {temporal_sampling_comparison} '
                'for temporal sampling comparison')
        cugraph_sampling_set_temporal_sampling_comparison(
            sampling_options, temporal_sampling_comparison_e)

    # Explicit edge-bias arrays are not yet supported from pylibcugraph; pass NULL
    # and optionally fall back to graph edge weights via use_edge_weights_as_biases.
    error_code = cugraph_neighbor_sample(
        c_resource_handle_ptr,
        rng_state_ptr,
        c_graph_ptr,
        <cugraph_edge_property_view_t*>NULL,
        start_vertex_list_ptr,
        starting_vertex_start_times_ptr,
        starting_vertex_end_times_ptr,
        starting_vertex_label_offsets_ptr,
        vertex_type_offsets_ptr,
        fan_out_ptr,
        num_edge_types,
        sampling_options,
        do_expensive_check,
        &result_ptr,
        &error_ptr)
    assert_success(error_code, error_ptr, "cugraph_neighbor_sample")

    cugraph_sampling_options_free(sampling_options)

    cugraph_type_erased_device_array_view_free(start_vertex_list_ptr)
    if starting_vertex_start_times is not None:
        cugraph_type_erased_device_array_view_free(starting_vertex_start_times_ptr)
    if starting_vertex_end_times is not None:
        cugraph_type_erased_device_array_view_free(starting_vertex_end_times_ptr)
    if starting_vertex_label_offsets is not None:
        cugraph_type_erased_device_array_view_free(starting_vertex_label_offsets_ptr)
    if vertex_type_offsets is not None:
        cugraph_type_erased_device_array_view_free(vertex_type_offsets_ptr)
    cugraph_type_erased_host_array_view_free(fan_out_ptr)

    result = SamplingResult()
    result.set_ptr(result_ptr)

    cupy_majors = result.get_majors()
    cupy_major_offsets = result.get_major_offsets()
    cupy_minors = result.get_minors()
    cupy_edge_weights = result.get_edge_weights()
    cupy_edge_ids = result.get_edge_ids()
    cupy_edge_types = result.get_edge_types()
    cupy_edge_start_time = result.get_edge_start_time()
    cupy_edge_end_time = result.get_edge_end_time()
    cupy_batch_ids = result.get_batch_ids()
    cupy_label_hop_offsets = result.get_label_hop_offsets()
    cupy_label_type_hop_offsets = result.get_label_type_hop_offsets()

    if renumber:
        cupy_renumber_map = result.get_renumber_map()
        cupy_renumber_map_offsets = result.get_renumber_map_offsets()
        cupy_edge_renumber_map = result.get_edge_renumber_map()
        cupy_edge_renumber_map_offsets = result.get_edge_renumber_map_offsets()

        sampling_results = {
            'major_offsets': cupy_major_offsets,
            'majors': cupy_majors,
            'minors': cupy_minors,
            'weight': cupy_edge_weights,
            'edge_id': cupy_edge_ids,
            'edge_type': cupy_edge_types,
            'edge_start_time': cupy_edge_start_time,
            'edge_end_time': cupy_edge_end_time,
            'batch_id': cupy_batch_ids,
            'label_hop_offsets': cupy_label_hop_offsets,
            'label_type_hop_offsets': cupy_label_type_hop_offsets,
            'renumber_map': cupy_renumber_map,
            'renumber_map_offsets': cupy_renumber_map_offsets,
            'edge_renumber_map': cupy_edge_renumber_map,
            'edge_renumber_map_offsets': cupy_edge_renumber_map_offsets,
        }
    else:
        sampling_results = {
            'major_offsets': cupy_major_offsets,
            'majors': cupy_majors,
            'minors': cupy_minors,
            'weight': cupy_edge_weights,
            'edge_id': cupy_edge_ids,
            'edge_type': cupy_edge_types,
            'edge_start_time': cupy_edge_start_time,
            'edge_end_time': cupy_edge_end_time,
            'batch_id': cupy_batch_ids,
            'label_hop_offsets': cupy_label_hop_offsets,
            'label_type_hop_offsets': cupy_label_type_hop_offsets,
        }

    return {k: v for k, v in sampling_results.items() if v is not None}
