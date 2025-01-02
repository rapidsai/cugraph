# Copyright (c) 2022-2025, NVIDIA CORPORATION.
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
from pylibcugraph._cugraph_c.types cimport (
    bool_t,
    SIZE_T
)
from pylibcugraph._cugraph_c.resource_handle cimport (
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
                                          starting_vertex_label_offsets,
                                          vertex_type_offsets,
                                          h_fan_out,
                                          *,
                                          num_edge_types,
                                          bool_t with_replacement,
                                          bool_t do_expensive_check,
                                          prior_sources_behavior=None,
                                          deduplicate_sources=False,
                                          return_hops=False,
                                          renumber=False,
                                          retain_seeds=False,
                                          compression='COO',
                                          compress_per_hop=False,
                                          random_state=None):
    """
    Performs uniform neighborhood sampling, which samples nodes from
    a graph based on the current node's neighbors, with a corresponding fan_out
    value at each hop. The edges are sampled uniformly. Heterogeneous
    neighborhood sampling translates to more than 1 edge types.

    Parameters
    ----------
    resource_handle: ResourceHandle
        Handle to the underlying device and host resources needed for
        referencing data and running algorithms.

    input_graph : SGGraph or MGGraph
        The input graph, for either Single or Multi-GPU operations.

    start_vertex_list: device array type
        Device array containing the list of starting vertices for sampling.

    starting_vertex_label_offsets: device array type (Optional)
        Offsets of each label within the start vertex list. Expanding
        'starting_vertex_label_offsets' must lead to an array of
        len(start_vertex_list)

    vertex_type_offsets: device array type (Optional)
        Offsets for each vertex type in the graph.

    h_fan_out: numpy array type
        Device array containing the branching out (fan-out) degrees per
        starting vertex for each hop level. The fanout value at each hop for each
        edge type is given by the relationship
        h_fanout[x*num_edge_types + edge_type_id]

        The sampling method can use different fan_out values for each edge type
        which is not the case for homogeneous neighborhood sampling (both biased
        and uniform).

    num_edge_types: int
        Number of edge types where a value of 1 translates to homogeneous neighbor
        sample whereas a value greater than 1 translates to heterogeneous neighbor
        sample.

    with_replacement: bool
        If true, sampling procedure is done with replacement (the same vertex
        can be selected multiple times in the same step).

    do_expensive_check: bool
        If True, performs more extensive tests on the inputs to ensure
        validitity, at the expense of increased run time.

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

    Examples
    --------
    >>> import pylibcugraph, cupy, numpy
    >>> srcs = cupy.asarray([0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5],
    ...                     dtype=numpy.int32)
    >>> dsts = cupy.asarray([1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4],
    ...                     dtype=numpy.int32)
    >>> weights = cupy.asarray([0.1, 2.1, 1.1, 5.1, 3.1, 4.1, 7.2, 3.2, 0.1, 2.1,
    ...                         1.1, 5.1, 3.1,  4.1, 7.2, 3.2], dtype=numpy.float32)
    >>> edge_types = cupy.asarray([0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1],
    ...                            dtype=numpy.int32)
    >>> start_vertices = cupy.asarray([2, 5, 1]).astype(numpy.int32)
    >>> num_edge_types = 2
    >>> starting_vertex_label_offsets = cupy.asarray([0, 2, 3])
    >>> h_fan_out = numpy.array([2]).astype(numpy.int32)
    >>> resource_handle = pylibcugraph.ResourceHandle()
    >>> graph_props = pylibcugraph.GraphProperties(
    ...     is_symmetric=False, is_multigraph=False)
    >>> G = pylibcugraph.SGGraph(
    ...     resource_handle, graph_props, srcs, dsts, weight_array=weights,
    ...     store_transposed=False, renumber=False, do_expensive_check=False)
    >>> sampling_results = pylibcugraph.heterogeneous_uniform_neighbor_sample(
    ...         resource_handle, G, start_vertices, starting_vertex_label_offsets,
    ...         h_fan_out, num_edge_types, False, True)
    >>> sampling_results
    {'majors': array([2, 2, 2, 5, 5, 1, 1, 1, 1], dtype=int32),
     'minors': array([0, 1, 3, 3, 4, 0, 2, 3, 4], dtype=int32),
     'weight': array([5.1, 3.1, 4.1, 7.2, 3.2, 0.1, 3.1, 2.1, 1.1], dtype=float32)}

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
    assert_CAI_type(starting_vertex_label_offsets, "starting_vertex_label_offsets", True)
    assert_CAI_type(vertex_type_offsets, "vertex_type_offsets", True)

    assert_AI_type(h_fan_out, "h_fan_out")

    if starting_vertex_label_offsets is not None:
        if starting_vertex_label_offsets[-1] != len(start_vertex_list):
            raise ValueError(
                "'starting_vertex_label_offsets' and 'start_vertex_list' must be proportional")

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


    cdef cugraph_type_erased_device_array_view_t* starting_vertex_label_offsets_ptr = <cugraph_type_erased_device_array_view_t*>NULL
    if starting_vertex_label_offsets is not None:
        starting_vertex_label_offsets_ptr = \
            cugraph_type_erased_device_array_view_create(
                <void*>cai_starting_vertex_label_offsets_ptr,
                len(starting_vertex_label_offsets),
                SIZE_T
            )

    cdef cugraph_type_erased_device_array_view_t* vertex_type_offsets_ptr = <cugraph_type_erased_device_array_view_t*>NULL
    if vertex_type_offsets is not None:
        vertex_type_offsets_ptr = \
            cugraph_type_erased_device_array_view_create(
                <void*>cai_vertex_type_offsets_ptr,
                len(vertex_type_offsets),
                SIZE_T
            )

    cdef cugraph_type_erased_device_array_view_t* label_offsets_ptr = <cugraph_type_erased_device_array_view_t*>NULL
    if retain_seeds:
        if starting_vertex_label_offsets is None:
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
        starting_vertex_label_offsets_ptr,
        vertex_type_offsets_ptr,
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

    if starting_vertex_label_offsets is not None:
        cugraph_type_erased_device_array_view_free(starting_vertex_label_offsets_ptr)

    # Have the SamplingResult instance assume ownership of the result data.
    result = SamplingResult()
    result.set_ptr(result_ptr)

    # Get cupy "views" of the individual arrays to return. These each increment
    # the refcount on the SamplingResult instance which will keep the data alive
    # until all references are removed and the GC runs.

    cupy_majors = result.get_majors()
    cupy_major_offsets = result.get_major_offsets()
    cupy_minors = result.get_minors()
    cupy_edge_weights = result.get_edge_weights()
    cupy_edge_ids = result.get_edge_ids()
    cupy_edge_types = result.get_edge_types()
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
            'batch_id': cupy_batch_ids,
            'label_hop_offsets': cupy_label_hop_offsets,
            'label_type_hop_offsets': cupy_label_type_hop_offsets,
            'hop_id': None,
            'renumber_map': cupy_renumber_map,
            'renumber_map_offsets': cupy_renumber_map_offsets,
            'edge_renumber_map' : cupy_edge_renumber_map,
            'edge_renumber_map_offsets' : cupy_edge_renumber_map_offsets
        }

    else:
        sampling_results = {
            'major_offsets': cupy_major_offsets,
            'majors': cupy_majors,
            'minors': cupy_minors,
            'weight': cupy_edge_weights,
            'edge_id': cupy_edge_ids,
            'edge_type': cupy_edge_types,
            'batch_id': cupy_batch_ids,
            'label_hop_offsets': cupy_label_hop_offsets,
            'label_type_hop_offsets': cupy_label_type_hop_offsets,
        }

    # Return everything that isn't null
    return {k: v for k, v in sampling_results.items() if v is not None}
