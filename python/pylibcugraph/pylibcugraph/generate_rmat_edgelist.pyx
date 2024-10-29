# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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


from pylibcugraph._cugraph_c.types cimport (
    bool_t,
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
)
from pylibcugraph._cugraph_c.coo cimport (
    cugraph_coo_t,
    cugraph_coo_get_sources,
    cugraph_coo_get_destinations,
    cugraph_coo_get_edge_weights,
    cugraph_coo_get_edge_id,
    cugraph_coo_get_edge_type,
    cugraph_coo_free,
)
from pylibcugraph._cugraph_c.graph_generators cimport (
    cugraph_generate_rmat_edgelist,
    cugraph_generate_edge_weights,
    cugraph_generate_edge_ids,
    cugraph_generate_edge_types,
)
from pylibcugraph.resource_handle cimport (
    ResourceHandle,
)
from pylibcugraph.utils cimport (
    assert_success,
    copy_to_cupy_array,
    get_c_type_from_numpy_type,
)
from pylibcugraph._cugraph_c.random cimport (
    cugraph_rng_state_t
)
from pylibcugraph.random cimport (
    CuGraphRandomState
)


def generate_rmat_edgelist(ResourceHandle resource_handle,
                           random_state,
                           size_t scale,
                           size_t num_edges,
                           double a,
                           double b,
                           double c,
                           bool_t clip_and_flip,
                           bool_t scramble_vertex_ids,
                           bool_t include_edge_weights,
                           minimum_weight,
                           maximum_weight,
                           dtype,
                           bool_t include_edge_ids,
                           bool_t include_edge_types,
                           min_edge_type_value,
                           max_edge_type_value,
                           bool_t multi_gpu,
                           ):
    """
    Generate RMAT edge list

    Parameters
    ----------
    resource_handle : ResourceHandle
        Handle to the underlying device resources needed for referencing data
        and running algorithms.

    random_state : int , optional
        Random state to use when generating samples. Optional argument,
        defaults to a hash of process id, time, and hostname.
        (See pylibcugraph.random.CuGraphRandomState)

    scale : size_t
        Scale factor to set the number of vertices in the graph Vertex IDs have
        values in [0, V), where V = 1 << 'scale'

    num_edges : size_t
        Number of edges to generate

    a : double
        Probability of the edge being in the first partition
        The Graph 500 spec sets this value to 0.57

    b : double
        Probability of the edge being in the second partition
        The Graph 500 spec sets this value to 0.19

    c : double
        Probability of the edge being in the third partition
        The Graph 500 spec sets this value to 0.19

    clip_and_flip : bool
        Flag controlling whether to generate edges only in the lower triangular
        part (including the diagonal) of the graph adjacency matrix
        (if set to 'true') or not (if set to 'false).

    scramble_vertex_ids : bool
        Flag controlling whether to scramble vertex ID bits (if set to `true`)
        or not (if set to `false`); scrambling vertex ID bits breaks
        correlation between vertex ID values and vertex degrees.

    include_edge_weights : bool
        Flag controlling whether to generate edges with weights
        (if set to 'true') or not (if set to 'false').

    minimum_weight : double
        Minimum weight value to generate (if 'include_edge_weights' is 'true')

    maximum_weight : double
        Maximum weight value to generate (if 'include_edge_weights' is 'true')

    dtype : string
        The type of weight to generate ("FLOAT32" or "FLOAT64"), ignored unless
        include_weights is true

    include_edge_ids : bool
        Flag controlling whether to generate edges with ids
        (if set to 'true') or not (if set to 'false').

    include_edge_types : bool
        Flag controlling whether to generate edges with types
        (if set to 'true') or not (if set to 'false').

    min_edge_type_value : int
        Minimum edge type to generate if 'include_edge_types' is 'true'
        otherwise, this parameter is ignored.

    max_edge_type_value : int
        Maximum edge type to generate if 'include_edge_types' is 'true'
        otherwise, this paramter is ignored.

    multi_gpu : bool
        Flag if the COO is being created on multiple GPUs


    Returns
    -------
    return a tuple containing the sources and destinations with their corresponding
    weights, ids and types if the flags 'include_edge_weights', 'include_edge_ids'
    and 'include_edge_types' are respectively set to 'true'
    """

    cdef cugraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr

    cdef cugraph_coo_t* result_coo_ptr
    cdef cugraph_error_code_t error_code
    cdef cugraph_error_t* error_ptr

    cg_rng_state = CuGraphRandomState(resource_handle, random_state)

    cdef cugraph_rng_state_t* rng_state_ptr = \
        cg_rng_state.rng_state_ptr

    error_code = cugraph_generate_rmat_edgelist(c_resource_handle_ptr,
                                                rng_state_ptr,
                                                scale,
                                                num_edges,
                                                a,
                                                b,
                                                c,
                                                clip_and_flip,
                                                scramble_vertex_ids,
                                                &result_coo_ptr,
                                                &error_ptr)
    assert_success(error_code, error_ptr, "generate_rmat_edgelist")

    cdef cugraph_type_erased_device_array_view_t* \
        sources_view_ptr = cugraph_coo_get_sources(result_coo_ptr)

    cdef cugraph_type_erased_device_array_view_t* \
        destinations_view_ptr = cugraph_coo_get_destinations(result_coo_ptr)

    cdef cugraph_type_erased_device_array_view_t* edge_weights_view_ptr

    cupy_edge_weights = None
    cupy_edge_ids = None
    cupy_edge_types = None

    if include_edge_weights:
        dtype = get_c_type_from_numpy_type(dtype)
        error_code =  cugraph_generate_edge_weights(c_resource_handle_ptr,
                                                    rng_state_ptr,
                                                    result_coo_ptr,
                                                    dtype,
                                                    minimum_weight,
                                                    maximum_weight,
                                                    &error_ptr)
        assert_success(error_code, error_ptr, "generate_edge_weights")

        edge_weights_view_ptr = cugraph_coo_get_edge_weights(result_coo_ptr)
        cupy_edge_weights = copy_to_cupy_array(c_resource_handle_ptr, edge_weights_view_ptr)


    if include_edge_ids:
        error_code = cugraph_generate_edge_ids(c_resource_handle_ptr,
                                               result_coo_ptr,
                                               multi_gpu,
                                               &error_ptr)

        assert_success(error_code, error_ptr, "generate_edge_ids")

        edge_ids_view_ptr = cugraph_coo_get_edge_id(result_coo_ptr)
        cupy_edge_ids = copy_to_cupy_array(c_resource_handle_ptr, edge_ids_view_ptr)

    if include_edge_types:
        error_code = cugraph_generate_edge_types(c_resource_handle_ptr,
                                                 rng_state_ptr,
                                                 result_coo_ptr,
                                                 min_edge_type_value,
                                                 max_edge_type_value,
                                                 &error_ptr)

        assert_success(error_code, error_ptr, "generate_edge_types")

        edge_type_view_ptr = cugraph_coo_get_edge_type(result_coo_ptr)
        cupy_edge_types = copy_to_cupy_array(c_resource_handle_ptr, edge_type_view_ptr)





    cupy_sources = copy_to_cupy_array(c_resource_handle_ptr, sources_view_ptr)
    cupy_destinations = copy_to_cupy_array(c_resource_handle_ptr, destinations_view_ptr)

    cugraph_coo_free(result_coo_ptr)

    return cupy_sources, cupy_destinations, cupy_edge_weights, cupy_edge_ids, cupy_edge_types
