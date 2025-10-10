# Copyright (c) 2025, NVIDIA CORPORATION.
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
from pylibcugraph._cugraph_c.layout_algorithms cimport (
    cugraph_force_atlas2,
    cugraph_layout_result_t,
    cugraph_layout_result_get_vertices,
    cugraph_layout_result_get_x_axis,
    cugraph_layout_result_get_y_axis,
    cugraph_layout_result_free
)
from pylibcugraph._cugraph_c.graph cimport (
    cugraph_graph_t,
)
from pylibcugraph.resource_handle cimport (
    ResourceHandle,
)
from pylibcugraph.graphs cimport (
    _GPUGraph,
)
from pylibcugraph.utils cimport (
    assert_success,
    copy_to_cupy_array,
    assert_CAI_type,
    create_cugraph_type_erased_device_array_view_from_py_obj
)
from pylibcugraph._cugraph_c.array cimport (
    cugraph_type_erased_device_array_t,
)
from pylibcugraph._cugraph_c.random cimport (
    cugraph_rng_state_t
)
from pylibcugraph.random cimport (
    CuGraphRandomState
)


def force_atlas2(ResourceHandle resource_handle,
                 random_state,
                 _GPUGraph graph,
                 int max_iter,
                 start_vertices,
                 x_start,
                 y_start,
                 bool_t outbound_attraction_distribution,
                 bool_t lin_log_mode,
                 bool_t prevent_overlapping,
                 vertex_radius_vertices,
                 vertex_radius_values,
                 double overlap_scaling_ratio,
                 double edge_weight_influence,
                 double jitter_tolerance,
                 bool_t barnes_hut_optimize,
                 double barnes_hut_theta,
                 double scaling_ratio,
                 bool_t strong_gravity_mode,
                 double gravity,
                 vertex_mobility_vertices,
                 vertex_mobility_values,
                 vertex_mass_vertices,
                 vertex_mass_values,
                 bool_t verbose,
                 bool_t do_expensive_check,
                ):
    """
    ForceAtlas2 is a continuous graph layout algorithm for handy network
    visualization.

    Parameters
    ----------
    resource_handle : ResourceHandle
        Handle to the underlying device resources needed for referencing data
        and running algorithms.

    random_state : int , optional
        Random state to use when generating samples. Optional argument,
        defaults to a hash of process id, time, and hostname.
        (See pylibcugraph.random.CuGraphRandomState)

    graph : SGGraph or MGGraph
        The input graph, for either Single or Multi-GPU operations.

    max_iter: int
        Maximum number of Katz Centrality iterations

    start_vertices : device array type, optional (default=None)
        Vertices of graph for x_start and y_start

    x_start : device array type, optional (default=None)
        Initial vertex positioning (x-axis)

    y_start : device array type, optional (default=None)
        Initial vertex positioning (y-axis)

    outbound_attraction_distribution : bool_t
        Distributes attraction along outbound edges
        Hubs attract less and thus are pushed to the borders.

    lin_log_mode : bool_t
        Switch Force Atlas model from lin-lin to lin-log.
        Makes clusters more tight.

    prevent_overlapping : bool_t
        Prevent nodes to overlap.

    vertex_radius_vertices : device array type, optional (default=None)
        Vertices of graph for vertex_radius_values.

    vertex_radius_values : device array type, optional (default=None)
        Radius of each vertex, used when prevent_overlapping is set.

    overlap_scaling_ratio : double
        When prevent_overlapping is set, scales the repulsion force
        between two nodes that are overlapping.

    edge_weight_influence : double
        How much influence you give to the edges weight.
        0 is “no influence” and 1 is “normal”.

    jitter_tolerance : double
        How much swinging you allow. Above 1 discouraged.
        Lower gives less speed and more precision.

    barnes_hut_optimize : bool_t
        Whether to use the Barnes Hut approximation or the slower exact version.

    barnes_hut_theta : double
        Float between 0 and 1. Tradeoff for speed (1) vs accuracy (0) for Barnes Hut only.

    scaling_ratio : double
        How much repulsion you want. More makes a more sparse graph.
        Switching from regular mode to LinLog mode needs a readjustment of the scaling parameter.

    strong_gravity_mode : bool_t
        Sets a force that attracts the nodes that are distant from the
        center more. It is so strong that it can sometimes dominate other forces.

    gravity : double
        Attracts nodes to the center. Prevents islands from drifting away.

    vertex_mobility_vertices : device array type, optional (default=None)
        Vertices of graph for vertex_mobility_values.

    vertex_mobility_values : device array type, optional (default=None)
        Mobility of each vertex, scaling its speed in each iteration.
        If not provided, all vertices will have a mobility of 1.0.

    vertex_mass_vertices : device array type, optional (default=None)
        Vertices of graph for vertex_mass_values.

    vertex_mass_values : device array type, optional (default=None)
        Mass of each vertex, which controls the attraction to other vertices.
        If not provided, the mass of each vertex will be its degree plus one.

    verbose : bool_t
        Output convergence info at each interation.

    do_expensive_check : bool_t
        A flag to run expensive checks for input arguments (if set to true)

    callback: # FIXME: NOT IMPLEMENTED YET
        intercept the internal state of positions while they are being trained.

    Returns
    -------
    return the position of each vertices

    Examples
    --------
    >>> import pylibcugraph, cupy, numpy
    >>> srcs = cupy.asarray([0, 1, 1, 2, 2, 2, 3, 3, 4], dtype=numpy.int32)
    >>> dsts = cupy.asarray([1, 3, 4, 0, 1, 3, 4, 5, 5], dtype=numpy.int32)
    >>> weights = cupy.asarray(
    ...     [0.1, 2.1, 1.1, 5.1, 3.1, 4.1, 7.2, 3.2, 6.1], dtype=numpy.float32)
    >>> resource_handle = pylibcugraph.ResourceHandle()
    >>> graph_props = pylibcugraph.GraphProperties(is_symmetric=False, is_multigraph=False)
    >>> G = pylibcugraph.SGGraph(
    ...     resource_handle, graph_props, srcs, dsts, weight_array=weights,
    ...     store_transposed=False, renumber=False, do_expensive_check=False)
    >>> (vertices, x_axis, y_axis) = pylibcugraph.force_atlas2(
    ...     resource_handle, 42, G, 500, None, None, None, True, False, False,
    ...     None, None, 100.0, 1.0, 1.0, False, 0.5, 2.0, False, 1.0, None, None,
    ...     None, None, False, False)
    >>> vertices
    [   0  1   2   3   4   5    ]
    >>> x_axis
    [-7.7015705   7.6763854  -2.7651896  -0.02446137  1.6971487  -1.2822613 ]
    >>> y_axis
    [ 23.276543     7.9067745   13.618961    -0.07172047  -6.8321953  -11.90544   ]


    """

    cdef cugraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef cugraph_graph_t* c_graph_ptr = graph.c_graph_ptr

    cdef cugraph_type_erased_device_array_t* pos_ptr

    assert_CAI_type(start_vertices, "start_vertices", True)

    cdef cugraph_type_erased_device_array_view_t* \
        start_vertices_view_ptr = \
            create_cugraph_type_erased_device_array_view_from_py_obj(
                start_vertices)

    assert_CAI_type(x_start, "x_start", True)

    cdef cugraph_type_erased_device_array_view_t* \
        x_start_view_ptr = \
            create_cugraph_type_erased_device_array_view_from_py_obj(
                x_start)

    assert_CAI_type(y_start, "y_start", True)

    cdef cugraph_type_erased_device_array_view_t* \
        y_start_view_ptr = \
            create_cugraph_type_erased_device_array_view_from_py_obj(
                y_start)

    assert_CAI_type(vertex_radius_vertices, "vertex_radius_vertices", True)

    cdef cugraph_type_erased_device_array_view_t* \
        vertex_radius_vertices_view_ptr = \
            create_cugraph_type_erased_device_array_view_from_py_obj(
                vertex_radius_vertices)

    assert_CAI_type(vertex_radius_values, "vertex_radius_values", True)

    cdef cugraph_type_erased_device_array_view_t* \
        vertex_radius_values_view_ptr = \
            create_cugraph_type_erased_device_array_view_from_py_obj(
                vertex_radius_values)

    assert_CAI_type(vertex_mobility_vertices, "vertex_mobility_vertices", True)

    cdef cugraph_type_erased_device_array_view_t* \
        vertex_mobility_vertices_view_ptr = \
            create_cugraph_type_erased_device_array_view_from_py_obj(
                vertex_mobility_vertices)

    assert_CAI_type(vertex_mobility_values, "vertex_mobility_values", True)

    cdef cugraph_type_erased_device_array_view_t* \
        vertex_mobility_values_view_ptr = \
            create_cugraph_type_erased_device_array_view_from_py_obj(
                vertex_mobility_values)

    assert_CAI_type(vertex_mass_vertices, "vertex_mass_vertices", True)

    cdef cugraph_type_erased_device_array_view_t* \
        vertex_mass_vertices_view_ptr = \
            create_cugraph_type_erased_device_array_view_from_py_obj(
                vertex_mass_vertices)

    assert_CAI_type(vertex_mass_values, "vertex_mass_values", True)

    cdef cugraph_type_erased_device_array_view_t* \
        vertex_mass_values_view_ptr = \
            create_cugraph_type_erased_device_array_view_from_py_obj(
                vertex_mass_values)

    cdef cugraph_layout_result_t* result_ptr
    cdef cugraph_error_code_t error_code
    cdef cugraph_error_t* error_ptr

    cg_rng_state = CuGraphRandomState(resource_handle, random_state)

    cdef cugraph_rng_state_t* rng_state_ptr = cg_rng_state.rng_state_ptr

    error_code = cugraph_force_atlas2(c_resource_handle_ptr,
                                      rng_state_ptr,
                                      c_graph_ptr,
                                      max_iter,
                                      start_vertices_view_ptr,
                                      x_start_view_ptr,
                                      y_start_view_ptr,
                                      outbound_attraction_distribution,
                                      lin_log_mode,
                                      prevent_overlapping,
                                      vertex_radius_vertices_view_ptr,
                                      vertex_radius_values_view_ptr,
                                      overlap_scaling_ratio,
                                      edge_weight_influence,
                                      jitter_tolerance,
                                      barnes_hut_optimize,
                                      barnes_hut_theta,
                                      scaling_ratio,
                                      strong_gravity_mode,
                                      gravity,
                                      vertex_mobility_vertices_view_ptr,
                                      vertex_mobility_values_view_ptr,
                                      vertex_mass_vertices_view_ptr,
                                      vertex_mass_values_view_ptr,
                                      verbose,
                                      do_expensive_check,
                                      &result_ptr,
                                      &error_ptr)
    assert_success(error_code, error_ptr, "force_atlas2")

    cdef cugraph_type_erased_device_array_view_t* vertices_ptr = \
        cugraph_layout_result_get_vertices(result_ptr)

    cdef cugraph_type_erased_device_array_view_t* x_axis_ptr = \
        cugraph_layout_result_get_x_axis(result_ptr)

    cdef cugraph_type_erased_device_array_view_t* y_axis_ptr = \
        cugraph_layout_result_get_y_axis(result_ptr)


    cupy_vertices = copy_to_cupy_array(c_resource_handle_ptr, vertices_ptr)

    cupy_x_axis = copy_to_cupy_array(c_resource_handle_ptr, x_axis_ptr)

    cupy_y_axis = copy_to_cupy_array(c_resource_handle_ptr, y_axis_ptr)

    cugraph_layout_result_free(result_ptr)

    return (cupy_vertices, cupy_x_axis, cupy_y_axis)
