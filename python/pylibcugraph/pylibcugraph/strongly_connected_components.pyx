# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Have cython use python 3 syntax
# cython: language_level = 3

from pylibcugraph import GraphProperties, SGGraph

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
    cugraph_type_erased_device_array_view_copy,
)
from pylibcugraph._cugraph_c.graph cimport (
    cugraph_graph_t,
)
from pylibcugraph._cugraph_c.labeling_algorithms cimport (
    cugraph_labeling_result_t,
    cugraph_strongly_connected_components,
    cugraph_labeling_result_get_vertices,
    cugraph_labeling_result_get_labels,
    cugraph_labeling_result_free,
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
    copy_to_cupy_array,
    create_cugraph_type_erased_device_array_view_from_py_obj,
)


def _ensure_args(graph, offsets, indices, weights):
    i = 0
    if graph is not None:
        # ensure the remaining parametes are None
        invalid_input = [i for p in [offsets, indices, weights] if p is not None]
        input_type = "graph"
    else:
        invalid_input = [i for p in [offsets, indices] if p is None]
        input_type = "csr_arrays"


    if len(invalid_input) != 0:
        raise TypeError("Invalid input combination: Must set either 'graph' or "
                        "a combination of 'offsets', 'indices' and 'weights', not both")
    else:
        if input_type == "csr_arrays":
            assert_CAI_type(offsets, "offsets")
            assert_CAI_type(indices, "indices")
            assert_CAI_type(weights, "weights", True)

    return input_type


def strongly_connected_components(ResourceHandle resource_handle,
                                  _GPUGraph graph,
                                  offsets,
                                  indices,
                                  weights,
                                  labels,
                                  bool_t do_expensive_check):
    """
    Generate the Strongly Connected Components from either an input graph or
    CSR arrays ('offsets', 'indices', 'weights') and attach a component label
    to each vertex.

    Parameters
    ----------
    resource_handle : ResourceHandle
        Handle to the underlying device resources needed for referencing data
        and running algorithms.

    graph : SGGraph or MGGraph
        The input graph.

    offsets : object supporting a __cuda_array_interface__ interface
        Array containing the offsets values of a Compressed Sparse Row matrix
        that represents the graph.

    indices : object supporting a __cuda_array_interface__ interface
        Array containing the indices values of a Compressed Sparse Row matrix
        that represents the graph.

    weights : object supporting a __cuda_array_interface__ interface
        Array containing the weights values of a Compressed Sparse Row matrix
        that represents the graph

    labels : optional, object supporting __cuda_array_interface__
        If provided, component labels are copied into this array; otherwise
        labels are returned in the result tuple.

    do_expensive_check : bool_t
        If True, performs more extensive tests on the inputs to ensure
        validitity, at the expense of increased run time.

    Returns
    -------
    tuple or None
        If ``labels`` is None, returns ``(vertices, labels)`` as device arrays.
        If ``labels`` is provided, returns None (output written in-place).

    Examples
    --------
    >>> import pylibcugraph, cupy, numpy
    >>> from pylibcugraph import strongly_connected_components
    >>> srcs = cupy.asarray([0, 0, 1, 1, 2, 2], dtype=numpy.int32)
    >>> dsts = cupy.asarray([1, 2, 1, 2, 0, 1], dtype=numpy.int32)
    >>> weights = cupy.asarray(
    ...     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=numpy.float32)
    >>> resource_handle = pylibcugraph.ResourceHandle()
    >>> graph_props = pylibcugraph.GraphProperties(
    ...      is_symmetric=False, is_multigraph=False)
    >>> G = pylibcugraph.SGGraph(
    ...     resource_handle, graph_props, srcs, dsts, weight_array=weights,
    ...     store_transposed=False, renumber=True, do_expensive_check=False)
    >>> (vertices, labels) = strongly_connected_components(
    ...     resource_handle, G, None, None, None, None, False)

    >>> vertices
    [0, 1, 2]
    >>> labels
    [0, 0, 0]

    >>> import cupy as cp
    >>> import numpy as np
    >>> from scipy.sparse import csr_matrix
    >>>
    >>> graph = [
    ... [0, 1, 1, 0, 0],
    ... [0, 0, 1, 0, 0],
    ... [0, 0, 0, 0, 0],
    ... [0, 0, 0, 0, 1],
    ... [0, 0, 0, 0, 0],
    ... ]
    >>> scipy_csr = csr_matrix(graph)
    >>>
    >>> cp_offsets = cp.asarray(scipy_csr.indptr)
    >>> cp_indices = cp.asarray(scipy_csr.indices, dtype=np.int32)
    >>>
    >>> resource_handle = pylibcugraph.ResourceHandle()
    >>> _, cp_labels = strongly_connected_components(
    ...     resource_handle=resource_handle,
    ...     graph=None,
    ...     offsets=cp_offsets,
    ...     indices=cp_indices,
    ...     weights=None,
    ...     labels=None,
    ...     do_expensive_check=False,
    ... )
    >>> print(f"{len(set(cp_labels.tolist()))} - {cp_labels}")
    5 - [0 1 2 3 4]

    """

    input_type = _ensure_args(graph, offsets, indices, weights)

    if input_type == "csr_arrays":
        if resource_handle is None:
            resource_handle = ResourceHandle()

        graph_props = GraphProperties(
        is_symmetric=False, is_multigraph=False)
        graph = SGGraph(
                resource_handle,
                graph_props,
                offsets,
                indices,
                weights,
                store_transposed=False,
                renumber=False,
                do_expensive_check=True,
                input_array_format="CSR"
            )

    cdef cugraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr

    cdef cugraph_graph_t* c_graph_ptr = graph.c_graph_ptr
    cdef cugraph_labeling_result_t* result_ptr
    cdef cugraph_error_code_t error_code
    cdef cugraph_error_t* error_ptr

    error_code = cugraph_strongly_connected_components(c_resource_handle_ptr,
                                                       c_graph_ptr,
                                                       do_expensive_check,
                                                       &result_ptr,
                                                       &error_ptr)
    assert_success(error_code, error_ptr, "cugraph_strongly_connected_components")

    cdef cugraph_type_erased_device_array_view_t* vertices_ptr = \
        cugraph_labeling_result_get_vertices(result_ptr)
    cdef cugraph_type_erased_device_array_view_t* labels_ptr = \
        cugraph_labeling_result_get_labels(result_ptr)

    cdef cugraph_type_erased_device_array_view_t* labels_view_ptr
    if labels is not None:
        labels_view_ptr = create_cugraph_type_erased_device_array_view_from_py_obj(
            labels)
        error_code = cugraph_type_erased_device_array_view_copy(
            c_resource_handle_ptr,
            labels_view_ptr,
            labels_ptr,
            &error_ptr
        )
        assert_success(
            error_code, error_ptr, "cugraph_type_erased_device_array_view_copy")
        cugraph_labeling_result_free(result_ptr)
    else:
        cupy_vertices = copy_to_cupy_array(c_resource_handle_ptr, vertices_ptr)
        cupy_labels = copy_to_cupy_array(c_resource_handle_ptr, labels_ptr)
        cugraph_labeling_result_free(result_ptr)
        return (cupy_vertices, cupy_labels)
