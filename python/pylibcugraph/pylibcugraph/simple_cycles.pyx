# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Have cython use python 3 syntax
# cython: language_level = 3

from libc.stddef cimport size_t

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
    cugraph_type_erased_device_array_view_free,
)
from pylibcugraph._cugraph_c.graph cimport (
    cugraph_graph_t,
)
from pylibcugraph._cugraph_c.components_algorithms cimport (
    cugraph_simple_cycles_result_t,
    cugraph_simple_cycles,
    cugraph_simple_cycles_result_get_cycle_vertices,
    cugraph_simple_cycles_result_get_cycle_offsets,
    cugraph_simple_cycles_result_get_num_cycles,
    cugraph_simple_cycles_result_free,
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
    create_cugraph_type_erased_device_array_view_from_py_obj,
)


def simple_cycles(ResourceHandle resource_handle,
                  _GPUGraph graph,
                  seed_vertices,
                  size_t length_bound,
                  bool_t do_expensive_check):
    """
    Enumerate simple cycles (elementary circuits) in a directed graph.

    The algorithm is defined for directed (asymmetric) graphs.

    Parameters
    ----------
    resource_handle : ResourceHandle
        Handle to the underlying device resources needed for referencing data
        and running algorithms.

    graph : SGGraph or MGGraph
        The input graph, for either Single or Multi-GPU operations.

    seed_vertices : cupy array or None
        Optional device array of seed vertex ids in external id space. If None,
        all simple cycles with length <= ``length_bound`` are returned. If
        non-None, only cycles that contain at least one seed are returned.
        Multi-GPU: if any rank uses seed filtering, every rank must pass a
        non-None array; ranks with no local seeds should pass an empty array
        (size 0), not None.

    length_bound : int
        Maximum cycle length (must be > 0). Intended for small values (for
        example, no larger than 10).

    do_expensive_check : bool
        If True, performs more extensive tests on the inputs to ensure
        validity, at the expense of increased run time.

    Returns
    -------
    tuple
        (cycle_vertices, cycle_offsets, num_cycles) where ``cycle_vertices`` is
        a device array of vertex ids (flat concatenation of cycles in cyclic
        order), ``cycle_offsets`` is a device array of type uint64 with length
        ``num_cycles + 1`` indexing into ``cycle_vertices``, and ``num_cycles``
        is the number of cycles on this GPU.
    """
    if length_bound == 0:
        raise ValueError("length_bound must be greater than 0")

    cdef cugraph_resource_handle_t* c_resource_handle_ptr = \
        resource_handle.c_resource_handle_ptr
    cdef cugraph_graph_t* c_graph_ptr = graph.c_graph_ptr

    cdef cugraph_simple_cycles_result_t* result_ptr
    cdef cugraph_error_code_t error_code
    cdef cugraph_error_t* error_ptr

    cdef cugraph_type_erased_device_array_view_t* seed_vertices_view_ptr = \
        create_cugraph_type_erased_device_array_view_from_py_obj(seed_vertices)

    error_code = cugraph_simple_cycles(c_resource_handle_ptr,
                                       c_graph_ptr,
                                       seed_vertices_view_ptr,
                                       length_bound,
                                       do_expensive_check,
                                       &result_ptr,
                                       &error_ptr)
    assert_success(error_code, error_ptr, "cugraph_simple_cycles")

    cdef cugraph_type_erased_device_array_view_t* cycle_vertices_ptr = \
        cugraph_simple_cycles_result_get_cycle_vertices(result_ptr)
    cdef cugraph_type_erased_device_array_view_t* cycle_offsets_ptr = \
        cugraph_simple_cycles_result_get_cycle_offsets(result_ptr)

    cdef size_t num_cycles = cugraph_simple_cycles_result_get_num_cycles(result_ptr)

    cupy_cycle_vertices = copy_to_cupy_array(c_resource_handle_ptr, cycle_vertices_ptr)
    cupy_cycle_offsets = copy_to_cupy_array(c_resource_handle_ptr, cycle_offsets_ptr)

    cugraph_simple_cycles_result_free(result_ptr)
    if seed_vertices is not None:
        cugraph_type_erased_device_array_view_free(seed_vertices_view_ptr)

    return cupy_cycle_vertices, cupy_cycle_offsets, num_cycles
