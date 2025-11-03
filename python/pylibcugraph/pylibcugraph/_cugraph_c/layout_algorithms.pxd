# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

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
from pylibcugraph._cugraph_c.graph cimport (
    cugraph_graph_t,
)
from pylibcugraph._cugraph_c.random cimport (
    cugraph_rng_state_t,
)


cdef extern from "cugraph_c/layout_algorithms.h":
    ###########################################################################
    # force_atlas_2
    ctypedef struct cugraph_layout_result_t:
        pass

    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_layout_result_get_vertices(
            cugraph_layout_result_t* result
        )

    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_layout_result_get_x_axis(
            cugraph_layout_result_t* result
        )

    cdef cugraph_type_erased_device_array_view_t* \
        cugraph_layout_result_get_y_axis(
            cugraph_layout_result_t* result
        )

    cdef void \
        cugraph_layout_result_free(
            cugraph_layout_result_t* result
        )

    cdef cugraph_error_code_t cugraph_force_atlas2(
        const cugraph_resource_handle_t* handle,
        cugraph_rng_state_t* rng_state,
        cugraph_graph_t* graph,
        int max_iter,
        cugraph_type_erased_device_array_view_t* start_vertices,
        cugraph_type_erased_device_array_view_t* x_start,
        cugraph_type_erased_device_array_view_t* y_start,
        bool_t outbound_attraction_distribution,
        bool_t lin_log_mode,
        bool_t prevent_overlapping,
        cugraph_type_erased_device_array_view_t* vertex_radius_vertices,
        cugraph_type_erased_device_array_view_t* vertex_radius_values,
        double overlap_scaling_ratio,
        double edge_weight_influence,
        double jitter_tolerance,
        bool_t barnes_hut_optimize,
        double barnes_hut_theta,
        double scaling_ratio,
        bool_t strong_gravity_mode,
        double gravity,
        cugraph_type_erased_device_array_view_t* vertex_mobility_vertices,
        cugraph_type_erased_device_array_view_t* vertex_mobility_values,
        cugraph_type_erased_device_array_view_t* vertex_mass_vertices,
        cugraph_type_erased_device_array_view_t* vertex_mass_values,
        bool_t verbose,
        bool_t do_expensive_check,
        cugraph_layout_result_t** result,
        cugraph_error_t** error
    )
