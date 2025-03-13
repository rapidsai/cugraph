/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cugraph_c/error.h>
#include <cugraph_c/graph.h>
#include <cugraph_c/graph_functions.h>
#include <cugraph_c/random.h>
#include <cugraph_c/resource_handle.h>

/** @defgroup community Community algorithms
 */

#ifdef __cplusplus
extern "C" {
#endif


/**
 * @brief     Opaque layout output
 */
typedef struct {
  int32_t align_;
} cugraph_layout_result_t;


/**
 * @brief     Opaque clustering output
 */
typedef struct { // FIXME: Remove this
  int32_t align_;
} cugraph_clustering_result_t;


/**
 * @ingroup layout
 * @brief     Get layout x-axis
 */
cugraph_type_erased_device_array_view_t* cugraph_layout_get_x_axis( // FIXME: Might not be used/necessary
  cugraph_layout_result_t* result);

/**
 * @ingroup layout
 * @brief     Get layout y-axis
 */
cugraph_type_erased_device_array_view_t* cugraph_layout_get_y_axis( // FIXME: Might not be used/necessary
  cugraph_layout_result_t* result);

/**
 * @ingroup community
 * @brief     Free a layout result
 *
 * @param [in] result     The result from a layout algorithm
 */
void cugraph_layout_result_free(cugraph_layout_result_t* result); // FIXME: Might not be used/necessary

/**
 * @brief   Force Atlas 2
 *
 * NOTE: This currently wraps the legacy force atlas 2 implementation and is only
 * available in Single GPU implementation.
 *
 * @param [in]   handle          Handle for accessing resources
 * @param [in]   graph           Pointer to graph.  NOTE: Graph might be modified if the storage
 *                               needs to be transposed
 * @param [out]  pos             Position of the vertices
 * @param [in]   max_iter        Maximum number of iterations. Initial vertex positioning
 * @param [in]   x_start         Initial vertex positioning (x-axis)
 * @param [in]   y_start         Initial vertex positioning (y-axis)
 * @param [in]   outbound_attraction_distribution
 *                               Distributes attraction along outbound edges
 *                               Hubs attract less and thus are pushed to the borders.
 * @param [in]   lin_log_mode    Switch Force Atlas model from lin-lin to lin-log.
 *                               Makes clusters more tight.
 * @param [in]   prevent_overlapping
 *                               Prevent nodes to overlap.
 * @param [in]   edge_weight_influence
 *                               How much influence you give to the edges weight.
 *                               0 is “no influence” and 1 is “normal”.
 * @param [in]   jitter_tolerance
 *                               How much swinging you allow. Above 1 discouraged.
 *                               Lower gives less speed and more precision.
 * @param [in]   barnes_hut_optimize
 *                               Whether to use the Barnes Hut approximation or the slower exact version.
 * @param [in]   barnes_hut_theta
 *                               Float between 0 and 1. Tradeoff for speed (1) vs accuracy (0) for Barnes Hut only.
 * @param [in]   scaling_ratio
 *                               How much repulsion you want. More makes a more sparse graph.
 *                               Switching from regular mode to LinLog mode needs a readjustment of the scaling parameter.
 * @param [in]   strong_gravity_mode
 *                               Sets a force that attracts the nodes that are distant from the
 *                               center more. It is so strong that it can sometimes dominate other forces.
 * @param [in]   gravity         Attracts nodes to the center. Prevents islands from drifting away.
 * @param [in]   verbose         Output convergence info at each interation.
 * @param [in]   do_expensive_check
 *                               A flag to run expensive checks for input arguments (if set to true)
 * @param [out]  result          Opaque object containing the clustering result
 * @param [out]  error           Pointer to an error object storing details of any error.  Will
 *                               be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
cugraph_error_code_t cugraph_force_atlas2(const cugraph_resource_handle_t* handle,
                                          cugraph_graph_t* graph,
                                          cugraph_type_erased_device_array_t** pos,
                                          const int max_iter,
                                          cugraph_type_erased_device_array_view_t* x_start,
                                          cugraph_type_erased_device_array_view_t* y_start,
                                          bool_t outbound_attraction_distribution,
                                          bool_t lin_log_mode,
                                          bool_t prevent_overlapping,
                                          const double edge_weight_influence,
                                          const double jitter_tolerance,
                                          bool_t barnes_hut_optimize,
                                          const double barnes_hut_theta,
                                          const double scaling_ratio,
                                          bool_t strong_gravity_mode,
                                          const double gravity,
                                          bool_t verbose,
                                          bool_t do_expensive_check,
                                          cugraph_error_t** error);



#ifdef __cplusplus
}
#endif
