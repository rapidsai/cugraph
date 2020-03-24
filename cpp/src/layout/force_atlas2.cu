/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>
#include "cub/cub.cuh"
#include <algorithm>
#include <iomanip>

#include <rmm_utils.h>

#include "utilities/graph_utils.cuh"
#include "utilities/error_utils.h"
#include <cugraph.h>
#include <graph.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <rmm/device_buffer.hpp>

#include "barnes_hut.h" 
#include "exact_fa2.h"

namespace cugraph {

template <typename VT, typename ET, typename WT>
void force_atlas2(experimental::GraphCSR<VT, ET, WT> const &graph,
                  float *x_pos, float *y_pos, const int max_iter,
                  float *x_start,
                  float *y_start, bool outbound_attraction_distribution,
                  bool lin_log_mode, bool prevent_overlapping,
                  const float edge_weight_influence,
                  const float jitter_tolerance, bool barnes_hut_optimize,
                  const float barnes_hut_theta, const float scaling_ratio,
                  bool strong_gravity_mode, const float gravity) {

    CUGRAPH_EXPECTS( x_pos != nullptr,
            "Invid API parameter: X_pos array should be of size V" );
    CUGRAPH_EXPECTS( y_pos != nullptr ,
            "Invid API parameter: Y_pos array should be of size V" );

    const VT *csrPtr = graph.offsets;
    const ET *csrInd = graph.indices;
    const WT *v = graph.edge_data;
    const VT n = graph.number_of_vertices;

    if (barnes_hut_optimize) {
        cugraph::detail::barnes_hut<VT, ET, WT>(csrPtr, csrInd, v, n,
                x_pos, y_pos, max_iter, x_start,
                y_start, outbound_attraction_distribution,
                lin_log_mode, prevent_overlapping, edge_weight_influence,
                jitter_tolerance,barnes_hut_theta, scaling_ratio,
                strong_gravity_mode, gravity);
    } else {
        cugraph::detail::exact_fa2<VT, ET, WT>(csrPtr, csrInd, v, n,
                x_pos, y_pos, max_iter, x_start,
                y_start, outbound_attraction_distribution,
                lin_log_mode, prevent_overlapping, edge_weight_influence,
                jitter_tolerance, scaling_ratio,
                strong_gravity_mode, gravity);
    }
}

template void force_atlas2<int, int, float>(
        experimental::GraphCSR<int, int, float> const &graph,
        float *x_pos, float *y_pos, const int max_iter,
        float *x_start, float *y_start,
        bool outbound_attraction_distribution,
        bool lin_log_mode, bool prevent_overlapping,
        const float edge_weight_influence, const float jitter_tolerance,
        bool barnes_hut_optimize, const float barnes_hut_theta,
        const float scaling_ratio, bool strong_gravity_mode,
        const float gravity);

template void cugraph::detail::barnes_hut<int, int, float>(
        const int *csrPtr, const int *csrInd, const float *v, const int n,
        float *x_pos, float *y_pos, int max_iter,
        float *x_start, float * y_start,
        bool outbount_attraction_distribution,
        bool lin_log_mode, bool prevent_overlapping,
        const float edge_weight_influence, const float jitter_tolerance,
        const float barnes_hut_theta, const float scaling_ratio,
        bool strong_gravity_mode,
		const float gravity);

template void cugraph::detail::exact_fa2<int, int, float>(
        const int *csrPtr, const int *csrInd, const float *v, const int n,
        float *x_pos, float *y_pos, const int max_iter,
        float *x_start, float *y_start,
        bool outbound_attraction_distribution,
        bool lin_log_mode, bool prevent_overlapping,
        const float edge_weight_influence, const float jitter_tolerance,
        const float scaling_ratio, bool strong_gravity_mode,
        const float gravity);

} // namespace cugraph
