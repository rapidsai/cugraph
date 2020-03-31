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

#pragma once

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

#include "exact_kernels.h"

namespace cugraph {
namespace detail {

template <typename vertex_t, typename edge_t, typename weight_t>
void exact_fa2(const edge_t *csrPtr, const vertex_t *csrInd,
               const weight_t *v, const vertex_t n,
               float *x_pos, float *y_pos, const int max_iter=1000,
               float *x_start=nullptr, float *y_start=nullptr,
               bool outbound_attraction_distribution=false,
               bool lin_log_mode=false, bool prevent_overlapping=false,
               const float edge_weight_influence=1.0,
               const float jitter_tolerance=1.0,
               const float scaling_ratio=2.0, bool strong_gravity_mode=false,
               const float gravity=1.0) { 
    
    float *d_repel_x{nullptr};
    float *d_repel_y{nullptr};
    float *d_gravity{nullptr};
    float *d_attract_x{nullptr};
    float *d_attract_y{nullptr};
    float *d_dx{nullptr};
    float *d_dy{nullptr};
    float *d_old_dx{nullptr};
    float *d_old_dy{nullptr};
    int *d_mass{nullptr};

    rmm::device_vector<float> repel_x(n, 0);
    rmm::device_vector<float> repel_y(n, 0);
    rmm::device_vector<float> attract_x(n, 0);
    rmm::device_vector<float> attract_y(n, 0);
    rmm::device_vector<float> dx(n, 0);
    rmm::device_vector<float> dy(n, 0);
    rmm::device_vector<float> old_dx(n, 0);
    rmm::device_vector<float> old_dy(n, 0);
    rmm::device_vector<int> mass(n, 0);

    d_repel_x = repel_x.data().get();
    d_repel_y = repel_y.data().get();
    d_attract_x = attract_x.data().get();
    d_attract_y = attract_y.data().get();
    d_dx = dx.data().get();
    d_dy = dy.data().get();
    d_old_dx = dx.data().get();
    d_old_dy = dy.data().get();
    d_mass = mass.data().get();

    if (x_start == nullptr || y_start == nullptr) {
        // TODO: generate random numbers
        return;
    } else {
        copy(n, x_start, x_pos);
        copy(n, y_start, y_pos);
    }

    float speed = 1.0;
    float speed_efficiency = 1.0;
    float outbound_at_compensation = 1.0; // FIXME: Compute mean
    init_mass<vertex_t, edge_t><<<ceil(NTHREADS / n), NTHREADS>>>(csrPtr,
            csrInd, d_mass, n);

    for (int iter=0; iter < max_iter; ++iter) {
        apply_repulsion<vertex_t>(x_pos,
                y_pos, d_dx, d_dy, d_mass,
                d_repel_x, d_repel_y, scaling_ratio, n);

        apply_gravity<vertex_t>(x_pos, y_pos, d_mass, d_dx, d_dy, gravity,
                strong_gravity_mode, scaling_ratio, n);
        
        /*
        apply_attraction<vertex_t, edge_t, weight_t>(csrPtr,
                csrInd, v, n, x_pos, y_pos, d_dx, d_dy, d_mass,
                outbound_attraction_distribution,
                edge_weight_influence, outbound_at_compensation);
        
        speed = apply_forces<vertex_t>(x_pos,
                y_pos, d_dx, d_dy, d_old_dx, d_old_dy, d_mass,
                jitter_tolerance, speed, speed_efficiency, n);
        */
        printf("speed at iteration %i: %f\n", iter, speed);
    }
}

} // namespace detail
}  // namespace cugraph
