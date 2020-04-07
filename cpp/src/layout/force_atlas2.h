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

#include <cugraph.h>
#include <graph.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/thrust_rmm_allocator.h>
#include <rmm_utils.h>
#include <stdio.h>

#include "utilities/error_utils.h"
#include "utilities/graph_utils.cuh"

#include "barnes_hut.h"
#include "exact_repulsion.h"
#include "fa2_kernels.h"
#include "utils.h"

namespace cugraph {
namespace detail {

template <bool weighted, typename vertex_t, typename edge_t, typename weight_t>
void exact_fa2(const vertex_t *row, const vertex_t *col,
        const weight_t *v, const edge_t e, const vertex_t n,
        float *x_pos, float *y_pos, const int max_iter=1000,
        float *x_start=nullptr, float *y_start=nullptr,
        bool outbound_attraction_distribution=false,
        bool lin_log_mode=false, bool prevent_overlapping=false,
        const float edge_weight_influence=1.0,
        const float jitter_tolerance=1.0,
		const float scaling_ratio=2.0, bool strong_gravity_mode=false,
        const float gravity=1.0) { 
    
    float *d_dx{nullptr};
    float *d_dy{nullptr};
    float *d_old_dx{nullptr};
    float *d_old_dy{nullptr};
    int *d_mass{nullptr};
    float *d_swinging{nullptr};
    float *d_traction{nullptr};

    rmm::device_vector<float> dx(n, 0);
    rmm::device_vector<float> dy(n, 0);
    rmm::device_vector<float> old_dx(n, 0);
    rmm::device_vector<float> old_dy(n, 0);
    rmm::device_vector<int> mass(n, 1);
    rmm::device_vector<float> swinging(n, 0);
    rmm::device_vector<float> traction(n, 0);

    d_dx = dx.data().get();
    d_dy = dy.data().get();
    d_old_dx = old_dx.data().get();
    d_old_dy = old_dy.data().get();
    d_mass = mass.data().get();
    d_swinging = swinging.data().get();
    d_traction = traction.data().get();

    int random_state = 0;
    float *YY = random_vector(n * 2, random_state);

    if (x_start && y_start) {
        copy(n, x_start, YY);
        copy(n, y_start, YY + n);
    }

    vertex_t* srcs{nullptr};
    vertex_t* dests{nullptr};
    weight_t* weights{nullptr};
    cudaStream_t stream = sort_coo<weighted, vertex_t, edge_t, weight_t>(row,
            col, v, &srcs, &dests, &weights, e);
    init_mass<vertex_t, edge_t>(&dests, d_mass, e, n);

    float speed = 1.f;
    float speed_efficiency = 1.f;
    float outbound_att_compensation = 1.f;
    float jt = 0.f;
    if (outbound_attraction_distribution) {
        int sum = thrust::reduce(mass.begin(), mass.end());
        outbound_att_compensation = sum / (float)n;
    }

    for (int iter=0; iter < max_iter; ++iter) {
        copy(n, d_dx, d_old_dx);
        copy(n, d_dy, d_old_dy);
        fill(n, d_dx, 0.f);
        fill(n, d_dy, 0.f);
        fill(n, d_swinging, 0.f);
        fill(n, d_traction, 0.f);

		apply_repulsion<vertex_t>(YY,
					YY + n, d_dx, d_dy, d_mass, scaling_ratio, n);

		apply_gravity<vertex_t>(YY, YY + n, d_mass, d_dx, d_dy, gravity,
				strong_gravity_mode, scaling_ratio, n);

		apply_attraction<weighted, vertex_t, edge_t, weight_t>(srcs,
				dests, weights, e, YY, YY + n, d_dx, d_dy, d_mass,
				outbound_attraction_distribution,
				edge_weight_influence, outbound_att_compensation);

       compute_local_speed(YY, YY + n, d_dx, d_dy,
                d_old_dx, d_old_dy, d_mass, d_swinging, d_traction, n);

       float s = thrust::reduce(swinging.begin(), swinging.end());
       float t = thrust::reduce(traction.begin(), traction.end());

       adapt_speed<vertex_t>(jitter_tolerance, &jt, &speed, &speed_efficiency,
               s, t, n);

       apply_forces<vertex_t>(YY, YY + n, d_dx, d_dy,
                d_old_dx, d_old_dy, d_swinging, d_mass, speed, n);
        //printf("speed at iteration %i: %f, speed_efficiency: %f, ",
        //       iter, speed, speed_efficiency);
        //printf("jt: %f, ", jt);
        //printf("swinging: %f, traction: %f\n", s, t);
    }
    copy(n, YY, x_pos);
    copy(n, YY + n, y_pos);

    ALLOC_FREE_TRY(srcs, stream);
    ALLOC_FREE_TRY(dests, stream);
    if (weighted)
        ALLOC_FREE_TRY(weights, stream);
    ALLOC_FREE_TRY(YY, nullptr);
}

} // namespace detail
}  // namespace cugraph
