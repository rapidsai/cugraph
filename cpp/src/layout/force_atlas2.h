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
#include <thrust/random.h>

#include <rmm_utils.h>

#include "utilities/graph_utils.cuh"
#include "utilities/error_utils.h"
#include <cugraph.h>
#include <graph.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <rmm/device_buffer.hpp>

#include "barnes_hut.h"
#include "fa2_kernels.h"
#include "exact_repulsion.h"
#include "utils.h"

namespace cugraph {
namespace detail {

template <bool weighted, typename vertex_t, typename edge_t, typename weight_t>
void fa2(const vertex_t *row, const vertex_t *col,
        const weight_t *v, const edge_t e, const vertex_t n,
        float *x_pos, float *y_pos, const int max_iter=1000,
        float *x_start=nullptr, float *y_start=nullptr,
        bool outbound_attraction_distribution=false,
        bool lin_log_mode=false, bool prevent_overlapping=false,
        const float edge_weight_influence=1.0,
        const float jitter_tolerance=1.0, bool barnes_hut_optimize=0.5,
        const float barnes_hut_theta=1.2,
        const float scaling_ratio=2.0, bool strong_gravity_mode=false,
        const float gravity=1.0) { 
    
    // Temporary fix
    // For weighted graph number_of_edges returns half the vertices
    // but does not adapt the datastructure accordingly.
    const edge_t tmp_e = e * 2;
    bool random_start = false;

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

    if (x_start == nullptr || y_start == nullptr) {
        random_start = true;
        x_start = random_vector(n, 0);
        y_start = random_vector(n, 1);
    }

    copy(n, x_start, x_pos);
    copy(n, y_start, y_pos);

    vertex_t* srcs{nullptr};
    vertex_t* dests{nullptr};
    weight_t* weights{nullptr};
    cudaStream_t stream = sort_coo<weighted, vertex_t, edge_t, weight_t>(row,
            col, v, &srcs, &dests, &weights, tmp_e);
    init_mass<vertex_t, edge_t>(&dests, d_mass, tmp_e, n);

    float speed = 1.f;
    float speed_efficiency = 1.f;
    float outbound_att_compensation = 1.f;
    float jt = 0.f;
    if (outbound_attraction_distribution) {
        int sum = thrust::reduce(mass.begin(), mass.end());
        outbound_att_compensation = sum / (float)n;
    }

    dim3 nthreads, nblocks;
    nthreads.x = min(n, CUDA_MAX_KERNEL_THREADS);
    nthreads.y = 1;
    nthreads.z = 1;
    nblocks.x = min((n + nthreads.x - 1) / nthreads.x, CUDA_MAX_BLOCKS);
    nblocks.y = 1;
    nblocks.z = 1;


    for (int iter=0; iter < max_iter; ++iter) {
        copy(n, d_dx, d_old_dx);
        copy(n, d_dy, d_old_dy);
        fill(n, d_dx, 0.f);
        fill(n, d_dy, 0.f);
        fill(n, d_swinging, 0.f);
        fill(n, d_traction, 0.f);

        if (barnes_hut_optimize) {
           return;
        } else {
            apply_repulsion<vertex_t>(x_pos,
                    y_pos, d_dx, d_dy, d_mass, scaling_ratio, n);
        }

        apply_gravity<vertex_t>(x_pos, y_pos, d_mass, d_dx, d_dy, gravity,
                strong_gravity_mode, scaling_ratio, n);

        apply_attraction<weighted, vertex_t, edge_t, weight_t>(srcs,
                dests, v, tmp_e, x_pos, y_pos, d_dx, d_dy, d_mass,
                outbound_attraction_distribution,
                edge_weight_influence, outbound_att_compensation);

        compute_local_speed(x_pos, y_pos, d_dx, d_dy,
                d_old_dx, d_old_dy, d_mass, d_swinging, d_traction, n);

       float s = thrust::reduce(swinging.begin(), swinging.end());
       float t = thrust::reduce(traction.begin(), traction.end());

       adapt_speed<vertex_t>(jitter_tolerance, &jt, &speed, &speed_efficiency,
               s, t, n);

       apply_forces<vertex_t>(x_pos, y_pos, d_dx, d_dy,
                d_old_dx, d_old_dy, d_swinging, d_mass, speed, n);
       
        printf("speed at iteration %i: %f, speed_efficiency: %f, ",
               iter, speed, speed_efficiency);
        printf("jt: %f, ", jt);
        printf("swinging: %f, traction: %f\n", s, t);
 
    }
    ALLOC_FREE_TRY(srcs, stream);
    ALLOC_FREE_TRY(dests, stream);
    if (weighted)
        ALLOC_FREE_TRY(weights, stream);
    if (random_start) {
        ALLOC_FREE_TRY(x_start, nullptr);
        ALLOC_FREE_TRY(y_start, nullptr);
    }
}

} // namespace detail
}  // namespace cugraph
