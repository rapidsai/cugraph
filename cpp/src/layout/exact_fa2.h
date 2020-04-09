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
#include <internals.h>
#include <stdio.h>

#include "utilities/error_utils.h"
#include "utilities/graph_utils.cuh"

#include "exact_repulsion.h"
#include "fa2_kernels.h"
#include "utils.h"

namespace cugraph {
namespace detail {

template <bool weighted, typename vertex_t, typename edge_t, typename weight_t>
void exact_fa2(const vertex_t *row, const vertex_t *col,
        const weight_t *v, const edge_t e, const vertex_t n,
        float *pos, const int max_iter=1000,
        float *x_start=nullptr, float *y_start=nullptr,
        bool outbound_attraction_distribution=false,
        bool lin_log_mode=false, bool prevent_overlapping=false,
        const float edge_weight_influence=1.0,
        const float jitter_tolerance=1.0,
        const float scaling_ratio=2.0, bool strong_gravity_mode=false,
        const float gravity=1.0, bool verbose=false,
        internals::GraphBasedDimRedCallback* callback=nullptr) { 

    float *d_repel{nullptr};
    float *d_attract{nullptr};
    float *d_old_forces{nullptr};
    int *d_mass{nullptr};
    float *d_swinging{nullptr};
    float *d_traction{nullptr};

    rmm::device_vector<float> repel(n * 2, 0);
    rmm::device_vector<float> attract(n * 2, 0);
    rmm::device_vector<float> old_forces(n * 2, 0);
    rmm::device_vector<int> mass(n, 1);
    rmm::device_vector<float> swinging(n, 0);
    rmm::device_vector<float> traction(n, 0);

    d_repel = repel.data().get();
    d_attract = attract.data().get();
    d_old_forces = old_forces.data().get();
    d_mass = mass.data().get();
    d_swinging = swinging.data().get();
    d_traction = traction.data().get();

    int random_state = 0;
    random_vector(pos, n * 2, random_state);

    if (x_start && y_start) {
        copy(n, x_start, pos);
        copy(n, y_start, pos + n);
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

    if (callback) {
        callback->setup<float>(n, 2);
        callback->on_preprocess_end(pos);
    }

    for (int iter = 0; iter < max_iter; ++iter) {
        fill(n * 2, d_repel, 0.f);
        fill(n * 2, d_attract, 0.f);
        fill(n * 2, d_swinging, 0.f);
        fill(n * 2, d_traction, 0.f);

        apply_repulsion<vertex_t>(pos,
                pos + n, d_repel, d_repel + n, d_mass, scaling_ratio, n);

        apply_gravity<vertex_t>(pos, pos + n, d_attract, d_attract + n, d_mass,
                gravity, strong_gravity_mode, scaling_ratio, n);

        apply_attraction<weighted, vertex_t, edge_t, weight_t>(srcs,
                dests, weights, e, pos, pos + n, d_attract, d_attract + n, d_mass,
                outbound_attraction_distribution, lin_log_mode,
                edge_weight_influence, outbound_att_compensation);

        compute_local_speed(d_repel, d_repel + n,
                d_attract, d_attract + n,
                d_old_forces, d_old_forces + n,
                d_mass, d_swinging, d_traction, n);

        const float s = thrust::reduce(swinging.begin(), swinging.end());
        const float t = thrust::reduce(traction.begin(), traction.end());

        adapt_speed<vertex_t>(jitter_tolerance, &jt, &speed, &speed_efficiency,
                s, t, n);

        apply_forces<vertex_t>(pos, pos + n,  d_repel, d_repel + n,
                d_attract, d_attract + n,
                d_old_forces, d_old_forces + n,
                d_swinging, speed, n);

        if (callback)
            callback->on_epoch_end(pos);

        if (verbose) {
            printf("speed at iteration %i: %f, speed_efficiency: %f, ",
                    iter, speed, speed_efficiency);
            printf("jt: %f, ", jt);
            printf("swinging: %f, traction: %f\n", s, t);
        }
    }

    if (callback)
        callback->on_train_end(pos);

    ALLOC_FREE_TRY(srcs, stream);
    ALLOC_FREE_TRY(dests, stream);
    if (weighted)
        ALLOC_FREE_TRY(weights, stream);
}

} // namespace detail
}  // namespace cugraph
