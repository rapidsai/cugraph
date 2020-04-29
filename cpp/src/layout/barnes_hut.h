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
                                                                                
#include <thrust/transform.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/functional.h>

#include "utilities/error_utils.h"                                              
#include "utilities/graph_utils.cuh"     
#include "bh_kernels.h"
#include "fa2_kernels.h"
#include "utils.h"

namespace cugraph {
namespace detail {

template <typename vertex_t, typename edge_t, typename weight_t>
void barnes_hut(const vertex_t *row, const vertex_t *col,
        const weight_t *v, const edge_t e, const vertex_t n,
        float *pos, const int max_iter=1000,
        float *x_start=nullptr, float *y_start=nullptr,
        bool outbound_attraction_distribution=false,
        bool lin_log_mode=false, bool prevent_overlapping=false,
        const float edge_weight_influence=1.0,
        const float jitter_tolerance=1.0, const float theta=0.5,
        const float scaling_ratio=2.0, bool strong_gravity_mode=false,
        const float gravity=1.0,
        bool verbose=false,
        internals::GraphBasedDimRedCallback* callback=nullptr) {

    const int blocks = getMultiProcessorCount();
    const float epssq = 0.0025;
    int nnodes = n * 2;
    if (nnodes < 1024 * blocks) nnodes = 1024 * blocks;
    while ((nnodes & (32 - 1)) != 0) nnodes++;
    nnodes--;

    // Allocate more space
    //---------------------------------------------------
    rmm::device_vector<unsigned>d_limiter(1);
    rmm::device_vector<int>d_maxdepthd(1);
    rmm::device_vector<int>d_bottomd(1);
    rmm::device_vector<float>d_radiusd(1);

    unsigned *limiter = d_limiter.data().get();
    int *maxdepthd = d_maxdepthd.data().get();
    int *bottomd = d_bottomd.data().get();
    float *radiusd = d_radiusd.data().get();

    InitializationKernel<<<1, 1>>>(/*errl,*/ limiter, maxdepthd,
          radiusd);
    CUDA_CHECK_LAST();

    const int FOUR_NNODES = 4 * nnodes;
    const int FOUR_N = 4 * n;
    const float theta_squared = theta * theta;
    const int NNODES = nnodes;

    rmm::device_vector<int>d_startl(nnodes + 1, 0);
    rmm::device_vector<int>d_childl((nnodes + 1) * 4, 0);
    rmm::device_vector<int>d_massl(nnodes + 1, 1.f);

    rmm::device_vector<float>d_maxxl(blocks * FACTOR1, 0);
    rmm::device_vector<float>d_maxyl(blocks * FACTOR1, 0);
    rmm::device_vector<float>d_minxl(blocks * FACTOR1, 0);
    rmm::device_vector<float>d_minyl(blocks * FACTOR1, 0);


    // Actual mallocs
    int *startl = d_startl.data().get();
    int *childl = d_childl.data().get();
    int *massl = d_massl.data().get();

    float *maxxl = d_maxxl.data().get();
    float *maxyl = d_maxyl.data().get();
    float *minxl = d_minxl.data().get();
    float *minyl = d_minyl.data().get();

    // SummarizationKernel
    rmm::device_vector<int>d_countl(nnodes + 1, 0);
    int *countl = d_countl.data().get();

    // SortKernel
    rmm::device_vector<int>d_sortl(nnodes + 1, 0);
    int *sortl = d_sortl.data().get();

    // RepulsionKernel
    rmm::device_vector<float>d_rep_forces((nnodes + 1) * 2, 0);
    float *rep_forces = d_rep_forces.data().get();

    rmm::device_vector<float>d_radius_squared(1, 0);
    float *radiusd_squared = d_radius_squared.data().get();

    rmm::device_vector<float>d_YY((nnodes + 1) * 2, 0);
    float *YY = d_YY.data().get();

    int random_state = 0;
    random_vector(YY, (nnodes + 1) * 2, random_state);

    if (x_start && y_start) {
        copy(n, x_start, YY);
        copy(n, y_start, YY + nnodes + 1);
    }

    float *d_attract{nullptr};
    float *d_old_forces{nullptr};
    float *d_swinging{nullptr};
    float *d_traction{nullptr};

    rmm::device_vector<float> attract(n * 2, 0);
    rmm::device_vector<float> old_forces(n * 2, 0);
    rmm::device_vector<int> mass(n, 1);
    rmm::device_vector<float> swinging(n, 0);
    rmm::device_vector<float> traction(n, 0);

    d_attract = attract.data().get();
    d_old_forces = old_forces.data().get();
    d_swinging = swinging.data().get();
    d_traction = traction.data().get();

    vertex_t* srcs{nullptr};
    vertex_t* dests{nullptr};
    weight_t* weights{nullptr};

    sort_coo<vertex_t, edge_t, weight_t>(row,
                        col, v, &srcs, &dests, &weights, e);
    init_mass<vertex_t, edge_t, int>(dests, massl, e, n);

    float speed = 1.f;
    float speed_efficiency = 1.f;
    float outbound_att_compensation = 1.f;
    float jt = 0.f;
 
    if (outbound_attraction_distribution) {
        int sum = thrust::reduce(d_massl.begin(), d_massl.end());
        const float div = 1.f / n;
        outbound_att_compensation = sum * div;
    }
    //
    // Set cache levels for faster algorithm execution
    //---------------------------------------------------
    cudaFuncSetCacheConfig(BoundingBoxKernel, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(TreeBuildingKernel, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(ClearKernel1, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(ClearKernel2, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(SummarizationKernel, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(SortKernel, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(RepulsionKernel, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(apply_forces_bh, cudaFuncCachePreferL1);

    if(callback) {
        callback->setup<float>(n, 2);
        callback->on_preprocess_end(pos);
    }

    for (int iter = 0; iter < max_iter; ++iter) {
        fill((nnodes + 1) * 2, rep_forces, 0.f);
        fill(n * 2, d_attract, 0.f);
        fill(n, d_swinging, 0.f);
        fill(n, d_traction, 0.f);

        Reset_Normalization<<<1, 1>>>(radiusd_squared,
                bottomd, NNODES, radiusd);
        CUDA_CHECK_LAST();

        BoundingBoxKernel<<<blocks * FACTOR1, THREADS1>>>(
                startl, childl, massl, YY, YY + nnodes + 1, maxxl, maxyl, minxl, minyl,
                FOUR_NNODES, NNODES, n, limiter, radiusd);
        CUDA_CHECK_LAST();

        ClearKernel1<<<blocks, 1024>>>(childl, FOUR_NNODES,
                FOUR_N);
        CUDA_CHECK_LAST();

        TreeBuildingKernel<<<blocks * FACTOR2, THREADS2>>>(
                childl, YY, YY + nnodes + 1, NNODES, n, maxdepthd, bottomd,
                radiusd);
        CUDA_CHECK_LAST();

        ClearKernel2<<<blocks, 1024>>>(startl, massl, NNODES,
                bottomd);
        CUDA_CHECK_LAST();

        SummarizationKernel<<<blocks * FACTOR3, THREADS3>>>(
                countl, childl, massl, YY, YY + nnodes + 1, NNODES, n, bottomd);
        CUDA_CHECK_LAST();

        SortKernel<<<blocks * FACTOR4, THREADS4>>>(
                sortl, countl, startl, childl, NNODES, n, bottomd);
        CUDA_CHECK_LAST();

        RepulsionKernel<<<blocks * FACTOR5, THREADS5>>>(
                scaling_ratio,
                theta, epssq, sortl, childl, massl, YY, YY + nnodes + 1,
                rep_forces, rep_forces + nnodes + 1, theta_squared, NNODES,
                FOUR_NNODES, n, radiusd_squared, maxdepthd);
        CUDA_CHECK_LAST();

        apply_gravity<vertex_t>(YY, YY + nnodes + 1, d_attract, d_attract + n,
                massl, gravity, strong_gravity_mode, scaling_ratio, n);

        apply_attraction<vertex_t, edge_t, weight_t>(srcs,
                dests, weights, e, YY, YY + nnodes + 1,
                d_attract, d_attract + n, massl,
                outbound_attraction_distribution, lin_log_mode,
                edge_weight_influence, outbound_att_compensation);

        compute_local_speed(rep_forces, rep_forces + nnodes + 1,
                d_attract, d_attract + n,
                d_old_forces, d_old_forces + n,
                massl, d_swinging, d_traction, n);

        const float s = thrust::reduce(rmm::exec_policy(nullptr)->on(nullptr),
                swinging.begin(), swinging.end());

        const float t = thrust::reduce(rmm::exec_policy(nullptr)->on(nullptr),
                traction.begin(), traction.end());

        adapt_speed<vertex_t>(jitter_tolerance, &jt, &speed, &speed_efficiency,
                s, t, n);

        apply_forces_bh<<<blocks * FACTOR6, THREADS6>>>(pos, pos + n,
                YY, YY + nnodes + 1, d_attract, d_attract + n,
                rep_forces, rep_forces + nnodes + 1,
                d_old_forces, d_old_forces + n,
                d_swinging, speed, n);

        if (callback)
            callback->on_epoch_end(pos);

        if (verbose) {
            printf("iteration %i: speed: %f, speed_efficiency: %f, ",
                    iter, speed, speed_efficiency);
            printf("jt: %f, ", jt);
            printf("swinging: %f, traction: %f\n", s, t);
        }

    }

    if (callback)
        callback->on_epoch_end(pos);

    ALLOC_FREE_TRY(srcs, nullptr);
    ALLOC_FREE_TRY(dests, nullptr);
    if (v)
        ALLOC_FREE_TRY(weights, nullptr);
}

} // namespace detail
} // namespace cugraph
