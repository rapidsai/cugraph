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

namespace cugraph {
namespace detail {

template <typename vertex_t>
__global__ void __launch_bounds__(CUDA_MAX_KERNEL_THREADS)
repulsion_kernel(float *x_pos, float *y_pos,
        float *d_dx, float *d_dy, int *d_mass, const float scaling_ratio,
        const vertex_t n) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n)
        return;

    float fx = 0.0f; float fy = 0.0f;

    for (int j = 0; j < n; ++j) {
        float x_dist = x_pos[i] - x_pos[j];
        float y_dist = y_pos[i] - y_pos[j];
        float distance = x_dist * x_dist + y_dist * y_dist;
        distance += FLT_EPSILON;
        float factor = scaling_ratio * d_mass[i] * d_mass[j] / distance;
    //    printf("tid: %i, (%f, %f), (%f, %f) d_mass[i]: %i, d_mass[j]: %i, factor: %f\n",
    //            i, x_pos[i], y_pos[i], x_pos[j], y_pos[j], d_mass[i], d_mass[j], factor);
        fx += x_dist * factor;
        fy += y_dist * factor;
    }
    atomicAdd(&d_dx[i], fx);
    atomicAdd(&d_dy[i], fy);

    //d_dx[i] += fx;
    //d_dy[i] += fy;
    //printf("tid: %i, pos_x: %f, pos_y: %f, dx: %f, dy: %f\n",
    //        i, x_pos[i], y_pos[i], d_dx[i], d_dy[i]);
}

template <typename vertex_t>
void apply_repulsion(float *x_pos, float *y_pos,
        float *d_dx, float *d_dy, int *d_mass, const float scaling_ratio,
        const vertex_t n) {

    dim3 nthreads, nblocks;
    nthreads.x = min(n, CUDA_MAX_KERNEL_THREADS);
    nthreads.y = 1;
    nthreads.z = 1;
    nblocks.x = min((n + nthreads.x - 1) / nthreads.x, CUDA_MAX_BLOCKS);
    nblocks.y = 1;
    nblocks.z = 1;

    repulsion_kernel<vertex_t><<<nthreads, nblocks>>>(x_pos, y_pos,
            d_dx, d_dy, d_mass, scaling_ratio, n);
} 

} // namespace detail
} // namespace cugraph
