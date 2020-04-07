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
	const int j =
		(blockIdx.x * blockDim.x) + threadIdx.x;  // for every item in row
	const int i = (blockIdx.y * blockDim.y) + threadIdx.y;  // for every row
	if (j >= i || i >= n || j >= n) return;

	float x_dist = x_pos[i] - x_pos[j];
	float y_dist = y_pos[i] - y_pos[j];
	float distance = x_dist * x_dist + y_dist * y_dist;
	distance += FLT_EPSILON;
	float factor = scaling_ratio * d_mass[i] * d_mass[j] / distance;
	// Add forces
	atomicAdd(&d_dx[i], x_dist * factor);
	atomicAdd(&d_dx[j], -x_dist * factor);
	atomicAdd(&d_dy[i], y_dist * factor);
	atomicAdd(&d_dy[j], -y_dist * factor);
}

template <typename vertex_t, int TPB_X = 32, int TPB_Y = 32>
void apply_repulsion(float *x_pos, float *y_pos,
        float *d_dx, float *d_dy, int *d_mass, const float scaling_ratio,
        const vertex_t n) {

    dim3 nthreads, nblocks;
    nthreads.x = min(TPB_X, CUDA_MAX_KERNEL_THREADS);
    nthreads.y = min(TPB_Y, CUDA_MAX_KERNEL_THREADS);
    nthreads.z = 1;
    nblocks.x = min((n + nthreads.x - 1) / nthreads.x, CUDA_MAX_BLOCKS);
    nblocks.y = min((n + nthreads.y - 1) / nthreads.y, CUDA_MAX_BLOCKS);
    nblocks.z = 1;

    repulsion_kernel<vertex_t><<<nthreads, nblocks>>>(x_pos, y_pos,
            d_dx, d_dy, d_mass, scaling_ratio, n);
    CUDA_CHECK_LAST();
} 

} // namespace detail
} // namespace cugraph
