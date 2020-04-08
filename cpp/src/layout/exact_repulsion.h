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
#define restrict __restrict__ 

namespace cugraph {
namespace detail {

template <typename vertex_t>
__global__ void
repulsion_kernel(const float *restrict x_pos, const float *restrict y_pos,
        float *restrict repel_x, float *restrict repel_y,
        const int *restrict mass, const float scaling_ratio,
        const vertex_t n) {
    const int j =
		(blockIdx.x * blockDim.x) + threadIdx.x;  // for every item in row
	const int i = (blockIdx.y * blockDim.y) + threadIdx.y;  // for every row
	if (j >= i || i >= n || j >= n) return;

	float x_dist = x_pos[i] - x_pos[j];
	float y_dist = y_pos[i] - y_pos[j];
	float distance = x_dist * x_dist + y_dist * y_dist;
	distance += FLT_EPSILON;
	float factor = scaling_ratio * mass[i] * mass[j] / distance;
	// Add forces
	atomicAdd(&repel_x[i], x_dist * factor);
	atomicAdd(&repel_x[j], -x_dist * factor);
	atomicAdd(&repel_y[i], y_dist * factor);
	atomicAdd(&repel_y[j], -y_dist * factor);
}

template <typename vertex_t, int TPB_X = 32, int TPB_Y = 32>
void apply_repulsion(const float *restrict x_pos, const float *restrict y_pos,
        float *restrict repel_x, float *restrict repel_y,
        const int *restrict mass, const float scaling_ratio,
        const vertex_t n) {

    dim3 nthreads(TPB_X, TPB_Y);
    dim3 nblocks(min((n + nthreads.x - 1) / nthreads.x, CUDA_MAX_BLOCKS),
    min((n + nthreads.y - 1) / nthreads.y, CUDA_MAX_BLOCKS));

    repulsion_kernel<vertex_t><<<nblocks, nthreads>>>(x_pos, y_pos,
            repel_x, repel_y, mass, scaling_ratio, n);
    CUDA_CHECK_LAST();
} 

} // namespace detail
} // namespace cugraph
