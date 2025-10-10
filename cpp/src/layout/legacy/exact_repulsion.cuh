/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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
#define restrict           __restrict__
#define CUDA_MAX_BLOCKS_2D 256

namespace cugraph {
namespace detail {

template <typename vertex_t>
__global__ static void repulsion_kernel(const float* restrict x_pos,
                                        const float* restrict y_pos,
                                        float* restrict repel_x,
                                        float* restrict repel_y,
                                        const float* restrict mass,
                                        const float scaling_ratio,
                                        bool prevent_overlapping,
                                        const float* restrict vertex_radius,
                                        const float overlap_scaling_ratio,
                                        const vertex_t n)
{
  int i = (blockIdx.y * blockDim.y) + threadIdx.y;  // for every row
  for (; i < n; i += gridDim.y * blockDim.y) {
    int j = (blockIdx.x * blockDim.x) + threadIdx.x;  // for every item in row
    for (; j < i; j += gridDim.x * blockDim.x) {
      float factor;
      float x_dist      = x_pos[i] - x_pos[j];
      float y_dist      = y_pos[i] - y_pos[j];
      float distance_sq = x_dist * x_dist + y_dist * y_dist + FLT_EPSILON;
      if (prevent_overlapping) {
        float radius_i = vertex_radius[i];
        float radius_j = vertex_radius[j];
        float distance = sqrt(distance_sq);
        if (distance <= radius_i + radius_j) {
          // Overlapping
          factor = overlap_scaling_ratio * mass[i] * mass[j] / distance;
        } else {
          // Non-overlapping
          float distance_inter = distance - radius_i - radius_j + FLT_EPSILON;
          factor               = scaling_ratio * mass[i] * mass[j] / (distance * distance_inter);
        }
      } else {
        factor = scaling_ratio * mass[i] * mass[j] / distance_sq;
      }
      // Add forces
      atomicAdd(&repel_x[i], x_dist * factor);
      atomicAdd(&repel_x[j], -x_dist * factor);
      atomicAdd(&repel_y[i], y_dist * factor);
      atomicAdd(&repel_y[j], -y_dist * factor);
    }
  }
}

template <typename vertex_t, int TPB_X = 32, int TPB_Y = 32>
void apply_repulsion(const float* restrict x_pos,
                     const float* restrict y_pos,
                     float* restrict repel_x,
                     float* restrict repel_y,
                     const float* restrict mass,
                     const float scaling_ratio,
                     bool prevent_overlapping,
                     const float* restrict vertex_radius,
                     const float overlap_scaling_ratio,
                     const vertex_t n,
                     cudaStream_t stream)
{
  dim3 nthreads(TPB_X, TPB_Y);
  dim3 nblocks(static_cast<int>(
                 min(n + static_cast<vertex_t>(nthreads.x) - 1 / static_cast<vertex_t>(nthreads.x),
                     static_cast<vertex_t>(CUDA_MAX_BLOCKS_2D))),
               static_cast<int>(
                 min(n + static_cast<vertex_t>(nthreads.y) - 1 / static_cast<vertex_t>(nthreads.y),
                     static_cast<vertex_t>(CUDA_MAX_BLOCKS_2D))));

  repulsion_kernel<vertex_t><<<nblocks, nthreads, 0, stream>>>(x_pos,
                                                               y_pos,
                                                               repel_x,
                                                               repel_y,
                                                               mass,
                                                               scaling_ratio,
                                                               prevent_overlapping,
                                                               vertex_radius,
                                                               overlap_scaling_ratio,
                                                               n);
  RAFT_CHECK_CUDA(stream);
}

}  // namespace detail
}  // namespace cugraph
