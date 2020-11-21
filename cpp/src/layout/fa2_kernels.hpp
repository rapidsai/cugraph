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

#include "utilities/graph_utils.cuh"

namespace cugraph {
namespace detail {

template <typename vertex_t, typename edge_t, typename weight_t>
__global__ void attraction_kernel(const vertex_t *restrict row,
                                  const vertex_t *restrict col,
                                  const weight_t *restrict v,
                                  const edge_t e,
                                  const float *restrict x_pos,
                                  const float *restrict y_pos,
                                  float *restrict attract_x,
                                  float *restrict attract_y,
                                  const int *restrict mass,
                                  bool outbound_attraction_distribution,
                                  bool lin_log_mode,
                                  const float edge_weight_influence,
                                  const float coef)
{
  vertex_t i, src, dst;
  weight_t weight = 1;
  // For every edge
  for (i = threadIdx.x + blockIdx.x * blockDim.x; i < e; i += gridDim.x * blockDim.x) {
    src = row[i];
    dst = col[i];

    // We only need the lower triangular part
    if (dst <= src) return;

    if (v) { weight = v[i]; }
    weight = pow(weight, edge_weight_influence);

    float x_dist = x_pos[src] - x_pos[dst];
    float y_dist = y_pos[src] - y_pos[dst];
    float factor = -coef * weight;

    if (lin_log_mode) {
      float distance = pow(x_dist, 2) + pow(y_dist, 2);
      distance += FLT_EPSILON;
      distance = sqrt(distance);
      factor *= log(1 + distance) / distance;
    }
    if (outbound_attraction_distribution) factor /= mass[src];

    // Force computation
    atomicAdd(&attract_x[src], x_dist * factor);
    atomicAdd(&attract_y[src], y_dist * factor);
    atomicAdd(&attract_x[dst], -x_dist * factor);
    atomicAdd(&attract_y[dst], -y_dist * factor);
  }
}

template <typename vertex_t, typename edge_t, typename weight_t>
void apply_attraction(const vertex_t *restrict row,
                      const vertex_t *restrict col,
                      const weight_t *restrict v,
                      const edge_t e,
                      const float *restrict x_pos,
                      const float *restrict y_pos,
                      float *restrict attract_x,
                      float *restrict attract_y,
                      const int *restrict mass,
                      bool outbound_attraction_distribution,
                      bool lin_log_mode,
                      const float edge_weight_influence,
                      const float coef,
                      cudaStream_t stream)
{
  // 0 edge graph.
  if (!e) return;

  dim3 nthreads, nblocks;
  nthreads.x = min(e, CUDA_MAX_KERNEL_THREADS);
  nthreads.y = 1;
  nthreads.z = 1;
  nblocks.x  = min((e + nthreads.x - 1) / nthreads.x, CUDA_MAX_BLOCKS);
  nblocks.y  = 1;
  nblocks.z  = 1;

  attraction_kernel<vertex_t, edge_t, weight_t>
    <<<nblocks, nthreads, 0, stream>>>(row,
                                       col,
                                       v,
                                       e,
                                       x_pos,
                                       y_pos,
                                       attract_x,
                                       attract_y,
                                       mass,
                                       outbound_attraction_distribution,
                                       lin_log_mode,
                                       edge_weight_influence,
                                       coef);

  CHECK_CUDA(stream);
}

template <typename vertex_t>
__global__ void linear_gravity_kernel(const float *restrict x_pos,
                                      const float *restrict y_pos,
                                      float *restrict attract_x,
                                      float *restrict attract_y,
                                      const int *restrict mass,
                                      const float gravity,
                                      const vertex_t n)
{
  // For every node.
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += gridDim.x * blockDim.x) {
    float x_dist   = x_pos[i];
    float y_dist   = y_pos[i];
    float distance = sqrt(x_dist * x_dist + y_dist * y_dist + FLT_EPSILON);
    float factor   = mass[i] * gravity / distance;
    attract_x[i] -= x_dist * factor;
    attract_y[i] -= y_dist * factor;
  }
}

template <typename vertex_t>
__global__ void strong_gravity_kernel(const float *restrict x_pos,
                                      const float *restrict y_pos,
                                      float *restrict attract_x,
                                      float *restrict attract_y,
                                      const int *restrict mass,
                                      const float gravity,
                                      const float scaling_ratio,
                                      const vertex_t n)
{
  // For every node.
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += gridDim.x * blockDim.x) {
    float x_dist = x_pos[i];
    float y_dist = y_pos[i];

    float factor = scaling_ratio * mass[i] * gravity;
    attract_x[i] -= x_dist * factor;
    attract_y[i] -= y_dist * factor;
  }
}

template <typename vertex_t>
void apply_gravity(const float *restrict x_pos,
                   const float *restrict y_pos,
                   float *restrict attract_x,
                   float *restrict attract_y,
                   const int *restrict mass,
                   const float gravity,
                   bool strong_gravity_mode,
                   const float scaling_ratio,
                   const vertex_t n,
                   cudaStream_t stream)
{
  dim3 nthreads, nblocks;
  nthreads.x = min(n, CUDA_MAX_KERNEL_THREADS);
  nthreads.y = 1;
  nthreads.z = 1;
  nblocks.x  = min((n + nthreads.x - 1) / nthreads.x, CUDA_MAX_BLOCKS);
  nblocks.y  = 1;
  nblocks.z  = 1;

  if (strong_gravity_mode) {
    strong_gravity_kernel<vertex_t><<<nblocks, nthreads, 0, stream>>>(
      x_pos, y_pos, attract_x, attract_y, mass, gravity, scaling_ratio, n);
  } else {
    linear_gravity_kernel<vertex_t>
      <<<nblocks, nthreads, 0, stream>>>(x_pos, y_pos, attract_x, attract_y, mass, gravity, n);
  }
  CHECK_CUDA(stream);
}

template <typename vertex_t>
__global__ void local_speed_kernel(const float *restrict repel_x,
                                   const float *restrict repel_y,
                                   const float *restrict attract_x,
                                   const float *restrict attract_y,
                                   const float *restrict old_dx,
                                   const float *restrict old_dy,
                                   const int *restrict mass,
                                   float *restrict swinging,
                                   float *restrict traction,
                                   const vertex_t n)
{
  // For every node.
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += gridDim.x * blockDim.x) {
    const float dx      = repel_x[i] + attract_x[i];
    const float dy      = repel_y[i] + attract_y[i];
    float node_swinging = mass[i] * sqrt(pow(old_dx[i] - dx, 2) + pow(old_dy[i] - dy, 2));
    float node_traction = 0.5 * mass[i] * sqrt(pow(old_dx[i] + dx, 2) + pow(old_dy[i] + dy, 2));
    swinging[i]         = node_swinging;
    traction[i]         = node_traction;
  }
}

template <typename vertex_t>
void compute_local_speed(const float *restrict repel_x,
                         const float *restrict repel_y,
                         const float *restrict attract_x,
                         const float *restrict attract_y,
                         float *restrict old_dx,
                         float *restrict old_dy,
                         const int *restrict mass,
                         float *restrict swinging,
                         float *restrict traction,
                         const vertex_t n,
                         cudaStream_t stream)
{
  dim3 nthreads, nblocks;
  nthreads.x = min(n, CUDA_MAX_KERNEL_THREADS);
  nthreads.y = 1;
  nthreads.z = 1;
  nblocks.x  = min((n + nthreads.x - 1) / nthreads.x, CUDA_MAX_BLOCKS);
  nblocks.y  = 1;
  nblocks.z  = 1;

  local_speed_kernel<<<nblocks, nthreads, 0, stream>>>(
    repel_x, repel_y, attract_x, attract_y, old_dx, old_dy, mass, swinging, traction, n);
  CHECK_CUDA(stream);
}

template <typename vertex_t>
void adapt_speed(const float jitter_tolerance,
                 float *restrict jt,
                 float *restrict speed,
                 float *restrict speed_efficiency,
                 const float s,
                 const float t,
                 const vertex_t n)
{
  float estimated_jt = 0.05 * sqrt(n);
  float min_jt       = sqrt(estimated_jt);
  float max_jt       = 10;
  float target_speed;
  float min_speed_efficiency = 0.05;
  const float max_rise       = 0.5;

  *jt = jitter_tolerance * max(min_jt, min(max_jt, estimated_jt * t / (n * n)));

  if (s / t > 2.0) {
    if (*speed_efficiency > min_speed_efficiency) { *speed_efficiency *= 0.5; }
    *jt = max(*jt, jitter_tolerance);
  }

  if (s == 0)
    target_speed = FLT_MAX;
  else
    target_speed = (*jt * *speed_efficiency * t) / s;

  if (s > *jt * t) {
    if (*speed_efficiency > min_speed_efficiency) *speed_efficiency *= .7;
  } else if (*speed < 1000)
    *speed_efficiency *= 1.3;

  *speed = *speed + min(target_speed - *speed, max_rise * *speed);
}

template <typename vertex_t>
__global__ void update_positions_kernel(float *restrict x_pos,
                                        float *restrict y_pos,
                                        const float *restrict repel_x,
                                        const float *restrict repel_y,
                                        const float *restrict attract_x,
                                        const float *restrict attract_y,
                                        float *restrict old_dx,
                                        float *restrict old_dy,
                                        const float *restrict swinging,
                                        const float speed,
                                        const vertex_t n)
{
  // For every node.
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += gridDim.x * blockDim.x) {
    const float factor = speed / (1.0 + sqrt(speed * swinging[i]));
    const float dx     = (repel_x[i] + attract_x[i]);
    const float dy     = (repel_y[i] + attract_y[i]);

    x_pos[i] += dx * factor;
    y_pos[i] += dy * factor;
    old_dx[i] = dx;
    old_dy[i] = dy;
  }
}

template <typename vertex_t>
void apply_forces(float *restrict x_pos,
                  float *restrict y_pos,
                  const float *restrict repel_x,
                  const float *restrict repel_y,
                  const float *restrict attract_x,
                  const float *restrict attract_y,
                  float *restrict old_dx,
                  float *restrict old_dy,
                  const float *restrict swinging,
                  const float speed,
                  const vertex_t n,
                  cudaStream_t stream)
{
  dim3 nthreads, nblocks;
  nthreads.x = min(n, CUDA_MAX_KERNEL_THREADS);
  nthreads.y = 1;
  nthreads.z = 1;
  nblocks.x  = min((n + nthreads.x - 1) / nthreads.x, CUDA_MAX_BLOCKS);
  nblocks.y  = 1;
  nblocks.z  = 1;

  update_positions_kernel<vertex_t><<<nblocks, nthreads, 0, stream>>>(
    x_pos, y_pos, repel_x, repel_y, attract_x, attract_y, old_dx, old_dy, swinging, speed, n);
  CHECK_CUDA(stream);
}

}  // namespace detail
}  // namespace cugraph
