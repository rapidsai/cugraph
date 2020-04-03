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

template <typename vertex_t, typename edge_t, typename weight_t>
__global__ void __launch_bounds__(CUDA_MAX_KERNEL_THREADS)
attraction_kernel(const vertex_t *row, const vertex_t *col,
        const weight_t *v, const vertex_t n, float *x_pos,
        float *y_pos, float *d_dx, float *d_dy, int *d_mass,
        bool outbound_attraction_distribution,
        const float edge_weight_influence, const float coef) {
    vertex_t i, src, dst;
    weight_t weight;
    for (i = threadIdx.x + blockIdx.x * blockDim.x;
            i < n;
            i += gridDim.x * blockDim.x) {
        src = row[i];
        dst = col[i];

        if (src > dst)
            return;

        if (v != nullptr)
            weight = v[i];

        if (v == nullptr || edge_weight_influence == 0)
            weight = 1;
        else
            weight = pow(weight, edge_weight_influence);

        float x_dist = x_pos[src] - x_pos[dst];
        float y_dist = y_pos[src] - y_pos[dst];
        float factor = 0;

        if (outbound_attraction_distribution)
            factor = -coef * weight / d_mass[src]; 
        else
            factor = -coef * weight;

        atomicAdd(&d_dx[src], x_dist * factor);
        atomicAdd(&d_dy[src], y_dist * factor);
        atomicAdd(&d_dx[dst], -x_dist * factor);
        atomicAdd(&d_dy[dst], -y_dist * factor);
        printf("tid: %i, pos_x: %f, pos_y: %f, dx: %f, dy: %f\n",
                i, x_pos[i], y_pos[i], d_dx[i], d_dy[i]);
    }
}


template <typename vertex_t, typename edge_t, typename weight_t>
void apply_attraction(const vertex_t *row, const vertex_t *col,
        const weight_t *v, const vertex_t n, float *x_pos,
        float *y_pos, float *d_dx, float *d_dy, int *d_mass,
        bool outbound_attraction_distribution,
        const float edge_weight_influence, const float coef) {
    dim3 nthreads, nblocks;
    nthreads.x = min(n, CUDA_MAX_KERNEL_THREADS);
    nthreads.y = 1;
    nthreads.z = 1;
    nblocks.x = min((n + nthreads.x - 1) / nthreads.x, CUDA_MAX_BLOCKS);
    nblocks.y = 1;
    nblocks.z = 1;

    attraction_kernel<vertex_t, edge_t, weight_t><<<nthreads, nblocks>>>(
            row,
            col, v, n, x_pos, y_pos, d_dx, d_dy, d_mass,
            outbound_attraction_distribution,
            edge_weight_influence, coef);
}

template <typename vertex_t>
__global__ void __launch_bounds__(CUDA_MAX_KERNEL_THREADS)
linear_gravity_kernel(float *x_pos, float *y_pos, int *d_mass, float *d_dx,
        float *d_dy, const float gravity, const vertex_t n) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x;
            i < n;
            i += gridDim.x * blockDim.x) {

        float x_dist = x_pos[i];
        float y_dist = y_pos[i];
        float distance = sqrt(x_dist * x_dist + y_dist * y_dist);
        distance += FLT_EPSILON;
        float factor = d_mass[i] * gravity / distance;
        d_dx[i] -= x_dist * factor;
        d_dy[i] -= y_dist * factor;

        //printf("tid: %i, pos_x: %f, pos_y: %f, dx: %f, dy: %f\n",
        //    i, x_pos[i], y_pos[i], d_dx[i], d_dy[i]);
    }

}

template <typename vertex_t>
__global__ void __launch_bounds__(CUDA_MAX_KERNEL_THREADS)
strong_gravity_kernel(float *x_pos, float *y_pos, int *d_mass, float *d_dx,
        float *d_dy, const float gravity, const float scaling_ratio,
        const vertex_t n) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x;
            i < n;
            i += gridDim.x * blockDim.x) {
        float x_dist = x_pos[i];
        float y_dist = y_pos[i];

        float factor = scaling_ratio * d_mass[i] * gravity;
        d_dx[i] -= x_dist * factor;
        d_dy[i] -= y_dist * factor;
    }
}

template <typename vertex_t>
void apply_gravity(float *x_pos, float *y_pos, int *d_mass, float *d_dx,
        float *d_dy, const float gravity, bool strong_gravity_mode,
        const float scaling_ratio, const vertex_t n) {
    dim3 nthreads, nblocks;
    nthreads.x = min(n, CUDA_MAX_KERNEL_THREADS);
    nthreads.y = 1;
    nthreads.z = 1;
    nblocks.x = min((n + nthreads.x - 1) / nthreads.x, CUDA_MAX_BLOCKS);
    nblocks.y = 1;
    nblocks.z = 1;

    if (strong_gravity_mode)
        strong_gravity_kernel<vertex_t><<<nthreads, nblocks>>>(
                x_pos, y_pos, d_mass,
                d_dx, d_dy, gravity, scaling_ratio, n);
    else
        linear_gravity_kernel<vertex_t><<<nthreads, nblocks>>>(
                x_pos, y_pos, d_mass,
                d_dx, d_dy, gravity, n);
}

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
    d_dx[i] += fx;
    d_dy[i] += fy;
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

template <typename vertex_t>
__global__ void __launch_bounds__(CUDA_MAX_KERNEL_THREADS)
local_speed_kernel(float *d_dx, float *d_dy, float *d_old_dx, float *d_old_dy,
        int *d_mass, float *d_swinging, float *d_traction, vertex_t n) {
    // TODO: Use shared memory reduction
    for (int i = threadIdx.x + blockIdx.x * blockDim.x;
            i < n;
            i += gridDim.x * blockDim.x) {
        float tmp_x = d_old_dx[i] - d_dx[i];
        float tmp_y = d_old_dy[i] - d_dy[i];
        float node_swinging = sqrt(tmp_x * tmp_x + tmp_y * tmp_y);
        atomicAdd(d_swinging, d_mass[i] * node_swinging);
        atomicAdd(d_traction, 0.5 * d_mass[i] * \
        sqrt((d_old_dx[i] + d_dx[i]) * (d_old_dx[i] + d_dx[i]) + \
            (d_old_dy[i] + d_dy[i]) * (d_old_dy[i] + d_dy[i])));
    }
}

template <typename vertex_t>
int compute_jitter_tolerance(const float jitter_tolerance,
        float speed_efficiency, float *d_swinging, float *d_traction,
        const vertex_t n) {
    float estimated_jt = 0.05 * sqrt(n);
    float min_jt = sqrt(estimated_jt);
    float max_jt = 10;
    float jt = jitter_tolerance * \
        max(min_jt, min(max_jt, estimated_jt * *d_traction / (n * n)));
    float min_speed_efficiency = 0.05;
    if (*d_swinging / *d_traction > 2.0) {
        if (speed_efficiency > min_speed_efficiency) {
            speed_efficiency *= 0.5;
        }
        jt = max(jt, jitter_tolerance);
    }
    return jt;
}

float compute_global_speed(float speed, float speed_efficiency,
        const float jt, float *d_swinging, float *d_traction) {

    float target_speed;
    float min_speed_efficiency = 0.05;

    if (*d_swinging == 0)
        target_speed = FLT_MAX;
    else
        target_speed = (jt * speed_efficiency * *d_traction) / *d_swinging;

    if (*d_swinging > jt * *d_traction) {
        if (speed_efficiency > min_speed_efficiency)
            speed_efficiency *= .7;
        else if (speed < 1000)
            speed_efficiency *= 1.3;
    }
    const float max_rise = 0.5;
    speed = speed + min(target_speed - speed, max_rise * speed);
    return speed;
}

template <typename vertex_t>
__global__ void __launch_bounds__(CUDA_MAX_KERNEL_THREADS)
update_positions_kernel(float *x_pos, float *y_pos,
        float *d_dx, float *d_dy, float * d_old_dx, float *d_old_dy,
        const float speed, vertex_t n) {

    for (int i = threadIdx.x + blockIdx.x * blockDim.x;
            i < n;
            i += gridDim.x * blockDim.x) {
        float tmp_x = d_old_dx[i] - d_dx[i];
        float tmp_y = d_old_dy[i] - d_dy[i];
        float local_swinging = sqrt(tmp_x * tmp_x + tmp_y * tmp_y);
        float factor = speed / (1.0 + sqrt(speed * local_swinging));
        x_pos[i] += d_dx[i] * factor;
        y_pos[i] += d_dy[i] * factor;
    }
}

} // namespace detail
} // namespace cugraph
