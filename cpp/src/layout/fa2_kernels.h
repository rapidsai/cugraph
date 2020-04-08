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

template <typename vertex_t, typename edge_t, typename value_t>
void init_mass(vertex_t **dests, value_t *d_mass, const edge_t e, const vertex_t n) {
    dim3 nthreads, nblocks;
    nthreads.x = 1024;
    nthreads.y = 1;
    nthreads.z = 1;
    nblocks.x = min((e + nthreads.x - 1) / nthreads.x, CUDA_MAX_BLOCKS);
    nblocks.y = 1;
    nblocks.z = 1;
    degree_coo<vertex_t, value_t><<<nthreads, nblocks>>>(n, e, *dests, d_mass);
}

template <bool weighted, typename vertex_t, typename edge_t, typename weight_t>
cudaStream_t sort_coo(const vertex_t *row, const vertex_t *col,
        const weight_t *v, vertex_t **srcs, vertex_t **dests,
        weight_t **weights, const edge_t e) {

    cudaStream_t stream {nullptr};
    ALLOC_TRY((void**)srcs, sizeof(vertex_t) * e, stream);
    ALLOC_TRY((void**)dests, sizeof(vertex_t) * e, stream);

    CUDA_TRY(cudaMemcpy(*srcs, row, sizeof(vertex_t) * e, cudaMemcpyDefault));
    CUDA_TRY(cudaMemcpy(*dests, col, sizeof(vertex_t) * e, cudaMemcpyDefault));

    if (!weighted) {
        thrust::stable_sort_by_key(rmm::exec_policy(stream)->on(stream),
                *dests, *dests + e, *srcs);
        thrust::stable_sort_by_key(rmm::exec_policy(stream)->on(stream),
                *srcs, *srcs + e, *dests);
    } else {
        ALLOC_TRY((void**)weights, sizeof(weight_t) * e, stream);
        CUDA_TRY(cudaMemcpy(*weights, v, sizeof(weight_t) * e, cudaMemcpyDefault));

        thrust::stable_sort_by_key(rmm::exec_policy(stream)->on(stream),
                *dests, *dests + e,
                thrust::make_zip_iterator(thrust::make_tuple(*srcs, *weights)));
        thrust::stable_sort_by_key(rmm::exec_policy(stream)->on(stream),
                *srcs, *srcs + e,
                thrust::make_zip_iterator(thrust::make_tuple(*dests, *weights)));
    }
    return stream;
}

template <bool weighted, typename vertex_t, typename edge_t, typename weight_t>
__global__ void __launch_bounds__(CUDA_MAX_KERNEL_THREADS)
attraction_kernel(const vertex_t *row, const vertex_t *col,
        const weight_t *v, const edge_t e, float *x_pos,
        float *y_pos, float *d_dx, float *d_dy, int *d_mass,
        bool outbound_attraction_distribution,
        const float edge_weight_influence, const float coef) {
    vertex_t i, src, dst;
    weight_t weight;
    for (i = threadIdx.x + blockIdx.x * blockDim.x;
            i < e;
            i += gridDim.x * blockDim.x) {
        src = row[i];
        dst = col[i];

        if (dst <= src)
            return;

        if (weighted)
            weight = v[i];
        else
            weight = 1;
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
    }
}


template <bool weighted, typename vertex_t, typename edge_t, typename weight_t>
void apply_attraction(const vertex_t *row, const vertex_t *col,
        const weight_t *v, const edge_t e, float *x_pos,
        float *y_pos, float *d_dx, float *d_dy, int *d_mass,
        bool outbound_attraction_distribution,
        const float edge_weight_influence, const float coef) {
    dim3 nthreads, nblocks;
    nthreads.x = 1024;
    nthreads.y = 1;
    nthreads.z = 1;
    nblocks.x = min((e + nthreads.x - 1) / nthreads.x, CUDA_MAX_BLOCKS);
    nblocks.y = 1;
    nblocks.z = 1;

    attraction_kernel<weighted, vertex_t, edge_t, weight_t><<<nthreads, nblocks>>>(
            row,
            col, v, e, x_pos, y_pos, d_dx, d_dy, d_mass,
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
        atomicAdd(&d_dx[i], -x_dist * factor);
        atomicAdd(&d_dy[i], -y_dist * factor);
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
    nthreads.x = 1024;
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
local_speed_kernel(float *x_pos, float *y_pos, float *d_dx, float *d_dy,
        float *d_old_dx, float *d_old_dy, int *d_mass, float *d_swinging,
        float *d_traction, vertex_t n) {

    for (int i = threadIdx.x + blockIdx.x * blockDim.x;
            i < n;
            i += gridDim.x * blockDim.x) {

        float node_swinging = d_mass[i] * sqrt(pow(d_old_dx[i] - d_dx[i], 2) + pow(d_old_dy[i] - d_dy[i], 2));
        float node_traction = 0.5 * d_mass[i] * \
        sqrt((d_old_dx[i] + d_dx[i]) * (d_old_dx[i] + d_dx[i]) + \
            (d_old_dy[i] + d_dy[i]) * (d_old_dy[i] + d_dy[i]));
        d_swinging[i] = node_swinging;
        d_traction[i] = node_traction;
    }
}

template <typename vertex_t>
void compute_local_speed(float *x_pos, float *y_pos, float *d_dx, float *d_dy,
        float *d_old_dx, float *d_old_dy, int *d_mass, float *d_swinging,
        float *d_traction, vertex_t n) {
    dim3 nthreads, nblocks;
    nthreads.x = 1024;
    nthreads.y = 1;
    nthreads.z = 1;
    nblocks.x = min((n + nthreads.x - 1) / nthreads.x, CUDA_MAX_BLOCKS);
    nblocks.y = 1;
    nblocks.z = 1;

    local_speed_kernel<<<nthreads, nblocks>>>(x_pos, y_pos, d_dx, d_dy,
            d_old_dx, d_old_dy, d_mass, d_swinging, d_traction, n);
}

template <typename vertex_t>
void adapt_speed(const float jitter_tolerance, float *jt,
        float *speed,
        float *speed_efficiency, float s, float t,
        const vertex_t n) {

    float estimated_jt = 0.05 * sqrt(n);
    float min_jt = sqrt(estimated_jt);
    float max_jt = 10;
    float target_speed;
    float min_speed_efficiency = 0.05;
    const float max_rise = 0.5;

    *jt = jitter_tolerance * \
        max(min_jt, min(max_jt, estimated_jt * t / (n * n)));

   if (s / t > 2.0) {
        if (*speed_efficiency > min_speed_efficiency) {
            *speed_efficiency *= 0.5;
        }
        *jt = max(*jt, jitter_tolerance);
    }

    if (s == 0)
        target_speed = FLT_MAX;
    else
        target_speed = (*jt * *speed_efficiency * t) / s;

    if (s > *jt * t) {
        if (*speed_efficiency > min_speed_efficiency)
            *speed_efficiency *= .7;
    }
    else if (*speed < 1000)
        *speed_efficiency *= 1.3;

    *speed = *speed + min(target_speed - *speed, max_rise * *speed);
}

template <typename vertex_t>
__global__ void __launch_bounds__(CUDA_MAX_KERNEL_THREADS)
update_positions_kernel(float *x_pos, float *y_pos,
        float *d_dx, float *d_dy, float * d_old_dx, float *d_old_dy,
        float *d_swinging, int *d_mass, const float speed, vertex_t n) {

    for (int i = threadIdx.x + blockIdx.x * blockDim.x;
            i < n;
            i += gridDim.x * blockDim.x) {
        float factor = speed / (1.0 + sqrt(speed * d_swinging[i]));
        x_pos[i] += d_dx[i] * factor;
        y_pos[i] += d_dy[i] * factor;
    }
}

template <typename vertex_t>
void apply_forces(float *x_pos, float *y_pos,
        float *d_dx, float *d_dy, float * d_old_dx, float *d_old_dy,
        float *d_swinging, int *d_mass, const float speed, vertex_t n) {
    dim3 nthreads, nblocks;
    nthreads.x = 1024;
    nthreads.y = 1;
    nthreads.z = 1;
    nblocks.x = min((n + nthreads.x - 1) / nthreads.x, CUDA_MAX_BLOCKS);
    nblocks.y = 1;
    nblocks.z = 1;

	update_positions_kernel<vertex_t><<<nthreads, nblocks>>>(
			x_pos, y_pos, d_dx, d_dy,
			d_old_dx, d_old_dy, d_swinging, d_mass, speed, n);
}

} // namespace detail
} // namespace cugraph
