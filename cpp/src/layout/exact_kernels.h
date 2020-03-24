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

template <typename vertex_t, typename edge_t>
__global__ void init_mass(const edge_t *csrPtr, const vertex_t *csrInd,
        int *d_mass, const vertex_t n) {
    vertex_t row;
    edge_t start, end, degree;
    for (row = threadIdx.x + blockIdx.x * blockDim.x;
            row < n;
            row += gridDim.x * blockDim.x) {
        start = csrPtr[row];
        end = csrPtr[row + 1];
        degree = start - end;
        // FA2's model is based on mass being deg(n) + 1.
        d_mass[row] = degree + 1;
    } 
}

template <typename vertex_t, typename edge_t, typename weight_t>
__global__ void attraction_kernel(const edge_t *csrPtr, const vertex_t *csrInd,
        const weight_t *v, const vertex_t n, float *x_pos,
        float *y_pos, float *d_dx, float *d_dy, int *d_mass,
        bool outbound_attraction_distribution,
        const float edge_weight_influence, const float coef) {

    vertex_t row, col, j;
    edge_t start, end;
    weight_t weight;
    for (row = threadIdx.x + blockIdx.x * blockDim.x;
            row < n;
            row += gridDim.x * blockDim.x) {
        start = csrPtr[row];
        end = csrPtr[row + 1];
        for (j = start;
                j < end;
                ++j) {
            col = csrInd[j];
            weight = v[j];

            if (edge_weight_influence == 0)
                weight = 1;
            else
                weight = pow(weight, edge_weight_influence);

            float x_dist = x_pos[row] - x_pos[col];
            float y_dist = y_pos[row] - y_pos[col];
            float factor = 0;

            if (outbound_attraction_distribution)
                factor = -(coef * weight);
            else
                factor = -(coef * weight) / d_mass[row]; 

            d_dx[row] += x_dist * factor;
            d_dy[row] += y_dist * factor;
            d_dx[col] += -(x_dist * factor);
            d_dy[col] += -(y_dist * factor);
        }
    }
}


template <typename vertex_t, typename edge_t, typename weight_t>
void apply_attraction(const edge_t *csrPtr, const vertex_t *csrInd,
        const weight_t *v, const vertex_t n, float *x_pos,
        float *y_pos, float *d_dx, float *d_dy, int *d_mass,
        bool outbound_attraction_distribution,
        const float edge_weight_influence, const float coef) {
    attraction_kernel<vertex_t, edge_t, weight_t><<<ceil(1024 / n), 1024>>>(
            csrPtr,
            csrInd, v, n, x_pos, y_pos, d_dx, d_dy, d_mass,
            outbound_attraction_distribution,
            edge_weight_influence, coef);
}

template <typename vertex_t>
__global__ void
linear_gravity_kernel(float *x_pos, float *y_pos, int *d_mass,
        const float gravity, const vertex_t n) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x;
            i < n;
            i += blockIdx.x * blockDim.x) {
        float x_dist = x_pos[i];
        float y_dist = y_pos[i];
        float distance = std::sqrt(x_dist * x_dist + y_dist * y_dist);

        if (distance == 0)
            return;
        float factor = (d_mass[i] * gravity) / distance;
        x_pos[i] += -(x_dist * factor);
        y_pos[i] += -(y_dist * factor);
    }
}

template <typename vertex_t>
__global__ void
strong_gravity_kernel(float *x_pos, float *y_pos, int *d_mass,
        const float gravity, const float scaling_ratio, const vertex_t n) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x;
            i < n;
            i += blockIdx.x * blockDim.x) {
        float x_dist = x_pos[i];
        float y_dist = y_pos[i];

        float factor = scaling_ratio * d_mass[i] * gravity;
        x_pos[i] += -(x_dist * factor);
        y_pos[i] += -(y_dist * factor);
    }
}

template <typename vertex_t>
void apply_gravity(float *x_pos, float *y_pos, int *d_mass,
        const float gravity, bool strong_gravity_mode,
        const float scaling_ratio, const vertex_t n) {
    if (strong_gravity_mode)
        strong_gravity_kernel<vertex_t><<<ceil(1024 / n), 1024>>>(x_pos, y_pos, d_mass,
                gravity, scaling_ratio, n);
    else
        linear_gravity_kernel<vertex_t><<<ceil(1024 / n), 1024>>>(x_pos, y_pos, d_mass,
                gravity, n);
}

template <typename vertex_t>
__global__ void
repulsion_kernel(float *x_pos, float *y_pos,
        float *d_dx, float *d_dy, int *d_mass, const float scaling_ratio,
        const vertex_t n) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n)
        return;

    float fx = 0.0f; float fy = 0.0f;

    for (int j = 0; j < n; ++j) {
        float x_dist = x_pos[j] - x_pos[i];
        float y_dist = y_pos[j] - y_pos[i];
        float distance = x_dist * x_dist + y_dist * y_dist;
        float factor = scaling_ratio * d_mass[i] * d_mass[j] / distance;
        fx += x_dist * factor;
        fy += y_dist * factor;
    }
    x_pos[i] += fx;
    y_pos[i] += fy;
}

template <typename vertex_t>
void apply_repulsion(float *x_pos, float *y_pos,
        float *d_dx, float *d_dy, int *d_mass, const float scaling_ratio,
        const vertex_t n) {
    repulsion_kernel<vertex_t><<<1, 1>>>(x_pos, y_pos,
            d_dx, d_dy, d_mass, scaling_ratio, n);
} 

template <typename vertex_t>
__global__ void
local_speed_kernel(float *d_dx, float *d_dy, float *d_old_dx, float *d_old_dy,
        int *d_mass, float *total_swinging, float *traction, vertex_t n) {
    // For every node
    // TODO: Use shared memory /  parallel sum
    for (int i = threadIdx.x + blockIdx.x * blockDim.x;
            i < n;
            i += blockIdx.x * blockDim.x) {
        float tmp_x = d_old_dx[i] - d_dx[i];
        float tmp_y = d_old_dy[i] - d_dy[i];
        float node_swinging = std::sqrt(tmp_x * tmp_x + tmp_y * tmp_y);
        atomicAdd(total_swinging, d_mass[i] * node_swinging);
        atomicAdd(traction, 0.5 * d_mass[i] * \
        std::sqrt((d_old_dx[i] + d_dx[i]) * (d_old_dx[i] + d_dx[i]) + \
            (d_old_dy[i] + d_dy[i]) * (d_old_dy[i] + d_dy[i])));
    }
}

template <typename vertex_t>
int compute_jitter_tolerance(const float jitter_tolerance,
        float speed_efficiency, float swinging, float traction,
        const vertex_t n) {
    float estimated_jt = 0.05 * std::sqrt(n);
    float min_jt = std::sqrt(estimated_jt);
    float max_jt = 10;
    float jt = jitter_tolerance * \
        max(min_jt, min(max_jt, estimated_jt * traction / (n * n)));
    float min_speed_efficiency = 0.05;
    if (swinging / traction > 2.0) {
        if (speed_efficiency > min_speed_efficiency) {
            speed_efficiency *= 0.5;
        }
        jt = max(jt, jitter_tolerance);
    }
    return jt;
}

float compute_global_speed(float speed, float speed_efficiency,
        const float jt, const float swinging, const float traction) {

    float target_speed;
    float min_speed_efficiency = 0.05;

    if (swinging == 0)
        target_speed = FLT_MAX;
    else
        target_speed = (jt * speed_efficiency * traction) / swinging;

    if (swinging > jt * traction) {
        if (speed_efficiency > min_speed_efficiency)
            speed_efficiency *= .7;
        else if (speed < 1000)
            speed_efficiency *= 1.3;
    }
    const float max_rise = 0.5;
    speed = speed + std::min(target_speed - speed, max_rise * speed);
    return speed;
}

template <typename vertex_t>
__global__ void
update_positions_kernel(float *x_pos, float *y_pos,
        float *d_dx, float *d_dy, float * d_old_dx, float *d_old_dy,
        const float speed, vertex_t n) {

    for (int i = threadIdx.x + blockIdx.x * blockDim.x;
            i < n;
            i += blockIdx.x * blockDim.x) {
        float tmp_x = d_old_dx[i] - d_dx[i];
        float tmp_y = d_old_dy[i] - d_dy[i];
        float local_swinging = std::sqrt(tmp_x * tmp_x + tmp_y * tmp_y);
        float factor = speed / (1.0 + std::sqrt(speed * local_swinging));
        x_pos[i] += d_dx[i] * factor;
        y_pos[i] += d_dy[i] * factor;
    }
}

template <typename vertex_t>
float apply_forces(float *x_pos, float *y_pos, float *d_dx, float *d_dy, 
        float *d_old_dx, float *d_old_dy, int *d_mass,
        const float jitter_tolerance,
        float speed, float speed_efficiency, const vertex_t n){
    float swinging = 0;
    float traction = 0;
    local_speed_kernel<<<ceil(1024 / n), 1024>>>(d_dx, d_dy, d_old_dx, d_old_dy, d_mass,
            &swinging, &traction, n);
    float jt = compute_jitter_tolerance<vertex_t>(jitter_tolerance,
            speed_efficiency, swinging, traction, n);
   speed = compute_global_speed(speed, speed_efficiency, jt, swinging,
           traction); 
   update_positions_kernel<vertex_t><<<ceil(1024 / n), 1024>>>(x_pos, y_pos, d_dx, d_dy,
           d_old_dx, d_old_dy, speed, n);
   return speed;
}

} // namespace detail
} // namespace cugraph
