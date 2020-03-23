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
__global__ void attraction_kernel(const edge_t *csrPtr, const vertex_t *csrInd,
        const weight_t *v, const vertex_t n, float *x_pos,
        float *y_pos, float *x_start, float *y_start,
        bool outbount_attraction_distribution, bool lin_log_mode,
        bool prevent_overlapping, float edge_weight_influence,
        float jitter_tolerance, float scaling_ratio,
        bool strong_gravity_mode, float gravity, float *attraction) {

}

template <typename vertex_t, typename edge_t, typename weight_t>
__global__ void repulsion_kernel(const edge_t *csrPtr, const vertex_t *csrInd,
        const weight_t *v, const vertex_t n, float *x_pos,
        float *y_pos, float *x_start, float *y_start,
        bool outbount_attraction_distribution, bool lin_log_mode,
        bool prevent_overlapping, float edge_weight_influence,
        float jitter_tolerance, float scaling_ratio,
        bool strong_gravity_mode, float gravity) {
    return;
}

__global__ void linear_gravity_kernel() {

}

__global__ void strong_gravity_kernel() {

}


template <typename vertex_t, typename edge_t, typename weight_t>
__global__ void apply_kernel(const edge_t *csrPtr, const vertex_t *csrInd,
        const weight_t *v, const vertex_t n,
        float *x_pos, float *y_pos, float *x_start, float *y_start,
        bool outbount_attraction_distribution, bool lin_log_mode,
        bool prevent_overlapping, float edge_weight_influence,
        float jitter_tolerance, float scaling_ratio,
        bool strong_gravity_mode, float gravity) {
    return;
}

template <typename vertex_t, typename edge_t, typename weight_t>
void compute_attraction(const edge_t *csrPtr, const vertex_t *csrInd,
        const weight_t *v, const vertex_t n, float *x_pos,
        float *y_pos, float *x_start, float *y_start,
        bool outbount_attraction_distribution, bool lin_log_mode,
        bool prevent_overlapping, float edge_weight_influence,
        float jitter_tolerance, float scaling_ratio,
        bool strong_gravity_mode, float gravity,
        float *d_attraction) {
    return;
}

template <typename vertex_t, typename edge_t, typename weight_t>
void compute_repulsion(const edge_t *csrPtr, const vertex_t *csrInd,
        const weight_t *v, const vertex_t n, float *x_pos,
        float *y_pos, float *x_start, float *y_start,
        bool outbount_attraction_distribution, bool lin_log_mode,
        bool prevent_overlapping, float edge_weight_influence,
        float jitter_tolerance, float scaling_ratio,
        bool strong_gravity_mode, float gravity,
        float *d_repulsion) {
    return;
}

template <typename vertex_t, typename edge_t, typename weight_t>
void apply_forces(const edge_t *csrPtr, const vertex_t *csrInd,
        const weight_t *v, const vertex_t n, float *x_pos,
        float *y_pos, float *x_start, float *y_start,
        bool outbount_attraction_distribution, bool lin_log_mode,
        bool prevent_overlapping, float edge_weight_influence,
        float jitter_tolerance, float scaling_ratio,
        bool strong_gravity_mode, float gravity,
        float *d_attraction, float *d_repulsion) {
    return;
}

} // namespace detail
} // namespace cugraph
