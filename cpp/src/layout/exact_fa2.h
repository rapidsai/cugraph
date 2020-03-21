#pragma once

#include <rmm_utils.h>
#include <rmm/thrust_rmm_allocator.h>
#include <rmm/device_buffer.hpp>

#include "exact_kernels.h"
#include "utils.h"

namespace cugraph {
namespace ForceAtlas2 {

template <typename IndexType, typename ValueType>
void exact_fa2(const IndexType *row, const IndexType *col,
               const ValueType *val, const int nnz,
               float *x_pos, float *y_pos, int max_iter=1000,
               float *x_start=nullptr, float *y_start=nullptr,
               bool outbound_attraction_distribution=false,
               bool lin_log_mode=false, bool prevent_overlapping=false,
               float edge_weight_influence=1.0, float jitter_tolerance=1.0,
               float scaling_ratio=2.0, bool strong_gravity_mode=false,
               float gravity=1.0) { 
    
    float *d_attraction{nullptr};
    float *d_repulsion{nullptr};

    rmm::device_vector<float> attraction(nnz, 0);
    rmm::device_vector<float> repulsion(nnz, 0);

    d_attraction = attraction.data().get();
    d_repulsion = repulsion.data().get();

    if (x_start == nullptr || y_start == nullptr) {
        // TODO: generate random numbers
        return;
    }

    for (int iter=0; iter < max_iter; ++iter) {
        ForceAtlas2::compute_attraction<IndexType, ValueType>(
                row, col, val, nnz,
                x_pos, y_pos, x_start, y_start,
                outbound_attraction_distribution,
                lin_log_mode,
                prevent_overlapping,
                edge_weight_influence,
                jitter_tolerance,
                scaling_ratio,
                strong_gravity_mode,
                gravity,
                d_attraction);

        ForceAtlas2::compute_repulsion<IndexType, ValueType>(
                row, col, val, nnz,
                x_pos, y_pos, x_start, y_start,
                outbound_attraction_distribution,
                lin_log_mode,
                prevent_overlapping,
                edge_weight_influence,
                jitter_tolerance,
                scaling_ratio,
                strong_gravity_mode,
                gravity,
                d_repulsion);

        ForceAtlas2::apply_forces<IndexType, ValueType>(
                row, col, val, nnz,
                x_pos, y_pos, x_start, y_start,
                outbound_attraction_distribution,
                lin_log_mode,
                prevent_overlapping,
                edge_weight_influence,
                jitter_tolerance,
                scaling_ratio,
                strong_gravity_mode,
                gravity,
                d_attraction, d_repulsion);
    }

}

} // namespace ForceAtlas2
}  // namespace cugraph

