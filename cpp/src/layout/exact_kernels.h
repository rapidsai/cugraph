#pragma once

namespace cugraph {
namespace ForceAtlas2 {

template <typename IndexType, typename ValueType>
__global__ void attraction_kernel(const IndexType *row, const IndexType *col,
        const ValueType *val, const int nnz, float *x_pos,
        float *y_pos, float *x_start, float *y_start,
        bool outbount_attraction_distribution, bool lin_log_mode,
        bool prevent_overlapping, float edge_weight_influence,
        float jitter_tolerance, float scaling_ratio,
        bool strong_gravity_mode, float gravity) {
    return;

}

template <typename IndexType, typename ValueType>
__global__ void repulsion_kernel(const IndexType *row, const IndexType *col,
        const ValueType *val, const int nnz, float *x_pos,
        float *y_pos, float *x_start, float *y_start,
        bool outbount_attraction_distribution, bool lin_log_mode,
        bool prevent_overlapping, float edge_weight_influence,
        float jitter_tolerance, float scaling_ratio,
        bool strong_gravity_mode, float gravity) {
    return;
}



template <typename IndexType, typename ValueType>
__global__ void apply_kernel(const IndexType *row, const IndexType *col,
        const ValueType *val, const int nnz,
        float *x_pos, float *y_pos, float *x_start, float *y_start,
        bool outbount_attraction_distribution, bool lin_log_mode,
        bool prevent_overlapping, float edge_weight_influence,
        float jitter_tolerance, float scaling_ratio,
        bool strong_gravity_mode, float gravity) {
    return;
}

template <typename IndexType, typename ValueType>
void compute_attraction(const IndexType *row, const IndexType *col,
        const ValueType *val, const int nnz, float *x_pos,
        float *y_pos, float *x_start, float *y_start,
        bool outbount_attraction_distribution, bool lin_log_mode,
        bool prevent_overlapping, float edge_weight_influence,
        float jitter_tolerance, float scaling_ratio,
        bool strong_gravity_mode, float gravity,
        float *d_attraction) {
    return;
}

template <typename IndexType, typename ValueType>
void compute_repulsion(const IndexType *row, const IndexType *col,
        const ValueType *val, const int nnz, float *x_pos,
        float *y_pos, float *x_start, float *y_start,
        bool outbount_attraction_distribution, bool lin_log_mode,
        bool prevent_overlapping, float edge_weight_influence,
        float jitter_tolerance, float scaling_ratio,
        bool strong_gravity_mode, float gravity,
        float *d_repulsion) {
    return;
}

template <typename IndexType, typename ValueType>
void apply_forces(const IndexType *row, const IndexType *col,
        const ValueType *val, const int nnz, float *x_pos,
        float *y_pos, float *x_start, float *y_start,
        bool outbount_attraction_distribution, bool lin_log_mode,
        bool prevent_overlapping, float edge_weight_influence,
        float jitter_tolerance, float scaling_ratio,
        bool strong_gravity_mode, float gravity,
        float *d_attraction, float *d_repulsion) {
    return;
}

} // namespace ForceAtlas2
} // namespace cugraph
