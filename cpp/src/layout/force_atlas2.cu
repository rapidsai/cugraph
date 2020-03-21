#include <algorithm>
#include <cugraph.h>
#include <graph.hpp>
#include <iomanip>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "barnes_hut.h" 
#include "cub/cub.cuh"
#include "exact_fa2.h"
#include "utilities/error_utils.h"
#include "utilities/graph_utils.cuh"

namespace cugraph {

template <typename VT, typename ET, typename WT>
void force_atlas2(experimental::GraphCOO<VT, ET, WT> const &graph,
                  float *x_pos, float *y_pos, int max_iter, float *x_start,
                  float *y_start, bool outbound_attraction_distribution,
                  bool lin_log_mode, bool prevent_overlapping,
                  float edge_weight_influence,
                  float jitter_tolerance, bool barnes_hut_optimize,
                  float barnes_hut_theta, float scaling_ratio,
                  bool strong_gravity_mode, float gravity) {

    CUGRAPH_EXPECTS( x_pos != nullptr,
            "Invalid API parameter: X_pos array should be of size V" );
    CUGRAPH_EXPECTS( y_pos != nullptr ,
            "Invalid API parameter: Y_pos array should be of size V" );

    const VT *row = graph.src_indices;
    const VT *col = graph.dst_indices;
    const WT *val = graph.edge_data;
    const int nnz = graph.number_of_edges;

    if (barnes_hut_optimize) {
        ForceAtlas2::barnes_hut<VT, WT>(row, col, val, nnz,
                x_pos, y_pos, max_iter, x_start,
                y_start, outbound_attraction_distribution,
                lin_log_mode, prevent_overlapping, edge_weight_influence,
                jitter_tolerance,barnes_hut_theta, scaling_ratio,
                strong_gravity_mode, gravity);
    } else {
        ForceAtlas2::exact_fa2<VT, WT>(row, col, val, nnz,
                x_pos, y_pos, max_iter, x_start,
                y_start, outbound_attraction_distribution,
                lin_log_mode, prevent_overlapping, edge_weight_influence,
                jitter_tolerance, scaling_ratio,
                strong_gravity_mode, gravity);
    }
}

// Explicit Instantiation
template void force_atlas2<int, int, float>(
        experimental::GraphCOO<int, int, float> const &graph,
        float *x_pos, float *y_pos, int max_iter,
        float *x_start, float *y_start,
        bool outbound_attraction_distribution,
        bool lin_log_mode, bool prevent_overlapping,
        float edge_weight_influence, float jitter_tolerance,
        bool barnes_hut_optimize, float barnes_hut_theta, float scaling_ratio,
        bool strong_gravity_mode, float gravity);

template void ForceAtlas2::barnes_hut<int, float>(
        const int *row,const int *col, const float *val, const int nnz,
        float *x_pos, float *y_pos, int max_iter,
        float *x_start, float * y_start,
        bool outbount_attraction_distribution,
        bool lin_log_mode, bool prevent_overlapping,
        float edge_weight_influence, float jitter_tolerance,
        float barnes_hut_theta, float scaling_ratio, bool strong_gravity_mode,
		float gravity);

template void ForceAtlas2::exact_fa2<int, float>(
        const int *row, const int *col, const float *val, const int nnz,
        float *x_pos, float *y_pos, int max_iter,
        float *x_start, float *y_start,
        bool outbound_attraction_distribution,
        bool lin_log_mode, bool prevent_overlapping,
        float edge_weight_influence, float jitter_tolerance,
        float scaling_ratio, bool strong_gravity_mode,
        float gravity);

template void ForceAtlas2::compute_attraction<int, float>(
        const int *row, const int *col, const float *val, const int nnz,
        float *x_pos, float *y_pos,
        float *x_start, float *y_start,
        bool outbound_attraction_distribution,
        bool lin_log_mode, bool prevent_overlapping,
        float edge_weight_influence, float jitter_tolerance,
        float scaling_ratio, bool strong_gravity_mode,
        float gravity, float *d_attraction);

template void ForceAtlas2::compute_repulsion<int, float>(
        const int *row, const int *col, const float *val, const int nnz,
        float *x_pos, float *y_pos,
        float *x_start, float *y_start,
        bool outbound_attraction_distribution,
        bool lin_log_mode, bool prevent_overlapping,
        float edge_weight_influence, float jitter_tolerance,
        float scaling_ratio, bool strong_gravity_mode,
        float gravity, float *d_attraction);

template void ForceAtlas2::apply_forces<int, float>(
        const int *row, const int *col, const float *val, const int nnz,
        float *x_pos, float *y_pos,
        float *x_start, float *y_start,
        bool outbound_attraction_distribution,
        bool lin_log_mode, bool prevent_overlapping,
        float edge_weight_influence, float jitter_tolerance,
        float scaling_ratio, bool strong_gravity_mode,
        float gravity, float *d_attraction, float *d_repulsion);
} //namespace cugraph
