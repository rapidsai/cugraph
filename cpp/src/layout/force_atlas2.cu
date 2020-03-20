#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>
#include "cub/cub.cuh"
#include <algorithm>
#include <iomanip>


#include "utilities/graph_utils.cuh"
#include "utilities/error_utils.h"
#include <cugraph.h>
#include <graph.hpp>

namespace cugraph {
namespace detail {

template <typename VT, typename ET, typename WT>
void force_atlas2_impl(experimental::GraphCOO<VT, ET, WT> const &graph,
                  float *x_pos, float *y_pos, int max_iter=1000,
                  float *x_start=nullptr, float *y_start=nullptr,
                  bool outbound_attraction_distribution=false,
                  bool lin_log_mode=false, bool prevent_overlapping=false,
                  float edge_weight_influence=1.0, float jitter_tolerance=1.0,
                  bool barnes_hut_optimize=true,
                  float barnes_hut_theta=0.5, float scaling_ratio=2.0,
                  bool strong_gravity_mode=false, float gravity=1.0) {

    if (x_start == nullptr || y_start == nullptr)
        return;

    int n = graph.number_of_vertices;
    copy(n, x_start, x_pos);
    copy(n, y_start, y_pos);
}
} //namespace detail

template <typename VT, typename ET, typename WT>
void force_atlas2(experimental::GraphCOO<VT, ET, WT> const &graph,
                  float *x_pos, float *y_pos, int max_iter, float *x_start,
                  float *y_start, bool outbound_attraction_distribution,
                  bool lin_log_mode, bool prevent_overlapping, float edge_weight_influence,
                  float jitter_tolerance, bool barnes_hut_optimize,
                  float barnes_hut_theta, float scaling_ratio,
                  bool strong_gravity_mode, float gravity) {

    CUGRAPH_EXPECTS( x_pos != nullptr , "Invalid API parameter: X_pos array should be of size V" );
    CUGRAPH_EXPECTS( y_pos != nullptr , "Invalid API parameter: Y_pos array should be of size V" );

    return detail::force_atlas2_impl<VT, ET, WT>(graph,
                                                 x_pos,
                                                 y_pos,
                                                 max_iter,
                                                 x_start,
                                                 y_start,
                                                 outbound_attraction_distribution,
                                                 lin_log_mode,
                                                 prevent_overlapping,
                                                 edge_weight_influence,
                                                 jitter_tolerance,
                                                 barnes_hut_optimize,
                                                 barnes_hut_theta,
                                                 scaling_ratio,
                                                 strong_gravity_mode,
                                                 gravity);

}

template void force_atlas2<int, int, float>(experimental::GraphCOO<int, int, float> const &graph,
                float *x_pos, float *y_pos, int max_iter, float *x_start, float *y_start,
                bool outbound_attraction_distribution, bool lin_log_mode, bool prevent_overlapping,
                float edge_weight_influence, float jitter_tolerance, bool barnes_hut_optimize,
                float barnes_hut_theta, float scaling_ratio, bool strong_gravity_mode, float gravity);
} //namespace cugraph
