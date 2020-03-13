#include <cugraph.h>
#include <graph.hpp>

#include "utilities/error_utils.h"

namespace cugraph {
namespace detail {

template <typename VT, typename ET, typename WT>
void force_atlas2_impl(experimental::GraphCOO<VT, ET, WT> const &graph,
                  float *x_pos,
                  float *y_pos,
                  float *x_start=nullptr,
                  float *y_start=nullptr,
                  int max_iter=1000,
                  float gravity=1.0,
                  float scaling_ratio=2.0,
                  float edge_weight_influence=1.0,
                  bool lin_log_mode=false,
                  bool prevent_overlapping=false) {
    return;
}
} //namespace detail

template <typename VT, typename ET, typename WT>
void force_atlas2(experimental::GraphCOO<VT, ET, WT> const &graph,
                  float *x_pos,
                  float *y_pos,
                  float *x_start,
                  float *y_start,
                  int max_iter,
                  float gravity,
                  float scaling_ratio,
                  float edge_weight_influence,
                  bool lin_log_mode,
                  bool prevent_overlapping) {

    CUGRAPH_EXPECTS( x_pos != nullptr , "Invalid API parameter: X_pos array should be of size V" );
    CUGRAPH_EXPECTS( y_pos != nullptr , "Invalid API parameter: Y_pos array should be of size V" );

    return detail::force_atlas2_impl<VT, ET, WT>(graph,
                                                 x_pos,
                                                 y_pos,
                                                 x_start,
                                                 y_start,
                                                 max_iter,
                                                 gravity,
                                                 scaling_ratio,
                                                 edge_weight_influence,
                                                 lin_log_mode,
                                                 prevent_overlapping);

}

template void force_atlas2<int, int, float>(experimental::GraphCOO<int, int, float> const &graph,
                float *x_pos, float *y_pos, float *x_start, float *y_start, int max_iter,
                float gravity, float scaling_ratio, float edge_weight_influence, bool lin_log_mode,
                bool prevent_overlapping);
} //namespace cugraph
