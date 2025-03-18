/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include "barnes_hut.cuh"
#include "exact_fa2.cuh"

namespace cugraph {

template <typename vertex_t, typename edge_t, typename weight_t>
void force_atlas2(raft::handle_t const& handle,
                  legacy::GraphCOOView<vertex_t, edge_t, weight_t>& graph,
                  float* pos,
                  const int max_iter,
                  float* x_start,
                  float* y_start,
                  bool outbound_attraction_distribution,
                  bool lin_log_mode,
                  bool prevent_overlapping,
                  const float edge_weight_influence,
                  const float jitter_tolerance,
                  bool barnes_hut_optimize,
                  const float barnes_hut_theta,
                  const float scaling_ratio,
                  bool strong_gravity_mode,
                  const float gravity,
                  bool verbose,
                  internals::GraphBasedDimRedCallback* callback)
{
  CUGRAPH_EXPECTS(pos != nullptr, "Invalid input argument: pos array should be of size 2 * V");
  CUGRAPH_EXPECTS(graph.number_of_vertices != 0, "Invalid input: Graph is empty");

  if (!barnes_hut_optimize) {
    cugraph::detail::exact_fa2<vertex_t, edge_t, weight_t>(handle,
                                                           graph,
                                                           pos,
                                                           max_iter,
                                                           x_start,
                                                           y_start,
                                                           outbound_attraction_distribution,
                                                           lin_log_mode,
                                                           prevent_overlapping,
                                                           edge_weight_influence,
                                                           jitter_tolerance,
                                                           scaling_ratio,
                                                           strong_gravity_mode,
                                                           gravity,
                                                           verbose,
                                                           callback);
  } else {
    cugraph::detail::barnes_hut<vertex_t, edge_t, weight_t>(handle,
                                                            graph,
                                                            pos,
                                                            max_iter,
                                                            x_start,
                                                            y_start,
                                                            outbound_attraction_distribution,
                                                            lin_log_mode,
                                                            prevent_overlapping,
                                                            edge_weight_influence,
                                                            jitter_tolerance,
                                                            barnes_hut_theta,
                                                            scaling_ratio,
                                                            strong_gravity_mode,
                                                            gravity,
                                                            verbose,
                                                            callback);
  }
}

template void force_atlas2<int, int, float>(raft::handle_t const& handle,
                                            legacy::GraphCOOView<int, int, float>& graph,
                                            float* pos,
                                            const int max_iter,
                                            float* x_start,
                                            float* y_start,
                                            bool outbound_attraction_distribution,
                                            bool lin_log_mode,
                                            bool prevent_overlapping,
                                            const float edge_weight_influence,
                                            const float jitter_tolerance,
                                            bool barnes_hut_optimize,
                                            const float barnes_hut_theta,
                                            const float scaling_ratio,
                                            bool strong_gravity_mode,
                                            const float gravity,
                                            bool verbose,
                                            internals::GraphBasedDimRedCallback* callback);

template void force_atlas2<int, int, double>(raft::handle_t const& handle,
                                             legacy::GraphCOOView<int, int, double>& graph,
                                             float* pos,
                                             const int max_iter,
                                             float* x_start,
                                             float* y_start,
                                             bool outbound_attraction_distribution,
                                             bool lin_log_mode,
                                             bool prevent_overlapping,
                                             const float edge_weight_influence,
                                             const float jitter_tolerance,
                                             bool barnes_hut_optimize,
                                             const float barnes_hut_theta,
                                             const float scaling_ratio,
                                             bool strong_gravity_mode,
                                             const float gravity,
                                             bool verbose,
                                             internals::GraphBasedDimRedCallback* callback);

}  // namespace cugraph
