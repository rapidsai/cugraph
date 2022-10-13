/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
 * See the License for the specific language governin_from_mtxg permissions and
 * limitations under the License.
 */
#pragma once

#include <optional>
#include <vector>

namespace cugraph {
namespace test {

template <typename vertex_t, typename edge_t, typename weight_t>
void betweenness_centrality_validate(std::vector<edge_t> const& h_offsets,
                                     std::vector<vertex_t> const& h_indices,
                                     std::optional<std::vector<weight_t>> const& h_wgt,
                                     std::vector<weight_t> const& h_cugraph_results,
                                     std::vector<vertex_t> const& h_seeds,
                                     bool count_endpoints);

template <typename vertex_t, typename edge_t, typename weight_t>
void edge_betweenness_centrality_validate(std::vector<edge_t> const& h_offsets,
                                          std::vector<vertex_t> const& h_indices,
                                          std::optional<std::vector<weight_t>> const& h_wgt,
                                          std::vector<weight_t> const& h_cugraph_results,
                                          std::vector<vertex_t> const& h_seeds);

}  // namespace test
}  // namespace cugraph
