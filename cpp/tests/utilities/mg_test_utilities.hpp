/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <experimental/graph.hpp>

namespace cugraph {
namespace test {

// FIXME: This type is defined in <utilities/test_utilities.hpp> which cannot be
// included here since it must also be included in tests. If included here,
// multiple definitions of various other utility functions will occur.
#ifndef DEFINED_edgelist_from_market_matrix_file_t
template <typename vertex_t, typename weight_t>
struct edgelist_from_market_matrix_file_t {
  std::vector<vertex_t> h_rows{};
  std::vector<vertex_t> h_cols{};
  std::vector<weight_t> h_weights{};
  vertex_t number_of_vertices{};
  bool is_symmetric{};
};
#define DEFINED_edgelist_from_market_matrix_file_t
#endif

// FIXME: it might be nicer to take a path to a graph input file instead of the
// edgelist struct, but creating a edgelist struct from a graph file requires a
// utility in test_utilities.hpp, which cannot be included here without
// resulting in multiple definition errors when linked with a test that also
// needs that include.
template <typename vertex_t, typename edge_t, typename weight_t, bool store_transposed>
cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, store_transposed, true> // multi_gpu=true
create_graph_for_gpu(raft::handle_t& handle,
                     edgelist_from_market_matrix_file_t<vertex_t, weight_t> edgelist_from_mm,
                     bool input_is_weighted);

} // namespace test
} // namespace cugraph
