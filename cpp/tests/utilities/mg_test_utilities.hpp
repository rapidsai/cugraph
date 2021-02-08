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

#include <utilities/test_utilities.hpp>

namespace cugraph {
namespace test {

// FIXME: it might be nicer to take a path to a graph input file instead of the
// edgelist struct, but creating a edgelist struct from a graph file requires a
// utility in test_utilities.hpp, which cannot be included here without
// resulting in multiple definition errors when linked with a test that also
// needs that include.
//
// Given a raft handle and an edgelist from reading a dataset (.mtx in this
// case), returns a tuple containing:
//  * graph_t instance for the partition accesible from the raft handle
//  * 4-tuple containing renumber info resulting from renumbering the
//    edgelist for the partition
//

   //   std::tuple<rmm::device_uvector<vertex_t>, cugraph::experimental::partition_t<vertex_t>, vertex_t, edge_t>

template <typename vertex_t, typename edge_t, typename weight_t, bool store_transposed>
std::tuple<
   std::unique_ptr<cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, store_transposed, true>>, // multi_gpu=true
   rmm::device_uvector<vertex_t>
>
create_graph_for_gpu(raft::handle_t& handle,
                     edgelist_from_market_matrix_file_t<vertex_t, weight_t> edgelist_from_mm);

#if 0
template <typename vertex_t, typename edge_t, typename weight_t, bool store_transposed>
   std::tuple<rmm::device_uvector<vertex_t>, cugraph::experimental::partition_t<vertex_t>, vertex_t, edge_t>
prep_edgelist_for_gpu(raft::handle_t& handle,
                     edgelist_from_market_matrix_file_t<vertex_t, weight_t> edgelist_from_mm);


template <typename vertex_t, typename edge_t, typename weight_t, bool store_transposed>
   cugraph::experimental::graph_t<vertex_t, edge_t, weight_t, store_transposed, true>&&, // multi_gpu=true
build_graph_for_gpu(raft::handle_t& handle,
                     edgelist_from_market_matrix_file_t<vertex_t, weight_t> edgelist_from_mm);
#endif
} // namespace test
} // namespace cugraph
