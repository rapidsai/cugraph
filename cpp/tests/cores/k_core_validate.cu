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

#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>

#include <thrust/copy.h>
#include <thrust/count.h>

#include <gtest/gtest.h>

namespace cugraph {
namespace test {

template <typename vertex_t, typename edge_t, typename weight_t>
void check_correctness(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, false> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  rmm::device_uvector<edge_t> const& core_numbers,
  std::tuple<rmm::device_uvector<vertex_t>,
             rmm::device_uvector<vertex_t>,
             std::optional<rmm::device_uvector<weight_t>>> const& subgraph,
  size_t k)
{
  auto const& [subgraph_src, subgraph_dst, subgraph_wgt] = subgraph;

  // Check that all edges in the subgraph are appropriate
  auto error_count = thrust::count_if(
    handle.get_thrust_policy(),
    subgraph_src.begin(),
    subgraph_src.end(),
    [k, d_core_numbers = core_numbers.data()] __device__(auto v) { return d_core_numbers[v] < k; });

  EXPECT_EQ(error_count, 0) << "source error count is non-zero";

  error_count = thrust::count_if(
    handle.get_thrust_policy(),
    subgraph_dst.begin(),
    subgraph_dst.end(),
    [k, d_core_numbers = core_numbers.data()] __device__(auto v) { return d_core_numbers[v] < k; });

  EXPECT_EQ(error_count, 0) << "destination error count is non-zero";

  auto [graph_src, graph_dst, graph_wgt] =
    cugraph::decompress_to_edgelist(handle,
                                    graph_view,
                                    edge_weight_view,
                                    std::optional<raft::device_span<vertex_t const>>{std::nullopt},
                                    false);

  // Now we'll count how many edges should be in the subgraph
  auto expected_edge_count =
    thrust::count_if(handle.get_thrust_policy(),
                     thrust::make_zip_iterator(graph_src.begin(), graph_dst.begin()),
                     thrust::make_zip_iterator(graph_src.end(), graph_dst.end()),
                     [k, d_core_numbers = core_numbers.data()] __device__(auto tuple) {
                       vertex_t src = thrust::get<0>(tuple);
                       vertex_t dst = thrust::get<1>(tuple);
                       return ((d_core_numbers[src] >= k) && (d_core_numbers[dst] >= k));
                     });

  EXPECT_EQ(expected_edge_count, subgraph_src.size());
}

template void check_correctness(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
  rmm::device_uvector<int32_t> const& core_numbers,
  std::tuple<rmm::device_uvector<int32_t>,
             rmm::device_uvector<int32_t>,
             std::optional<rmm::device_uvector<float>>> const& subgraph,
  size_t k);

template void check_correctness(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int64_t, false, false> const& graph_view,
  std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
  rmm::device_uvector<int64_t> const& core_numbers,
  std::tuple<rmm::device_uvector<int32_t>,
             rmm::device_uvector<int32_t>,
             std::optional<rmm::device_uvector<float>>> const& subgraph,
  size_t k);

template void check_correctness(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, false> const& graph_view,
  std::optional<edge_property_view_t<int64_t, float const*>> edge_weight_view,
  rmm::device_uvector<int64_t> const& core_numbers,
  std::tuple<rmm::device_uvector<int64_t>,
             rmm::device_uvector<int64_t>,
             std::optional<rmm::device_uvector<float>>> const& subgraph,
  size_t k);

}  // namespace test
}  // namespace cugraph
