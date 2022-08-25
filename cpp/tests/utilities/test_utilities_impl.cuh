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

#include <utilities/device_comm_wrapper.hpp>
#include <utilities/test_utilities.hpp>

namespace cugraph {
namespace test {

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool is_multi_gpu>
std::tuple<std::vector<vertex_t>, std::vector<vertex_t>, std::optional<std::vector<weight_t>>>
graph_to_host_coo(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, weight_t, store_transposed, is_multi_gpu> const& graph_view)
{
  auto [d_src, d_dst, d_wgt] = graph_view.decompress_to_edgelist(handle, std::nullopt);

  if constexpr (is_multi_gpu) {
    d_src = cugraph::test::device_gatherv(
      handle, raft::device_span<vertex_t const>{d_src.data(), d_src.size()});
    d_dst = cugraph::test::device_gatherv(
      handle, raft::device_span<vertex_t const>{d_dst.data(), d_dst.size()});
    if (d_wgt)
      *d_wgt = cugraph::test::device_gatherv(
        handle, raft::device_span<weight_t const>{d_wgt->data(), d_wgt->size()});
  }

  std::vector<vertex_t> h_src(d_src.size());
  std::vector<vertex_t> h_dst(d_dst.size());
  std::optional<std::vector<weight_t>> h_wgt(std::nullopt);

  raft::update_host(h_src.data(), d_src.data(), d_src.size(), handle.get_stream());
  raft::update_host(h_dst.data(), d_dst.data(), d_dst.size(), handle.get_stream());

  if (d_wgt) {
    h_wgt = std::make_optional<std::vector<weight_t>>(d_wgt->size());
    raft::update_host(h_wgt->data(), d_wgt->data(), d_wgt->size(), handle.get_stream());
  }

  return std::make_tuple(std::move(h_src), std::move(h_dst), std::move(h_wgt));
}

}  // namespace test
}  // namespace cugraph
