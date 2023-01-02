/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <structure/detail/structure_utils.cuh>
#include <utilities/device_comm_wrapper.hpp>
#include <utilities/test_utilities.hpp>

#include <cugraph/graph_functions.hpp>

#include <raft/core/device_span.hpp>

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
  cugraph::graph_view_t<vertex_t, edge_t, store_transposed, is_multi_gpu> const& graph_view,
  std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>> edge_weight_view)
{
  auto [d_src, d_dst, d_wgt] =
    cugraph::decompress_to_edgelist(handle,
                                    graph_view,
                                    edge_weight_view,
                                    std::optional<raft::device_span<vertex_t const>>{std::nullopt});

  if constexpr (is_multi_gpu) {
    d_src = cugraph::test::device_gatherv(
      handle, raft::device_span<vertex_t const>{d_src.data(), d_src.size()});
    d_dst = cugraph::test::device_gatherv(
      handle, raft::device_span<vertex_t const>{d_dst.data(), d_dst.size()});
    if (d_wgt)
      *d_wgt = cugraph::test::device_gatherv(
        handle, raft::device_span<weight_t const>{d_wgt->data(), d_wgt->size()});
    if (handle.get_comms().get_rank() != 0) {
      d_src.resize(0, handle.get_stream());
      d_src.shrink_to_fit(handle.get_stream());
      d_dst.resize(0, handle.get_stream());
      d_dst.shrink_to_fit(handle.get_stream());
      if (d_wgt) {
        (*d_wgt).resize(0, handle.get_stream());
        (*d_wgt).shrink_to_fit(handle.get_stream());
      }
    }
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

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool is_multi_gpu>
std::tuple<std::vector<edge_t>, std::vector<vertex_t>, std::optional<std::vector<weight_t>>>
graph_to_host_csr(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, store_transposed, is_multi_gpu> const& graph_view,
  std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>> edge_weight_view)
{
  auto [d_src, d_dst, d_wgt] =
    cugraph::decompress_to_edgelist(handle,
                                    graph_view,
                                    edge_weight_view,
                                    std::optional<raft::device_span<vertex_t const>>{std::nullopt});

  if constexpr (is_multi_gpu) {
    d_src = cugraph::test::device_gatherv(
      handle, raft::device_span<vertex_t const>{d_src.data(), d_src.size()});
    d_dst = cugraph::test::device_gatherv(
      handle, raft::device_span<vertex_t const>{d_dst.data(), d_dst.size()});
    if (d_wgt)
      *d_wgt = cugraph::test::device_gatherv(
        handle, raft::device_span<weight_t const>{d_wgt->data(), d_wgt->size()});
    if (handle.get_comms().get_rank() != 0) {
      d_src.resize(0, handle.get_stream());
      d_src.shrink_to_fit(handle.get_stream());
      d_dst.resize(0, handle.get_stream());
      d_dst.shrink_to_fit(handle.get_stream());
      if (d_wgt) {
        (*d_wgt).resize(0, handle.get_stream());
        (*d_wgt).shrink_to_fit(handle.get_stream());
      }
    }
  }

  rmm::device_uvector<edge_t> d_offsets(0, handle.get_stream());

  if (d_wgt) {
    std::tie(d_offsets, d_dst, *d_wgt, std::ignore) =
      detail::compress_edgelist<edge_t, store_transposed>(d_src.begin(),
                                                          d_src.end(),
                                                          d_dst.begin(),
                                                          d_wgt->begin(),
                                                          vertex_t{0},
                                                          std::optional<vertex_t>{std::nullopt},
                                                          graph_view.number_of_vertices(),
                                                          vertex_t{0},
                                                          graph_view.number_of_vertices(),
                                                          handle.get_stream());

    // segmented sort neighbors
    detail::sort_adjacency_list(handle,
                                raft::device_span<edge_t const>(d_offsets.data(), d_offsets.size()),
                                d_dst.begin(),
                                d_dst.end(),
                                d_wgt->begin());
  } else {
    std::tie(d_offsets, d_dst, std::ignore) =
      detail::compress_edgelist<edge_t, store_transposed>(d_src.begin(),
                                                          d_src.end(),
                                                          d_dst.begin(),
                                                          vertex_t{0},
                                                          std::optional<vertex_t>{std::nullopt},
                                                          graph_view.number_of_vertices(),
                                                          vertex_t{0},
                                                          graph_view.number_of_vertices(),
                                                          handle.get_stream());
    // segmented sort neighbors
    detail::sort_adjacency_list(handle,
                                raft::device_span<edge_t const>(d_offsets.data(), d_offsets.size()),
                                d_dst.begin(),
                                d_dst.end());
  }

  return std::make_tuple(
    to_host(handle, raft::device_span<edge_t const>(d_offsets.data(), d_offsets.size())),
    to_host(handle, raft::device_span<vertex_t const>(d_dst.data(), d_dst.size())),
    d_wgt ? to_host(handle, raft::device_span<weight_t const>(d_wgt->data(), d_wgt->size()))
          : std::optional<std::vector<weight_t>>(std::nullopt));
}

template <typename vertex_t, typename edge_t, typename weight_t, bool store_transposed>
std::tuple<
  cugraph::graph_t<vertex_t, edge_t, store_transposed, false>,
  std::optional<edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, false>, weight_t>>,
  std::optional<rmm::device_uvector<vertex_t>>>
mg_graph_to_sg_graph(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, store_transposed, true> const& graph_view,
  std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<rmm::device_uvector<vertex_t>> const& number_map,
  bool renumber)
{
  auto [d_src, d_dst, d_wgt] = cugraph::decompress_to_edgelist(
    handle,
    graph_view,
    edge_weight_view,
    number_map ? std::make_optional<raft::device_span<vertex_t const>>((*number_map).data(),
                                                                       (*number_map).size())
               : std::nullopt);

  d_src = cugraph::test::device_gatherv(
    handle, raft::device_span<vertex_t const>{d_src.data(), d_src.size()});
  d_dst = cugraph::test::device_gatherv(
    handle, raft::device_span<vertex_t const>{d_dst.data(), d_dst.size()});
  if (d_wgt)
    *d_wgt = cugraph::test::device_gatherv(
      handle, raft::device_span<weight_t const>{d_wgt->data(), d_wgt->size()});
  if (handle.get_comms().get_rank() != 0) {
    d_src.resize(0, handle.get_stream());
    d_src.shrink_to_fit(handle.get_stream());
    d_dst.resize(0, handle.get_stream());
    d_dst.shrink_to_fit(handle.get_stream());
    if (d_wgt) {
      (*d_wgt).resize(0, handle.get_stream());
      (*d_wgt).shrink_to_fit(handle.get_stream());
    }
  }

  graph_t<vertex_t, edge_t, store_transposed, false> graph(handle);
  std::optional<edge_property_t<graph_view_t<vertex_t, edge_t, store_transposed, false>, weight_t>>
    edge_weights{std::nullopt};
  std::optional<rmm::device_uvector<vertex_t>> new_number_map;
  std::tie(graph, edge_weights, std::ignore, new_number_map) = cugraph::
    create_graph_from_edgelist<vertex_t, edge_t, weight_t, int32_t, store_transposed, false>(
      handle,
      std::optional<rmm::device_uvector<vertex_t>>{std::nullopt},
      std::move(d_src),
      std::move(d_dst),
      std::move(d_wgt),
      std::nullopt,
      cugraph::graph_properties_t{graph_view.is_symmetric(), graph_view.is_multigraph()},
      renumber);

  return std::make_tuple(std::move(graph), std::move(edge_weights), std::move(new_number_map));
}

}  // namespace test
}  // namespace cugraph
