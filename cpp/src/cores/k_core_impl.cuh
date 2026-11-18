/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/algorithms.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/prims/extract_transform_if_v.cuh>
#include <cugraph/utilities/error.hpp>

#include <cuda/functional>

#include <optional>
#include <variant>

namespace cugraph {

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>>
k_core(raft::handle_t const& handle,
       graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
       std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
       size_t k,
       std::variant<vertex_property_view_t<vertex_t, edge_t const*>, k_core_degree_type_t>
         core_numbers_or_degree_type,
       bool do_expensive_check)
{
  std::optional<vertex_property_t<vertex_t, edge_t>> computed_core_numbers{std::nullopt};
  vertex_property_view_t<vertex_t, edge_t const*> core_numbers{};

  if (auto degree_type = std::get_if<k_core_degree_type_t>(&core_numbers_or_degree_type)) {
    computed_core_numbers = cugraph::core_number(handle,
                                                 graph_view,
                                                 *degree_type,
                                                 size_t{0},
                                                 std::numeric_limits<size_t>::max(),
                                                 do_expensive_check);
    core_numbers          = computed_core_numbers->view();
  } else {
    core_numbers =
      std::get<vertex_property_view_t<vertex_t, edge_t const*>>(core_numbers_or_degree_type);
  }

  auto subgraph_vertices = extract_transform_if_v(
    handle,
    graph_view,
    core_numbers,
    cuda::proclaim_return_type<vertex_t>([] __device__(vertex_t v, edge_t) { return v; }),
    cuda::proclaim_return_type<bool>([k] __device__(vertex_t, edge_t c) { return k <= c; }));

  rmm::device_uvector<size_t> subgraph_offsets(2, handle.get_stream());
  std::vector<size_t> h_subgraph_offsets{{0, subgraph_vertices.size()}};
  raft::update_device(subgraph_offsets.data(),
                      h_subgraph_offsets.data(),
                      h_subgraph_offsets.size(),
                      handle.get_stream());
  handle.sync_stream();

  auto [src, dst, wgt, offsets] = extract_induced_subgraphs(
    handle,
    graph_view,
    edge_weight_view,
    raft::device_span<size_t const>{subgraph_offsets.data(), subgraph_offsets.size()},
    raft::device_span<vertex_t const>{subgraph_vertices.data(), subgraph_vertices.size()},
    do_expensive_check);

  return std::make_tuple(std::move(src), std::move(dst), std::move(wgt));
}

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>>
k_core(raft::handle_t const& handle,
       graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
       std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
       size_t k,
       std::optional<k_core_degree_type_t> degree_type,
       std::optional<raft::device_span<edge_t const>> core_numbers,
       bool do_expensive_check)
{
  using core_numbers_or_degree_type_t =
    std::variant<vertex_property_view_t<vertex_t, edge_t const*>, k_core_degree_type_t>;

  CUGRAPH_EXPECTS(core_numbers.has_value() || degree_type.has_value(),
                  "If core_numbers is not specified then degree_type must be specified");

  return cugraph::k_core(
    handle,
    graph_view,
    edge_weight_view,
    k,
    core_numbers ? core_numbers_or_degree_type_t{vertex_property_view_t<vertex_t, edge_t const*>{
                     core_numbers->data(), static_cast<vertex_t>(core_numbers->size())}}
                 : core_numbers_or_degree_type_t{*degree_type},
    do_expensive_check);
}

}  // namespace cugraph
