/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Vertex preparation and breadth-first search.
 */
#include "cugraph_python.hpp"

#include <raft/util/cudart_utils.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/sort.h>
#include <thrust/unique.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace cugraph_python {
namespace {

template <typename vertex_t>
std::vector<vertex_t> narrow_vertex_ids(std::vector<std::int64_t> const& external_ids)
{
  std::vector<vertex_t> narrowed;
  narrowed.reserve(external_ids.size());
  for (auto id : external_ids) {
    if (id < static_cast<std::int64_t>(std::numeric_limits<vertex_t>::min()) ||
        id > static_cast<std::int64_t>(std::numeric_limits<vertex_t>::max())) {
      throw std::invalid_argument(
        "prepare_vertices: a vertex ID does not fit in the graph vertex type");
    }
    narrowed.push_back(static_cast<vertex_t>(id));
  }
  return narrowed;
}

template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
void check_prepared_for(graph_instance_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph,
                        prepared_vertices_instance_t<vertex_t> const& starts)
{
  if (starts.graph_id != graph.graph_id) {
    throw std::invalid_argument("PreparedVertices was prepared for a different graph");
  }
  if (starts.numbering_id != graph.numbering_id) {
    throw std::invalid_argument("PreparedVertices uses a different vertex numbering map");
  }
  if (starts.version != graph.version) {
    throw std::invalid_argument("PreparedVertices was prepared for a different graph version");
  }
  if (starts.partition_rank != 0 || starts.partition_count != 1) {
    throw std::invalid_argument("PreparedVertices partition does not match this single-GPU graph");
  }
}

template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
prepared_vertices_instance_t<vertex_t> prepare_vertices_impl(
  graph_instance_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph,
  std::vector<std::int64_t> const& external_ids)
{
  if (external_ids.empty()) {
    throw std::invalid_argument("prepare_vertices: external_ids must not be empty");
  }
  if (!graph.number_map) {
    throw std::logic_error("prepare_vertices: graph has no number map (renumbering disabled)");
  }

  auto host_ids          = narrow_vertex_ids<vertex_t>(external_ids);
  auto& handle           = graph.context->handle;
  auto stream            = handle.get_stream();
  auto const& number_map = *graph.number_map;

  rmm::device_uvector<vertex_t> external(host_ids.size(), stream);
  raft::copy(external.data(), host_ids.data(), host_ids.size(), stream);

  auto view = graph.native.view();

  // Translate external -> internal. Unknown external IDs come back as invalid_vertex_id.
  auto internal = cugraph::renumber_ext_vertices(
    handle,
    view,
    std::move(external),
    cugraph::vertex_property_view_t<vertex_t, vertex_t const*>(number_map.data(), number_map.size()),
    /*do_expensive_check=*/false);

  // Starts must be sorted and unique (required by cugraph::bfs for the frontier overload).
  auto policy = handle.get_thrust_policy();
  thrust::sort(policy, internal.begin(), internal.end());
  auto new_end = thrust::unique(policy, internal.begin(), internal.end());
  internal.resize(static_cast<std::size_t>(new_end - internal.begin()), stream);

  std::vector<vertex_t> host(internal.size());
  raft::copy(host.data(), internal.data(), internal.size(), stream);
  handle.sync_stream();

  if (std::any_of(host.begin(), host.end(), [](vertex_t v) {
        return v == cugraph::invalid_vertex_id<vertex_t>::value;
      })) {
    throw std::invalid_argument("prepare_vertices: a vertex ID is not present in the graph");
  }

  return prepared_vertices_instance_t<vertex_t>{graph.context,
                                                std::move(internal),
                                                graph.graph_id,
                                                graph.numbering_id,
                                                graph.version,
                                                /*partition_rank=*/0,
                                                /*partition_count=*/1};
}

template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
std::tuple<device_array_t, device_array_t, std::optional<device_array_t>> bfs_impl(
  graph_instance_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph,
  prepared_vertices_instance_t<vertex_t> const& starts,
  bool compute_predecessors)
{
  if constexpr (store_transposed) {
    throw std::invalid_argument("bfs: requires a graph created with store_transposed=false");
  } else {
  check_prepared_for(graph, starts);
  if (starts.internal_ids.size() == 0) {
    throw std::invalid_argument("bfs: starts must not be empty");
  }
  if (!graph.number_map) {
    throw std::logic_error("bfs: graph has no number map (renumbering disabled)");
  }

  auto& handle           = graph.context->handle;
  auto stream            = handle.get_stream();
  auto view              = graph.native.view();
  auto const& number_map = *graph.number_map;

  auto [distance_property, predecessor_property] = cugraph::bfs<vertex_t, edge_t, multi_gpu>(
    handle,
    view,
    raft::device_span<vertex_t const>(starts.internal_ids.data(), starts.internal_ids.size()),
    /*direction_optimizing=*/false,
    /*depth_limit=*/std::numeric_limits<vertex_t>::max(),
    compute_predecessors,
    /*do_expensive_check=*/false);

  // External IDs, one per internal vertex, aligned with distances and predecessors.
  auto vertices = std::make_shared<rmm::device_uvector<vertex_t>>(number_map.size(), stream);
  raft::copy(vertices->data(), number_map.data(), number_map.size(), stream);

  auto distances =
    std::make_shared<rmm::device_uvector<vertex_t>>(distance_property.release_buffer());

  std::optional<device_array_t> predecessors;
  if (predecessor_property) {
    // Internal parent IDs -> external parent IDs. Invalid (no parent) stays invalid.
    auto unrenumbered = cugraph::unrenumber_int_vertices(
      handle,
      view,
      predecessor_property->release_buffer(),
      cugraph::vertex_property_view_t<vertex_t, vertex_t const*>(number_map.data(),
                                                                 number_map.size()),
      /*do_expensive_check=*/false);
    predecessors = device_array_t{
      std::make_shared<rmm::device_uvector<vertex_t>>(std::move(unrenumbered))};
  }

  handle.sync_stream();
  return {device_array_t{std::move(vertices)}, device_array_t{std::move(distances)},
          std::move(predecessors)};
  }
}

}  // namespace

std::unique_ptr<prepared_vertices_t> prepare_vertices(graph_t const& graph,
                                                      std::vector<std::int64_t> const& external_ids)
{
  return std::visit(
    [&](auto const& instance) -> std::unique_ptr<prepared_vertices_t> {
      using instance_t = std::decay_t<decltype(instance)>;
      if constexpr (instance_t::is_storage_transposed) {
        throw std::invalid_argument(
          "prepare_vertices: requires a graph created with store_transposed=false");
      } else {
        return std::make_unique<prepared_vertices_t>(prepare_vertices_impl(instance, external_ids));
      }
    },
    graph.impl);
}

std::tuple<device_array_t, device_array_t, std::optional<device_array_t>>
bfs(graph_t const& graph, prepared_vertices_t const& starts, bool compute_predecessors)
{
  return std::visit(
    [&](auto const& instance)
      -> std::tuple<device_array_t, device_array_t, std::optional<device_array_t>> {
      using vertex_t = typename std::decay_t<decltype(instance)>::vertex_type;
      auto const* prepared = std::get_if<prepared_vertices_instance_t<vertex_t>>(&starts.impl);
      if (prepared == nullptr) {
        throw std::invalid_argument("PreparedVertices vertex type does not match the graph");
      }
      return bfs_impl(instance, *prepared, compute_predecessors);
    },
    graph.impl);
}

}  // namespace cugraph_python
