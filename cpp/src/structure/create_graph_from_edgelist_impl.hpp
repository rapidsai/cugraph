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

#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/error.hpp>

#include <rmm/thrust_rmm_allocator.h>

#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#include <cstdint>
#include <numeric>

namespace cugraph {

namespace {

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::enable_if_t<
  multi_gpu,
  std::tuple<cugraph::graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>,
             std::optional<rmm::device_uvector<vertex_t>>>>
create_graph_from_edgelist_impl(
  raft::handle_t const& handle,
  std::optional<std::tuple<vertex_t const*, vertex_t>> optional_local_vertex_span,
  rmm::device_uvector<vertex_t>&& edgelist_rows,
  rmm::device_uvector<vertex_t>&& edgelist_cols,
  std::optional<rmm::device_uvector<weight_t>>&& edgelist_weights,
  graph_properties_t graph_properties,
  bool renumber)
{
  CUGRAPH_EXPECTS(renumber, "renumber should be true if multi_gpu is true.");

  auto& comm               = handle.get_comms();
  auto const comm_size     = comm.get_size();
  auto const comm_rank     = comm.get_rank();
  auto& row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
  auto const row_comm_size = row_comm.get_size();
  auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
  auto const col_comm_size = col_comm.get_size();

  auto edge_counts =
    cugraph::detail::groupby_and_count_by_edge(handle,
                                               store_transposed ? edgelist_cols : edgelist_rows,
                                               store_transposed ? edgelist_rows : edgelist_cols,
                                               edgelist_weights,
                                               col_comm_size);

  std::vector<size_t> h_edge_counts(edge_counts.size());
  raft::update_host(
    h_edge_counts.data(), edge_counts.data(), edge_counts.size(), handle.get_stream());
  handle.get_stream_view().synchronize();

  std::vector<size_t> h_displacements(h_edge_counts.size(), size_t{0});
  std::partial_sum(h_edge_counts.begin(), h_edge_counts.end() - 1, h_displacements.begin() + 1);

  // 3. renumber

  rmm::device_uvector<vertex_t> renumber_map_labels(0, handle.get_stream());
  cugraph::partition_t<vertex_t> partition{};
  vertex_t number_of_vertices{};
  edge_t number_of_edges{};
  auto segment_offsets = std::make_optional<std::vector<vertex_t>>(0);
  {
    std::vector<vertex_t*> major_ptrs(h_edge_counts.size());
    std::vector<vertex_t*> minor_ptrs(major_ptrs.size());
    std::vector<edge_t> counts(major_ptrs.size());
    for (size_t i = 0; i < h_edge_counts.size(); ++i) {
      major_ptrs[i] =
        (store_transposed ? edgelist_cols.begin() : edgelist_rows.begin()) + h_displacements[i];
      minor_ptrs[i] =
        (store_transposed ? edgelist_rows.begin() : edgelist_cols.begin()) + h_displacements[i];
      counts[i] = static_cast<edge_t>(h_edge_counts[i]);
    }
    std::tie(
      renumber_map_labels, partition, number_of_vertices, number_of_edges, *segment_offsets) =
      cugraph::renumber_edgelist<vertex_t, edge_t, multi_gpu>(
        handle, optional_local_vertex_span, major_ptrs, minor_ptrs, counts);
  }

  // 4. create a graph

  std::vector<cugraph::edgelist_t<vertex_t, edge_t, weight_t>> edgelists(h_edge_counts.size());
  for (size_t i = 0; i < h_edge_counts.size(); ++i) {
    edgelists[i] = cugraph::edgelist_t<vertex_t, edge_t, weight_t>{
      edgelist_rows.data() + h_displacements[i],
      edgelist_cols.data() + h_displacements[i],
      edgelist_weights
        ? std::optional<weight_t const*>{(*edgelist_weights).data() + h_displacements[i]}
        : std::nullopt,
      static_cast<edge_t>(h_edge_counts[i])};
  }

  return std::make_tuple(
    cugraph::graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
      handle,
      edgelists,
      partition,
      number_of_vertices,
      number_of_edges,
      graph_properties,
      std::optional<std::vector<vertex_t>>{segment_offsets}),
    std::optional<rmm::device_uvector<vertex_t>>{std::move(renumber_map_labels)});
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::enable_if_t<
  !multi_gpu,
  std::tuple<cugraph::graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>,
             std::optional<rmm::device_uvector<vertex_t>>>>
create_graph_from_edgelist_impl(
  raft::handle_t const& handle,
  std::optional<std::tuple<vertex_t const*, vertex_t>> optional_vertex_span,
  rmm::device_uvector<vertex_t>&& edgelist_rows,
  rmm::device_uvector<vertex_t>&& edgelist_cols,
  std::optional<rmm::device_uvector<weight_t>>&& edgelist_weights,
  graph_properties_t graph_properties,
  bool renumber)
{
  auto renumber_map_labels =
    renumber ? std::make_optional<rmm::device_uvector<vertex_t>>(0, handle.get_stream())
             : std::nullopt;
  std::optional<std::vector<vertex_t>> segment_offsets{std::nullopt};
  if (renumber) {
    segment_offsets = std::vector<vertex_t>{};
    std::tie(*renumber_map_labels, *segment_offsets) =
      cugraph::renumber_edgelist<vertex_t, edge_t, multi_gpu>(
        handle,
        optional_vertex_span,
        store_transposed ? edgelist_cols.data() : edgelist_rows.data(),
        store_transposed ? edgelist_rows.data() : edgelist_cols.data(),
        static_cast<edge_t>(edgelist_rows.size()));
  }

  vertex_t num_vertices{};
  if (renumber) {
    num_vertices = static_cast<vertex_t>((*renumber_map_labels).size());
  } else {
    if (optional_vertex_span) {
      num_vertices = std::get<1>(*optional_vertex_span);
    } else {
      num_vertices = 1 + cugraph::detail::compute_maximum_vertex_id(
                           handle.get_stream_view(), edgelist_rows, edgelist_cols);
    }
  }

  return std::make_tuple(
    cugraph::graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
      handle,
      cugraph::edgelist_t<vertex_t, edge_t, weight_t>{
        edgelist_rows.data(),
        edgelist_cols.data(),
        edgelist_weights ? std::optional<weight_t const*>{(*edgelist_weights).data()}
                         : std::nullopt,
        static_cast<edge_t>(edgelist_rows.size())},
      num_vertices,
      graph_properties,
      std::optional<std::vector<vertex_t>>{segment_offsets}),
    std::move(renumber_map_labels));
}

}  // namespace

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<cugraph::graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>,
           std::optional<rmm::device_uvector<vertex_t>>>
create_graph_from_edgelist(
  raft::handle_t const& handle,
  std::optional<std::tuple<vertex_t const*, vertex_t>> optional_vertex_span,
  rmm::device_uvector<vertex_t>&& edgelist_rows,
  rmm::device_uvector<vertex_t>&& edgelist_cols,
  std::optional<rmm::device_uvector<weight_t>>&& edgelist_weights,
  graph_properties_t graph_properties,
  bool renumber)
{
  return create_graph_from_edgelist_impl<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
    handle,
    optional_vertex_span,
    std::move(edgelist_rows),
    std::move(edgelist_cols),
    std::move(edgelist_weights),
    graph_properties,
    renumber);
}

}  // namespace cugraph
