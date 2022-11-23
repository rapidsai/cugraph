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

#pragma once

#include <detail/graph_utils.cuh>

#include <cugraph/detail/decompress_edge_partition.cuh>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/binary_search.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

#include <algorithm>
#include <optional>
#include <tuple>
#include <type_traits>
#include <vector>

namespace cugraph {

namespace {

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::enable_if_t<multi_gpu,
                 std::tuple<rmm::device_uvector<vertex_t>,
                            rmm::device_uvector<vertex_t>,
                            std::optional<rmm::device_uvector<weight_t>>>>
decompress_to_edgelist_impl(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<raft::device_span<vertex_t const>> renumber_map,
  bool do_expensive_check)
{
  CUGRAPH_EXPECTS(!renumber_map.has_value() ||
                    ((*renumber_map).size() ==
                     static_cast<size_t>(graph_view.local_vertex_partition_range_size())),
                  "Invalid input arguments: (*renumber_map).size() should match with the local "
                  "vertex partition range size.");

  if (do_expensive_check) { /* currently, nothing to do */
  }

  auto& row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
  auto const row_comm_size = row_comm.get_size();
  auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
  auto const col_comm_rank = col_comm.get_rank();

  std::vector<size_t> edgelist_edge_counts(graph_view.number_of_local_edge_partitions(), size_t{0});
  for (size_t i = 0; i < edgelist_edge_counts.size(); ++i) {
    edgelist_edge_counts[i] =
      static_cast<size_t>(graph_view.number_of_local_edge_partition_edges(i));
  }
  auto number_of_local_edges =
    std::reduce(edgelist_edge_counts.begin(), edgelist_edge_counts.end());

  rmm::device_uvector<vertex_t> edgelist_majors(number_of_local_edges, handle.get_stream());
  rmm::device_uvector<vertex_t> edgelist_minors(edgelist_majors.size(), handle.get_stream());
  auto edgelist_weights = edge_weight_view ? std::make_optional<rmm::device_uvector<weight_t>>(
                                               edgelist_majors.size(), handle.get_stream())
                                           : std::nullopt;

  size_t cur_size{0};
  for (size_t i = 0; i < edgelist_edge_counts.size(); ++i) {
    detail::decompress_edge_partition_to_edgelist(
      handle,
      edge_partition_device_view_t<vertex_t, edge_t, multi_gpu>(
        graph_view.local_edge_partition_view(i)),
      edge_weight_view
        ? std::make_optional<
            detail::edge_partition_edge_property_device_view_t<edge_t, weight_t const*>>(
            (*edge_weight_view), i)
        : std::nullopt,
      edgelist_majors.data() + cur_size,
      edgelist_minors.data() + cur_size,
      edgelist_weights ? std::optional<weight_t*>{(*edgelist_weights).data() + cur_size}
                       : std::nullopt,
      graph_view.local_edge_partition_segment_offsets(i));
    cur_size += edgelist_edge_counts[i];
  }

  if (renumber_map) {
    std::vector<vertex_t> h_thresholds(row_comm_size - 1, vertex_t{0});
    for (int i = 0; i < row_comm_size - 1; ++i) {
      h_thresholds[i] = graph_view.vertex_partition_range_last(col_comm_rank * row_comm_size + i);
    }
    rmm::device_uvector<vertex_t> d_thresholds(h_thresholds.size(), handle.get_stream());
    raft::update_device(
      d_thresholds.data(), h_thresholds.data(), h_thresholds.size(), handle.get_stream());

    std::vector<vertex_t*> major_ptrs(edgelist_edge_counts.size());
    std::vector<vertex_t*> minor_ptrs(major_ptrs.size());
    auto edgelist_intra_partition_segment_offsets =
      std::make_optional<std::vector<std::vector<size_t>>>(
        major_ptrs.size(), std::vector<size_t>(row_comm_size + 1, size_t{0}));
    size_t cur_size{0};
    for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
      major_ptrs[i] = edgelist_majors.data() + cur_size;
      minor_ptrs[i] = edgelist_minors.data() + cur_size;
      if (edgelist_weights) {
        thrust::sort_by_key(handle.get_thrust_policy(),
                            minor_ptrs[i],
                            minor_ptrs[i] + edgelist_edge_counts[i],
                            thrust::make_zip_iterator(thrust::make_tuple(
                              major_ptrs[i], (*edgelist_weights).data() + cur_size)));
      } else {
        thrust::sort_by_key(handle.get_thrust_policy(),
                            minor_ptrs[i],
                            minor_ptrs[i] + edgelist_edge_counts[i],
                            major_ptrs[i]);
      }
      rmm::device_uvector<size_t> d_segment_offsets(d_thresholds.size(), handle.get_stream());
      thrust::lower_bound(handle.get_thrust_policy(),
                          minor_ptrs[i],
                          minor_ptrs[i] + edgelist_edge_counts[i],
                          d_thresholds.begin(),
                          d_thresholds.end(),
                          d_segment_offsets.begin(),
                          thrust::less<vertex_t>{});
      (*edgelist_intra_partition_segment_offsets)[i][0]     = size_t{0};
      (*edgelist_intra_partition_segment_offsets)[i].back() = edgelist_edge_counts[i];
      raft::update_host((*edgelist_intra_partition_segment_offsets)[i].data() + 1,
                        d_segment_offsets.data(),
                        d_segment_offsets.size(),
                        handle.get_stream());
      handle.sync_stream();
      cur_size += edgelist_edge_counts[i];
    }

    unrenumber_local_int_edges<vertex_t, store_transposed, multi_gpu>(
      handle,
      store_transposed ? minor_ptrs : major_ptrs,
      store_transposed ? major_ptrs : minor_ptrs,
      edgelist_edge_counts,
      (*renumber_map).data(),
      graph_view.vertex_partition_range_lasts(),
      edgelist_intra_partition_segment_offsets);
  }

  return std::make_tuple(store_transposed ? std::move(edgelist_minors) : std::move(edgelist_majors),
                         store_transposed ? std::move(edgelist_majors) : std::move(edgelist_minors),
                         std::move(edgelist_weights));
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::enable_if_t<!multi_gpu,
                 std::tuple<rmm::device_uvector<vertex_t>,
                            rmm::device_uvector<vertex_t>,
                            std::optional<rmm::device_uvector<weight_t>>>>
decompress_to_edgelist_impl(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<raft::device_span<vertex_t const>> renumber_map,
  bool do_expensive_check)
{
  CUGRAPH_EXPECTS(
    !renumber_map.has_value() ||
      (*renumber_map).size() == static_cast<size_t>(graph_view.local_vertex_partition_range_size()),
    "Invalid input arguments: if renumber_map.has_value() == true, (*renumber_map).size() should "
    "match with the local vertex partition range size.");

  if (do_expensive_check) { /* currently, nothing to do */
  }

  rmm::device_uvector<vertex_t> edgelist_majors(graph_view.number_of_local_edge_partition_edges(),
                                                handle.get_stream());
  rmm::device_uvector<vertex_t> edgelist_minors(edgelist_majors.size(), handle.get_stream());
  auto edgelist_weights = edge_weight_view ? std::make_optional<rmm::device_uvector<weight_t>>(
                                               edgelist_majors.size(), handle.get_stream())
                                           : std::nullopt;
  detail::decompress_edge_partition_to_edgelist(
    handle,
    edge_partition_device_view_t<vertex_t, edge_t, multi_gpu>(
      graph_view.local_edge_partition_view()),
    edge_weight_view
      ? std::make_optional<
          detail::edge_partition_edge_property_device_view_t<edge_t, weight_t const*>>(
          (*edge_weight_view), 0)
      : std::nullopt,
    edgelist_majors.data(),
    edgelist_minors.data(),
    edgelist_weights ? std::optional<weight_t*>{(*edgelist_weights).data()} : std::nullopt,
    graph_view.local_edge_partition_segment_offsets());

  if (renumber_map) {
    unrenumber_local_int_edges<vertex_t, store_transposed, multi_gpu>(
      handle,
      store_transposed ? edgelist_minors.data() : edgelist_majors.data(),
      store_transposed ? edgelist_majors.data() : edgelist_minors.data(),
      edgelist_majors.size(),
      (*renumber_map).data(),
      (*renumber_map).size());
  }

  return std::make_tuple(store_transposed ? std::move(edgelist_minors) : std::move(edgelist_majors),
                         store_transposed ? std::move(edgelist_majors) : std::move(edgelist_minors),
                         std::move(edgelist_weights));
}

}  // namespace

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>>
decompress_to_edgelist(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<raft::device_span<vertex_t const>> renumber_map,
  bool do_expensive_check)
{
  return decompress_to_edgelist_impl(
    handle, graph_view, edge_weight_view, renumber_map, do_expensive_check);
}

}  // namespace cugraph
