/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include "detail/graph_partition_utils.cuh"

#include <cugraph/detail/decompress_edge_partition.cuh>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/mask_utils.cuh>

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
          typename edge_type_t,
          bool store_transposed,
          bool multi_gpu>
std::enable_if_t<multi_gpu,
                 std::tuple<rmm::device_uvector<vertex_t>,
                            rmm::device_uvector<vertex_t>,
                            std::optional<rmm::device_uvector<weight_t>>,
                            std::optional<rmm::device_uvector<edge_t>>,
                            std::optional<rmm::device_uvector<edge_type_t>>>>
decompress_to_edgelist_impl(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<edge_property_view_t<edge_t, edge_t const*>> edge_id_view,
  std::optional<edge_property_view_t<edge_t, edge_type_t const*>> edge_type_view,
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

  auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
  auto const major_comm_rank = major_comm.get_rank();
  auto const major_comm_size = major_comm.get_size();
  auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
  auto const minor_comm_rank = minor_comm.get_rank();
  auto const minor_comm_size = minor_comm.get_size();

  std::vector<size_t> edgelist_edge_counts(graph_view.number_of_local_edge_partitions(), size_t{0});
  for (size_t i = 0; i < edgelist_edge_counts.size(); ++i) {
    edgelist_edge_counts[i] = graph_view.local_edge_partition_view(i).number_of_edges();
    if (graph_view.has_edge_mask()) {
      edgelist_edge_counts[i] = detail::count_set_bits(
        handle, (*(graph_view.edge_mask_view())).value_firsts()[i], edgelist_edge_counts[i]);
    }
  }
  auto number_of_local_edges =
    std::reduce(edgelist_edge_counts.begin(), edgelist_edge_counts.end());

  rmm::device_uvector<vertex_t> edgelist_majors(number_of_local_edges, handle.get_stream());
  rmm::device_uvector<vertex_t> edgelist_minors(edgelist_majors.size(), handle.get_stream());
  auto edgelist_ids     = edge_id_view ? std::make_optional<rmm::device_uvector<edge_t>>(
                                       edgelist_majors.size(), handle.get_stream())
                                       : std::nullopt;
  auto edgelist_weights = edge_weight_view ? std::make_optional<rmm::device_uvector<weight_t>>(
                                               edgelist_majors.size(), handle.get_stream())
                                           : std::nullopt;
  auto edgelist_types   = edge_type_view ? std::make_optional<rmm::device_uvector<edge_type_t>>(
                                           edgelist_majors.size(), handle.get_stream())
                                         : std::nullopt;

  size_t cur_size{0};
  for (size_t i = 0; i < edgelist_edge_counts.size(); ++i) {
    detail::decompress_edge_partition_to_edgelist<vertex_t, edge_t, weight_t, int32_t, multi_gpu>(
      handle,
      edge_partition_device_view_t<vertex_t, edge_t, multi_gpu>(
        graph_view.local_edge_partition_view(i)),
      edge_weight_view
        ? std::make_optional<
            detail::edge_partition_edge_property_device_view_t<edge_t, weight_t const*>>(
            (*edge_weight_view), i)
        : std::nullopt,
      edge_id_view ? std::make_optional<
                       detail::edge_partition_edge_property_device_view_t<edge_t, edge_t const*>>(
                       (*edge_id_view), i)
                   : std::nullopt,
      edge_type_view
        ? std::make_optional<
            detail::edge_partition_edge_property_device_view_t<edge_t, edge_type_t const*>>(
            (*edge_type_view), i)
        : std::nullopt,
      graph_view.has_edge_mask()
        ? std::make_optional<
            detail::edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool>>(
            *(graph_view.edge_mask_view()), i)
        : std::nullopt,
      raft::device_span<vertex_t>(edgelist_majors.data() + cur_size, edgelist_edge_counts[i]),
      raft::device_span<vertex_t>(edgelist_minors.data() + cur_size, edgelist_edge_counts[i]),
      edgelist_weights ? std::make_optional<raft::device_span<weight_t>>(
                           (*edgelist_weights).data() + cur_size, edgelist_edge_counts[i])
                       : std::nullopt,
      edgelist_ids ? std::make_optional<raft::device_span<edge_t>>(
                       (*edgelist_ids).data() + cur_size, edgelist_edge_counts[i])
                   : std::nullopt,
      edgelist_types ? std::make_optional<raft::device_span<edge_type_t>>(
                         (*edgelist_types).data() + cur_size, edgelist_edge_counts[i])
                     : std::nullopt,
      graph_view.local_edge_partition_segment_offsets(i));
    cur_size += edgelist_edge_counts[i];
  }

  if (renumber_map) {
    std::vector<vertex_t> h_thresholds(major_comm_size - 1, vertex_t{0});
    for (int i = 0; i < major_comm_size - 1; ++i) {
      auto minor_range_vertex_partition_id =
        detail::compute_local_edge_partition_minor_range_vertex_partition_id_t{
          major_comm_size, minor_comm_size, major_comm_rank, minor_comm_rank}(i);
      h_thresholds[i] = graph_view.vertex_partition_range_last(minor_range_vertex_partition_id);
    }
    rmm::device_uvector<vertex_t> d_thresholds(h_thresholds.size(), handle.get_stream());
    raft::update_device(
      d_thresholds.data(), h_thresholds.data(), h_thresholds.size(), handle.get_stream());

    std::vector<vertex_t*> major_ptrs(edgelist_edge_counts.size());
    std::vector<vertex_t*> minor_ptrs(major_ptrs.size());
    auto edgelist_intra_partition_segment_offsets =
      std::make_optional<std::vector<std::vector<size_t>>>(
        major_ptrs.size(), std::vector<size_t>(major_comm_size + 1, size_t{0}));
    size_t cur_size{0};
    for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
      major_ptrs[i] = edgelist_majors.data() + cur_size;
      minor_ptrs[i] = edgelist_minors.data() + cur_size;

      if (edgelist_weights) {
        if (edgelist_ids) {
          if (edgelist_types) {
            auto zip_itr =
              thrust::make_zip_iterator(thrust::make_tuple(major_ptrs[i],
                                                           (*edgelist_weights).data() + cur_size,
                                                           (*edgelist_ids).data() + cur_size,
                                                           (*edgelist_types).data() + cur_size));

            thrust::sort_by_key(handle.get_thrust_policy(),
                                minor_ptrs[i],
                                minor_ptrs[i] + edgelist_edge_counts[i],
                                zip_itr);

          } else {
            auto zip_itr =
              thrust::make_zip_iterator(thrust::make_tuple(major_ptrs[i],
                                                           (*edgelist_weights).data() + cur_size,
                                                           (*edgelist_ids).data() + cur_size));

            thrust::sort_by_key(handle.get_thrust_policy(),
                                minor_ptrs[i],
                                minor_ptrs[i] + edgelist_edge_counts[i],
                                zip_itr);
          }
        } else {
          if (edgelist_types) {
            auto zip_itr =
              thrust::make_zip_iterator(thrust::make_tuple(major_ptrs[i],
                                                           (*edgelist_weights).data() + cur_size,
                                                           (*edgelist_types).data() + cur_size));

            thrust::sort_by_key(handle.get_thrust_policy(),
                                minor_ptrs[i],
                                minor_ptrs[i] + edgelist_edge_counts[i],
                                zip_itr);

          } else {
            auto zip_itr = thrust::make_zip_iterator(
              thrust::make_tuple(major_ptrs[i], (*edgelist_weights).data() + cur_size));

            thrust::sort_by_key(handle.get_thrust_policy(),
                                minor_ptrs[i],
                                minor_ptrs[i] + edgelist_edge_counts[i],
                                zip_itr);
          }
        }
      } else {
        if (edgelist_ids) {
          if (edgelist_types) {
            auto zip_itr =
              thrust::make_zip_iterator(thrust::make_tuple(major_ptrs[i],
                                                           (*edgelist_ids).data() + cur_size,
                                                           (*edgelist_types).data() + cur_size));

            thrust::sort_by_key(handle.get_thrust_policy(),
                                minor_ptrs[i],
                                minor_ptrs[i] + edgelist_edge_counts[i],
                                zip_itr);

          } else {
            auto zip_itr = thrust::make_zip_iterator(
              thrust::make_tuple(major_ptrs[i], (*edgelist_ids).data() + cur_size));

            thrust::sort_by_key(handle.get_thrust_policy(),
                                minor_ptrs[i],
                                minor_ptrs[i] + edgelist_edge_counts[i],
                                zip_itr);
          }
        } else {
          if (edgelist_types) {
            auto zip_itr = thrust::make_zip_iterator(
              thrust::make_tuple(major_ptrs[i], (*edgelist_types).data() + cur_size));

            thrust::sort_by_key(handle.get_thrust_policy(),
                                minor_ptrs[i],
                                minor_ptrs[i] + edgelist_edge_counts[i],
                                zip_itr);

          } else {
            thrust::sort_by_key(handle.get_thrust_policy(),
                                minor_ptrs[i],
                                minor_ptrs[i] + edgelist_edge_counts[i],
                                major_ptrs[i]);
          }
        }
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
                         std::move(edgelist_weights),
                         std::move(edgelist_ids),
                         std::move(edgelist_types));
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          bool store_transposed,
          bool multi_gpu>
std::enable_if_t<!multi_gpu,
                 std::tuple<rmm::device_uvector<vertex_t>,
                            rmm::device_uvector<vertex_t>,
                            std::optional<rmm::device_uvector<weight_t>>,
                            std::optional<rmm::device_uvector<edge_t>>,
                            std::optional<rmm::device_uvector<edge_type_t>>>>
decompress_to_edgelist_impl(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<edge_property_view_t<edge_t, edge_t const*>> edge_id_view,
  std::optional<edge_property_view_t<edge_t, edge_type_t const*>> edge_type_view,
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

  auto num_edges = graph_view.local_edge_partition_view().number_of_edges();
  if (graph_view.has_edge_mask()) {
    num_edges =
      detail::count_set_bits(handle, (*(graph_view.edge_mask_view())).value_firsts()[0], num_edges);
  }

  rmm::device_uvector<vertex_t> edgelist_majors(num_edges, handle.get_stream());
  rmm::device_uvector<vertex_t> edgelist_minors(edgelist_majors.size(), handle.get_stream());
  auto edgelist_weights = edge_weight_view ? std::make_optional<rmm::device_uvector<weight_t>>(
                                               edgelist_majors.size(), handle.get_stream())
                                           : std::nullopt;
  auto edgelist_ids     = edge_id_view ? std::make_optional<rmm::device_uvector<edge_t>>(
                                       edgelist_majors.size(), handle.get_stream())
                                       : std::nullopt;

  auto edgelist_types = edge_type_view ? std::make_optional<rmm::device_uvector<edge_type_t>>(
                                           edgelist_majors.size(), handle.get_stream())
                                       : std::nullopt;

  detail::decompress_edge_partition_to_edgelist<vertex_t, edge_t, weight_t, int32_t, multi_gpu>(
    handle,
    edge_partition_device_view_t<vertex_t, edge_t, multi_gpu>(
      graph_view.local_edge_partition_view()),
    edge_weight_view
      ? std::make_optional<
          detail::edge_partition_edge_property_device_view_t<edge_t, weight_t const*>>(
          (*edge_weight_view), 0)
      : std::nullopt,
    edge_id_view ? std::make_optional<
                     detail::edge_partition_edge_property_device_view_t<edge_t, edge_t const*>>(
                     (*edge_id_view), 0)
                 : std::nullopt,
    edge_type_view
      ? std::make_optional<
          detail::edge_partition_edge_property_device_view_t<edge_t, edge_type_t const*>>(
          (*edge_type_view), 0)
      : std::nullopt,
    graph_view.has_edge_mask()
      ? std::make_optional<
          detail::edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool>>(
          *(graph_view.edge_mask_view()), 0)
      : std::nullopt,
    raft::device_span<vertex_t>(edgelist_majors.data(), edgelist_majors.size()),
    raft::device_span<vertex_t>(edgelist_minors.data(), edgelist_minors.size()),
    edgelist_weights ? std::make_optional<raft::device_span<weight_t>>((*edgelist_weights).data(),
                                                                       (*edgelist_weights).size())
                     : std::nullopt,
    edgelist_ids ? std::make_optional<raft::device_span<edge_t>>((*edgelist_ids).data(),
                                                                 (*edgelist_ids).size())
                 : std::nullopt,
    edgelist_types ? std::make_optional<raft::device_span<edge_type_t>>((*edgelist_types).data(),
                                                                        (*edgelist_types).size())
                   : std::nullopt,

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
                         std::move(edgelist_weights),
                         std::move(edgelist_ids),
                         std::move(edgelist_types));
}

}  // namespace

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           std::optional<rmm::device_uvector<edge_t>>,
           std::optional<rmm::device_uvector<edge_type_t>>>
decompress_to_edgelist(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<edge_property_view_t<edge_t, edge_t const*>> edge_id_view,
  std::optional<edge_property_view_t<edge_t, edge_type_t const*>> edge_type_view,
  std::optional<raft::device_span<vertex_t const>> renumber_map,
  bool do_expensive_check)
{
  return decompress_to_edgelist_impl<vertex_t,
                                     edge_t,
                                     weight_t,
                                     edge_type_t,
                                     store_transposed,
                                     multi_gpu>(handle,
                                                graph_view,
                                                edge_weight_view,
                                                edge_id_view,
                                                edge_type_view,
                                                renumber_map,
                                                do_expensive_check);
}

}  // namespace cugraph
