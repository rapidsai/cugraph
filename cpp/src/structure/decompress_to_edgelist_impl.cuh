/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cugraph/arithmetic_variant_types.hpp>
#include <cugraph/detail/decompress_edge_partition.cuh>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/large_buffer_manager.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/graph_partition_utils.cuh>
#include <cugraph/utilities/mask_utils.cuh>
#include <cugraph/utilities/thrust_wrappers/sequence.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/std/functional>
#include <cuda/std/tuple>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/gather.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>

#include <algorithm>
#include <numeric>
#include <optional>
#include <tuple>
#include <type_traits>
#include <vector>

namespace cugraph {

namespace {

template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::vector<arithmetic_device_uvector_t>>
decompress_to_edgelist_impl(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  std::vector<edge_arithmetic_property_view_t<edge_t>> const& edge_property_views,
  std::optional<vertex_property_view_t<vertex_t, vertex_t const*>> renumber_map,
  std::optional<large_buffer_type_t> large_buffer_type,
  bool do_expensive_check)
{
  CUGRAPH_EXPECTS(!large_buffer_type || cugraph::large_buffer_manager::memory_buffer_initialized(),
                  "Invalid input argument: large memory_buffer is not initialized.");

  if (do_expensive_check) { /* currently, nothing to do */
  }

  std::vector<size_t> edgelist_edge_counts(graph_view.number_of_local_edge_partitions(), size_t{0});
  for (size_t i = 0; i < edgelist_edge_counts.size(); ++i) {
    edgelist_edge_counts[i] = graph_view.local_edge_partition_view(i).number_of_edges();
    if (graph_view.has_edge_mask()) {
      edgelist_edge_counts[i] = count_set_bits(handle.get_thrust_policy(),
                                               graph_view.edge_mask_view()->value_firsts()[i],
                                               edgelist_edge_counts[i]);
    }
  }
  std::vector<size_t> edgelist_partition_offsets(edgelist_edge_counts.size() + 1, size_t{0});
  std::inclusive_scan(edgelist_edge_counts.begin(),
                      edgelist_edge_counts.end(),
                      edgelist_partition_offsets.begin() + 1);

  auto edgelist_majors =
    large_buffer_type
      ? large_buffer_manager::allocate_memory_buffer<vertex_t>(edgelist_partition_offsets.back(),
                                                               handle.get_stream())
      : rmm::device_uvector<vertex_t>(edgelist_partition_offsets.back(), handle.get_stream());
  auto edgelist_minors =
    large_buffer_type
      ? large_buffer_manager::allocate_memory_buffer<vertex_t>(edgelist_partition_offsets.back(),
                                                               handle.get_stream())
      : rmm::device_uvector<vertex_t>(edgelist_partition_offsets.back(), handle.get_stream());
  std::vector<arithmetic_device_uvector_t> edgelist_edge_properties{};
  edgelist_edge_properties.reserve(edge_property_views.size());
  for (auto view : edge_property_views) {
    edgelist_edge_properties.push_back(variant_type_dispatch(view, [&](auto& v) {
      using T = typename std::remove_reference_t<decltype(v)>::value_type;
      return arithmetic_device_uvector_t(
        large_buffer_type
          ? large_buffer_manager::allocate_memory_buffer<T>(edgelist_partition_offsets.back(),
                                                            handle.get_stream())
          : rmm::device_uvector<T>(edgelist_partition_offsets.back(), handle.get_stream()));
    }));
  }

  for (size_t i = 0; i < edgelist_edge_counts.size(); ++i) {
    auto edge_partition = edge_partition_device_view_t<vertex_t, edge_t, multi_gpu>(
      graph_view.local_edge_partition_view(i));
    auto edge_partition_mask_view =
      graph_view.has_edge_mask()
        ? std::make_optional<
            detail::edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool>>(
            *(graph_view.edge_mask_view()), i)
        : std::nullopt;
    auto copy_edge_values = [&handle, &edge_partition, &edge_partition_mask_view](
                              auto input_first, auto output_first) {
      if (edge_partition_mask_view) {
        cugraph::copy_if_mask_set(handle,
                                  input_first,
                                  input_first + edge_partition.number_of_edges(),
                                  edge_partition_mask_view->value_first(),
                                  output_first);
      } else {
        thrust::copy(handle.get_thrust_policy(),
                     input_first,
                     input_first + edge_partition.number_of_edges(),
                     output_first);
      }
    };

    detail::decompress_edge_partition_to_fill_edgelist_majors(
      handle,
      edge_partition,
      edge_partition_mask_view,
      raft::device_span<vertex_t>(edgelist_majors.data() + edgelist_partition_offsets[i],
                                  edgelist_edge_counts[i]),
      graph_view.local_edge_partition_segment_offsets(i));
    copy_edge_values(edge_partition.indices(),
                     edgelist_minors.begin() + edgelist_partition_offsets[i]);
    for (size_t j = 0; j < edge_property_views.size(); ++j) {
      auto view = edge_property_views[j];
      variant_type_dispatch(view, [&](auto& v) {
        using T = typename std::remove_reference_t<decltype(v)>::value_type;
        copy_edge_values(v.value_firsts()[i],
                         std::get<rmm::device_uvector<T>>(edgelist_edge_properties[j]).begin() +
                           edgelist_partition_offsets[i]);
      });
    }
  }

  auto& edgelist_srcs = store_transposed ? edgelist_minors : edgelist_majors;
  auto& edgelist_dsts = store_transposed ? edgelist_majors : edgelist_minors;

  if (renumber_map) {
    if constexpr (multi_gpu) {
      auto& major_comm = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
      auto const major_comm_rank = major_comm.get_rank();
      auto const major_comm_size = major_comm.get_size();
      auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
      auto const minor_comm_rank = minor_comm.get_rank();
      auto const minor_comm_size = minor_comm.get_size();

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

      auto edgelist_intra_partition_segment_offsets =
        std::make_optional<std::vector<std::vector<size_t>>>(
          edgelist_edge_counts.size(), std::vector<size_t>(major_comm_size + 1, size_t{0}));
      for (size_t i = 0; i < edgelist_edge_counts.size(); ++i) {
        auto major_ptr = edgelist_majors.data() + edgelist_partition_offsets[i];
        auto minor_ptr = edgelist_minors.data() + edgelist_partition_offsets[i];

        if (edgelist_edge_properties.empty()) {
          thrust::sort_by_key(
            handle.get_thrust_policy(), minor_ptr, minor_ptr + edgelist_edge_counts[i], major_ptr);
        } else {
          auto positions = large_buffer_type ? large_buffer_manager::allocate_memory_buffer<size_t>(
                                                 edgelist_edge_counts[i], handle.get_stream())
                                             : rmm::device_uvector<size_t>(edgelist_edge_counts[i],
                                                                           handle.get_stream());
          cugraph::sequence(
            handle.get_thrust_policy(), positions.begin(), positions.end(), size_t{0});
          thrust::sort_by_key(handle.get_thrust_policy(),
                              minor_ptr,
                              minor_ptr + edgelist_edge_counts[i],
                              thrust::make_zip_iterator(major_ptr, positions.begin()));
          for (auto& property : edgelist_edge_properties) {
            variant_type_dispatch(property, [&](auto& p) {
              using T  = typename std::remove_reference_t<decltype(p)>::value_type;
              auto tmp = large_buffer_type
                           ? large_buffer_manager::allocate_memory_buffer<T>(positions.size(),
                                                                             handle.get_stream())
                           : rmm::device_uvector<T>(positions.size(), handle.get_stream());
              thrust::gather(handle.get_thrust_policy(),
                             positions.begin(),
                             positions.end(),
                             p.begin() + edgelist_partition_offsets[i],
                             tmp.begin());
              thrust::copy(handle.get_thrust_policy(),
                           tmp.begin(),
                           tmp.end(),
                           p.begin() + edgelist_partition_offsets[i]);
            });
          }
        }

        rmm::device_uvector<size_t> d_segment_offsets(d_thresholds.size(), handle.get_stream());
        thrust::lower_bound(handle.get_thrust_policy(),
                            minor_ptr,
                            minor_ptr + edgelist_edge_counts[i],
                            d_thresholds.begin(),
                            d_thresholds.end(),
                            d_segment_offsets.begin(),
                            cuda::std::less<vertex_t>{});
        (*edgelist_intra_partition_segment_offsets)[i][0]     = size_t{0};
        (*edgelist_intra_partition_segment_offsets)[i].back() = edgelist_edge_counts[i];
        raft::update_host((*edgelist_intra_partition_segment_offsets)[i].data() + 1,
                          d_segment_offsets.data(),
                          d_segment_offsets.size(),
                          handle.get_stream());
        handle.sync_stream();
      }

      std::tie(edgelist_srcs, edgelist_dsts) =
        unrenumber_local_int_edges(handle,
                                   graph_view,
                                   std::move(edgelist_srcs),
                                   std::move(edgelist_dsts),
                                   edgelist_partition_offsets,
                                   *renumber_map,
                                   edgelist_intra_partition_segment_offsets);
    } else {
      std::tie(edgelist_srcs, edgelist_dsts) = unrenumber_local_int_edges(
        handle, graph_view, std::move(edgelist_srcs), std::move(edgelist_dsts), *renumber_map);
    }
  }

  return std::make_tuple(
    std::move(edgelist_srcs), std::move(edgelist_dsts), std::move(edgelist_edge_properties));
}

}  // namespace

template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::vector<arithmetic_device_uvector_t>>
decompress_to_edgelist(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  std::vector<edge_arithmetic_property_view_t<edge_t>> const& edge_property_views,
  std::optional<vertex_property_view_t<vertex_t, vertex_t const*>> renumber_map,
  std::optional<large_buffer_type_t> large_buffer_type,
  bool do_expensive_check)
{
  return decompress_to_edgelist_impl(
    handle, graph_view, edge_property_views, renumber_map, large_buffer_type, do_expensive_check);
}

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
  std::optional<large_buffer_type_t> large_buffer_type,
  bool do_expensive_check)
{
  std::vector<edge_arithmetic_property_view_t<edge_t>> edge_property_views{};
  if (edge_weight_view) { edge_property_views.push_back(*edge_weight_view); }
  if (edge_id_view) { edge_property_views.push_back(*edge_id_view); }
  if (edge_type_view) { edge_property_views.push_back(*edge_type_view); }

  auto [srcs, dsts, edge_properties] =
    decompress_to_edgelist<vertex_t, edge_t, store_transposed, multi_gpu>(
      handle,
      graph_view,
      edge_property_views,
      renumber_map ? std::make_optional(vertex_property_view_t<vertex_t, vertex_t const*>(
                       renumber_map->data(), static_cast<vertex_t>(renumber_map->size())))
                   : std::nullopt,
      large_buffer_type,
      do_expensive_check);

  size_t j{0};
  auto weights =
    edge_weight_view
      ? std::make_optional(std::move(std::get<rmm::device_uvector<weight_t>>(edge_properties[j++])))
      : std::nullopt;
  auto ids =
    edge_id_view
      ? std::make_optional(std::move(std::get<rmm::device_uvector<edge_t>>(edge_properties[j++])))
      : std::nullopt;
  auto types = edge_type_view ? std::make_optional(std::move(
                                  std::get<rmm::device_uvector<edge_type_t>>(edge_properties[j++])))
                              : std::nullopt;

  return std::make_tuple(
    std::move(srcs), std::move(dsts), std::move(weights), std::move(ids), std::move(types));
}

}  // namespace cugraph
