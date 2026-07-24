/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "gather_sampled_properties.cuh"
#include "sampling_utils.hpp"
#include "temporal_sampling_utils.cuh"

#include <cugraph/arithmetic_variant_types.hpp>
#include <cugraph/detail/device_comm_wrapper.hpp>
#include <cugraph/edge_property.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/prims/edge_bucket.cuh>
#include <cugraph/prims/extract_transform_if_v_frontier_incoming_outgoing_e.cuh>
#include <cugraph/prims/extract_transform_v_frontier_incoming_outgoing_e.cuh>
#include <cugraph/prims/kv_store.cuh>
#include <cugraph/prims/transform_gather_e.cuh>
#include <cugraph/prims/vertex_frontier.cuh>
#include <cugraph/sampling_functions.hpp>
#include <cugraph/utilities/assert.cuh>
#include <cugraph/utilities/collect_comm.cuh>
#include <cugraph/utilities/mask_utils.cuh>
#include <cugraph/utilities/thrust_wrappers/fill.hpp>
#include <cugraph/utilities/thrust_wrappers/sequence.hpp>
#include <cugraph/utilities/thrust_wrappers/sort.hpp>

#include <raft/util/cudart_utils.hpp>

#include <cuda/std/tuple>
#include <thrust/binary_search.h>
#include <thrust/gather.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>

#include <limits>
#include <optional>
#include <type_traits>
#include <variant>
#include <vector>

namespace cugraph {
namespace detail {

namespace {

struct return_edge_property_t {
  template <typename key_t, typename vertex_t, typename T>
  T __device__
  operator()(key_t, vertex_t, cuda::std::nullopt_t, cuda::std::nullopt_t, T edge_property) const
  {
    return edge_property;
  }
};

struct return_edges_with_label_op {
  template <typename vertex_t, typename label_t>
  cuda::std::tuple<vertex_t, vertex_t, label_t> __device__
  operator()(cuda::std::tuple<vertex_t, label_t> tagged_src,
             vertex_t dst,
             cuda::std::nullopt_t,
             cuda::std::nullopt_t,
             cuda::std::nullopt_t) const
  {
    return cuda::std::make_tuple(cuda::std::get<0>(tagged_src), dst, cuda::std::get<1>(tagged_src));
  }
};

struct return_edges_with_index_and_position_op {
  template <typename vertex_t, typename position_t, typename index_t>
  cuda::std::tuple<vertex_t, vertex_t, index_t, position_t> __device__
  operator()(cuda::std::tuple<vertex_t, position_t> tagged_src,
             vertex_t dst,
             cuda::std::nullopt_t,
             cuda::std::nullopt_t,
             index_t index) const
  {
    return cuda::std::make_tuple(
      cuda::std::get<0>(tagged_src), dst, index, cuda::std::get<1>(tagged_src));
  }

  template <typename vertex_t, typename index_t>
  cuda::std::tuple<vertex_t, vertex_t, index_t> __device__ operator()(
    vertex_t src, vertex_t dst, cuda::std::nullopt_t, cuda::std::nullopt_t, index_t index) const
  {
    return cuda::std::make_tuple(src, dst, index);
  }
};

struct return_edges_op {
  template <typename vertex_t>
  cuda::std::tuple<vertex_t, vertex_t> __device__ operator()(vertex_t src,
                                                             vertex_t dst,
                                                             cuda::std::nullopt_t,
                                                             cuda::std::nullopt_t,
                                                             cuda::std::nullopt_t) const
  {
    return cuda::std::make_tuple(src, dst);
  }
};

template <typename vertex_t, typename edge_t, typename tag_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           rmm::device_uvector<edge_t>,
           std::optional<rmm::device_uvector<tag_t>>>
simple_gather_one_hop_with_multi_edge_indices(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  edge_multi_index_property_view_t<edge_t, vertex_t> edge_multi_index_property_view,
  raft::device_span<vertex_t const> active_majors,
  std::optional<raft::device_span<tag_t const>> active_major_tags,
  bool do_expensive_check)
{
  rmm::device_uvector<vertex_t> majors(0, handle.get_stream());
  rmm::device_uvector<vertex_t> minors(0, handle.get_stream());
  rmm::device_uvector<edge_t> multi_index_position(0, handle.get_stream());
  std::optional<rmm::device_uvector<tag_t>> tags{std::nullopt};

  if (active_major_tags) {
    cugraph::vertex_frontier_t<vertex_t, tag_t, multi_gpu, false> vertex_tag_frontier(handle, 1);
    auto& key_list = vertex_tag_frontier.bucket(0);
    key_list.insert(thrust::make_zip_iterator(active_majors.begin(), active_major_tags->begin()),
                    thrust::make_zip_iterator(active_majors.end(), active_major_tags->end()));

    std::tie(majors, minors, multi_index_position, tags) =
      cugraph::extract_transform_v_frontier_outgoing_e(handle,
                                                       graph_view,
                                                       key_list,
                                                       edge_src_dummy_property_t{}.view(),
                                                       edge_dst_dummy_property_t{}.view(),
                                                       edge_multi_index_property_view,
                                                       return_edges_with_index_and_position_op{},
                                                       do_expensive_check);
  } else {
    cugraph::vertex_frontier_t<vertex_t, void, multi_gpu, false> vertex_tag_frontier(handle, 1);
    auto& key_list = vertex_tag_frontier.bucket(0);
    key_list.insert(active_majors.begin(), active_majors.end());

    std::tie(majors, minors, multi_index_position) =
      cugraph::extract_transform_v_frontier_outgoing_e(handle,
                                                       graph_view,
                                                       key_list,
                                                       edge_src_dummy_property_t{}.view(),
                                                       edge_dst_dummy_property_t{}.view(),
                                                       edge_multi_index_property_view,
                                                       return_edges_with_index_and_position_op{},
                                                       do_expensive_check);
  }

  return std::make_tuple(
    std::move(majors), std::move(minors), std::move(multi_index_position), std::move(tags));
}

template <typename vertex_t, typename edge_t, typename label_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<label_t>>>
simple_gather_one_hop_without_multi_edge_indices(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  raft::device_span<vertex_t const> active_majors,
  std::optional<raft::device_span<label_t const>> active_major_labels,
  bool do_expensive_check)
{
  rmm::device_uvector<vertex_t> majors(0, handle.get_stream());
  rmm::device_uvector<vertex_t> minors(0, handle.get_stream());
  std::optional<rmm::device_uvector<label_t>> labels{std::nullopt};

  if (active_major_labels) {
    cugraph::vertex_frontier_t<vertex_t, label_t, multi_gpu, false> vertex_label_frontier(handle,
                                                                                          1);
    auto& key_list = vertex_label_frontier.bucket(0);
    key_list.insert(thrust::make_zip_iterator(active_majors.begin(), active_major_labels->begin()),
                    thrust::make_zip_iterator(active_majors.end(), active_major_labels->end()));

    std::tie(majors, minors, labels) =
      cugraph::extract_transform_v_frontier_outgoing_e(handle,
                                                       graph_view,
                                                       key_list,
                                                       edge_src_dummy_property_t{}.view(),
                                                       edge_dst_dummy_property_t{}.view(),
                                                       edge_dummy_property_view_t{},
                                                       return_edges_with_label_op{},
                                                       do_expensive_check);
  } else {
    cugraph::vertex_frontier_t<vertex_t, void, multi_gpu, false> vertex_label_frontier(handle, 1);
    auto& key_list = vertex_label_frontier.bucket(0);
    key_list.insert(active_majors.begin(), active_majors.end());

    std::tie(majors, minors) =
      cugraph::extract_transform_v_frontier_outgoing_e(handle,
                                                       graph_view,
                                                       key_list,
                                                       edge_src_dummy_property_t{}.view(),
                                                       edge_dst_dummy_property_t{}.view(),
                                                       edge_dummy_property_view_t{},
                                                       return_edges_op{},
                                                       do_expensive_check);
  }

  return std::make_tuple(std::move(majors), std::move(minors), std::move(labels));
}

template <typename vertex_t, typename edge_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           arithmetic_device_uvector_t,
           std::optional<rmm::device_uvector<int32_t>>>
filter_edge_by_type(raft::handle_t const& handle,
                    graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
                    edge_property_view_t<edge_t, int32_t const*> edge_type_view,
                    rmm::device_uvector<vertex_t>&& majors,
                    rmm::device_uvector<vertex_t>&& minors,
                    arithmetic_device_uvector_t&& edge_indices,
                    std::optional<rmm::device_uvector<int32_t>>&& labels,
                    raft::device_span<bool const> gather_flags)
{
  // Filter by type
  using edge_type_t = int32_t;

  constexpr bool store_transposed = false;

  rmm::device_uvector<edge_type_t> edge_type(majors.size(), handle.get_stream());

  cugraph::edge_bucket_t<vertex_t, edge_t, !store_transposed, multi_gpu, false> edge_list(
    handle, graph_view.is_multigraph());

  edge_list.insert(
    majors.begin(),
    majors.end(),
    minors.begin(),
    std::holds_alternative<rmm::device_uvector<edge_t>>(edge_indices)
      ? std::make_optional(std::get<rmm::device_uvector<edge_t>>(edge_indices).begin())
      : std::nullopt);

  cugraph::transform_gather_e(handle,
                              graph_view,
                              edge_list,
                              edge_src_dummy_property_t{}.view(),
                              edge_dst_dummy_property_t{}.view(),
                              edge_type_view,
                              return_edge_property_t{},
                              edge_type.begin());

  auto [keep_count, marked_entries] = mark_entries(
    edge_type.size(),
    [d_edge_type = edge_type.data(), gather_flags] __device__(auto idx) {
      return gather_flags[d_edge_type[idx]];
    },
    handle.get_stream());

  raft::device_span<uint32_t const> marked_entry_span{marked_entries.data(), marked_entries.size()};

  majors = keep_marked_entries(handle, std::move(majors), marked_entry_span, keep_count);
  minors = keep_marked_entries(handle, std::move(minors), marked_entry_span, keep_count);
  if (std::holds_alternative<rmm::device_uvector<edge_t>>(edge_indices)) {
    edge_indices =
      keep_marked_entries(handle,
                          std::move(std::get<rmm::device_uvector<edge_t>>(edge_indices)),
                          marked_entry_span,
                          keep_count);
  }
  if (labels) {
    *labels = keep_marked_entries(handle, std::move(*labels), marked_entry_span, keep_count);
  }

  return std::make_tuple(
    std::move(majors), std::move(minors), std::move(edge_indices), std::move(labels));
}

}  // namespace

template <typename vertex_t, typename edge_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           arithmetic_device_uvector_t,
           std::optional<rmm::device_uvector<int32_t>>>
gather_one_hop_edgelist(raft::handle_t const& handle,
                        graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
                        size_t number_of_edge_properties,
                        std::optional<edge_property_view_t<edge_t, int32_t const*>> edge_type_view,
                        raft::device_span<vertex_t const> active_majors,
                        std::optional<raft::device_span<int32_t const>> active_major_labels,
                        std::optional<raft::device_span<bool const>> gather_flags,
                        bool do_expensive_check)
{
  rmm::device_uvector<vertex_t> result_majors(0, handle.get_stream());
  rmm::device_uvector<vertex_t> result_minors(0, handle.get_stream());
  arithmetic_device_uvector_t tmp_edge_indices{std::monostate{}};
  std::optional<rmm::device_uvector<int32_t>> result_labels{std::nullopt};

  if (number_of_edge_properties == 0) {
    CUGRAPH_EXPECTS(!edge_type_view,
                    "Can't specify type filtering without edge type as a property");

    // Don't care if graph is multigraph, since we don't need multi-edge indices
    std::tie(result_majors, result_minors, result_labels) =
      simple_gather_one_hop_without_multi_edge_indices(
        handle, graph_view, active_majors, active_major_labels, do_expensive_check);
  } else {
    if (graph_view.is_multigraph()) {
      cugraph::edge_multi_index_property_t<edge_t, vertex_t> multi_index_property(handle,
                                                                                  graph_view);

      std::tie(result_majors, result_minors, tmp_edge_indices, result_labels) =
        simple_gather_one_hop_with_multi_edge_indices(handle,
                                                      graph_view,
                                                      multi_index_property.view(),
                                                      active_majors,
                                                      active_major_labels,
                                                      do_expensive_check);
    } else {
      std::tie(result_majors, result_minors, result_labels) =
        simple_gather_one_hop_without_multi_edge_indices(
          handle, graph_view, active_majors, active_major_labels, do_expensive_check);
    }

    if (edge_type_view) {
      std::tie(result_majors, result_minors, tmp_edge_indices, result_labels) =
        filter_edge_by_type(handle,
                            graph_view,
                            *edge_type_view,
                            std::move(result_majors),
                            std::move(result_minors),
                            std::move(tmp_edge_indices),
                            std::move(result_labels),
                            *gather_flags);
    }
  }

  return std::make_tuple(std::move(result_majors),
                         std::move(result_minors),
                         std::move(tmp_edge_indices),
                         std::move(result_labels));
}

template <typename vertex_t, typename edge_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           arithmetic_device_uvector_t,
           std::optional<rmm::device_uvector<int32_t>>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<int32_t>>>
gather_one_hop_edgelist_to_unvisited_neighbors(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  size_t number_of_edge_properties,
  std::optional<edge_property_view_t<edge_t, int32_t const*>> edge_type_view,
  raft::device_span<vertex_t const> active_majors,
  std::optional<raft::device_span<int32_t const>> active_major_labels,
  std::optional<raft::device_span<bool const>> gather_flags,
  rmm::device_uvector<vertex_t>&& visited_minors,
  std::optional<rmm::device_uvector<int32_t>>&& visited_minor_labels,
  bool do_expensive_check)
{
  auto [result_majors, result_minors, tmp_edge_indices, result_labels] =
    gather_one_hop_edgelist<vertex_t, edge_t, multi_gpu>(handle,
                                                         graph_view,
                                                         number_of_edge_properties,
                                                         edge_type_view,
                                                         active_majors,
                                                         active_major_labels,
                                                         gather_flags,
                                                         do_expensive_check);

  // Remove any edges that lead to a visited vertex
  {
    auto [keep_count, marked_entries] =
      visited_minor_labels
        ? mark_entries(
            result_minors.size(),
            [minors              = result_minors.data(),
             labels              = result_labels->data(),
             visited_minors      = visited_minors.data(),
             visited_labels      = visited_minor_labels->data(),
             visited_minors_size = visited_minors.size()] __device__(auto index) {
              auto iter_begin = thrust::make_zip_iterator(visited_minors, visited_labels);
              return !thrust::binary_search(thrust::seq,
                                            iter_begin,
                                            iter_begin + visited_minors_size,
                                            cuda::std::make_tuple(minors[index], labels[index]));
            },
            handle.get_stream())
        : mark_entries(
            result_minors.size(),
            [minors              = result_minors.data(),
             visited_minors      = visited_minors.data(),
             visited_minors_size = visited_minors.size()] __device__(auto index) {
              return !thrust::binary_search(
                thrust::seq, visited_minors, visited_minors + visited_minors_size, minors[index]);
            },
            handle.get_stream());

    raft::device_span<uint32_t const> marked_entry_span{marked_entries.data(),
                                                        marked_entries.size()};
    result_majors =
      keep_marked_entries(handle, std::move(result_majors), marked_entry_span, keep_count);
    result_minors =
      keep_marked_entries(handle, std::move(result_minors), marked_entry_span, keep_count);
    if (result_labels) {
      *result_labels =
        keep_marked_entries(handle, std::move(*result_labels), marked_entry_span, keep_count);
    }
    if (std::holds_alternative<rmm::device_uvector<edge_t>>(tmp_edge_indices)) {
      tmp_edge_indices =
        keep_marked_entries(handle,
                            std::move(std::get<rmm::device_uvector<edge_t>>(tmp_edge_indices)),
                            marked_entry_span,
                            keep_count);
    }
  }

  std::tie(result_majors,
           result_minors,
           tmp_edge_indices,
           result_labels,
           std::ignore,
           std::ignore,
           std::ignore,
           std::ignore,
           std::ignore) = deduplicate_edges_by_minor(handle,
                                                     graph_view,
                                                     std::move(result_majors),
                                                     std::move(result_minors),
                                                     std::move(tmp_edge_indices),
                                                     arithmetic_device_uvector_t{std::monostate{}},
                                                     std::move(result_labels));

  std::tie(visited_minors, visited_minor_labels) =
    detail::update_dst_visited_vertices_and_labels<vertex_t, edge_t, multi_gpu>(
      handle,
      graph_view,
      std::move(visited_minors),
      std::move(visited_minor_labels),
      raft::device_span<vertex_t const>{result_minors.data(), result_minors.size()},
      result_labels ? std::make_optional(raft::device_span<int32_t const>{result_labels->data(),
                                                                          result_labels->size()})
                    : std::nullopt);

  return std::make_tuple(std::move(result_majors),
                         std::move(result_minors),
                         std::move(tmp_edge_indices),
                         std::move(result_labels),
                         std::move(visited_minors),
                         std::move(visited_minor_labels));
}

template <typename vertex_t, typename edge_t, typename time_stamp_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           arithmetic_device_uvector_t,
           std::optional<rmm::device_uvector<int32_t>>>
temporal_gather_one_hop_edgelist(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  edge_property_view_t<edge_t, time_stamp_t const*> edge_time_view,
  std::optional<edge_property_view_t<edge_t, int32_t const*>> edge_type_view,
  raft::device_span<vertex_t const> active_majors,
  raft::device_span<time_stamp_t const> active_major_times,
  std::optional<raft::device_span<time_stamp_t const>> active_major_window_ends,
  std::optional<raft::device_span<int32_t const>> active_major_labels,
  std::optional<raft::device_span<bool const>> gather_flags,
  temporal_sampling_comparison_t temporal_sampling_comparison,
  bool do_expensive_check)
{
  constexpr bool store_transposed = false;

  rmm::device_uvector<vertex_t> result_majors(0, handle.get_stream());
  rmm::device_uvector<vertex_t> result_minors(0, handle.get_stream());
  std::optional<rmm::device_uvector<int32_t>> result_labels{std::nullopt};

  using label_t = int32_t;

  // Sorted per-source side tables keyed by (major, label).  Used when labels are present.
  std::optional<rmm::device_uvector<vertex_t>> labeled_side_majors{std::nullopt};
  std::optional<rmm::device_uvector<label_t>> labeled_side_labels{std::nullopt};
  std::optional<rmm::device_uvector<time_stamp_t>> labeled_side_times{std::nullopt};
  std::optional<rmm::device_uvector<time_stamp_t>> labeled_side_window_ends{std::nullopt};
  std::optional<rmm::device_uvector<label_t>> edge_src_labels{std::nullopt};

  // Only used if active_major_labels is not set
  std::optional<rmm::device_uvector<time_stamp_t>> tmp_times{std::nullopt};
  std::optional<rmm::device_uvector<time_stamp_t>> tmp_window_ends{std::nullopt};

  // Only used if graph_view is a multi_graph
  arithmetic_device_uvector_t tmp_edge_indices{std::monostate{}};

  if (active_major_labels) {
    // Key the per-source (time, window_end) by (major, label).  Under disjoint sampling that key is
    // unique in the frontier; on MG, allgather across minor_comm and keep the time extremum
    // required by the temporal mode so gather filtering matches the sample path.
    rmm::device_uvector<vertex_t> kv_majors(active_majors.size(), handle.get_stream());
    rmm::device_uvector<label_t> kv_labels(active_major_labels->size(), handle.get_stream());
    rmm::device_uvector<time_stamp_t> kv_times(active_major_times.size(), handle.get_stream());
    rmm::device_uvector<time_stamp_t> kv_window_ends(active_majors.size(), handle.get_stream());

    raft::copy(kv_majors.data(), active_majors.data(), active_majors.size(), handle.get_stream());
    raft::copy(kv_labels.data(),
               active_major_labels->data(),
               active_major_labels->size(),
               handle.get_stream());
    raft::copy(
      kv_times.data(), active_major_times.data(), active_major_times.size(), handle.get_stream());
    if (active_major_window_ends) {
      raft::copy(kv_window_ends.data(),
                 active_major_window_ends->data(),
                 active_major_window_ends->size(),
                 handle.get_stream());
    } else {
      cugraph::fill(handle.get_thrust_policy(),
                    kv_window_ends.begin(),
                    kv_window_ends.end(),
                    unbounded_temporal_window_end<time_stamp_t>(temporal_sampling_comparison));
    }

    if constexpr (multi_gpu) {
      auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
      kv_majors        = device_allgatherv(
        handle, minor_comm, raft::device_span<vertex_t const>{kv_majors.data(), kv_majors.size()});
      kv_labels = device_allgatherv(
        handle, minor_comm, raft::device_span<label_t const>{kv_labels.data(), kv_labels.size()});
      kv_times =
        device_allgatherv(handle,
                          minor_comm,
                          raft::device_span<time_stamp_t const>{kv_times.data(), kv_times.size()});
      kv_window_ends = device_allgatherv(
        handle,
        minor_comm,
        raft::device_span<time_stamp_t const>{kv_window_ends.data(), kv_window_ends.size()});
    }

    cugraph::sort(handle.get_thrust_policy(),
                  thrust::make_zip_iterator(
                    kv_majors.begin(), kv_labels.begin(), kv_times.begin(), kv_window_ends.begin()),
                  thrust::make_zip_iterator(
                    kv_majors.end(), kv_labels.end(), kv_times.end(), kv_window_ends.end()));

    if constexpr (multi_gpu) {
      bool const increasing =
        temporal_sampling_comparison == temporal_sampling_comparison_t::MONOTONICALLY_INCREASING ||
        temporal_sampling_comparison == temporal_sampling_comparison_t::STRICTLY_INCREASING;
      auto const n = kv_majors.size();
      rmm::device_uvector<vertex_t> out_majors(n, handle.get_stream());
      rmm::device_uvector<label_t> out_labels(n, handle.get_stream());
      rmm::device_uvector<time_stamp_t> out_times(n, handle.get_stream());
      rmm::device_uvector<time_stamp_t> out_window_ends(n, handle.get_stream());

      size_t new_size{};
      if (increasing) {
        auto ends = thrust::reduce_by_key(
          handle.get_thrust_policy(),
          thrust::make_zip_iterator(kv_majors.begin(), kv_labels.begin()),
          thrust::make_zip_iterator(kv_majors.end(), kv_labels.end()),
          thrust::make_zip_iterator(kv_times.begin(), kv_window_ends.begin()),
          thrust::make_zip_iterator(out_majors.begin(), out_labels.begin()),
          thrust::make_zip_iterator(out_times.begin(), out_window_ends.begin()),
          thrust::equal_to<cuda::std::tuple<vertex_t, label_t>>{},
          [] __device__(cuda::std::tuple<time_stamp_t, time_stamp_t> a,
                        cuda::std::tuple<time_stamp_t, time_stamp_t> b) {
            return cuda::std::get<0>(a) >= cuda::std::get<0>(b) ? a : b;
          });
        new_size = static_cast<size_t>(cuda::std::distance(
          thrust::make_zip_iterator(out_majors.begin(), out_labels.begin()), ends.first));
      } else {
        auto ends = thrust::reduce_by_key(
          handle.get_thrust_policy(),
          thrust::make_zip_iterator(kv_majors.begin(), kv_labels.begin()),
          thrust::make_zip_iterator(kv_majors.end(), kv_labels.end()),
          thrust::make_zip_iterator(kv_times.begin(), kv_window_ends.begin()),
          thrust::make_zip_iterator(out_majors.begin(), out_labels.begin()),
          thrust::make_zip_iterator(out_times.begin(), out_window_ends.begin()),
          thrust::equal_to<cuda::std::tuple<vertex_t, label_t>>{},
          [] __device__(cuda::std::tuple<time_stamp_t, time_stamp_t> a,
                        cuda::std::tuple<time_stamp_t, time_stamp_t> b) {
            return cuda::std::get<0>(a) <= cuda::std::get<0>(b) ? a : b;
          });
        new_size = static_cast<size_t>(cuda::std::distance(
          thrust::make_zip_iterator(out_majors.begin(), out_labels.begin()), ends.first));
      }

      kv_majors      = std::move(out_majors);
      kv_labels      = std::move(out_labels);
      kv_times       = std::move(out_times);
      kv_window_ends = std::move(out_window_ends);
      kv_majors.resize(new_size, handle.get_stream());
      kv_labels.resize(new_size, handle.get_stream());
      kv_times.resize(new_size, handle.get_stream());
      kv_window_ends.resize(new_size, handle.get_stream());
    }

    labeled_side_majors      = std::move(kv_majors);
    labeled_side_labels      = std::move(kv_labels);
    labeled_side_times       = std::move(kv_times);
    labeled_side_window_ends = std::move(kv_window_ends);

    if (graph_view.is_multigraph()) {
      cugraph::edge_multi_index_property_t<edge_t, vertex_t> multi_index_property(handle,
                                                                                  graph_view);

      std::tie(result_majors, result_minors, tmp_edge_indices, edge_src_labels) =
        simple_gather_one_hop_with_multi_edge_indices(
          handle,
          graph_view,
          multi_index_property.view(),
          active_majors,
          std::make_optional(raft::device_span<label_t const>{active_major_labels->data(),
                                                              active_major_labels->size()}),
          do_expensive_check);
    } else {
      std::tie(result_majors, result_minors, edge_src_labels) =
        simple_gather_one_hop_without_multi_edge_indices(
          handle,
          graph_view,
          active_majors,
          std::make_optional(raft::device_span<label_t const>{active_major_labels->data(),
                                                              active_major_labels->size()}),
          do_expensive_check);
    }
  } else {
    // Can use time directly
    if (active_major_window_ends) {
      tmp_window_ends = rmm::device_uvector<time_stamp_t>(0, handle.get_stream());
    }

    if (graph_view.is_multigraph()) {
      cugraph::edge_multi_index_property_t<edge_t, vertex_t> multi_index_property(handle,
                                                                                  graph_view);

      //. Note that this is irrelevant to gather_one_hop_edgelist and only relevant to
      //. temporal_gather_one_hop_edgelist
      //
      std::tie(result_majors, result_minors, tmp_edge_indices, tmp_times) =
        simple_gather_one_hop_with_multi_edge_indices(handle,
                                                      graph_view,
                                                      multi_index_property.view(),
                                                      active_majors,
                                                      std::make_optional(active_major_times),
                                                      do_expensive_check);
      if (active_major_window_ends) {
        rmm::device_uvector<vertex_t> unused_majors(0, handle.get_stream());
        rmm::device_uvector<vertex_t> unused_minors(0, handle.get_stream());
        rmm::device_uvector<edge_t> unused_edge_indices(0, handle.get_stream());
        std::tie(unused_majors, unused_minors, unused_edge_indices, tmp_window_ends) =
          simple_gather_one_hop_with_multi_edge_indices(handle,
                                                        graph_view,
                                                        multi_index_property.view(),
                                                        active_majors,
                                                        active_major_window_ends,
                                                        do_expensive_check);
      }
    } else {
      std::tie(result_majors, result_minors, tmp_times) =
        simple_gather_one_hop_without_multi_edge_indices(handle,
                                                         graph_view,
                                                         active_majors,
                                                         std::make_optional(active_major_times),
                                                         do_expensive_check);
      if (active_major_window_ends) {
        rmm::device_uvector<vertex_t> unused_majors(0, handle.get_stream());
        rmm::device_uvector<vertex_t> unused_minors(0, handle.get_stream());
        std::tie(unused_majors, unused_minors, tmp_window_ends) =
          simple_gather_one_hop_without_multi_edge_indices(
            handle, graph_view, active_majors, active_major_window_ends, do_expensive_check);
      }
    }
  }

  if (edge_type_view) {
    std::tie(result_majors, result_minors, tmp_edge_indices, result_labels) =
      filter_edge_by_type(handle,
                          graph_view,
                          *edge_type_view,
                          std::move(result_majors),
                          std::move(result_minors),
                          std::move(tmp_edge_indices),
                          std::move(result_labels),
                          *gather_flags);
  }

  // Filter by time
  {
    rmm::device_uvector<time_stamp_t> edge_times(result_majors.size(), handle.get_stream());

    cugraph::edge_bucket_t<vertex_t, edge_t, !store_transposed, multi_gpu, false> edge_list(
      handle, graph_view.is_multigraph());

    edge_list.insert(
      result_majors.begin(),
      result_majors.end(),
      result_minors.begin(),
      std::holds_alternative<rmm::device_uvector<edge_t>>(tmp_edge_indices)
        ? std::make_optional(std::get<rmm::device_uvector<edge_t>>(tmp_edge_indices).begin())
        : std::nullopt);

    cugraph::transform_gather_e(handle,
                                graph_view,
                                edge_list,
                                edge_src_dummy_property_t{}.view(),
                                edge_dst_dummy_property_t{}.view(),
                                edge_time_view,
                                return_edge_property_t{},
                                edge_times.begin());

    auto [keep_count, marked_entries] =
      edge_src_labels
        ? mark_entries(
            edge_times.size(),
            [temporal_sampling_comparison,
             d_tmp            = edge_times.data(),
             d_srcs           = result_majors.data(),
             d_src_labels     = edge_src_labels->data(),
             side_majors      = raft::device_span<vertex_t const>{labeled_side_majors->data(),
                                                                  labeled_side_majors->size()},
             side_labels      = raft::device_span<label_t const>{labeled_side_labels->data(),
                                                                 labeled_side_labels->size()},
             side_times       = labeled_side_times->data(),
             side_window_ends = labeled_side_window_ends->data()] __device__(auto index) {
              auto const edge_time = d_tmp[index];
              size_t side_idx{};
              if (!try_find_temporal_key_index(
                    side_majors, side_labels, d_srcs[index], d_src_labels[index], side_idx)) {
                return false;
              }
              return passes_temporal_filter(temporal_sampling_comparison,
                                            side_times[side_idx],
                                            side_window_ends[side_idx],
                                            edge_time);
            },
            handle.get_stream())
        : mark_entries(
            edge_times.size(),
            [temporal_sampling_comparison,
             d_tmp      = edge_times.data(),
             d_tmp_time = tmp_times->data(),
             d_tmp_window_end =
               tmp_window_ends ? tmp_window_ends->data() : nullptr] __device__(auto index) {
              auto const edge_time = d_tmp[index];
              auto const key_time  = d_tmp_time[index];
              auto const window_end =
                d_tmp_window_end
                  ? d_tmp_window_end[index]
                  : unbounded_temporal_window_end<time_stamp_t>(temporal_sampling_comparison);
              return passes_temporal_filter(
                temporal_sampling_comparison, key_time, window_end, edge_time);
            },
            handle.get_stream());

    raft::device_span<uint32_t const> marked_entry_span{marked_entries.data(),
                                                        marked_entries.size()};

    result_majors =
      keep_marked_entries(handle, std::move(result_majors), marked_entry_span, keep_count);
    result_minors =
      keep_marked_entries(handle, std::move(result_minors), marked_entry_span, keep_count);
    if (std::holds_alternative<rmm::device_uvector<edge_t>>(tmp_edge_indices)) {
      tmp_edge_indices =
        keep_marked_entries(handle,
                            std::move(std::get<rmm::device_uvector<edge_t>>(tmp_edge_indices)),
                            marked_entry_span,
                            keep_count);
    }
    if (tmp_times) {
      *tmp_times =
        keep_marked_entries(handle, std::move(*tmp_times), marked_entry_span, keep_count);
    }
    if (edge_src_labels) {
      *edge_src_labels =
        keep_marked_entries(handle, std::move(*edge_src_labels), marked_entry_span, keep_count);
    }
    if (tmp_window_ends) {
      *tmp_window_ends =
        keep_marked_entries(handle, std::move(*tmp_window_ends), marked_entry_span, keep_count);
    }

    if (edge_src_labels) { result_labels = std::move(*edge_src_labels); }
  }

  // Return endpoints and graph edge indices.  Output edge properties are gathered by the caller
  // after visited filtering / minor deduplication so MG shuffles carry real edge indices rather
  // than rank-local property-row numbers.
  return std::make_tuple(std::move(result_majors),
                         std::move(result_minors),
                         std::move(tmp_edge_indices),
                         std::move(result_labels));
}

// Temporal gather-one-hop to unvisited neighbors.  Gathers all temporally-valid one-hop edges, then
// (1) drops edges whose destination is already visited, (2) deduplicates by (minor[, label])
// keeping one edge per destination while carrying graph edge indices through the MG shuffle, (3)
// gathers the requested output properties from the surviving indices, and (4) updates the visited
// sets.  Matches the disjoint semantics of gather_one_hop_edgelist_to_unvisited_neighbors.
template <typename vertex_t, typename edge_t, typename time_stamp_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::vector<arithmetic_device_uvector_t>,
           std::optional<rmm::device_uvector<int32_t>>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<int32_t>>>
temporal_gather_one_hop_edgelist_to_unvisited_neighbors(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  raft::host_span<edge_arithmetic_property_view_t<edge_t>> edge_property_views,
  edge_property_view_t<edge_t, time_stamp_t const*> edge_time_view,
  std::optional<edge_property_view_t<edge_t, int32_t const*>> edge_type_view,
  raft::device_span<vertex_t const> active_majors,
  raft::device_span<time_stamp_t const> active_major_times,
  std::optional<raft::device_span<time_stamp_t const>> active_major_window_ends,
  std::optional<raft::device_span<int32_t const>> active_major_labels,
  std::optional<raft::device_span<bool const>> gather_flags,
  rmm::device_uvector<vertex_t>&& visited_minors,
  std::optional<rmm::device_uvector<int32_t>>&& visited_minor_labels,
  temporal_sampling_comparison_t temporal_sampling_comparison,
  bool do_expensive_check)
{
  CUGRAPH_EXPECTS(active_major_labels.has_value() == visited_minor_labels.has_value(),
                  "Active major labels and visited vertex labels must both be specified or both "
                  "be unspecified");
  CUGRAPH_EXPECTS(edge_property_views.size() > 0,
                  "Temporal requires at least one edge property - the time");

  auto [result_majors, result_minors, tmp_edge_indices, result_labels] =
    temporal_gather_one_hop_edgelist<vertex_t, edge_t, time_stamp_t, multi_gpu>(
      handle,
      graph_view,
      edge_time_view,
      edge_type_view,
      active_majors,
      active_major_times,
      active_major_window_ends,
      active_major_labels,
      gather_flags,
      temporal_sampling_comparison,
      do_expensive_check);

  // Drop edges whose destination has already been visited.  Keep graph edge indices aligned so
  // subsequent MG deduplication and property gather remain valid.
  {
    auto [keep_count, marked_entries] =
      visited_minor_labels
        ? mark_entries(
            result_minors.size(),
            [minors              = result_minors.data(),
             labels              = result_labels->data(),
             visited_minors      = visited_minors.data(),
             visited_labels      = visited_minor_labels->data(),
             visited_minors_size = visited_minors.size()] __device__(auto index) {
              auto iter_begin = thrust::make_zip_iterator(visited_minors, visited_labels);
              return !thrust::binary_search(thrust::seq,
                                            iter_begin,
                                            iter_begin + visited_minors_size,
                                            cuda::std::make_tuple(minors[index], labels[index]));
            },
            handle.get_stream())
        : mark_entries(
            result_minors.size(),
            [minors              = result_minors.data(),
             visited_minors      = visited_minors.data(),
             visited_minors_size = visited_minors.size()] __device__(auto index) {
              return !thrust::binary_search(
                thrust::seq, visited_minors, visited_minors + visited_minors_size, minors[index]);
            },
            handle.get_stream());

    raft::device_span<uint32_t const> marked_entry_span{marked_entries.data(),
                                                        marked_entries.size()};
    result_majors =
      keep_marked_entries(handle, std::move(result_majors), marked_entry_span, keep_count);
    result_minors =
      keep_marked_entries(handle, std::move(result_minors), marked_entry_span, keep_count);
    if (result_labels) {
      *result_labels =
        keep_marked_entries(handle, std::move(*result_labels), marked_entry_span, keep_count);
    }
    if (std::holds_alternative<rmm::device_uvector<edge_t>>(tmp_edge_indices)) {
      tmp_edge_indices =
        keep_marked_entries(handle,
                            std::move(std::get<rmm::device_uvector<edge_t>>(tmp_edge_indices)),
                            marked_entry_span,
                            keep_count);
    }
  }

  // Deduplicate by (minor[, label]) while shuffling the real graph edge indices with the endpoints.
  // Gathering properties before this step is incorrect on MG: rank-local property-row numbers are
  // not meaningful after edges move between GPUs.
  std::tie(result_majors,
           result_minors,
           tmp_edge_indices,
           result_labels,
           std::ignore,
           std::ignore,
           std::ignore,
           std::ignore,
           std::ignore) = deduplicate_edges_by_minor(handle,
                                                     graph_view,
                                                     std::move(result_majors),
                                                     std::move(result_minors),
                                                     std::move(tmp_edge_indices),
                                                     arithmetic_device_uvector_t{std::monostate{}},
                                                     std::move(result_labels));

  std::vector<arithmetic_device_uvector_t> result_properties{};
  std::tie(result_majors, result_minors, result_properties) =
    gather_sampled_properties(handle,
                              graph_view,
                              std::move(result_majors),
                              std::move(result_minors),
                              std::move(tmp_edge_indices),
                              raft::host_span<edge_arithmetic_property_view_t<edge_t>>{
                                edge_property_views.data(), edge_property_views.size()});

  std::tie(visited_minors, visited_minor_labels) =
    detail::update_dst_visited_vertices_and_labels<vertex_t, edge_t, multi_gpu>(
      handle,
      graph_view,
      std::move(visited_minors),
      std::move(visited_minor_labels),
      raft::device_span<vertex_t const>{result_minors.data(), result_minors.size()},
      result_labels ? std::make_optional(raft::device_span<int32_t const>{result_labels->data(),
                                                                          result_labels->size()})
                    : std::nullopt);

  return std::make_tuple(std::move(result_majors),
                         std::move(result_minors),
                         std::move(result_properties),
                         std::move(result_labels),
                         std::move(visited_minors),
                         std::move(visited_minor_labels));
}

}  // namespace detail
}  // namespace cugraph
