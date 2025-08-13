/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "gather_sampled_properties.cuh"
#include "prims/edge_bucket.cuh"
#include "prims/extract_transform_if_v_frontier_outgoing_e.cuh"
#include "prims/extract_transform_v_frontier_outgoing_e.cuh"
#include "prims/kv_store.cuh"
#include "prims/transform_gather_e.cuh"
#include "prims/vertex_frontier.cuh"
#include "utilities/collect_comm.cuh"

#include <cugraph/arithmetic_variant_types.hpp>
#include <cugraph/edge_property.hpp>
#include <cugraph/utilities/mask_utils.cuh>

#include <raft/util/cudart_utils.hpp>

#include <variant>

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
  rmm::device_uvector<vertex_t> srcs(0, handle.get_stream());
  rmm::device_uvector<vertex_t> dsts(0, handle.get_stream());
  rmm::device_uvector<edge_t> multi_index_position(0, handle.get_stream());
  std::optional<rmm::device_uvector<tag_t>> tags{std::nullopt};

  if (active_major_tags) {
    cugraph::vertex_frontier_t<vertex_t, tag_t, multi_gpu, false> vertex_tag_frontier(handle, 1);
    auto& key_list = vertex_tag_frontier.bucket(0);
    key_list.insert(thrust::make_zip_iterator(active_majors.begin(), active_major_tags->begin()),
                    thrust::make_zip_iterator(active_majors.end(), active_major_tags->end()));

    std::tie(srcs, dsts, multi_index_position, tags) =
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

    std::tie(srcs, dsts, multi_index_position) =
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
    std::move(srcs), std::move(dsts), std::move(multi_index_position), std::move(tags));
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
  rmm::device_uvector<vertex_t> srcs(0, handle.get_stream());
  rmm::device_uvector<vertex_t> dsts(0, handle.get_stream());
  std::optional<rmm::device_uvector<label_t>> labels{std::nullopt};

  if (active_major_labels) {
    cugraph::vertex_frontier_t<vertex_t, label_t, multi_gpu, false> vertex_label_frontier(handle,
                                                                                          1);
    auto& key_list = vertex_label_frontier.bucket(0);
    key_list.insert(thrust::make_zip_iterator(active_majors.begin(), active_major_labels->begin()),
                    thrust::make_zip_iterator(active_majors.end(), active_major_labels->end()));

    std::tie(srcs, dsts, labels) =
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

    std::tie(srcs, dsts) =
      cugraph::extract_transform_v_frontier_outgoing_e(handle,
                                                       graph_view,
                                                       key_list,
                                                       edge_src_dummy_property_t{}.view(),
                                                       edge_dst_dummy_property_t{}.view(),
                                                       edge_dummy_property_view_t{},
                                                       return_edges_op{},
                                                       do_expensive_check);
  }

  return std::make_tuple(std::move(srcs), std::move(dsts), std::move(labels));
}

template <typename vertex_t, typename edge_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           arithmetic_device_uvector_t,
           std::optional<rmm::device_uvector<int32_t>>>
filter_edge_by_type(raft::handle_t const& handle,
                    graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
                    edge_property_view_t<edge_t, int32_t const*> edge_type_view,
                    rmm::device_uvector<vertex_t>&& srcs,
                    rmm::device_uvector<vertex_t>&& dsts,
                    arithmetic_device_uvector_t&& edge_indices,
                    std::optional<rmm::device_uvector<int32_t>>&& labels,
                    raft::device_span<uint8_t const> gather_flags)
{
  // Filter by type
  using edge_type_t = int32_t;

  constexpr bool store_transposed = false;

  rmm::device_uvector<edge_type_t> edge_type(srcs.size(), handle.get_stream());

  cugraph::edge_bucket_t<vertex_t, edge_t, !store_transposed, multi_gpu, false> edge_list(
    handle, graph_view.is_multigraph());

  edge_list.insert(
    srcs.begin(),
    srcs.end(),
    dsts.begin(),
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

  auto [keep_count, marked_entries] = detail::mark_entries(
    handle, edge_type.size(), [d_edge_type = edge_type.data(), gather_flags] __device__(auto idx) {
      return (gather_flags[d_edge_type[idx]] == static_cast<uint8_t>(true));
    });

  raft::device_span<uint32_t const> marked_entry_span{marked_entries.data(), marked_entries.size()};

  srcs = detail::keep_marked_entries(handle, std::move(srcs), marked_entry_span, keep_count);
  dsts = detail::keep_marked_entries(handle, std::move(dsts), marked_entry_span, keep_count);
  if (std::holds_alternative<rmm::device_uvector<edge_t>>(edge_indices)) {
    edge_indices =
      detail::keep_marked_entries(handle,
                                  std::move(std::get<rmm::device_uvector<edge_t>>(edge_indices)),
                                  marked_entry_span,
                                  keep_count);
  }
  if (labels) {
    *labels =
      detail::keep_marked_entries(handle, std::move(*labels), marked_entry_span, keep_count);
  }

  return std::make_tuple(
    std::move(srcs), std::move(dsts), std::move(edge_indices), std::move(labels));
}

}  // namespace

template <typename vertex_t, typename edge_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::vector<arithmetic_device_uvector_t>,
           std::optional<rmm::device_uvector<int32_t>>>
gather_one_hop_edgelist(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  raft::host_span<edge_arithmetic_property_view_t<edge_t>> edge_property_views,
  std::optional<edge_property_view_t<edge_t, int32_t const*>> edge_type_view,
  raft::device_span<vertex_t const> active_majors,
  std::optional<raft::device_span<int32_t const>> active_major_labels,
  std::optional<raft::device_span<uint8_t const>> gather_flags,
  bool do_expensive_check)
{
  rmm::device_uvector<vertex_t> result_srcs(0, handle.get_stream());
  rmm::device_uvector<vertex_t> result_dsts(0, handle.get_stream());
  std::vector<arithmetic_device_uvector_t> result_properties{};
  std::optional<rmm::device_uvector<int32_t>> result_labels{std::nullopt};

  if (edge_property_views.size() == 0) {
    CUGRAPH_EXPECTS(!edge_type_view,
                    "Can't specify type filtering without edge type as a property");

    // Don't care if graph is multigraph, since we don't need to transform_gather_e any properties
    std::tie(result_srcs, result_dsts, result_labels) =
      simple_gather_one_hop_without_multi_edge_indices(
        handle, graph_view, active_majors, active_major_labels, do_expensive_check);
  } else {
    arithmetic_device_uvector_t tmp_edge_indices{std::monostate{}};

    if (graph_view.is_multigraph()) {
      cugraph::edge_multi_index_property_t<edge_t, vertex_t> multi_index_property(handle,
                                                                                  graph_view);

      std::tie(result_srcs, result_dsts, tmp_edge_indices, result_labels) =
        simple_gather_one_hop_with_multi_edge_indices(handle,
                                                      graph_view,
                                                      multi_index_property.view(),
                                                      active_majors,
                                                      active_major_labels,
                                                      do_expensive_check);
    } else {
      std::tie(result_srcs, result_dsts, result_labels) =
        simple_gather_one_hop_without_multi_edge_indices(
          handle, graph_view, active_majors, active_major_labels, do_expensive_check);
    }

    if (gather_flags) {
      std::tie(result_srcs, result_dsts, tmp_edge_indices, result_labels) =
        filter_edge_by_type(handle,
                            graph_view,
                            *edge_type_view,
                            std::move(result_srcs),
                            std::move(result_dsts),
                            std::move(tmp_edge_indices),
                            std::move(result_labels),
                            *gather_flags);
    }

    std::tie(result_srcs, result_dsts, result_properties) =
      gather_sampled_properties(handle,
                                graph_view,
                                std::move(result_srcs),
                                std::move(result_dsts),
                                std::move(tmp_edge_indices),
                                raft::host_span<edge_arithmetic_property_view_t<edge_t>>{
                                  edge_property_views.data(), edge_property_views.size()});
  }

  return std::make_tuple(std::move(result_srcs),
                         std::move(result_dsts),
                         std::move(result_properties),
                         std::move(result_labels));
}

template <typename vertex_t, typename edge_t, typename edge_time_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::vector<arithmetic_device_uvector_t>,
           std::optional<rmm::device_uvector<int32_t>>>
temporal_gather_one_hop_edgelist(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  raft::host_span<edge_arithmetic_property_view_t<edge_t>> edge_property_views,
  edge_property_view_t<edge_t, edge_time_t const*> edge_time_view,
  std::optional<edge_property_view_t<edge_t, int32_t const*>> edge_type_view,
  raft::device_span<vertex_t const> active_majors,
  raft::device_span<edge_time_t const> active_major_times,
  std::optional<raft::device_span<int32_t const>> active_major_labels,
  std::optional<raft::device_span<uint8_t const>> gather_flags,
  bool do_expensive_check)
{
  constexpr bool store_transposed = false;

  rmm::device_uvector<vertex_t> result_srcs(0, handle.get_stream());
  rmm::device_uvector<vertex_t> result_dsts(0, handle.get_stream());
  std::vector<arithmetic_device_uvector_t> result_properties{};
  std::optional<rmm::device_uvector<int32_t>> result_labels{std::nullopt};

  // FIXME:  Modify sampling to take raft::host_span of arithmetic edge property views directly
  // rather than creating and unpacking them
  CUGRAPH_EXPECTS(edge_property_views.size() > 0,
                  "Temporal requires at least one edge property - the time");
  using label_t = int32_t;

  // Only used if active_major_labels is set
  kv_store_t<size_t, thrust::tuple<edge_time_t, label_t>, true> kv_store{handle.get_stream()};
  std::optional<rmm::device_uvector<size_t>> tmp_positions{std::nullopt};

  // Only used if active_major_labels is not set
  std::optional<rmm::device_uvector<edge_time_t>> tmp_times{std::nullopt};

  // Only used if graph_view is a multi_graph
  arithmetic_device_uvector_t tmp_edge_indices{std::monostate{}};

  if (active_major_labels) {
    //
    // Each active_majors entry has a time and a label.  In order to do time filtering, I need
    // to access the correct time.  In order to apply labels, I need to get the exact label.
    // This code creates a kvstore mapping a unique id to each label/time instance.  Then I can
    // use the position index into the kvstore to look up the label and the time at the
    // appropriate point in the code.
    //
    rmm::device_uvector<size_t> vertex_label_time_positions(active_majors.size(),
                                                            handle.get_stream());

    size_t starting_pos{0};
    if (multi_gpu) {
      auto sizes = cugraph::host_scalar_allgather(
        handle.get_comms(), active_majors.size(), handle.get_stream());
      std::exclusive_scan(sizes.begin(), sizes.end(), sizes.begin(), size_t{0});
      starting_pos = sizes[handle.get_comms().get_rank()];
    }

    thrust::sequence(handle.get_thrust_policy(),
                     vertex_label_time_positions.begin(),
                     vertex_label_time_positions.end(),
                     starting_pos);

    if (multi_gpu) {
      auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());

      auto all_minor_keys =
        device_allgatherv(handle,
                          minor_comm,
                          raft::device_span<size_t const>{vertex_label_time_positions.data(),
                                                          vertex_label_time_positions.size()});
      auto all_minor_times = device_allgatherv(
        handle,
        minor_comm,
        raft::device_span<edge_time_t const>{active_major_times.data(), active_major_times.size()});
      auto all_minor_labels = device_allgatherv(
        handle,
        minor_comm,
        raft::device_span<label_t const>{active_major_labels->data(), active_major_labels->size()});

      kv_store = kv_store_t<size_t, thrust::tuple<edge_time_t, label_t>, true>(
        all_minor_keys.begin(),
        all_minor_keys.end(),
        thrust::make_zip_iterator(all_minor_times.begin(), all_minor_labels.begin()),
        thrust::make_tuple(edge_time_t{-1}, label_t{-1}),
        true,
        handle.get_stream());

    } else {
      kv_store = kv_store_t<size_t, thrust::tuple<edge_time_t, label_t>, true>(
        vertex_label_time_positions.begin(),
        vertex_label_time_positions.end(),
        thrust::make_zip_iterator(active_major_times.begin(),
                                  active_major_labels->begin()),  // multi_gpu is different
        thrust::make_tuple(edge_time_t{-1}, label_t{-1}),
        true,
        handle.get_stream());
    }

    if (graph_view.is_multigraph()) {
      cugraph::edge_multi_index_property_t<edge_t, vertex_t> multi_index_property(handle,
                                                                                  graph_view);

      //. Note that this is irrelevant to gather_one_hop_edgelist and only relevant to
      //. temporal_gather_one_hop_edgelist
      //
      std::tie(result_srcs, result_dsts, tmp_edge_indices, tmp_positions) =
        simple_gather_one_hop_with_multi_edge_indices(
          handle,
          graph_view,
          multi_index_property.view(),
          active_majors,
          std::make_optional(raft::device_span<size_t const>{vertex_label_time_positions.data(),
                                                             vertex_label_time_positions.size()}),
          do_expensive_check);
    } else {
      std::tie(result_srcs, result_dsts, tmp_positions) =
        simple_gather_one_hop_without_multi_edge_indices(
          handle,
          graph_view,
          active_majors,
          std::make_optional(raft::device_span<size_t const>{vertex_label_time_positions.data(),
                                                             vertex_label_time_positions.size()}),
          do_expensive_check);
    }
  } else {
    // Can use time directly
    if (graph_view.is_multigraph()) {
      cugraph::edge_multi_index_property_t<edge_t, vertex_t> multi_index_property(handle,
                                                                                  graph_view);

      //. Note that this is irrelevant to gather_one_hop_edgelist and only relevant to
      //. temporal_gather_one_hop_edgelist
      //
      std::tie(result_srcs, result_dsts, tmp_edge_indices, tmp_times) =
        simple_gather_one_hop_with_multi_edge_indices(handle,
                                                      graph_view,
                                                      multi_index_property.view(),
                                                      active_majors,
                                                      std::make_optional(active_major_times),
                                                      do_expensive_check);
    } else {
      std::tie(result_srcs, result_dsts, tmp_times) =
        simple_gather_one_hop_without_multi_edge_indices(handle,
                                                         graph_view,
                                                         active_majors,
                                                         std::make_optional(active_major_times),
                                                         do_expensive_check);
    }
  }

  if (gather_flags) {
    std::tie(result_srcs, result_dsts, tmp_edge_indices, result_labels) =
      filter_edge_by_type(handle,
                          graph_view,
                          *edge_type_view,
                          std::move(result_srcs),
                          std::move(result_dsts),
                          std::move(tmp_edge_indices),
                          std::move(result_labels),
                          *gather_flags);
  }

  // Filter by time
  {
    rmm::device_uvector<edge_time_t> edge_times(result_srcs.size(), handle.get_stream());

    cugraph::edge_bucket_t<vertex_t, edge_t, !store_transposed, multi_gpu, false> edge_list(
      handle, graph_view.is_multigraph());

    edge_list.insert(
      result_srcs.begin(),
      result_srcs.end(),
      result_dsts.begin(),
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
      tmp_positions
        ? detail::mark_entries(handle,
                               edge_times.size(),
                               [d_tmp           = edge_times.data(),
                                d_tmp_positions = tmp_positions->data(),
                                kv_store_view =
                                  kv_binary_search_store_device_view_t<decltype(kv_store.view())>{
                                    kv_store.view()}] __device__(auto index) {
                                 auto edge_time = d_tmp[index];
                                 auto key_time =
                                   cuda::std::get<0>(kv_store_view.find(d_tmp_positions[index]));
                                 return (edge_time > key_time);
                               })
        : detail::mark_entries(
            handle,
            edge_times.size(),
            [d_tmp = edge_times.data(), d_tmp_time = tmp_times->data()] __device__(auto index) {
              auto edge_time = d_tmp[index];
              auto key_time  = d_tmp_time[index];
              return (edge_time > key_time);
            });

    raft::device_span<uint32_t const> marked_entry_span{marked_entries.data(),
                                                        marked_entries.size()};

    result_srcs =
      detail::keep_marked_entries(handle, std::move(result_srcs), marked_entry_span, keep_count);
    result_dsts =
      detail::keep_marked_entries(handle, std::move(result_dsts), marked_entry_span, keep_count);
    if (std::holds_alternative<rmm::device_uvector<edge_t>>(tmp_edge_indices)) {
      tmp_edge_indices = detail::keep_marked_entries(
        handle,
        std::move(std::get<rmm::device_uvector<edge_t>>(tmp_edge_indices)),
        marked_entry_span,
        keep_count);
    }
    if (tmp_times) {
      *tmp_times =
        detail::keep_marked_entries(handle, std::move(*tmp_times), marked_entry_span, keep_count);
    }
    if (tmp_positions) {
      *tmp_positions = detail::keep_marked_entries(
        handle, std::move(*tmp_positions), marked_entry_span, keep_count);
    }

    result_labels = rmm::device_uvector<label_t>(keep_count, handle.get_stream());
    kv_store.view().find(
      tmp_positions->begin(),
      tmp_positions->end(),
      thrust::make_zip_iterator(thrust::make_discard_iterator(), result_labels->begin()),
      handle.get_stream());
  }

  std::tie(result_srcs, result_dsts, result_properties) =
    gather_sampled_properties(handle,
                              graph_view,
                              std::move(result_srcs),
                              std::move(result_dsts),
                              std::move(tmp_edge_indices),
                              raft::host_span<edge_arithmetic_property_view_t<edge_t>>{
                                edge_property_views.data(), edge_property_views.size()});

  return std::make_tuple(std::move(result_srcs),
                         std::move(result_dsts),
                         std::move(result_properties),
                         std::move(result_labels));
}

}  // namespace detail
}  // namespace cugraph
