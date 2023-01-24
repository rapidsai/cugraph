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
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <sampling/detail/graph_functions.hpp>

#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph.hpp>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

namespace cugraph {
namespace detail {

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           rmm::device_uvector<weight_t>,
           rmm::device_uvector<edge_t>>
uniform_nbr_sample_impl(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  rmm::device_uvector<vertex_t>& d_in,
  raft::host_span<const int> h_fan_out,
  bool with_replacement,
  uint64_t seed)
{
#ifdef NO_CUGRAPH_OPS
  CUGRAPH_FAIL(
    "uniform_nbr_sampl_impl not supported in this configuration, built with NO_CUGRAPH_OPS");
#else

  CUGRAPH_EXPECTS(h_fan_out.size() > 0,
                  "Invalid input argument: number of levels must be non-zero.");

  rmm::device_uvector<vertex_t> d_result_src(0, handle.get_stream());
  rmm::device_uvector<vertex_t> d_result_dst(0, handle.get_stream());
  auto d_result_weights =
    thrust::make_optional(rmm::device_uvector<weight_t>(0, handle.get_stream()));

  int32_t hop{0};
  size_t comm_size{1};

  if constexpr (multi_gpu) {
    seed += handle.get_comms().get_rank();
    comm_size = handle.get_comms().get_size();
  }

  for (auto&& k_level : h_fan_out) {
    // prep step for extracting out-degs(sources):
    if constexpr (multi_gpu) {
      d_in = shuffle_int_vertices_to_local_gpu_by_vertex_partitioning(
        handle, std::move(d_in), graph_view.vertex_partition_range_lasts());
    }

    rmm::device_uvector<vertex_t> d_out_src(0, handle.get_stream());
    rmm::device_uvector<vertex_t> d_out_dst(0, handle.get_stream());
    auto d_out_weights = std::make_optional(rmm::device_uvector<weight_t>(0, handle.get_stream()));

    if (k_level > 0) {
      raft::random::RngState rng_state(seed);
      seed += d_in.size() * k_level * comm_size;

      std::tie(d_out_src, d_out_dst, d_out_weights) = sample_edges(handle,
                                                                   graph_view,
                                                                   edge_weight_view,
                                                                   rng_state,
                                                                   d_in,
                                                                   static_cast<size_t>(k_level),
                                                                   with_replacement);
    } else {
      std::tie(d_out_src, d_out_dst, d_out_weights) =
        gather_one_hop_edgelist(handle, graph_view, edge_weight_view, d_in);
    }

    // resize accumulators:
    auto old_sz = d_result_dst.size();
    auto add_sz = d_out_dst.size();
    auto new_sz = old_sz + add_sz;

    d_result_src.resize(new_sz, handle.get_stream());
    d_result_dst.resize(new_sz, handle.get_stream());
    d_result_weights->resize(new_sz, handle.get_stream());

    raft::copy(
      d_result_src.begin() + old_sz, d_out_src.begin(), d_out_src.size(), handle.get_stream());
    raft::copy(
      d_result_dst.begin() + old_sz, d_out_dst.begin(), d_out_dst.size(), handle.get_stream());
    raft::copy(d_result_weights->begin() + old_sz,
               d_out_weights->begin(),
               d_out_weights->size(),
               handle.get_stream());

    d_in = std::move(d_out_dst);

    ++hop;
  }

  return count_and_remove_duplicates<vertex_t, edge_t, weight_t>(
    handle, std::move(d_result_src), std::move(d_result_dst), std::move(*d_result_weights));
#endif
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
           std::optional<rmm::device_uvector<edge_type_t>>,
           rmm::device_uvector<int32_t>,
           std::optional<rmm::device_uvector<int32_t>>>
uniform_neighbor_sample_impl(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<
    edge_property_view_t<edge_t,
                         thrust::zip_iterator<thrust::tuple<edge_t const*, edge_type_t const*>>>>
    edge_id_type_view,
  rmm::device_uvector<vertex_t>&& d_in,
  std::optional<rmm::device_uvector<int32_t>>&& d_labels,
  raft::host_span<int32_t const> h_fan_out,
  bool with_replacement,
  raft::random::RngState& rng_state)
{
#ifdef NO_CUGRAPH_OPS
  CUGRAPH_FAIL(
    "uniform_neighbor_sampl_impl not supported in this configuration, built with NO_CUGRAPH_OPS");
#else

  CUGRAPH_EXPECTS(h_fan_out.size() > 0,
                  "Invalid input argument: number of levels must be non-zero.");

  rmm::device_uvector<vertex_t> d_result_src(0, handle.get_stream());
  rmm::device_uvector<vertex_t> d_result_dst(0, handle.get_stream());
  auto d_result_edge_id =
    edge_id_type_view ? std::make_optional(rmm::device_uvector<edge_t>(0, handle.get_stream()))
                      : std::nullopt;
  auto d_result_weight =
    edge_weight_view ? std::make_optional(rmm::device_uvector<weight_t>(0, handle.get_stream()))
                     : std::nullopt;
  auto d_result_edge_type =
    edge_id_type_view ? std::make_optional(rmm::device_uvector<edge_type_t>(0, handle.get_stream()))
                      : std::nullopt;
  rmm::device_uvector<int32_t> d_result_hop(0, handle.get_stream());
  auto d_result_label = d_labels
                          ? std::make_optional(rmm::device_uvector<int32_t>(0, handle.get_stream()))
                          : std::nullopt;

  int32_t hop{0};
  size_t comm_size{1};

  if constexpr (multi_gpu) { comm_size = handle.get_comms().get_size(); }

  for (auto&& k_level : h_fan_out) {
    // prep step for extracting out-degs(sources):
    if constexpr (multi_gpu) {
      if (d_labels) {
        std::tie(d_in, *d_labels) =
          shuffle_int_vertex_value_pairs_to_local_gpu_by_vertex_partitioning(
            handle,
            std::move(d_in),
            std::move(*d_labels),
            graph_view.vertex_partition_range_lasts());
      } else {
        d_in = shuffle_int_vertices_to_local_gpu_by_vertex_partitioning(
          handle, std::move(d_in), graph_view.vertex_partition_range_lasts());
      }
    }

    rmm::device_uvector<vertex_t> d_out_src(0, handle.get_stream());
    rmm::device_uvector<vertex_t> d_out_dst(0, handle.get_stream());
    std::optional<rmm::device_uvector<edge_t>> d_out_edge_id{std::nullopt};
    std::optional<rmm::device_uvector<weight_t>> d_out_weight{std::nullopt};
    std::optional<rmm::device_uvector<edge_type_t>> d_out_edge_type{std::nullopt};
    std::optional<rmm::device_uvector<int32_t>> d_out_label{std::nullopt};

    if (k_level > 0) {
      std::tie(d_out_src, d_out_dst, d_out_weight, d_out_edge_id, d_out_edge_type, d_out_label) =
        sample_edges(handle,
                     graph_view,
                     edge_weight_view,
                     edge_id_type_view,
                     rng_state,
                     d_in,
                     d_labels,
                     static_cast<size_t>(k_level),
                     with_replacement);
    } else {
      std::tie(d_out_src, d_out_dst, d_out_weight, d_out_edge_id, d_out_edge_type, d_out_label) =
        gather_one_hop_edgelist(
          handle, graph_view, edge_weight_view, edge_id_type_view, d_in, d_labels);
    }

    // FIXME: potential optimization, continual resizing/copying
    //        of these device vectors is inefficient.  If we're going to
    //        do higher hop sampling it might be more efficient to create
    //        a separate vector for each hop and concatenate them all
    //        at the end rather than recopying each time.
    // resize accumulators:
    auto old_sz = d_result_dst.size();
    auto add_sz = d_out_dst.size();
    auto new_sz = old_sz + add_sz;

    d_result_src.resize(new_sz, handle.get_stream());
    d_result_dst.resize(new_sz, handle.get_stream());
    d_result_hop.resize(new_sz, handle.get_stream());
    if (d_result_weight) d_result_weight->resize(new_sz, handle.get_stream());
    if (d_result_edge_id) d_result_edge_id->resize(new_sz, handle.get_stream());
    if (d_result_edge_type) d_result_edge_type->resize(new_sz, handle.get_stream());
    if (d_result_label) d_result_label->resize(new_sz, handle.get_stream());

    raft::copy(
      d_result_src.begin() + old_sz, d_out_src.begin(), d_out_src.size(), handle.get_stream());
    raft::copy(
      d_result_dst.begin() + old_sz, d_out_dst.begin(), d_out_dst.size(), handle.get_stream());
    if (d_out_edge_id)
      raft::copy(d_result_edge_id->begin() + old_sz,
                 d_out_edge_id->begin(),
                 d_out_edge_id->size(),
                 handle.get_stream());

    detail::scalar_fill(handle, d_result_hop.data() + old_sz, add_sz, hop);

    if (d_result_weight)
      raft::copy(d_result_weight->begin() + old_sz,
                 d_out_weight->begin(),
                 d_out_weight->size(),
                 handle.get_stream());
    if (d_result_edge_type)
      raft::copy(d_result_edge_type->begin() + old_sz,
                 d_out_edge_type->begin(),
                 d_out_edge_type->size(),
                 handle.get_stream());
    if (d_result_label) {
      raft::copy(d_result_label->begin() + old_sz,
                 d_out_label->begin(),
                 d_out_label->size(),
                 handle.get_stream());
    }

    d_in     = std::move(d_out_dst);
    d_labels = std::move(d_out_label);

    ++hop;
  }

  return std::make_tuple(std::move(d_result_src),
                         std::move(d_result_dst),
                         std::move(d_result_weight),
                         std::move(d_result_edge_id),
                         std::move(d_result_edge_type),
                         std::move(d_result_hop),
                         std::move(d_result_label));
#endif
}

}  // namespace detail

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           rmm::device_uvector<weight_t>,
           rmm::device_uvector<edge_t>>
uniform_nbr_sample(raft::handle_t const& handle,
                   graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
                   std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
                   raft::device_span<vertex_t> starting_vertices,
                   raft::host_span<const int> fan_out,
                   bool with_replacement,
                   uint64_t seed)
{
  rmm::device_uvector<vertex_t> d_start_vs(starting_vertices.size(), handle.get_stream());
  raft::copy(
    d_start_vs.data(), starting_vertices.data(), starting_vertices.size(), handle.get_stream());

  return detail::uniform_nbr_sample_impl(
    handle, graph_view, edge_weight_view, d_start_vs, fan_out, with_replacement, seed);
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
           std::optional<rmm::device_uvector<edge_type_t>>,
           rmm::device_uvector<int32_t>,
           std::optional<rmm::device_uvector<int32_t>>>
uniform_neighbor_sample(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<
    edge_property_view_t<edge_t,
                         thrust::zip_iterator<thrust::tuple<edge_t const*, edge_type_t const*>>>>
    edge_id_type_view,
  rmm::device_uvector<vertex_t>&& starting_vertices,
  std::optional<rmm::device_uvector<int32_t>>&& starting_labels,
  raft::host_span<int32_t const> fan_out,
  raft::random::RngState& rng_state,
  bool with_replacement)
{
  return detail::uniform_neighbor_sample_impl(handle,
                                              graph_view,
                                              edge_weight_view,
                                              edge_id_type_view,
                                              std::move(starting_vertices),
                                              std::move(starting_labels),
                                              fan_out,
                                              with_replacement,
                                              rng_state);
}

}  // namespace cugraph
