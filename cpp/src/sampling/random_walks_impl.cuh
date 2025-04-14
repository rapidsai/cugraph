/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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
#include "prims/detail/nbr_intersection.cuh"
#include "prims/per_v_random_select_transform_outgoing_e.cuh"
#include "prims/property_op_utils.cuh"
#include "prims/update_edge_src_dst_property.cuh"
#include "prims/vertex_frontier.cuh"

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/shuffle_comm.cuh>

#include <raft/core/handle.hpp>
#include <raft/random/rng.cuh>

#include <rmm/device_uvector.hpp>

#include <cuda/std/iterator>
#include <cuda/std/optional>

#include <algorithm>
#include <limits>
#include <numeric>
#include <type_traits>
#include <vector>

namespace cugraph {
namespace detail {

template <typename vertex_t, typename weight_t>
struct sample_edges_op_t {
  template <typename W = weight_t>
  __device__ std::enable_if_t<std::is_same_v<W, void>, vertex_t> operator()(
    vertex_t, vertex_t dst, cuda::std::nullopt_t, cuda::std::nullopt_t, cuda::std::nullopt_t) const
  {
    return dst;
  }

  template <typename W = weight_t>
  __device__ std::enable_if_t<!std::is_same_v<W, void>, thrust::tuple<vertex_t, W>> operator()(
    vertex_t, vertex_t dst, cuda::std::nullopt_t, cuda::std::nullopt_t, W w) const
  {
    return thrust::make_tuple(dst, w);
  }
};

template <typename vertex_t, typename bias_t>
struct biased_random_walk_e_bias_op_t {
  __device__ bias_t operator()(
    vertex_t, vertex_t, bias_t src_out_weight_sum, cuda::std::nullopt_t, bias_t weight) const
  {
    return weight / src_out_weight_sum;
  }
};

template <typename vertex_t, typename weight_t>
struct biased_sample_edges_op_t {
  __device__ thrust::tuple<vertex_t, weight_t> operator()(
    vertex_t, vertex_t dst, weight_t, cuda::std::nullopt_t, weight_t weight) const
  {
    return thrust::make_tuple(dst, weight);
  }
};

template <typename vertex_t, typename bias_t, typename weight_t>
struct node2vec_random_walk_e_bias_op_t {
  bias_t p_{};
  bias_t q_{};
  raft::device_span<size_t const> intersection_offsets_{};
  raft::device_span<vertex_t const> intersection_indices_{};
  raft::device_span<vertex_t const> current_vertices_{};
  raft::device_span<vertex_t const> prev_vertices_{};

  // Unweighted Bias Operator
  template <typename W = weight_t>
  __device__ std::enable_if_t<std::is_same_v<W, void>, bias_t> operator()(
    thrust::tuple<vertex_t, vertex_t> tagged_src,
    vertex_t dst,
    cuda::std::nullopt_t,
    cuda::std::nullopt_t,
    cuda::std::nullopt_t) const
  {
    //  Check tag (prev vert) for destination
    if (dst == thrust::get<1>(tagged_src)) { return 1.0 / p_; }
    //  Search zipped vertices for tagged src
    auto lower_itr = thrust::lower_bound(
      thrust::seq,
      thrust::make_zip_iterator(current_vertices_.begin(), prev_vertices_.begin()),
      thrust::make_zip_iterator(current_vertices_.end(), prev_vertices_.end()),
      tagged_src);
    auto low_idx = cuda::std::distance(
      thrust::make_zip_iterator(current_vertices_.begin(), prev_vertices_.begin()), lower_itr);
    auto intersection_index_first = intersection_indices_.begin() + intersection_offsets_[low_idx];
    auto intersection_index_last =
      intersection_indices_.begin() + intersection_offsets_[low_idx + 1];
    auto itr =
      thrust::lower_bound(thrust::seq, intersection_index_first, intersection_index_last, dst);
    return (itr != intersection_index_last && *itr == dst) ? 1.0 : 1.0 / q_;
  }

  //  Weighted Bias Operator
  template <typename W = weight_t>
  __device__ std::enable_if_t<!std::is_same_v<W, void>, bias_t> operator()(
    thrust::tuple<vertex_t, vertex_t> tagged_src,
    vertex_t dst,
    cuda::std::nullopt_t,
    cuda::std::nullopt_t,
    W) const
  {
    //  Check tag (prev vert) for destination
    if (dst == thrust::get<1>(tagged_src)) { return 1.0 / p_; }
    //  Search zipped vertices for tagged src
    auto lower_itr = thrust::lower_bound(
      thrust::seq,
      thrust::make_zip_iterator(current_vertices_.begin(), prev_vertices_.begin()),
      thrust::make_zip_iterator(current_vertices_.end(), prev_vertices_.end()),
      tagged_src);
    auto low_idx = cuda::std::distance(
      thrust::make_zip_iterator(current_vertices_.begin(), prev_vertices_.begin()), lower_itr);
    auto intersection_index_first = intersection_indices_.begin() + intersection_offsets_[low_idx];
    auto intersection_index_last =
      intersection_indices_.begin() + intersection_offsets_[low_idx + 1];
    auto itr =
      thrust::lower_bound(thrust::seq, intersection_index_first, intersection_index_last, dst);
    return (itr != intersection_index_last && *itr == dst) ? 1.0 : 1.0 / q_;
  }
};

template <typename vertex_t, typename weight_t>
struct node2vec_sample_edges_op_t {
  template <typename W = weight_t>
  __device__ std::enable_if_t<std::is_same_v<W, void>, vertex_t> operator()(
    thrust::tuple<vertex_t, vertex_t> tagged_src,
    vertex_t dst,
    cuda::std::nullopt_t,
    cuda::std::nullopt_t,
    cuda::std::nullopt_t) const
  {
    return dst;
  }

  template <typename W = weight_t>
  __device__ std::enable_if_t<!std::is_same_v<W, void>, thrust::tuple<vertex_t, W>> operator()(
    thrust::tuple<vertex_t, vertex_t> tagged_src,
    vertex_t dst,
    cuda::std::nullopt_t,
    cuda::std::nullopt_t,
    W w) const
  {
    return thrust::make_tuple(dst, w);
  }
};

template <typename weight_t>
struct uniform_selector {
  raft::random::RngState& rng_state_;
  static constexpr bool is_second_order_ = false;

  template <typename GraphViewType>
  std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
             std::optional<rmm::device_uvector<typename GraphViewType::vertex_type>>,
             std::optional<rmm::device_uvector<weight_t>>>
  follow_random_edge(
    raft::handle_t const& handle,
    GraphViewType const& graph_view,
    std::optional<edge_property_view_t<typename GraphViewType::edge_type, weight_t const*>>
      edge_weight_view,
    rmm::device_uvector<typename GraphViewType::vertex_type>&& current_vertices,
    std::optional<rmm::device_uvector<typename GraphViewType::vertex_type>>&& previous_vertices)
  {
    using vertex_t = typename GraphViewType::vertex_type;

    // FIXME: add as a template parameter
    using tag_t = void;

    cugraph::vertex_frontier_t<vertex_t, tag_t, GraphViewType::is_multi_gpu, false> vertex_frontier(
      handle, 1);

    vertex_frontier.bucket(0).insert(current_vertices.begin(), current_vertices.end());

    rmm::device_uvector<vertex_t> minors(0, handle.get_stream());
    std::optional<rmm::device_uvector<weight_t>> weights{std::nullopt};
    if (edge_weight_view) {
      auto [sample_offsets, sample_e_op_results] =
        cugraph::per_v_random_select_transform_outgoing_e(
          handle,
          graph_view,
          vertex_frontier.bucket(0),
          edge_src_dummy_property_t{}.view(),
          edge_dst_dummy_property_t{}.view(),
          *edge_weight_view,
          sample_edges_op_t<vertex_t, weight_t>{},
          rng_state_,
          size_t{1},
          true,
          std::make_optional(
            thrust::make_tuple(cugraph::invalid_vertex_id<vertex_t>::value, weight_t{0.0})));

      minors  = std::move(std::get<0>(sample_e_op_results));
      weights = std::move(std::get<1>(sample_e_op_results));
    } else {
      auto [sample_offsets, sample_e_op_results] =
        cugraph::per_v_random_select_transform_outgoing_e(
          handle,
          graph_view,
          vertex_frontier.bucket(0),
          edge_src_dummy_property_t{}.view(),
          edge_dst_dummy_property_t{}.view(),
          edge_dummy_property_t{}.view(),
          sample_edges_op_t<vertex_t, void>{},
          rng_state_,
          size_t{1},
          true,
          std::make_optional(vertex_t{cugraph::invalid_vertex_id<vertex_t>::value}));

      minors = std::move(sample_e_op_results);
    }
    return std::make_tuple(std::move(minors), std::move(previous_vertices), std::move(weights));
  }
};

template <typename weight_t>
struct biased_selector {
  raft::random::RngState& rng_state_;
  static constexpr bool is_second_order_ = false;

  template <typename GraphViewType>
  std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
             std::optional<rmm::device_uvector<typename GraphViewType::vertex_type>>,
             std::optional<rmm::device_uvector<weight_t>>>
  follow_random_edge(
    raft::handle_t const& handle,
    GraphViewType const& graph_view,
    std::optional<edge_property_view_t<typename GraphViewType::edge_type, weight_t const*>>
      edge_weight_view,
    rmm::device_uvector<typename GraphViewType::vertex_type>&& current_vertices,
    std::optional<rmm::device_uvector<typename GraphViewType::vertex_type>>&& previous_vertices)
  {
    //  Create vertex frontier
    using vertex_t = typename GraphViewType::vertex_type;

    using tag_t = void;

    cugraph::vertex_frontier_t<vertex_t, tag_t, GraphViewType::is_multi_gpu, false> vertex_frontier(
      handle, 1);

    vertex_frontier.bucket(0).insert(current_vertices.begin(), current_vertices.end());

    auto vertex_weight_sum = compute_out_weight_sums(handle, graph_view, *edge_weight_view);
    edge_src_property_t<GraphViewType, weight_t> edge_src_out_weight_sums(handle, graph_view);
    update_edge_src_property(handle,
                             graph_view,
                             vertex_frontier.bucket(0).begin(),
                             vertex_frontier.bucket(0).end(),
                             vertex_weight_sum.data(),
                             edge_src_out_weight_sums.mutable_view());
    auto [sample_offsets, sample_e_op_results] = cugraph::per_v_random_select_transform_outgoing_e(
      handle,
      graph_view,
      vertex_frontier.bucket(0),
      edge_src_out_weight_sums.view(),
      cugraph::edge_dst_dummy_property_t{}.view(),
      *edge_weight_view,
      biased_random_walk_e_bias_op_t<vertex_t, weight_t>{},
      edge_src_out_weight_sums.view(),
      cugraph::edge_dst_dummy_property_t{}.view(),
      *edge_weight_view,
      biased_sample_edges_op_t<vertex_t, weight_t>{},
      rng_state_,
      size_t{1},
      true,
      std::make_optional(
        thrust::make_tuple(vertex_t{cugraph::invalid_vertex_id<vertex_t>::value}, weight_t{0.0})));

    //  Return results
    return std::make_tuple(std::move(std::get<0>(sample_e_op_results)),
                           std::move(previous_vertices),
                           std::move(std::get<1>(sample_e_op_results)));
  }
};

template <typename weight_t>
struct node2vec_selector {
  weight_t p_;
  weight_t q_;
  raft::random::RngState& rng_state_;
  static constexpr bool is_second_order_ = true;

  template <typename GraphViewType>
  std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
             std::optional<rmm::device_uvector<typename GraphViewType::vertex_type>>,
             std::optional<rmm::device_uvector<weight_t>>>
  follow_random_edge(
    raft::handle_t const& handle,
    GraphViewType const& graph_view,
    std::optional<edge_property_view_t<typename GraphViewType::edge_type, weight_t const*>>
      edge_weight_view,
    rmm::device_uvector<typename GraphViewType::vertex_type>&& current_vertices,
    std::optional<rmm::device_uvector<typename GraphViewType::vertex_type>>&& previous_vertices)
  {
    //  Create vertex frontier
    using vertex_t = typename GraphViewType::vertex_type;

    using tag_t = vertex_t;

    //  Zip previous and current vertices for nbr_intersection()
    auto intersection_pairs =
      thrust::make_zip_iterator(current_vertices.begin(), (*previous_vertices).begin());

    auto [intersection_offsets, intersection_indices] =
      detail::nbr_intersection(handle,
                               graph_view,
                               cugraph::edge_dummy_property_t{}.view(),
                               intersection_pairs,
                               intersection_pairs + current_vertices.size(),
                               std::array<bool, 2>{true, true},
                               false);

    rmm::device_uvector<size_t> intersection_counts(size_t{0}, handle.get_stream());
    rmm::device_uvector<size_t> aggregate_offsets(size_t{0}, handle.get_stream());
    rmm::device_uvector<vertex_t> aggregate_currents(size_t{0}, handle.get_stream());
    rmm::device_uvector<vertex_t> aggregate_previous(size_t{0}, handle.get_stream());
    rmm::device_uvector<vertex_t> aggregate_indices(size_t{0}, handle.get_stream());

    //  Aggregate intersection data across minor comm
    if constexpr (GraphViewType::is_multi_gpu) {
      intersection_counts.resize(intersection_offsets.size(), handle.get_stream());
      thrust::adjacent_difference(handle.get_thrust_policy(),
                                  intersection_offsets.begin(),
                                  intersection_offsets.end(),
                                  intersection_counts.begin());

      auto recv_counts = cugraph::host_scalar_allgather(
        handle.get_subcomm(cugraph::partition_manager::minor_comm_name()),
        current_vertices.size(),
        handle.get_stream());

      std::vector<size_t> displacements(recv_counts.size());
      std::exclusive_scan(recv_counts.begin(), recv_counts.end(), displacements.begin(), size_t{0});

      aggregate_offsets.resize(displacements.back() + recv_counts.back() + 1, handle.get_stream());
      aggregate_offsets.set_element_to_zero_async(aggregate_offsets.size() - 1,
                                                  handle.get_stream());

      cugraph::device_allgatherv(
        handle.get_subcomm(cugraph::partition_manager::minor_comm_name()),
        intersection_counts.begin() + 1,
        aggregate_offsets.begin(),
        raft::host_span<size_t const>(recv_counts.data(), recv_counts.size()),
        raft::host_span<size_t const>(displacements.data(), displacements.size()),
        handle.get_stream());

      thrust::exclusive_scan(handle.get_thrust_policy(),
                             aggregate_offsets.begin(),
                             aggregate_offsets.end(),
                             aggregate_offsets.begin());

      aggregate_currents.resize(displacements.back() + recv_counts.back(), handle.get_stream());

      cugraph::device_allgatherv(
        handle.get_subcomm(cugraph::partition_manager::minor_comm_name()),
        current_vertices.begin(),
        aggregate_currents.begin(),
        raft::host_span<size_t const>(recv_counts.data(), recv_counts.size()),
        raft::host_span<size_t const>(displacements.data(), displacements.size()),
        handle.get_stream());

      aggregate_previous.resize(displacements.back() + recv_counts.back(), handle.get_stream());

      cugraph::device_allgatherv(
        handle.get_subcomm(cugraph::partition_manager::minor_comm_name()),
        (*previous_vertices).begin(),
        aggregate_previous.begin(),
        raft::host_span<size_t const>(recv_counts.data(), recv_counts.size()),
        raft::host_span<size_t const>(displacements.data(), displacements.size()),
        handle.get_stream());

      recv_counts = cugraph::host_scalar_allgather(
        handle.get_subcomm(cugraph::partition_manager::minor_comm_name()),
        intersection_offsets.back_element(handle.get_stream()),
        handle.get_stream());

      displacements.resize(recv_counts.size());
      std::exclusive_scan(recv_counts.begin(), recv_counts.end(), displacements.begin(), size_t{0});

      aggregate_indices.resize(displacements.back() + recv_counts.back(), handle.get_stream());

      cugraph::device_allgatherv(
        handle.get_subcomm(cugraph::partition_manager::minor_comm_name()),
        intersection_indices.begin(),
        aggregate_indices.begin(),
        raft::host_span<size_t const>(recv_counts.data(), recv_counts.size()),
        raft::host_span<size_t const>(displacements.data(), displacements.size()),
        handle.get_stream());
    }

    cugraph::vertex_frontier_t<vertex_t, tag_t, GraphViewType::is_multi_gpu, false> vertex_frontier(
      handle, 1);
    vertex_frontier.bucket(0).insert(
      thrust::make_zip_iterator(current_vertices.begin(), (*previous_vertices).begin()),
      thrust::make_zip_iterator(current_vertices.end(), (*previous_vertices).end()));

    // Create data structs for results
    rmm::device_uvector<vertex_t> minors(0, handle.get_stream());
    std::optional<rmm::device_uvector<weight_t>> weights{std::nullopt};

    if (edge_weight_view) {
      auto [sample_offsets, sample_e_op_results] =
        cugraph::per_v_random_select_transform_outgoing_e(
          handle,
          graph_view,
          vertex_frontier.bucket(0),
          cugraph::edge_src_dummy_property_t{}.view(),
          cugraph::edge_dst_dummy_property_t{}.view(),
          *edge_weight_view,
          GraphViewType::is_multi_gpu
            ? node2vec_random_walk_e_bias_op_t<vertex_t,
                                               weight_t,
                                               weight_t>{p_,
                                                         q_,
                                                         raft::device_span<size_t const>(
                                                           aggregate_offsets.data(),
                                                           aggregate_offsets.size()),
                                                         raft::device_span<vertex_t const>(
                                                           aggregate_indices.data(),
                                                           aggregate_indices.size()),
                                                         raft::device_span<vertex_t const>(
                                                           aggregate_currents.data(),
                                                           aggregate_currents.size()),
                                                         raft::device_span<vertex_t const>(
                                                           aggregate_previous.data(),
                                                           aggregate_previous.size())}
            : node2vec_random_walk_e_bias_op_t<vertex_t,
                                               weight_t,
                                               weight_t>{p_,
                                                         q_,
                                                         raft::device_span<size_t const>(
                                                           intersection_offsets.data(),
                                                           intersection_offsets.size()),
                                                         raft::device_span<vertex_t const>(
                                                           intersection_indices.data(),
                                                           intersection_indices.size()),
                                                         raft::device_span<
                                                           vertex_t const>(current_vertices.data(),
                                                                           current_vertices.size()),
                                                         raft::device_span<vertex_t const>(
                                                           (*previous_vertices).data(),
                                                           (*previous_vertices).size())},
          cugraph::edge_src_dummy_property_t{}.view(),
          cugraph::edge_dst_dummy_property_t{}.view(),
          *edge_weight_view,
          node2vec_sample_edges_op_t<vertex_t, weight_t>{},
          rng_state_,
          size_t{1},
          true,
          std::make_optional(thrust::make_tuple(
            vertex_t{cugraph::invalid_vertex_id<vertex_t>::value}, weight_t{0.0})));
      minors  = std::move(std::get<0>(sample_e_op_results));
      weights = std::move(std::get<1>(sample_e_op_results));
    } else {
      auto [sample_offsets, sample_e_op_results] =
        cugraph::per_v_random_select_transform_outgoing_e(
          handle,
          graph_view,
          vertex_frontier.bucket(0),
          cugraph::edge_src_dummy_property_t{}.view(),
          cugraph::edge_dst_dummy_property_t{}.view(),
          cugraph::edge_dummy_property_t{}.view(),
          GraphViewType::is_multi_gpu
            ? node2vec_random_walk_e_bias_op_t<vertex_t,
                                               weight_t,
                                               weight_t>{p_,
                                                         q_,
                                                         raft::device_span<size_t const>(
                                                           aggregate_offsets.data(),
                                                           aggregate_offsets.size()),
                                                         raft::device_span<vertex_t const>(
                                                           aggregate_indices.data(),
                                                           aggregate_indices.size()),
                                                         raft::device_span<vertex_t const>(
                                                           aggregate_currents.data(),
                                                           aggregate_currents.size()),
                                                         raft::device_span<vertex_t const>(
                                                           aggregate_previous.data(),
                                                           aggregate_previous.size())}
            : node2vec_random_walk_e_bias_op_t<vertex_t,
                                               weight_t,
                                               weight_t>{p_,
                                                         q_,
                                                         raft::device_span<size_t const>(
                                                           intersection_offsets.data(),
                                                           intersection_offsets.size()),
                                                         raft::device_span<vertex_t const>(
                                                           intersection_indices.data(),
                                                           intersection_indices.size()),
                                                         raft::device_span<
                                                           vertex_t const>(current_vertices.data(),
                                                                           current_vertices.size()),
                                                         raft::device_span<vertex_t const>(
                                                           (*previous_vertices).data(),
                                                           (*previous_vertices).size())},
          cugraph::edge_src_dummy_property_t{}.view(),
          cugraph::edge_dst_dummy_property_t{}.view(),
          cugraph::edge_dummy_property_t{}.view(),
          node2vec_sample_edges_op_t<vertex_t, void>{},
          rng_state_,
          size_t{1},
          true,
          std::make_optional(vertex_t{cugraph::invalid_vertex_id<vertex_t>::value}));
      minors = std::move(sample_e_op_results);
    }

    *previous_vertices = std::move(current_vertices);

    return std::make_tuple(std::move(minors), std::move(previous_vertices), std::move(weights));
  }
};

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool multi_gpu,
          typename random_selector_t>
std::tuple<rmm::device_uvector<vertex_t>, std::optional<rmm::device_uvector<weight_t>>>
random_walk_impl(raft::handle_t const& handle,
                 graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
                 std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
                 raft::device_span<vertex_t const> start_vertices,
                 size_t max_length,
                 random_selector_t random_selector)
{
  rmm::device_uvector<vertex_t> result_vertices(start_vertices.size() * (max_length + 1),
                                                handle.get_stream());
  auto result_weights = edge_weight_view
                          ? std::make_optional<rmm::device_uvector<weight_t>>(
                              start_vertices.size() * max_length, handle.get_stream())
                          : std::nullopt;

  detail::scalar_fill(handle,
                      result_vertices.data(),
                      result_vertices.size(),
                      cugraph::invalid_vertex_id<vertex_t>::value);
  if (result_weights)
    detail::scalar_fill(handle, result_weights->data(), result_weights->size(), weight_t{0});

  rmm::device_uvector<vertex_t> current_vertices(start_vertices.size(), handle.get_stream());
  rmm::device_uvector<size_t> current_position(start_vertices.size(), handle.get_stream());
  rmm::device_uvector<int> current_gpu(0, handle.get_stream());
  auto new_weights = edge_weight_view
                       ? std::make_optional<rmm::device_uvector<weight_t>>(0, handle.get_stream())
                       : std::nullopt;

  auto previous_vertices = (random_selector.is_second_order_)
                             ? std::make_optional<rmm::device_uvector<vertex_t>>(
                                 current_vertices.size(), handle.get_stream())
                             : std::nullopt;
  if (previous_vertices) {
    raft::copy((*previous_vertices).data(),
               start_vertices.data(),
               start_vertices.size(),
               handle.get_stream());
  }
  raft::copy(
    current_vertices.data(), start_vertices.data(), start_vertices.size(), handle.get_stream());
  detail::sequence_fill(
    handle.get_stream(), current_position.data(), current_position.size(), size_t{0});

  if constexpr (multi_gpu) {
    current_gpu.resize(start_vertices.size(), handle.get_stream());

    detail::scalar_fill(
      handle, current_gpu.data(), current_gpu.size(), handle.get_comms().get_rank());
  }

  thrust::for_each(
    handle.get_thrust_policy(),
    thrust::make_counting_iterator<size_t>(0),
    thrust::make_counting_iterator<size_t>(current_vertices.size()),
    [current_verts = current_vertices.data(),
     result_verts  = result_vertices.data(),
     max_length] __device__(size_t i) { result_verts[i * (max_length + 1)] = current_verts[i]; });

  rmm::device_uvector<vertex_t> vertex_partition_range_lasts(
    graph_view.vertex_partition_range_lasts().size(), handle.get_stream());
  raft::update_device(vertex_partition_range_lasts.data(),
                      graph_view.vertex_partition_range_lasts().data(),
                      graph_view.vertex_partition_range_lasts().size(),
                      handle.get_stream());

  for (size_t level = 0; level < max_length; ++level) {
    if constexpr (multi_gpu) {
      auto& major_comm = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
      auto const major_comm_size = major_comm.get_size();
      auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
      auto const minor_comm_size = minor_comm.get_size();

      if (previous_vertices) {
        std::forward_as_tuple(
          std::tie(current_vertices, current_gpu, current_position, previous_vertices),
          std::ignore) =
          cugraph::groupby_gpu_id_and_shuffle_values(
            handle.get_comms(),
            thrust::make_zip_iterator(current_vertices.begin(),
                                      current_gpu.begin(),
                                      current_position.begin(),
                                      previous_vertices->begin()),
            thrust::make_zip_iterator(current_vertices.end(),
                                      current_gpu.end(),
                                      current_position.end(),
                                      previous_vertices->end()),
            [key_func =
               cugraph::detail::compute_gpu_id_from_int_vertex_t<vertex_t>{
                 {vertex_partition_range_lasts.begin(), vertex_partition_range_lasts.size()},
                 major_comm_size,
                 minor_comm_size}] __device__(auto val) { return key_func(thrust::get<0>(val)); },
            handle.get_stream());
      } else {
        // Shuffle vertices to correct GPU to compute random indices
        std::forward_as_tuple(std::tie(current_vertices, current_gpu, current_position),
                              std::ignore) =
          cugraph::groupby_gpu_id_and_shuffle_values(
            handle.get_comms(),
            thrust::make_zip_iterator(
              current_vertices.begin(), current_gpu.begin(), current_position.begin()),
            thrust::make_zip_iterator(
              current_vertices.end(), current_gpu.end(), current_position.end()),
            [key_func =
               cugraph::detail::compute_gpu_id_from_int_vertex_t<vertex_t>{
                 {vertex_partition_range_lasts.begin(), vertex_partition_range_lasts.size()},
                 major_comm_size,
                 minor_comm_size}] __device__(auto val) { return key_func(thrust::get<0>(val)); },
            handle.get_stream());
      }
    }

    //  Sort for nbr_intersection, must sort all together
    if (previous_vertices) {
      if constexpr (multi_gpu) {
        thrust::sort(handle.get_thrust_policy(),
                     thrust::make_zip_iterator(current_vertices.begin(),
                                               (*previous_vertices).begin(),
                                               current_position.begin(),
                                               current_gpu.begin()),
                     thrust::make_zip_iterator(current_vertices.end(),
                                               (*previous_vertices).end(),
                                               current_position.end(),
                                               current_gpu.end()));
      } else {
        thrust::sort(
          handle.get_thrust_policy(),
          thrust::make_zip_iterator(
            current_vertices.begin(), (*previous_vertices).begin(), current_position.begin()),
          thrust::make_zip_iterator(
            current_vertices.end(), (*previous_vertices).end(), current_position.end()));
      }
    }

    std::tie(current_vertices, previous_vertices, new_weights) =
      random_selector.follow_random_edge(handle,
                                         graph_view,
                                         edge_weight_view,
                                         std::move(current_vertices),
                                         std::move(previous_vertices));

    // FIXME: remove_if has a 32-bit overflow issue
    // (https://github.com/NVIDIA/thrust/issues/1302) Seems unlikely here (the goal of
    // sampling is to extract small graphs) so not going to work around this for now.
    CUGRAPH_EXPECTS(
      current_vertices.size() < static_cast<size_t>(std::numeric_limits<int32_t>::max()),
      "remove_if will fail, current_vertices.size() is too large");
    size_t compacted_length{0};
    if constexpr (multi_gpu) {
      if (result_weights) {
        if (previous_vertices) {
          auto input_iter = thrust::make_zip_iterator(current_vertices.begin(),
                                                      new_weights->begin(),
                                                      current_gpu.begin(),
                                                      current_position.begin(),
                                                      previous_vertices->begin());

          compacted_length = cuda::std::distance(
            input_iter,
            thrust::remove_if(handle.get_thrust_policy(),
                              input_iter,
                              input_iter + current_vertices.size(),
                              current_vertices.begin(),
                              [] __device__(auto dst) {
                                return (dst == cugraph::invalid_vertex_id<vertex_t>::value);
                              }));
        } else {
          auto input_iter = thrust::make_zip_iterator(current_vertices.begin(),
                                                      new_weights->begin(),
                                                      current_gpu.begin(),
                                                      current_position.begin());

          compacted_length = cuda::std::distance(
            input_iter,
            thrust::remove_if(handle.get_thrust_policy(),
                              input_iter,
                              input_iter + current_vertices.size(),
                              current_vertices.begin(),
                              [] __device__(auto dst) {
                                return (dst == cugraph::invalid_vertex_id<vertex_t>::value);
                              }));
        }
      } else {
        if (previous_vertices) {
          auto input_iter = thrust::make_zip_iterator(current_vertices.begin(),
                                                      current_gpu.begin(),
                                                      current_position.begin(),
                                                      previous_vertices->begin());

          compacted_length = cuda::std::distance(
            input_iter,
            thrust::remove_if(handle.get_thrust_policy(),
                              input_iter,
                              input_iter + current_vertices.size(),
                              current_vertices.begin(),
                              [] __device__(auto dst) {
                                return (dst == cugraph::invalid_vertex_id<vertex_t>::value);
                              }));
        } else {
          auto input_iter = thrust::make_zip_iterator(
            current_vertices.begin(), current_gpu.begin(), current_position.begin());

          compacted_length = cuda::std::distance(
            input_iter,
            thrust::remove_if(handle.get_thrust_policy(),
                              input_iter,
                              input_iter + current_vertices.size(),
                              current_vertices.begin(),
                              [] __device__(auto dst) {
                                return (dst == cugraph::invalid_vertex_id<vertex_t>::value);
                              }));
        }
      }
    } else {
      if (result_weights) {
        if (previous_vertices) {
          auto input_iter = thrust::make_zip_iterator(current_vertices.begin(),
                                                      new_weights->begin(),
                                                      current_position.begin(),
                                                      previous_vertices->begin());

          compacted_length = cuda::std::distance(
            input_iter,
            thrust::remove_if(handle.get_thrust_policy(),
                              input_iter,
                              input_iter + current_vertices.size(),
                              current_vertices.begin(),
                              [] __device__(auto dst) {
                                return (dst == cugraph::invalid_vertex_id<vertex_t>::value);
                              }));
        } else {
          auto input_iter = thrust::make_zip_iterator(
            current_vertices.begin(), new_weights->begin(), current_position.begin());

          compacted_length = cuda::std::distance(
            input_iter,
            thrust::remove_if(handle.get_thrust_policy(),
                              input_iter,
                              input_iter + current_vertices.size(),
                              current_vertices.begin(),
                              [] __device__(auto dst) {
                                return (dst == cugraph::invalid_vertex_id<vertex_t>::value);
                              }));
        }
      } else {
        if (previous_vertices) {
          auto input_iter = thrust::make_zip_iterator(
            current_vertices.begin(), current_position.begin(), previous_vertices->begin());

          compacted_length = cuda::std::distance(
            input_iter,
            thrust::remove_if(handle.get_thrust_policy(),
                              input_iter,
                              input_iter + current_vertices.size(),
                              current_vertices.begin(),
                              [] __device__(auto dst) {
                                return (dst == cugraph::invalid_vertex_id<vertex_t>::value);
                              }));
        } else {
          auto input_iter =
            thrust::make_zip_iterator(current_vertices.begin(), current_position.begin());

          compacted_length = cuda::std::distance(
            input_iter,
            thrust::remove_if(handle.get_thrust_policy(),
                              input_iter,
                              input_iter + current_vertices.size(),
                              current_vertices.begin(),
                              [] __device__(auto dst) {
                                return (dst == cugraph::invalid_vertex_id<vertex_t>::value);
                              }));
        }
      }
    }

    //  Moved out of if statements to cut down on code duplication
    current_vertices.resize(compacted_length, handle.get_stream());
    current_vertices.shrink_to_fit(handle.get_stream());
    current_position.resize(compacted_length, handle.get_stream());
    current_position.shrink_to_fit(handle.get_stream());
    if (result_weights) {
      new_weights->resize(compacted_length, handle.get_stream());
      new_weights->shrink_to_fit(handle.get_stream());
    }
    if (previous_vertices) {
      previous_vertices->resize(compacted_length, handle.get_stream());
      previous_vertices->shrink_to_fit(handle.get_stream());
    }
    if constexpr (multi_gpu) {
      current_gpu.resize(compacted_length, handle.get_stream());
      current_gpu.shrink_to_fit(handle.get_stream());

      // Shuffle back to original GPU
      if (previous_vertices) {
        if (result_weights) {
          auto current_iter = thrust::make_zip_iterator(current_vertices.begin(),
                                                        new_weights->begin(),
                                                        current_gpu.begin(),
                                                        current_position.begin(),
                                                        previous_vertices->begin());

          std::forward_as_tuple(
            std::tie(
              current_vertices, *new_weights, current_gpu, current_position, *previous_vertices),
            std::ignore) =
            cugraph::groupby_gpu_id_and_shuffle_values(
              handle.get_comms(),
              current_iter,
              current_iter + current_vertices.size(),
              [] __device__(auto val) { return thrust::get<2>(val); },
              handle.get_stream());
        } else {
          auto current_iter = thrust::make_zip_iterator(current_vertices.begin(),
                                                        current_gpu.begin(),
                                                        current_position.begin(),
                                                        previous_vertices->begin());

          std::forward_as_tuple(
            std::tie(current_vertices, current_gpu, current_position, *previous_vertices),
            std::ignore) =
            cugraph::groupby_gpu_id_and_shuffle_values(
              handle.get_comms(),
              current_iter,
              current_iter + current_vertices.size(),
              [] __device__(auto val) { return thrust::get<1>(val); },
              handle.get_stream());
        }
      } else {
        if (result_weights) {
          auto current_iter = thrust::make_zip_iterator(current_vertices.begin(),
                                                        new_weights->begin(),
                                                        current_gpu.begin(),
                                                        current_position.begin());

          std::forward_as_tuple(
            std::tie(current_vertices, *new_weights, current_gpu, current_position), std::ignore) =
            cugraph::groupby_gpu_id_and_shuffle_values(
              handle.get_comms(),
              current_iter,
              current_iter + current_vertices.size(),
              [] __device__(auto val) { return thrust::get<2>(val); },
              handle.get_stream());
        } else {
          auto current_iter = thrust::make_zip_iterator(
            current_vertices.begin(), current_gpu.begin(), current_position.begin());

          std::forward_as_tuple(std::tie(current_vertices, current_gpu, current_position),
                                std::ignore) =
            cugraph::groupby_gpu_id_and_shuffle_values(
              handle.get_comms(),
              current_iter,
              current_iter + current_vertices.size(),
              [] __device__(auto val) { return thrust::get<1>(val); },
              handle.get_stream());
        }
      }
    }

    if (result_weights) {
      thrust::for_each(handle.get_thrust_policy(),
                       thrust::make_zip_iterator(
                         current_vertices.begin(), new_weights->begin(), current_position.begin()),
                       thrust::make_zip_iterator(
                         current_vertices.end(), new_weights->end(), current_position.end()),
                       [result_verts = result_vertices.data(),
                        result_wgts  = result_weights->data(),
                        level,
                        max_length] __device__(auto tuple) {
                         vertex_t v                                       = thrust::get<0>(tuple);
                         weight_t w                                       = thrust::get<1>(tuple);
                         size_t pos                                       = thrust::get<2>(tuple);
                         result_verts[pos * (max_length + 1) + level + 1] = v;
                         result_wgts[pos * max_length + level]            = w;
                       });
    } else {
      thrust::for_each(
        handle.get_thrust_policy(),
        thrust::make_zip_iterator(current_vertices.begin(), current_position.begin()),
        thrust::make_zip_iterator(current_vertices.end(), current_position.end()),
        [result_verts = result_vertices.data(), level, max_length] __device__(auto tuple) {
          vertex_t v                                       = thrust::get<0>(tuple);
          size_t pos                                       = thrust::get<1>(tuple);
          result_verts[pos * (max_length + 1) + level + 1] = v;
        });
    }
  }

  return std::make_tuple(std::move(result_vertices), std::move(result_weights));
}

}  // namespace detail

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, std::optional<rmm::device_uvector<weight_t>>>
uniform_random_walks(raft::handle_t const& handle,
                     raft::random::RngState& rng_state,
                     graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
                     std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
                     raft::device_span<vertex_t const> start_vertices,
                     size_t max_length)
{
  CUGRAPH_EXPECTS(!graph_view.has_edge_mask(), "unimplemented.");

  return detail::random_walk_impl(handle,
                                  graph_view,
                                  edge_weight_view,
                                  start_vertices,
                                  max_length,
                                  detail::uniform_selector<weight_t>{rng_state});
}

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, std::optional<rmm::device_uvector<weight_t>>>
biased_random_walks(raft::handle_t const& handle,
                    raft::random::RngState& rng_state,
                    graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
                    edge_property_view_t<edge_t, weight_t const*> edge_weight_view,
                    raft::device_span<vertex_t const> start_vertices,
                    size_t max_length)
{
  CUGRAPH_EXPECTS(!graph_view.has_edge_mask(), "unimplemented.");

  return detail::random_walk_impl(
    handle,
    graph_view,
    std::optional<edge_property_view_t<edge_t, weight_t const*>>{edge_weight_view},
    start_vertices,
    max_length,
    detail::biased_selector<weight_t>{rng_state});
}

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, std::optional<rmm::device_uvector<weight_t>>>
node2vec_random_walks(raft::handle_t const& handle,
                      raft::random::RngState& rng_state,
                      graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
                      std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
                      raft::device_span<vertex_t const> start_vertices,
                      size_t max_length,
                      weight_t p,
                      weight_t q)
{
  CUGRAPH_EXPECTS(!graph_view.has_edge_mask(), "unimplemented.");

  return detail::random_walk_impl(handle,
                                  graph_view,
                                  edge_weight_view,
                                  start_vertices,
                                  max_length,
                                  detail::node2vec_selector<weight_t>{p, q, rng_state});
}

}  // namespace cugraph
