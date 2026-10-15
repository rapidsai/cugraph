/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/prims/copy_v.cuh>
#include <cugraph/prims/count_if_v.cuh>
#include <cugraph/prims/fill_edge_src_dst_property.cuh>
#include <cugraph/prims/fill_v.cuh>
#include <cugraph/prims/per_v_transform_reduce_incoming_outgoing_e.cuh>
#include <cugraph/prims/reduce_op.cuh>
#include <cugraph/prims/reduce_v.cuh>
#include <cugraph/prims/transform_reduce_v.cuh>
#include <cugraph/prims/transform_v.cuh>
#include <cugraph/prims/update_edge_src_dst_property.cuh>
#include <cugraph/vertex_property.hpp>

#include <cuda/std/tuple>
#include <thrust/copy.h>

namespace cugraph {
namespace detail {
template <typename GraphViewType, typename ReduceOp, typename result_t>
void normalize(raft::handle_t const& handle,
               GraphViewType const& graph_view,
               vertex_property_view_t<typename GraphViewType::vertex_type, result_t*> vertex_values,
               result_t init,
               ReduceOp reduce_op)
{
  auto hubs_norm = reduce_v(handle, graph_view, vertex_values, init, reduce_op);
  CUGRAPH_EXPECTS(hubs_norm > 0, "Norm is required to be a positive value.");
  transform_v(
    handle,
    graph_view,
    vertex_values,
    [hubs_norm] __device__(auto val) { return val / hubs_norm; },
    vertex_values);
}

template <typename GraphViewType, typename result_t>
centrality_algorithm_metadata_t hits(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  vertex_property_view_t<typename GraphViewType::vertex_type, result_t*> hubs,
  vertex_property_view_t<typename GraphViewType::vertex_type, result_t*> authorities,
  result_t epsilon,
  size_t max_iterations,
  bool has_initial_hubs_guess,
  bool normalize,
  bool do_expensive_check)
{
  using vertex_t = typename GraphViewType::vertex_type;
  static_assert(std::is_integral<vertex_t>::value,
                "GraphViewType::vertex_type should be integral.");
  static_assert(std::is_floating_point<result_t>::value,
                "result_t should be a floating-point type.");
  static_assert(GraphViewType::is_storage_transposed,
                "GraphViewType should support the pull model.");

  auto const num_vertices = graph_view.number_of_vertices();
  result_t diff_sum{std::numeric_limits<result_t>::max()};

  if (num_vertices == 0) { return centrality_algorithm_metadata_t{0, true}; }

  CUGRAPH_EXPECTS(epsilon >= 0.0, "Invalid input argument: epsilon should be non-negative.");
  auto tolerance = static_cast<result_t>(graph_view.number_of_vertices()) * epsilon;

  // Check validity of initial guess if supplied
  if (has_initial_hubs_guess && do_expensive_check) {
    auto num_negative_values =
      count_if_v(handle, graph_view, hubs, [] __device__(auto, auto val) { return val < 0.0; });
    CUGRAPH_EXPECTS(num_negative_values == 0,
                    "Invalid input argument: initial guess values should be non-negative.");
  }

  if (has_initial_hubs_guess) {
    detail::normalize(handle, graph_view, hubs, result_t{0.0}, reduce_op::plus<result_t>{});
  }

  // Property wrappers
  edge_src_property_t<vertex_t, result_t> prev_src_hubs(handle, graph_view);
  edge_dst_property_t<vertex_t, result_t> curr_dst_auth(handle, graph_view);
  vertex_property_t<vertex_t, result_t> temp_hubs(handle, graph_view);

  auto prev_hubs = hubs;
  auto curr_hubs = temp_hubs.mutable_view();

  // Initialize hubs from user input if provided
  if (has_initial_hubs_guess) {
    update_edge_src_property(handle, graph_view, prev_hubs, prev_src_hubs.mutable_view());
  } else {
    fill_edge_src_property(
      handle, graph_view, prev_src_hubs.mutable_view(), result_t{1.0} / num_vertices);
    fill_v(handle, graph_view, prev_hubs, result_t{1.0} / num_vertices);
  }

  size_t iter{0};
  while (true) {
    // Update current destination authorities property
    per_v_transform_reduce_incoming_e(
      handle,
      graph_view,
      prev_src_hubs.view(),
      edge_dst_dummy_property_t{}.view(),
      edge_dummy_property_t{}.view(),
      [] __device__(auto, auto, auto prev_src_hub_value, auto, auto) { return prev_src_hub_value; },
      result_t{0},
      reduce_op::plus<result_t>{},
      authorities);

    update_edge_dst_property(handle, graph_view, authorities, curr_dst_auth.mutable_view());

    // Update current source hubs property
    per_v_transform_reduce_outgoing_e(
      handle,
      graph_view,
      edge_src_dummy_property_t{}.view(),
      curr_dst_auth.view(),
      edge_dummy_property_t{}.view(),
      [] __device__(auto src, auto dst, auto, auto curr_dst_auth_value, auto) {
        return curr_dst_auth_value;
      },
      result_t{0},
      reduce_op::plus<result_t>{},
      curr_hubs);

    // Normalize current hub values
    detail::normalize(handle,
                      graph_view,
                      curr_hubs,
                      std::numeric_limits<result_t>::lowest(),
                      reduce_op::maximum<result_t>{});

    // Normalize current authority values
    detail::normalize(handle,
                      graph_view,
                      authorities,
                      std::numeric_limits<result_t>::lowest(),
                      reduce_op::maximum<result_t>{});

    // Test for exit condition
    diff_sum = transform_reduce_v(
      handle,
      graph_view,
      view_concat(curr_hubs, prev_hubs),
      [] __device__(auto, auto val) {
        return std::abs(cuda::std::get<0>(val) - cuda::std::get<1>(val));
      },
      result_t{0});

    update_edge_src_property(handle, graph_view, curr_hubs, prev_src_hubs.mutable_view());

    // After this swap, prev_hubs has the latest hub values
    std::swap(prev_hubs, curr_hubs);
    iter++;

    if (diff_sum < tolerance) {
      break;
    } else if (iter >= max_iterations) {
      break;
    }
  }

  if (normalize) {
    detail::normalize(handle, graph_view, prev_hubs, result_t{0.0}, reduce_op::plus<result_t>{});
    detail::normalize(handle, graph_view, authorities, result_t{0.0}, reduce_op::plus<result_t>{});
  }

  // Copy calculated hubs to the in/out property if the latest values are in the temporary buffer
  if (prev_hubs.value_first() != hubs.value_first()) {
    copy_v(handle, graph_view, prev_hubs, hubs);
  }

  return centrality_algorithm_metadata_t{iter, iter < max_iterations};
}

}  // namespace detail

template <typename vertex_t, typename edge_t, typename result_t, bool multi_gpu>
std::tuple<vertex_property_t<vertex_t, result_t>,
           vertex_property_t<vertex_t, result_t>,
           centrality_algorithm_metadata_t>
hits(raft::handle_t const& handle,
     graph_view_t<vertex_t, edge_t, true, multi_gpu> const& graph_view,
     std::optional<vertex_property_t<vertex_t, result_t>>&& initial_hubs,
     result_t epsilon,
     size_t max_iterations,
     bool normalize,
     bool do_expensive_check)
{
  auto const has_initial_hubs_guess = initial_hubs.has_value();
  auto hubs                         = has_initial_hubs_guess ? std::move(*initial_hubs)
                                                             : vertex_property_t<vertex_t, result_t>(handle, graph_view);
  vertex_property_t<vertex_t, result_t> authorities(handle, graph_view);

  auto metadata = detail::hits(handle,
                               graph_view,
                               hubs.mutable_view(),
                               authorities.mutable_view(),
                               epsilon,
                               max_iterations,
                               has_initial_hubs_guess,
                               normalize,
                               do_expensive_check);

  return std::make_tuple(std::move(hubs), std::move(authorities), metadata);
}

template <typename vertex_t, typename edge_t, typename result_t, bool multi_gpu>
std::tuple<result_t, size_t> hits(raft::handle_t const& handle,
                                  graph_view_t<vertex_t, edge_t, true, multi_gpu> const& graph_view,
                                  result_t* hubs,
                                  result_t* authorities,
                                  result_t epsilon,
                                  size_t max_iterations,
                                  bool has_initial_hubs_guess,
                                  bool normalize,
                                  bool do_expensive_check)
{
  auto const range_size = graph_view.local_vertex_partition_range_size();

  std::optional<vertex_property_t<vertex_t, result_t>> initial_hubs{std::nullopt};
  if (has_initial_hubs_guess) {
    rmm::device_uvector<result_t> hubs_values(static_cast<size_t>(range_size), handle.get_stream());
    thrust::copy(handle.get_thrust_policy(), hubs, hubs + range_size, hubs_values.begin());
    initial_hubs = vertex_property_t<vertex_t, result_t>(std::move(hubs_values), range_size);
  }

  auto [hubs_property, authorities_property, metadata] = hits(handle,
                                                              graph_view,
                                                              std::move(initial_hubs),
                                                              epsilon,
                                                              max_iterations,
                                                              normalize,
                                                              do_expensive_check);

  auto hubs_result        = hubs_property.release_buffer();
  auto authorities_result = authorities_property.release_buffer();
  thrust::copy(handle.get_thrust_policy(), hubs_result.begin(), hubs_result.end(), hubs);
  thrust::copy(
    handle.get_thrust_policy(), authorities_result.begin(), authorities_result.end(), authorities);

  CUGRAPH_EXPECTS(metadata.converged_, "HITS failed to converge.");
  return std::make_tuple(result_t{0}, metadata.number_of_iterations_);
}

}  // namespace cugraph
