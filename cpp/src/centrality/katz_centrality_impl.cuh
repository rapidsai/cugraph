/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/prims/copy_v.cuh>
#include <cugraph/prims/count_if_v.cuh>
#include <cugraph/prims/fill_v.cuh>
#include <cugraph/prims/per_v_transform_reduce_incoming_outgoing_e.cuh>
#include <cugraph/prims/reduce_op.cuh>
#include <cugraph/prims/transform_reduce_v.cuh>
#include <cugraph/prims/transform_v.cuh>
#include <cugraph/prims/update_edge_src_dst_property.cuh>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/error.hpp>
#include <cugraph/vertex_property.hpp>

#include <raft/core/handle.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/std/tuple>
#include <thrust/copy.h>

#include <cmath>

namespace cugraph {
namespace detail {

template <typename GraphViewType, typename weight_t, typename result_t>
std::tuple<vertex_property_t<typename GraphViewType::vertex_type, result_t>,
           iterative_algorithm_metadata_t>
katz_centrality(
  raft::handle_t const& handle,
  GraphViewType const& pull_graph_view,
  std::optional<edge_property_view_t<typename GraphViewType::edge_type, weight_t const*>>
    edge_weight_view,
  std::optional<vertex_property_view_t<typename GraphViewType::vertex_type, result_t const*>> betas,
  std::optional<vertex_property_t<typename GraphViewType::vertex_type, result_t>>&&
    initial_katz_centralities,
  result_t alpha,
  result_t beta,  // relevant only if betas == std::nullopt
  result_t epsilon,
  size_t max_iterations,
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

  auto const has_initial_guess = initial_katz_centralities.has_value();
  auto local_katz_centralities = has_initial_guess
                                   ? std::move(*initial_katz_centralities)
                                   : vertex_property_t<vertex_t, result_t>(handle, pull_graph_view);

  auto const num_vertices = pull_graph_view.number_of_vertices();
  if (num_vertices == 0) {
    return std::make_tuple(std::move(local_katz_centralities),
                           iterative_algorithm_metadata_t{0, true});
  }

  // 1. check input arguments

  CUGRAPH_EXPECTS((alpha >= 0.0) && (alpha <= 1.0),
                  "Invalid input argument: alpha should be in [0.0, 1.0].");
  CUGRAPH_EXPECTS(epsilon >= 0.0, "Invalid input argument: epsilon should be non-negative.");

  if (has_initial_guess && do_expensive_check) {
    auto num_negative_values = count_if_v(
      handle, pull_graph_view, local_katz_centralities.view(), [] __device__(auto, auto val) {
        return val < 0.0;
      });
    CUGRAPH_EXPECTS(num_negative_values == 0,
                    "Invalid input argument: initial guess values should be non-negative.");
  }

  // 2. initialize katz centrality values

  if (!has_initial_guess) {
    fill_v(handle, pull_graph_view, local_katz_centralities.mutable_view(), result_t{0.0});
  }

  // 3. katz centrality iteration

  auto katz_centralities = local_katz_centralities.mutable_view();
  vertex_property_t<vertex_t, result_t> tmp_katz_centralities(handle, pull_graph_view);
  edge_src_property_t<vertex_t, result_t> edge_src_katz_centralities(handle, pull_graph_view);
  auto new_katz_centralities = katz_centralities;
  auto old_katz_centralities = tmp_katz_centralities.mutable_view();
  size_t iter{0};
  while (true) {
    std::swap(new_katz_centralities, old_katz_centralities);

    update_edge_src_property(
      handle, pull_graph_view, old_katz_centralities, edge_src_katz_centralities.mutable_view());

    if (edge_weight_view) {
      per_v_transform_reduce_incoming_e(
        handle,
        pull_graph_view,
        edge_src_katz_centralities.view(),
        edge_dst_dummy_property_t{}.view(),
        *edge_weight_view,
        [alpha] __device__(vertex_t, vertex_t, auto src_val, auto, weight_t w) {
          return static_cast<result_t>(alpha * src_val * w);
        },
        betas ? result_t{0.0} : beta,
        reduce_op::plus<result_t>{},
        new_katz_centralities);
    } else {
      per_v_transform_reduce_incoming_e(
        handle,
        pull_graph_view,
        edge_src_katz_centralities.view(),
        edge_dst_dummy_property_t{}.view(),
        edge_dummy_property_t{}.view(),
        [alpha] __device__(vertex_t, vertex_t, auto src_val, auto, auto) {
          return static_cast<result_t>(alpha * src_val * 1.0);
        },
        betas ? result_t{0.0} : beta,
        reduce_op::plus<result_t>{},
        new_katz_centralities);
    }

    if (betas) {
      transform_v(
        handle,
        pull_graph_view,
        view_concat(new_katz_centralities, *betas),
        [] __device__(auto val) {
          auto const katz_centrality = cuda::std::get<0>(val);
          auto const beta            = cuda::std::get<1>(val);
          return katz_centrality + beta;
        },
        new_katz_centralities);
    }

    auto diff_sum = transform_reduce_v(
      handle,
      pull_graph_view,
      view_concat(new_katz_centralities, old_katz_centralities),
      [] __device__(auto, auto val) {
        return std::abs(cuda::std::get<0>(val) - cuda::std::get<1>(val));
      },
      result_t{0.0});

    iter++;

    if (diff_sum < epsilon) {
      break;
    } else if (iter >= max_iterations) {
      break;
    }
  }

  if (new_katz_centralities.value_first() != katz_centralities.value_first()) {
    copy_v(handle, pull_graph_view, new_katz_centralities, katz_centralities);
  }

  if (normalize) {
    auto l2_norm = std::sqrt(transform_reduce_v(
      handle,
      pull_graph_view,
      katz_centralities,
      [] __device__(auto, auto val) { return val * val; },
      result_t{0.0}));
    CUGRAPH_EXPECTS(l2_norm > 0.0,
                    "L2 norm of the computed Katz Centrality values should be positive.");
    transform_v(
      handle, pull_graph_view, katz_centralities, divider_t<result_t>{l2_norm}, katz_centralities);
  }

  return std::make_tuple(std::move(local_katz_centralities),
                         iterative_algorithm_metadata_t{iter, iter < max_iterations});
}

}  // namespace detail

template <typename vertex_t, typename edge_t, typename weight_t, typename result_t, bool multi_gpu>
std::tuple<vertex_property_t<vertex_t, result_t>, iterative_algorithm_metadata_t> katz_centrality(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, true, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<vertex_property_view_t<vertex_t, result_t const*>> betas,
  std::optional<vertex_property_t<vertex_t, result_t>>&& initial_katz_centralities,
  result_t alpha,
  result_t beta,
  result_t epsilon,
  size_t max_iterations,
  bool normalize,
  bool do_expensive_check)
{
  return detail::katz_centrality(handle,
                                 graph_view,
                                 edge_weight_view,
                                 betas,
                                 std::move(initial_katz_centralities),
                                 alpha,
                                 beta,
                                 epsilon,
                                 max_iterations,
                                 normalize,
                                 do_expensive_check);
}

template <typename vertex_t, typename edge_t, typename weight_t, typename result_t, bool multi_gpu>
void katz_centrality(raft::handle_t const& handle,
                     graph_view_t<vertex_t, edge_t, true, multi_gpu> const& graph_view,
                     std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
                     result_t const* betas,
                     result_t* katz_centralities,
                     result_t alpha,
                     result_t beta,
                     result_t epsilon,
                     size_t max_iterations,
                     bool has_initial_guess,
                     bool normalize,
                     bool do_expensive_check)
{
  auto const range_size = graph_view.local_vertex_partition_range_size();

  std::optional<vertex_property_view_t<vertex_t, result_t const*>> betas_view{std::nullopt};
  if (betas != nullptr) {
    betas_view = vertex_property_view_t<vertex_t, result_t const*>{betas, range_size};
  }

  std::optional<vertex_property_t<vertex_t, result_t>> initial_katz_centralities{std::nullopt};
  if (has_initial_guess) {
    rmm::device_uvector<result_t> values(static_cast<size_t>(range_size), handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 katz_centralities,
                 katz_centralities + range_size,
                 values.begin());
    initial_katz_centralities =
      vertex_property_t<vertex_t, result_t>(std::move(values), range_size);
  }

  auto [result_katz_centralities, metadata] =
    cugraph::katz_centrality(handle,
                             graph_view,
                             edge_weight_view,
                             betas_view,
                             std::move(initial_katz_centralities),
                             alpha,
                             beta,
                             epsilon,
                             max_iterations,
                             normalize,
                             do_expensive_check);

  auto katz_centrality_values = result_katz_centralities.release_buffer();
  thrust::copy(handle.get_thrust_policy(),
               katz_centrality_values.begin(),
               katz_centrality_values.end(),
               katz_centralities);

  CUGRAPH_EXPECTS(metadata.converged_, "Katz Centrality failed to converge.");
}

}  // namespace cugraph
