/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/prims/copy_v.cuh>
#include <cugraph/prims/count_if_e.cuh>
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

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<vertex_property_t<vertex_t, weight_t>, iterative_algorithm_metadata_t>
eigenvector_centrality(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, true, multi_gpu> const& pull_graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<vertex_property_t<vertex_t, weight_t>>&& initial_centralities,
  weight_t epsilon,
  size_t max_iterations,
  bool do_expensive_check)
{
  static_assert(std::is_integral<vertex_t>::value,
                "GraphViewType::vertex_type should be integral.");
  static_assert(std::is_floating_point<weight_t>::value,
                "weight_t should be a floating-point type.");

  auto const has_initial_guess = initial_centralities.has_value();
  auto local_centralities      = has_initial_guess
                                   ? std::move(*initial_centralities)
                                   : vertex_property_t<vertex_t, weight_t>(handle, pull_graph_view);

  auto const num_vertices = pull_graph_view.number_of_vertices();
  if (num_vertices == 0) {
    return std::make_tuple(std::move(local_centralities), iterative_algorithm_metadata_t{0, true});
  }

  CUGRAPH_EXPECTS(epsilon >= 0.0, "Invalid input argument: epsilon should be non-negative.");

  if (do_expensive_check) {
    if (edge_weight_view) {
      auto num_nonpositive_edge_weights =
        count_if_e(handle,
                   pull_graph_view,
                   edge_src_dummy_property_t{}.view(),
                   edge_dst_dummy_property_t{}.view(),
                   *edge_weight_view,
                   [] __device__(vertex_t, vertex_t, auto, auto, weight_t w) { return w <= 0.0; });
      CUGRAPH_EXPECTS(num_nonpositive_edge_weights == 0,
                      "Invalid input argument: input edge weights should have postive values.");
    }
  }

  if (!has_initial_guess) {
    fill_v(handle,
           pull_graph_view,
           local_centralities.mutable_view(),
           weight_t{1.0} / static_cast<weight_t>(num_vertices));
  }

  auto centralities = local_centralities.mutable_view();
  vertex_property_t<vertex_t, weight_t> old_centralities(handle, pull_graph_view);
  edge_src_property_t<vertex_t, weight_t> edge_src_centralities(handle, pull_graph_view);

  size_t iter{0};
  while (true) {
    copy_v(handle, pull_graph_view, centralities, old_centralities.mutable_view());

    update_edge_src_property(
      handle, pull_graph_view, old_centralities.view(), edge_src_centralities.mutable_view());

    if (edge_weight_view) {
      per_v_transform_reduce_incoming_e(
        handle,
        pull_graph_view,
        edge_src_centralities.view(),
        edge_dst_dummy_property_t{}.view(),
        *edge_weight_view,
        [] __device__(vertex_t, vertex_t, auto src_val, auto, weight_t w) { return src_val * w; },
        weight_t{0},
        reduce_op::plus<weight_t>{},
        centralities);
    } else {
      per_v_transform_reduce_incoming_e(
        handle,
        pull_graph_view,
        edge_src_centralities.view(),
        edge_dst_dummy_property_t{}.view(),
        edge_dummy_property_t{}.view(),
        [] __device__(vertex_t, vertex_t, auto src_val, auto, auto) { return src_val; },
        weight_t{0},
        reduce_op::plus<weight_t>{},
        centralities);
    }

    transform_v(
      handle,
      pull_graph_view,
      view_concat(centralities, old_centralities.view()),
      [] __device__(auto val) { return cuda::std::get<0>(val) + cuda::std::get<1>(val); },
      centralities);

    auto hypotenuse = std::sqrt(transform_reduce_v(
      handle,
      pull_graph_view,
      centralities,
      [] __device__(auto, auto val) { return val * val; },
      weight_t{0.0}));

    transform_v(
      handle, pull_graph_view, centralities, divider_t<weight_t>{hypotenuse}, centralities);

    auto diff_sum = transform_reduce_v(
      handle,
      pull_graph_view,
      view_concat(centralities, old_centralities.view()),
      [] __device__(auto, auto val) {
        return std::abs(cuda::std::get<0>(val) - cuda::std::get<1>(val));
      },
      weight_t{0.0});

    iter++;

    if (diff_sum < (num_vertices * epsilon)) {
      break;
    } else if (iter >= max_iterations) {
      break;
    }
  }

  return std::make_tuple(std::move(local_centralities),
                         iterative_algorithm_metadata_t{iter, iter < max_iterations});
}

}  // namespace detail

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<vertex_property_t<vertex_t, weight_t>, iterative_algorithm_metadata_t>
eigenvector_centrality(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, true, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<vertex_property_t<vertex_t, weight_t>>&& initial_centralities,
  weight_t epsilon,
  size_t max_iterations,
  bool do_expensive_check)
{
  return detail::eigenvector_centrality(handle,
                                        graph_view,
                                        edge_weight_view,
                                        std::move(initial_centralities),
                                        epsilon,
                                        max_iterations,
                                        do_expensive_check);
}

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
rmm::device_uvector<weight_t> eigenvector_centrality(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, true, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<raft::device_span<weight_t const>> initial_centralities,
  weight_t epsilon,
  size_t max_iterations,
  bool do_expensive_check)
{
  auto const range_size = graph_view.local_vertex_partition_range_size();

  std::optional<vertex_property_t<vertex_t, weight_t>> centralities{std::nullopt};
  if (initial_centralities) {
    CUGRAPH_EXPECTS(initial_centralities->size() == static_cast<size_t>(range_size),
                    "Centralities should be same size as vertex range");
    rmm::device_uvector<weight_t> values(static_cast<size_t>(range_size), handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 initial_centralities->begin(),
                 initial_centralities->end(),
                 values.begin());
    centralities = vertex_property_t<vertex_t, weight_t>(std::move(values), range_size);
  }

  auto [result_centralities, metadata] = cugraph::eigenvector_centrality(handle,
                                                                         graph_view,
                                                                         edge_weight_view,
                                                                         std::move(centralities),
                                                                         epsilon,
                                                                         max_iterations,
                                                                         do_expensive_check);

  CUGRAPH_EXPECTS(metadata.converged_, "Eigenvector Centrality failed to converge.");
  return result_centralities.release_buffer();
}

}  // namespace cugraph
