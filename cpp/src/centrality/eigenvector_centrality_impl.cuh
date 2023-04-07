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

#include <prims/count_if_e.cuh>
#include <prims/count_if_v.cuh>
#include <prims/per_v_transform_reduce_incoming_outgoing_e.cuh>
#include <prims/reduce_op.cuh>
#include <prims/reduce_v.cuh>
#include <prims/transform_reduce_v.cuh>
#include <prims/update_edge_src_dst_property.cuh>

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/core/handle.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

namespace cugraph {
namespace detail {

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
rmm::device_uvector<weight_t> eigenvector_centrality(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, true, multi_gpu> const& pull_graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<raft::device_span<weight_t const>> initial_centralities,
  weight_t epsilon,
  size_t max_iterations,
  bool do_expensive_check)
{
  using GraphViewType     = graph_view_t<vertex_t, edge_t, true, multi_gpu>;
  auto const num_vertices = pull_graph_view.number_of_vertices();
  if (num_vertices == 0) { return rmm::device_uvector<weight_t>(0, handle.get_stream()); }

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

  rmm::device_uvector<weight_t> centralities(pull_graph_view.local_vertex_partition_range_size(),
                                             handle.get_stream());
  if (initial_centralities) {
    thrust::copy(handle.get_thrust_policy(),
                 initial_centralities->begin(),
                 initial_centralities->end(),
                 centralities.begin());
  } else {
    thrust::fill(handle.get_thrust_policy(),
                 centralities.begin(),
                 centralities.end(),
                 weight_t{1.0} / static_cast<weight_t>(num_vertices));
  }

  // Power iteration
  rmm::device_uvector<weight_t> old_centralities(centralities.size(), handle.get_stream());

  edge_src_property_t<GraphViewType, weight_t> edge_src_centralities(handle, pull_graph_view);

  size_t iter{0};
  while (true) {
    thrust::copy(handle.get_thrust_policy(),
                 centralities.begin(),
                 centralities.end(),
                 old_centralities.data());

    update_edge_src_property(handle, pull_graph_view, centralities.begin(), edge_src_centralities);

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
        centralities.begin());
    } else {
      per_v_transform_reduce_incoming_e(
        handle,
        pull_graph_view,
        edge_src_centralities.view(),
        edge_dst_dummy_property_t{}.view(),
        edge_dummy_property_t{}.view(),
        [] __device__(vertex_t, vertex_t, auto src_val, auto, auto) { return src_val * 1.0; },
        weight_t{0},
        reduce_op::plus<weight_t>{},
        centralities.begin());
    }

    // Normalize the centralities
    auto hypotenuse = sqrt(transform_reduce_v(
      handle,
      pull_graph_view,
      centralities.begin(),
      [] __device__(auto, auto val) { return val * val; },
      weight_t{0.0}));

    thrust::transform(handle.get_thrust_policy(),
                      centralities.begin(),
                      centralities.end(),
                      centralities.begin(),
                      [hypotenuse] __device__(auto val) { return val / hypotenuse; });

    auto diff_sum = transform_reduce_v(
      handle,
      pull_graph_view,
      thrust::make_zip_iterator(thrust::make_tuple(centralities.begin(), old_centralities.data())),
      [] __device__(auto, auto val) { return std::abs(thrust::get<0>(val) - thrust::get<1>(val)); },
      weight_t{0.0});

    iter++;

    if (diff_sum < (pull_graph_view.number_of_vertices() * epsilon)) {
      break;
    } else if (iter >= max_iterations) {
      CUGRAPH_FAIL("Eigenvector Centrality failed to converge.");
    }
  }

  return centralities;
}

}  // namespace detail

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
  static_assert(std::is_integral<vertex_t>::value,
                "GraphViewType::vertex_type should be integral.");
  static_assert(std::is_floating_point<weight_t>::value,
                "weight_t should be a floating-point type.");

  CUGRAPH_EXPECTS(epsilon >= 0.0, "Invalid input argument: epsilon should be non-negative.");
  if (initial_centralities)
    CUGRAPH_EXPECTS(initial_centralities->size() ==
                      static_cast<size_t>(graph_view.local_vertex_partition_range_size()),
                    "Centralities should be same size as vertex range");

  return detail::eigenvector_centrality(handle,
                                        graph_view,
                                        edge_weight_view,
                                        initial_centralities,
                                        epsilon,
                                        max_iterations,
                                        do_expensive_check);
}

}  // namespace cugraph
