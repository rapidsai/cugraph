/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include <prims/count_if_v.cuh>
#include <prims/per_v_transform_reduce_incoming_outgoing_e.cuh>
#include <prims/reduce_op.cuh>
#include <prims/transform_reduce_v.cuh>
#include <prims/update_edge_src_dst_property.cuh>

#include <cugraph/algorithms.hpp>
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

template <typename GraphViewType, typename weight_t, typename result_t>
void katz_centrality(
  raft::handle_t const& handle,
  GraphViewType const& pull_graph_view,
  std::optional<edge_property_view_t<typename GraphViewType::edge_type, weight_t const*>>
    edge_weight_view,
  result_t const* betas,
  result_t* katz_centralities,
  result_t alpha,
  result_t beta,  // relevant only if betas == nullptr
  result_t epsilon,
  size_t max_iterations,
  bool has_initial_guess,
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

  auto const num_vertices = pull_graph_view.number_of_vertices();
  if (num_vertices == 0) { return; }

  // 1. check input arguments

  CUGRAPH_EXPECTS((alpha >= 0.0) && (alpha <= 1.0),
                  "Invalid input argument: alpha should be in [0.0, 1.0].");
  CUGRAPH_EXPECTS(epsilon >= 0.0, "Invalid input argument: epsilon should be non-negative.");

  if (do_expensive_check) {
    // FIXME: should I check for betas?

    if (has_initial_guess) {
      auto num_negative_values =
        count_if_v(handle, pull_graph_view, katz_centralities, [] __device__(auto, auto val) {
          return val < 0.0;
        });
      CUGRAPH_EXPECTS(num_negative_values == 0,
                      "Invalid input argument: initial guess values should be non-negative.");
    }
  }

  // 2. initialize katz centrality values

  if (!has_initial_guess) {
    thrust::fill(handle.get_thrust_policy(),
                 katz_centralities,
                 katz_centralities + pull_graph_view.local_vertex_partition_range_size(),
                 result_t{0.0});
  }

  // 3. katz centrality iteration

  // old katz centrality values
  rmm::device_uvector<result_t> tmp_katz_centralities(
    pull_graph_view.local_vertex_partition_range_size(), handle.get_stream());
  edge_src_property_t<GraphViewType, result_t> edge_src_katz_centralities(handle, pull_graph_view);
  auto new_katz_centralities = katz_centralities;
  auto old_katz_centralities = tmp_katz_centralities.data();
  size_t iter{0};
  while (true) {
    std::swap(new_katz_centralities, old_katz_centralities);

    update_edge_src_property(
      handle, pull_graph_view, old_katz_centralities, edge_src_katz_centralities);

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
        betas != nullptr ? result_t{0.0} : beta,
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
        betas != nullptr ? result_t{0.0} : beta,
        reduce_op::plus<result_t>{},
        new_katz_centralities);
    }

    if (betas != nullptr) {
      auto val_first = thrust::make_zip_iterator(thrust::make_tuple(new_katz_centralities, betas));
      thrust::transform(handle.get_thrust_policy(),
                        val_first,
                        val_first + pull_graph_view.local_vertex_partition_range_size(),
                        new_katz_centralities,
                        [] __device__(auto val) {
                          auto const katz_centrality = thrust::get<0>(val);
                          auto const beta            = thrust::get<1>(val);
                          return katz_centrality + beta;
                        });
    }

    auto diff_sum = transform_reduce_v(
      handle,
      pull_graph_view,
      thrust::make_zip_iterator(thrust::make_tuple(new_katz_centralities, old_katz_centralities)),
      [] __device__(auto, auto val) { return std::abs(thrust::get<0>(val) - thrust::get<1>(val)); },
      result_t{0.0});

    iter++;

    if (diff_sum < epsilon) {
      break;
    } else if (iter >= max_iterations) {
      CUGRAPH_FAIL("Katz Centrality failed to converge.");
    }
  }

  if (new_katz_centralities != katz_centralities) {
    thrust::copy(handle.get_thrust_policy(),
                 new_katz_centralities,
                 new_katz_centralities + pull_graph_view.local_vertex_partition_range_size(),
                 katz_centralities);
  }

  if (normalize) {
    auto l2_norm = transform_reduce_v(
      handle,
      pull_graph_view,
      katz_centralities,
      [] __device__(auto, auto val) { return val * val; },
      result_t{0.0});
    l2_norm = std::sqrt(l2_norm);
    CUGRAPH_EXPECTS(l2_norm > 0.0,
                    "L2 norm of the computed Katz Centrality values should be positive.");
    thrust::transform(handle.get_thrust_policy(),
                      katz_centralities,
                      katz_centralities + pull_graph_view.local_vertex_partition_range_size(),
                      katz_centralities,
                      [l2_norm] __device__(auto val) { return val / l2_norm; });
  }
}

}  // namespace detail

template <typename vertex_t, typename edge_t, typename weight_t, typename result_t, bool multi_gpu>
void katz_centrality(raft::handle_t const& handle,
                     graph_view_t<vertex_t, edge_t, true, multi_gpu> const& graph_view,
                     std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
                     result_t const* betas,
                     result_t* katz_centralities,
                     result_t alpha,
                     result_t beta,  // relevant only if beta == nullptr
                     result_t epsilon,
                     size_t max_iterations,
                     bool has_initial_guess,
                     bool normalize,
                     bool do_expensive_check)
{
  detail::katz_centrality(handle,
                          graph_view,
                          edge_weight_view,
                          betas,
                          katz_centralities,
                          alpha,
                          beta,
                          epsilon,
                          max_iterations,
                          has_initial_guess,
                          normalize,
                          do_expensive_check);
}

}  // namespace cugraph
