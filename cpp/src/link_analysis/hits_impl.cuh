/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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
#include <prims/fill_edge_src_dst_property.cuh>
#include <prims/per_v_transform_reduce_incoming_outgoing_e.cuh>
#include <prims/reduce_op.cuh>
#include <prims/reduce_v.cuh>
#include <prims/transform_reduce_v.cuh>
#include <prims/update_edge_src_dst_property.cuh>

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_view.hpp>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

namespace cugraph {
namespace detail {
template <typename GraphViewType, typename ReduceOp, typename result_t>
void normalize(raft::handle_t const& handle,
               GraphViewType const& graph_view,
               result_t* hubs,
               result_t init,
               ReduceOp reduce_op)
{
  auto hubs_norm = reduce_v(handle, graph_view, hubs, init, reduce_op);
  CUGRAPH_EXPECTS(hubs_norm > 0, "Norm is required to be a positive value.");
  thrust::transform(handle.get_thrust_policy(),
                    hubs,
                    hubs + graph_view.local_vertex_partition_range_size(),
                    thrust::make_constant_iterator(hubs_norm),
                    hubs,
                    thrust::divides<result_t>());
}

template <typename GraphViewType, typename result_t>
std::tuple<result_t, size_t> hits(raft::handle_t const& handle,
                                  GraphViewType const& graph_view,
                                  result_t* const hubs,
                                  result_t* const authorities,
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
  size_t final_iteration_count{max_iterations};

  if (num_vertices == 0) { return std::make_tuple(diff_sum, final_iteration_count); }

  CUGRAPH_EXPECTS(epsilon >= 0.0, "Invalid input argument: epsilon should be non-negative.");

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
  edge_src_property_t<GraphViewType, result_t> prev_src_hubs(handle, graph_view);
  edge_dst_property_t<GraphViewType, result_t> curr_dst_auth(handle, graph_view);
  rmm::device_uvector<result_t> temp_hubs(graph_view.local_vertex_partition_range_size(),
                                          handle.get_stream());

  result_t* prev_hubs = hubs;
  result_t* curr_hubs = temp_hubs.data();

  // Initialize hubs from user input if provided
  if (has_initial_hubs_guess) {
    update_edge_src_property(handle, graph_view, prev_hubs, prev_src_hubs);
  } else {
    fill_edge_src_property(handle, graph_view, result_t{1.0} / num_vertices, prev_src_hubs);
    thrust::fill(handle.get_thrust_policy(),
                 prev_hubs,
                 prev_hubs + graph_view.local_vertex_partition_range_size(),
                 result_t{1.0} / num_vertices);
  }
  for (size_t iter = 0; iter < max_iterations; ++iter) {
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

    update_edge_dst_property(handle, graph_view, authorities, curr_dst_auth);

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
      thrust::make_zip_iterator(thrust::make_tuple(curr_hubs, prev_hubs)),
      [] __device__(auto, auto val) { return std::abs(thrust::get<0>(val) - thrust::get<1>(val)); },
      result_t{0});
    if (diff_sum < epsilon) {
      final_iteration_count = iter;
      std::swap(prev_hubs, curr_hubs);
      break;
    }

    update_edge_src_property(handle, graph_view, curr_hubs, prev_src_hubs);

    // Swap pointers for the next iteration
    // After this swap call, prev_hubs has the latest value of hubs
    std::swap(prev_hubs, curr_hubs);
  }

  if (normalize) {
    detail::normalize(handle, graph_view, prev_hubs, result_t{0.0}, reduce_op::plus<result_t>{});
    detail::normalize(handle, graph_view, authorities, result_t{0.0}, reduce_op::plus<result_t>{});
  }

  // Copy calculated hubs to in/out parameter if necessary
  if (hubs != prev_hubs) {
    thrust::copy(handle.get_thrust_policy(),
                 prev_hubs,
                 prev_hubs + graph_view.local_vertex_partition_range_size(),
                 hubs);
  }

  return std::make_tuple(diff_sum, final_iteration_count);
}

}  // namespace detail

template <typename vertex_t, typename edge_t, typename result_t, bool multi_gpu>
std::tuple<result_t, size_t> hits(raft::handle_t const& handle,
                                  graph_view_t<vertex_t, edge_t, true, multi_gpu> const& graph_view,
                                  result_t* const hubs,
                                  result_t* const authorities,
                                  result_t epsilon,
                                  size_t max_iterations,
                                  bool has_initial_hubs_guess,
                                  bool normalize,
                                  bool do_expensive_check)
{
  return detail::hits(handle,
                      graph_view,
                      hubs,
                      authorities,
                      epsilon,
                      max_iterations,
                      has_initial_hubs_guess,
                      normalize,
                      do_expensive_check);
}

}  // namespace cugraph
