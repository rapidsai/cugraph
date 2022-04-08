/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <cugraph/algorithms.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/prims/copy_v_transform_reduce_in_out_nbr.cuh>
#include <cugraph/prims/count_if_v.cuh>
#include <cugraph/prims/edge_partition_src_dst_property.cuh>
#include <cugraph/prims/reduce_v.cuh>
#include <cugraph/prims/transform_reduce_v.cuh>
#include <cugraph/prims/update_edge_partition_src_dst_property.cuh>

#include <thrust/fill.h>
#include <thrust/transform.h>

namespace cugraph {
namespace detail {
template <typename GraphViewType, typename result_t>
void normalize(raft::handle_t const& handle,
               GraphViewType const& graph_view,
               result_t* hubs,
               raft::comms::op_t op)
{
  auto hubs_norm = reduce_v(handle,
                            graph_view,
                            hubs,
                            hubs + graph_view.local_vertex_partition_range_size(),
                            identity_element<result_t>(op),
                            op);
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
      count_if_v(handle, graph_view, hubs, [] __device__(auto val) { return val < 0.0; });
    CUGRAPH_EXPECTS(num_negative_values == 0,
                    "Invalid input argument: initial guess values should be non-negative.");
  }

  if (has_initial_hubs_guess) {
    detail::normalize(handle, graph_view, hubs, raft::comms::op_t::SUM);
  }

  // Property wrappers
  edge_partition_src_property_t<GraphViewType, result_t> prev_src_hubs(handle, graph_view);
  edge_partition_dst_property_t<GraphViewType, result_t> curr_dst_auth(handle, graph_view);
  rmm::device_uvector<result_t> temp_hubs(graph_view.local_vertex_partition_range_size(),
                                          handle.get_stream());

  result_t* prev_hubs = hubs;
  result_t* curr_hubs = temp_hubs.data();

  // Initialize hubs from user input if provided
  if (has_initial_hubs_guess) {
    update_edge_partition_src_property(handle, graph_view, prev_hubs, prev_src_hubs);
  } else {
    prev_src_hubs.fill(result_t{1.0} / num_vertices, handle.get_stream());
    thrust::fill(handle.get_thrust_policy(),
                 prev_hubs,
                 prev_hubs + graph_view.local_vertex_partition_range_size(),
                 result_t{1.0} / num_vertices);
  }
  for (size_t iter = 0; iter < max_iterations; ++iter) {
    // Update current destination authorities property
    copy_v_transform_reduce_in_nbr(
      handle,
      graph_view,
      prev_src_hubs.device_view(),
      dummy_property_t<result_t>{}.device_view(),
      [] __device__(auto, auto, auto, auto prev_src_hub_value, auto) { return prev_src_hub_value; },
      result_t{0},
      authorities);

    update_edge_partition_dst_property(handle, graph_view, authorities, curr_dst_auth);

    // Update current source hubs property
    copy_v_transform_reduce_out_nbr(
      handle,
      graph_view,
      dummy_property_t<result_t>{}.device_view(),
      curr_dst_auth.device_view(),
      [] __device__(auto src, auto dst, auto, auto, auto curr_dst_auth_value) {
        return curr_dst_auth_value;
      },
      result_t{0},
      curr_hubs);

    // Normalize current hub values
    detail::normalize(handle, graph_view, curr_hubs, raft::comms::op_t::MAX);

    // Normalize current authority values
    detail::normalize(handle, graph_view, authorities, raft::comms::op_t::MAX);

    // Test for exit condition
    diff_sum = transform_reduce_v(
      handle,
      graph_view,
      thrust::make_zip_iterator(thrust::make_tuple(curr_hubs, prev_hubs)),
      [] __device__(auto val) { return std::abs(thrust::get<0>(val) - thrust::get<1>(val)); },
      result_t{0});
    if (diff_sum < epsilon) {
      final_iteration_count = iter;
      std::swap(prev_hubs, curr_hubs);
      break;
    }

    update_edge_partition_src_property(handle, graph_view, curr_hubs, prev_src_hubs);

    // Swap pointers for the next iteration
    // After this swap call, prev_hubs has the latest value of hubs
    std::swap(prev_hubs, curr_hubs);
  }

  if (normalize) {
    detail::normalize(handle, graph_view, prev_hubs, raft::comms::op_t::SUM);
    detail::normalize(handle, graph_view, authorities, raft::comms::op_t::SUM);
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

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<weight_t, size_t> hits(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, weight_t, true, multi_gpu> const& graph_view,
  weight_t* const hubs,
  weight_t* const authorities,
  weight_t epsilon,
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
