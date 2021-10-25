/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include <cugraph/prims/row_col_properties.cuh>

namespace cugraph {
namespace detail {
}  // namespace detail

template <typename GraphViewType, typename result_t>
std::tuple<rmm::device_uvector<result_t>,  // hubs
           rmm::device_uvector<result_t>,  // authorities
           result_t,                       // error
           size_t>                         // iteration count
hits(raft::handle_t const& handle,
     GraphViewType const& graph_view,
     size_t max_iterations,  //  = 500,
     result_t epsilon,
     std::optional<result_t const*> starting_hub_values,
     bool normalized)
{  //       = true,
  using vertex_t = typename GraphViewType::vertex_type;
  static_assert(std::is_integral<vertex_t>::value,
                "GraphViewType::vertex_type should be integral.");
  static_assert(std::is_floating_point<result_t>::value,
                "result_t should be a floating-point type.");
  static_assert(GraphViewType::is_adj_matrix_transposed,
                "GraphViewType should support the pull model.");
  auto const num_vertices = graph_view.get_number_of_vertices();
  if (num_vertices == 0) { return; }

  CUGRAPH_EXPECTS(epsilon >= 0.0, "Invalid input argument: epsilon should be non-negative.");

  result_t hubs_error;
  size_t final_iteration_count{max_iterations};
  // Property wrappers
  row_properties_t<GraphViewType, result_t> prev_src_hubs(handle, graph_view);
  rmm::device_uvector<result_t> curr_authorities(graph_view.get_number_of_local_vertices(),
                                                 handle.get_stream());
  col_properties_t<GraphViewType, result_t> curr_dst_auth(handle, graph_view);
  rmm::device_uvector<result_t> prev_hubs(graph_view.get_number_of_local_vertices(),
                                          handle.get_stream());
  rmm::device_uvector<result_t> curr_hubs(graph_view.get_number_of_local_vertices(),
                                          handle.get_stream());

  if (starting_hub_values) {
    // TODO:
    // starting_hub_values are not guaranteed to be normalized to begin with
    // copy to prev_hubs and normalize them
    // TODO:
    // add check to make sure starting_hub_values are not negative
    copy_to_adj_matrix_row(handle, graph_view, (*starting_hub_values), prev_src_hubs);
    raft::copy_async(prev_hubs.data(),
                     (*starting_hub_values),
                     graph_view.get_number_of_local_vertices(),
                     handle.get_stream());
  } else {
    prev_src_hubs.fill(result_t{1.0} / num_vertices, handle.get_stream());
    thrust::fill(
      handle.get_stream(), prev_hubs.begin(), prev_hubs.end(), result_t{1.0} / num_vertices);
  }
  for (size_t iter = 0; iter < max_iterations; ++iter) {
    // Update current destination authorities property
    copy_v_transform_reduce_out_nbr(
      handle,
      graph_view,
      prev_src_hubs.device_view(),
      dummy_properties_t<result_t>{}.device_view(),
      [] __device__(auto, auto, auto, auto prev_src_hub_value, auto) { return prev_src_hub_value; },
      result_t{0},
      curr_authorities.data());

    copy_to_adj_matrix_col(handle, graph_view, curr_authorities.data(), curr_dst_auth);

    // Update current source hubs property
    copy_v_transform_reduce_in_nbr(
      handle,
      graph_view,
      curr_dst_auth.device_view(),
      dummy_properties_t<result_t>{}.device_view(),
      [] __device__(auto, auto, auto, auto, auto curr_dst_auth_value) {
        return curr_dst_auth_value;
      },
      result_t{0},
      curr_hubs.data());

    // Normalize current hub values
    if (normalized) {
      auto hubs_total =
        reduce_v(handle, graph_view, curr_hubs.begin(), curr_hubs.end(), result_t{});
      thrust::transform(handle.get_stream(),
                        curr_hubs.begin(),
                        curr_hubs.end(),
                        thrust::make_constant_iterator(hubs_total),
                        curr_hubs.begin(),
                        thrust::divides<result_t>());
    } else {
      // TODO : Divide by max - after PR 1902
    }

    // Test for exit condition
    hubs_error = transform_reduce_v(
      handle,
      graph_view,
      thrust::make_zip_iterator(thrust::make_tuple(curr_hubs.begin(), prev_hubs.begin())),
      [] __device__(auto val) { return std::abs(thrust::get<0>(val) - thrust::get<1>(val)); },
      result_t{0.0});
    if (hubs_error < epsilon) {
      final_iteration_count = iter;
      break;
    }

    copy_to_adj_matrix_row(handle, graph_view, curr_hubs.data(), prev_src_hubs);

    std::swap(prev_hubs, curr_hubs);
  }

  return std::make_tuple(
    std::move(curr_hubs), std::move(curr_authorities), hubs_error, final_iteration_count);
}

}  // namespace cugraph
