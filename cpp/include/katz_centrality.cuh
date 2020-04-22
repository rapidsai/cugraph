/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <patterns.hpp>

#include <rmm/rmm.h>

#include <thrust/fill.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>


namespace cugraph {
namespace experimental {

template <typename GraphType, typename VertexIterator, typename ResultIterator>
void katz_centrality(
    GraphType const& graph,
    ResultIteraotr dst_beta_first, ResultIteraotr dst_katz_centrality_first,
    double alpha = 0.1, double epsilon = 1e-5, size_t max_iterations = 500,
    bool has_initial_guess = false, normalize = true) {
  using vertex_t = typename std::iterator_traits<VertexIterator>::value_type;
  using result_t = typename std::iterator_traits<ResultIterator>::value_type;
  static_assert(
    std::is_integral<vertex_t>::value,
    "VertexIterator should point to an integral value.");
  static_assert(
    std::is_floating_point<result_t>::value,
    "ResultIterator should point to a floating-point value.");
  static_assert(
    is_csc<GraphType>::value,
    "cugraph::experimental::katz_centrality expects a CSC graph.");

  CUGRAPH_EXPECTS(
    graph.is_directed(), "cugraph::experimental::katz_centrality expects a directed graph.");

  auto const num_vertices = graph.get_number_of_vertices();
  vertex_t src_vertex_first{};
  vertex_t src_vertex_last{};
  vertex_t dst_vertex_first{};
  vertex_t dst_vertex_last{};
  std::tie(src_vertex_first, src_vertex_last) = graph.get_this_src_vertex_range();
  std::tie(dst_vertex_first, dst_vertex_last) = graph.get_this_dst_vertex_range();
  auto num_src_vertices = src_vertex_last - src_vertex_first;
  auto num_dst_vertices = dst_vertex_last - dst_vertex_first;

  if (!has_initial_guess) {
    thrust::fill(
      dst_katz_centrality_first, dst_katz_centrality_first + num_dst_vertices,
      static_cast<result_t>(0.0));
  }

  rmm::device_vector<result_t> src_katz_centralities(num_src_vertices, static_cast<result_t>(0.0));

  size_t iter = 0;
  while (true) {
    copy_dst_values_to_src(
      graph, dst_katz_centrality_first, src_katz_centralities.begin());

    if (graph.is_weighted()) {
      transform_dst_v_transform_reduce_e(
        graph,
        src_katz_centrality_first, thrust::make_constant_iterator(0)/* dummy */,
        dst_katz_centrality_first,
        [alpha] __device__ (auto src_val, auto dst_val, weight_t w) {
          return alpha * src_val * w;
        },
        static_cast<result_t>(0.0));
    }
    else {
      transform_dst_v_transform_reduce_e(
        graph,
        src_katz_centrality_first, thrust::make_constant_iterator(0)/* dummy */,
        dst_katz_centrality_first,
        [alpha] __device__ (auto src_val, auto dst_val) {
          return alpha * src_val * static_cast<result_t>(1.0);
        },
        unvarying_part);
    }
    auto dst_val_first =
      thrust::make_zip_iterator(thrust::make_tuple(dst_katz_centrality_first, dst_beta_first));
    thrust::transform(
      dst_val_first, dst_val_first + num_dst_vertices, dst_katz_centrality_first,
      [] __device__ (auto val) {
        auto const katz_centrality = thrust::get<0>(val);
        auto const beta = thrust::get<1>(val);
        return katz_centrality + beta;
      });

    auto diff_sum =
      transform_reduce_src_dst_v(
        graph, src_katz_centralities.begin(), dst_katz_centrality_first,
        [] __device__ (auto src_val, auto dst_val) {
          return std::abs(dst_val - src_val);
        },
        static_cast<result_t>(0.0));

    iter++;

    if (diff_sum < static_cast<result_t>(num_verticse) * static_cast<result_t>(epsilon)) {
      break;
    }
    else if(iter > max_iters) {
      CUGRAPH_FAIL("Katz Centrality failed to converge.");
    }
  }

  if (normalize) {
    auto l2_norm =
      transform_reduce_dst_v(
        graph, dst_katz_centrality_first,
        [] __device__ (auto val) {
          return val * val;
        });
    l2_norm = std::sqrt(l2_norm);
    CUGRAPH_EXPECTS(
      l2_norm > 0.0, "L2 norm of the computed Katz Centrality values should be positive.");
    thrust::transform(
      dst_katz_centrality_first, dst_katz_centrality_first + num_dst_vertices,
      dst_katz_centrality_first,
      [l2_norm] __device__ (auto val) {
        return val / l2_norm;
      });
  }

  return;
}

}  // namespace cugraph
}  // namespace experimental
