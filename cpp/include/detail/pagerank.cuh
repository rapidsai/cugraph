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
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>


namespace cugraph {
namespace experimental {
namespace detail {

template <typename GraphType, typename VertexIterator, typename ResultIterator>
void pagerank_this_graph_partition(
    raft::Handle handle, GraphType const& graph,
    VertexIterator src_out_degree_first,
    VertexIterator dst_personalization_vertex_first, VertexIterator dst_personalization_vertex_last,
    ResultIteraotr dst_personalization_value_first,
    ResultIteraotr dst_pagerank_first,
    double alpha = 0.85, double epsilon = 1e-5, size_t max_iterations = 500,
    bool has_initial_guess = false, bool is_personalized = false) {
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
    "cugraph::experimental::pagerank_this_graph_partition expects a CSC graph.");

  CUGRAPH_EXPECTS(
    graph.is_directed(),
    "cugraph::experimental::pagerank_this_graph_partition expects a directed graph.");

  auto const num_vertices = graph.get_number_of_vertices();
  vertex_t src_vertex_first{};
  vertex_t src_vertex_last{};
  vertex_t dst_vertex_first{};
  vertex_t dst_vertex_last{};
  std::tie(src_vertex_first, src_vertex_last) = graph.get_this_src_vertex_range();
  std::tie(dst_vertex_first, dst_vertex_last) = graph.get_this_dst_vertex_range();
  auto num_src_vertices = src_vertex_last - src_vertex_first;
  auto num_dst_vertices = dst_vertex_last - dst_vertex_first;

  rmm::device_vector<result_t> src_out_weight_sums{};
  if (graph.is_weighted()) {
    src_out_weight_sums.assign(num_src_vertices, static_cast<result_t>(0.0));
    transform_src_v_transform_reduce_e(
      handle, graph,
      thrust::make_constant_iterator(0),  // dummy
      thrust::make_constant_iterator(0),  // dummy
      src_out_weight_sums.begin(),
      [] __device__ (auto src_val, dst_val, weight_t w) {
        return w;
      },
      static_cast<result_t>(0.0));
  }

  if (has_initial_guess) {
    auto sum =
      reduce_dst_v(handle, graph, dst_pagerank_first, static_cast<result_t>(0.0));
    CUGRAPH_EXPECTS(sum > 0.0, "Sum of the PageRank initial guess values should be positive.");
    thrust::transform(
      dst_pagerank_first, dst_pagerank_first + num_dst_vertices, dst_pagerank_first,
      [sum] __device__ (auto val) { return val / sum; });
  }
  else {
    thrust::fill(
      dst_pagerank_first, dst_pagerank_first + num_dst_vertices,
      static_cast<result_t>(1.0) / static_cast<result_t>(num_vertices));
  }

  result_t personalization_sum{0.0};
  if (is_personalized) {
    personalization_sum =
      reduce_dst_v(
        handle, graph, dst_personalization_value_first, static_cast<result_t>(0.0));
    CUGRAPH_EXPECTS(
      personalization_sum > 0.0, "Sum of personalization valuese should be positive.");
  }

  rmm::device_vector<result_t> src_pageranks(num_src_vertices, static_cast<result_t>(0.0));

  size_t iter = 0;
  while (true) {
    copy_dst_values_to_src(handle, graph, dst_pagerank_first, src_pageranks.begin());

    if (graph.is_weighted()) {
      auto src_val_first =
        thrust::make_zip_iterator(
          thrust::make_tuple(
            src_pageranks.begin(), src_out_degree_first, src_out_weight_sums.begin()));
      thrust::transform(
        src_val_first, src_val_first + num_src_vertices, src_pageranks.begin(),
        [] __device__ (auto val) {
          auto const src_pagerank = thrust::get<0>(val);
          auto const out_degree = thrust::get<1>(val);
          auto const out_weight_sum = thrust::get<2>(val);
          auto const divisor =
            out_degree == 0 ? static_cast<result_t>(1.0) : out_weight_sum;
          return src_pagerank / divisor;
        });
    }
    else {
      auto src_val_first =
        thrust::make_zip_iterator(thrust::make_tuple(src_pageranks.begin(), src_out_degree_first));
      thrust::transform(
        src_val_first, src_val_first + num_src_vertices, src_pageranks.begin(),
        [] __device__ (auto val) {
          auto const src_pagerank = thrust::get<0>(val);
          auto const out_degree = thrust::get<1>(val);
          auto const divisor =
            out_degree == 0 ? static_cast<result_t>(1.0) : static_cast<result_t>(out_degree);
          return src_pagerank / divisor;
        });
    }
    auto src_val_first =
      thrust::make_zip_iterator(thrust::make_tuple(src_pageranks.begin(), src_out_degree_first));
    auto dangling_sum =
      cguraph::transform_reduce_src_v(
        handle, graph, src_val_first,
        [] __device__ (auto val) {
          auto const src_pagerank = thrust::get<0>(val);
          auto const out_degree = thrust::get<1>(val);
          return out_degree == 0 ? src_pagerank : static_cast<result_t>(0.0);
        },
        static_cast<result_t>(0.0));

    auto unvarying_part{0.0};
    if (!is_personalized) {
      unvarying_part =
        (static_cast<result_t>(1.0 - alpha)) / static_cast<result_t>(num_vertices) +
        static_cast<result_t>(alpha) * (dangling_sum / static_cast<result_t>(num_vertices));
    }
    transform_dst_v_transform_reduce_e(
      handle, graph,
      src_pageranks.begin(), thrust::make_constant_iterator(0)/* dummy */, dst_pagerank_first,
      [damping_factor] __device__ (auto src_val, auto dst_val) {
        return src_val * damping_factor;
      },
      unvarying_part);
    if (is_personalized) {
      auto dst_val_first =
        thrust::make_zip_iterator(
          thrust::make_tuple(dst_personalization_vertex_first, dst_personalization_value_first));
      thrust::for_each(
        dst_val_first, dst_val_first + num_dst_vertices,
        [dangling_sum, personalization_sum, dst_pagerank_first] __device__ (auto val) {
          auto dst_v = thrust::get<0>(val);
          auto dst_value = thrust::get<1>(val);
          *(dst_pagerank_first + (dst_v - dst_vertex_first)) +=
            (dangling_sum + static_cast<result_t>(1.0 - alpha)) * (dst_value / personalization_sum);
        });
    }

    auto diff_sum =
      transform_reduce_src_dst_v(
        handle, graph, src_pageranks.begin(), dst_pagerank_first,
        [] __device__ (auto src_val, auto dst_val) {
          return std::abs(dst_val - src_val);
        });

    iter++;

    if (diff_sum < static_cast<result_t>(num_verticse) * static_cast<result_t>(epsilon)) {
      break;
    }
    else if(iter >= max_iters) {
      CUGRAPH_FAIL("PageRank failed to converge.");
    }
  }

  return;
}

}  // namespace detail
}  // namespace experimental
}  // namespace cugraph
