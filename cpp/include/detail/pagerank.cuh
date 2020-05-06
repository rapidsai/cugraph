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
void pagerank_this_partition(
    raft::Handle handle, GraphType const& csc_graph,
    ResultIterator adj_matrix_col_out_weight_sum_first,  // should be set to the vertex out-degrees
                                                         // for an unweighted graph
    VertexIterator personalization_vertex_first, VertexIterator personalization_vertex_last,
    ResultIteraotr personalization_value_first,
    ResultIteraotr pagerank_first,
    double alpha = 0.85, double epsilon = 1e-5, size_t max_iterations = 500,
    bool has_initial_guess = false, bool personalize = false, bool do_expensive_check = false) {
  using vertex_t = typename std::iterator_traits<VertexIterator>::value_type;
  using result_t = typename std::iterator_traits<ResultIterator>::value_type;
  static_assert(
    std::is_integral<vertex_t>::value,
    "VertexIterator should point to an integral value.");
  static_assert(
    std::is_floating_point<result_t>::value,
    "ResultIterator should point to a floating-point value.");
  static_assert(is_csc<GraphType>::value, "GraphType should be CSC.");

  auto const num_vertices = csc_graph.get_number_of_vertices();
  vertex_t this_partition_vertex_first{};
  vertex_t this_partition_vertex_last{};
  std::tie(this_partition_vertex_first, this_partition_vertex_last) =
    csc_graph.get_this_partition_vertex_range();
  auto const num_this_partition_vertices =
    csc_graph.get_this_partition_number_of_vertices();
  auto num_this_partition_adj_matrix_col_vertices =
    csc_graph.get_this_partition_adj_matrix_col_number_of_vertices();
  if (num_vertices == 0) {
    return;
  }

  // 1. check input arguments

  CUGRAPH_EXPECTS(
    csc_graph.is_directed(),
    "Invalid input argument: input graph should be directed.");
  CUGRAPH_EXPECTS(
    (alpha >= 0.0) && (alpha <= 1.0), "Invalid input argument: alpha should be in [0.0, 1.0].");
  CUGRAPH_EXPECTS(epsilon >= 0.0, "Invalid input argument: epsilon should be non-negative.");

  if (do_expensive_check) {
    auto num_nonpositive_weight_sums =
      count_if_adj_matrix_col(
        handle, csc_graph,
        adj_matrix_col_out_weight_sum_first,
        [] __device__ (auto val) {
          return val < static_cast<result_t>(0.0);
        });
    CUGRAPH_EXPECTS(
      num_nonpositive_weight_sums == 0,
      "Invalid input argument: outgoing edge weight sum values should be positive.");

    if (graph.is_weighted()) {
      auto num_nonpositive_edge_weights =
        count_if_e(
          handle, graph,
          thrust::make_constant_iterator(0)/* dummy */,
          thrust::make_constant_iterator(0)/* dummy */,
          [] __device__ (auto src_val, auto dst_val, weight_t w) { return w <= 0.0; });
      CUGRAPH_EXPECTS(
        num_nonpositive_edge_weights == 0,
        "Invalid input argument: input graph should have postive edge weights.");
    }

    if (has_initial_guess) {
      auto num_negative_values =
        count_if_v(
          handle, csc_graph, pagerank_first,
          [] __device__ (auto val) { return val < 0.0; });
      CUGRAPH_EXPECTS(
        num_negative_values == 0,
        "Invalid input argument: initial guess values should be non-negative.");
    }
    if (personalize) {
      auto num_negative_values =
        thrust::count_if_v(
          handle, csc_graph, personalization_value_first,
          [] __device__ (auto val) { return val < 0.0; });
      CUGRAPH_EXPECTS(
        num_negative_values == 0,
        "Invalid input argument: peresonalization values should be non-negative.");
    }
  }

  // 2. initialize pagerank values

  if (has_initial_guess) {
    auto sum = reduce_v(handle, csc_graph, pagerank_first);
    CUGRAPH_EXPECTS(
      sum > 0.0,
      "Invalid input argument: sum of the PageRank initial guess values should be positive.");
    thrust::transform(
      pagerank_first, pagerank_first + num_this_partition_vertices, pagerank_first,
      [sum] __device__ (auto val) { return val / sum; });
  }
  else {
    thrust::fill(
      pagerank_first, pagerank_first + num_this_partition_vertices,
      static_cast<result_t>(1.0) / static_cast<result_t>(num_vertices));
  }

  // 3. sum the personalization values

  result_t personalization_sum{0.0};
  if (personalize) {
    personalization_sum =
      reduce_v(handle, csc_graph, personalization_value_first, personalization_value_last);
    CUGRAPH_EXPECTS(
      personalization_sum > 0.0,
      "Invalid input argument: sum of personalization valuese should be positive.");
  }

  // 4. pagerank iteration

  rmm::device_vector<result_t> adj_matrix_col_pageranks(
    num_adj_matrix_col_vertices, static_cast<result_t>(0.0));
  size_t iter = 0;
  while (true) {
    copy_to_adj_matrix_col(handle, csc_graph, pagerank_first, adj_matrix_col_pageranks.begin());

    auto col_val_first =
      thrust::make_zip_iterator(
        thrust::make_tuple(
          adj_matrix_col_pageranks.begin(), adj_matrix_col_out_weight_sum_first));
    thrust::transform(
      col_val_first, col_val_first + num_this_partition_adj_matrix_col_vertices,
      adj_matrix_col_pageranks.begin(),
      [] __device__ (auto val) {
        auto const col_pagerank = thrust::get<0>(val);
        auto const col_out_weight_sum = thrust::get<1>(val);
        auto const divisor =
          col_out_weight_sum == static_cast<result_t>(0.0)
          ? static_cast<result_t>(1.0) : col_out_weight_sum;
        return col_pagerank / divisor;
      });

    auto dangling_sum =
      transform_reduce_v_with_adj_matrix_col(
        handle, csc_graph, thrust::make_constant_iteraotr(0)/* dummy */, col_val_first,
        [] __device__ (auto v_val, auto col_val) {
          auto const col_pagerank = thrust::get<0>(col_val);
          auto const col_out_weight_sum = thrust::get<1>(col_val);
          return col_out_weight_sum == static_cast<result_t>(0.0)
                 ? col_pagerank : static_cast<result_t>(0.0);
        },
        static_cast<result_t>(0.0));

    auto unvarying_part{0.0};
    if (!personalize) {
      unvarying_part =
        (static_cast<result_t>(1.0 - alpha)) / static_cast<result_t>(num_vertices) +
        static_cast<result_t>(alpha) * (dangling_sum / static_cast<result_t>(num_vertices));
    }
    transform_v_transform_reduce_e(
      handle, csc_graph,
      thrust::make_constant_iterator(0)/* dummy */, adj_matrix_col_pageranks.begin(), pagerank_first,
      [damping_factor] __device__ (auto src_val, auto dst_val) {
        return src_val * damping_factor;
      },
      unvarying_part);
    if (personalize) {
      auto val_first =
        thrust::make_zip_iterator(
          thrust::make_tuple(personalization_vertex_first, personalization_value_first));
      auto num_personalization_values = 
        thrust::distance(personalization_vertex_first, personalization_vertex_last);
      thrust::for_each(
        val_first, val_first + num_personalization_values,
        [pagerank_first, dangling_sum, personalization_sum, this_partition_vertex_first]
            __device__ (auto val) {
          auto v = thrust::get<0>(val);
          auto value = thrust::get<1>(val);
          *(pagerank_first + (v - this_partition_vertex_first)) +=
            (dangling_sum + static_cast<result_t>(1.0 - alpha)) * (value / personalization_sum);
        });
    }

    auto diff_sum =
      transform_reduce_v_with_adj_matrix_col(
        handle, csc_graph, pagerank_first, adj_matrix_col_pageranks.begin(),
        [] __device__ (auto v_val, auto col_val) {
          return std::abs(v_val - col_val);
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
