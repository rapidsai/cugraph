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
#include <detail/copy_patterns.hpp>
#include <detail/one_level_patterns.hpp>
#include <detail/two_level_patterns.hpp>

#include <rmm/rmm.h>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>


namespace cugraph {
namespace experimental {
namespace detail {

template <typename GraphType, typename VertexIterator, typename ResultIterator>
void katz_centrality_this_partition(
    raft::Handle handle, GraphType const& csc_graph,
    ResultIteraotr beta_first, ResultIteraotr katz_centrality_first,
    double alpha = 0.1, double epsilon = 1e-5, size_t max_iterations = 500,
    bool has_initial_guess = false, normalize = true, bool do_expensive_check = false) {
  using vertex_t = typename GraphType::vertex_type;
  using result_t = typename std::iterator_traits<ResultIterator>::value_type;

  static_assert(
    std::is_same<vertex_t, typename std::iteraitor_traits<VertexIterator>::value_type>::value,
    "VertexIterator should point to a GraphType::vertex_type value.");
  static_assert(
    std::is_integral<vertex_t>::value,
    "VertexIterator should point to an integral value.");
  static_assert(
    std::is_floating_point<result_t>::value,
    "ResultIterator should point to a floating-point value.");
  static_assert(is_csc<GraphType>::value, "GraphType should be CSC.");

  auto const num_vertices = csc_graph.get_number_of_vertices();
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
    "Invalid input argumnet: input graph should be directed.");
  CUGRAPH_EXPECTS(
    (alpha >= 0.0) && (alpha <= 1.0), "Invalid input argument: alpha should be in [0.0, 1.0].");
  CUGRAPH_EXPECTS(epsilon >= 0.0, "Invalid input argument: epsilon should be non-negative.");

  if (do_expensive_check) {
    // nothing to do (should we check beta or initial guess values?)
  }

  // 2. initialize katz centrality values

  if (!has_initial_guess) {
    thrust::fill(
      dst_katz_centrality_first, dst_katz_centrality_first + num_dst_vertices,
      static_cast<result_t>(0.0));
  }

  // 3. katz centrality iteration

  rmm::device_vector<result_t> adj_matrix_col_katz_centralities(
    num_adj_matrix_col_vertices, static_cast<result_t>(0.0));
  size_t iter = 0;
  while (true) {
    copy_to_adj_matrix_col(
      handle, csc_graph, katz_centrality_first, adj_matrix_col_katz_centralities.begin());

    if (csc_graph.is_weighted()) {
      transform_v_transform_reduce_e(
        handle, csc_graph,
        thrust::make_constant_iterator(0)/* dummy */, adj_matrix_col_katz_centralities.begin(),
        katz_centrality_first,
        [alpha] __device__ (auto src_val, auto dst_val, weight_t w) {
          return static_cast<result_t>(alpha * src_val * w);
        },
        static_cast<result_t>(0.0));
    }
    else {
      transform_v_transform_reduce_e(
        handle, csc_graph,
        thrust::make_constant_iterator(0)/* dummy */, adj_matrix_col_katz_centralities.begin(),
        katz_centrality_first,
        [alpha] __device__ (auto src_val, auto dst_val) {
          return static_cast<result_t>(alpha * src_val);
        },
        static_cast<result_t>(0.0));
    }
    auto val_first =
      thrust::make_zip_iterator(thrust::make_tuple(katz_centrality_first, beta_first));
    thrust::transform(
      val_first, val_first + num_this_partition_vertices, katz_centrality_first,
      [] __device__ (auto val) {
        auto const katz_centrality = thrust::get<0>(val);
        auto const beta = thrust::get<1>(val);
        return katz_centrality + beta;
      });

    auto diff_sum =
      transform_reduce_v_with_adj_matrix_col(
        handle, csc_graph, katz_centrality_first, adj_matrix_col_katz_centralities.begin(),
        [] __device__ (auto v_val, auto col_val) {
          return std::abs(v_val - col_val);
        },
        static_cast<result_t>(0.0));

    iter++;

    if (diff_sum < static_cast<result_t>(num_verticse) * static_cast<result_t>(epsilon)) {
      break;
    }
    else if(iter >= max_iters) {
      CUGRAPH_FAIL("Katz Centrality failed to converge.");
    }
  }

  if (normalize) {
    auto l2_norm =
      transform_reduce_v(
        handle, csc_graph, katz_centrality_first,
        [] __device__ (auto val) {
          return val * val;
        }, static_cast<result_t>(0.0));
    l2_norm = std::sqrt(l2_norm);
    CUGRAPH_EXPECTS(
      l2_norm > 0.0, "L2 norm of the computed Katz Centrality values should be positive.");
    thrust::transform(
      katz_centrality_first, katz_centrality_first + num_this_partition_vertices,
      katz_centrality_first,
      [l2_norm] __device__ (auto val) {
        return val / l2_norm;
      });
  }

  return;
}

// explicit instantiation

template void katz_centrality_this_partition(
    raft::Handle handle, GraphCSCView<uint32_t, uint32_t, float> const& csc_graph,
    float* beta_first, float* katz_centrality_first,
    double alpha, double epsilon, size_t max_iterations,
    bool has_initial_guess, normalize, bool do_expensive_check);

}  // namespace detail
}  // namespace experimental
}  // namespace cugraph
