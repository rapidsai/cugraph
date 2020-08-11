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
#include <algorithms.hpp>
#include <graph.hpp>
#include <graph_device_view.cuh>
#include <patterns/copy_to_adj_matrix_row.cuh>
#include <patterns/copy_v_transform_reduce_nbr.cuh>
#include <patterns/count_if_v.cuh>
#include <patterns/transform_reduce_v.cuh>
#include <patterns/transform_reduce_v_with_adj_matrix_row.cuh>
#include <utilities/error.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <raft/handle.hpp>

#include <thrust/fill.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

namespace cugraph {
namespace experimental {
namespace detail {

template <typename GraphType, typename result_t>
void katz_centrality(raft::handle_t &handle,
                     GraphType const &pull_graph,
                     result_t *betas,
                     result_t *katz_centralities,
                     result_t alpha,
                     result_t beta,  // relevant only if betas == nullptr
                     result_t epsilon,
                     size_t max_iterations,
                     bool has_initial_guess,
                     bool normalize,
                     bool do_expensive_check)
{
  using vertex_t = typename GraphType::vertex_type;
  using weight_t = typename GraphType::weight_type;

  static_assert(std::is_integral<vertex_t>::value, "GraphType::vertex_type should be integral.");
  static_assert(std::is_floating_point<result_t>::value,
                "result_t should be a floating-point type.");
  static_assert(GraphType::is_adj_matrix_transposed, "GraphType should support the pull model.");

  auto p_graph_device_view     = graph_device_view_t<GraphType>::create(pull_graph);
  auto const graph_device_view = *p_graph_device_view;

  auto const num_vertices = graph_device_view.get_number_of_vertices();
  if (num_vertices == 0) { return; }

  // 1. check input arguments

  CUGRAPH_EXPECTS((alpha >= 0.0) && (alpha <= 1.0),
                  "Invalid input argument: alpha should be in [0.0, 1.0].");
  CUGRAPH_EXPECTS(epsilon >= 0.0, "Invalid input argument: epsilon should be non-negative.");

  if (do_expensive_check) {
    // FIXME: should I check for betas?

    if (has_initial_guess) {
      auto num_negative_values =
        count_if_v(handle, graph_device_view, katz_centralities, [] __device__(auto val) {
          return val < 0.0;
        });
      CUGRAPH_EXPECTS(num_negative_values == 0,
                      "Invalid input argument: initial guess values should be non-negative.");
    }
  }

  // 2. initialize katz centrality values

  if (!has_initial_guess) {
    thrust::fill(thrust::cuda::par.on(handle.get_stream()),
                 katz_centralities,
                 katz_centralities + graph_device_view.get_number_of_local_vertices(),
                 result_t{0.0});
  }

  // 3. katz centrality iteration

  // old katz centrality values
  rmm::device_vector<result_t> adj_matrix_row_katz_centralities(
    graph_device_view.get_number_of_adj_matrix_local_rows(), result_t{0.0});
  size_t iter{0};
  while (true) {
    copy_to_adj_matrix_row(
      handle, graph_device_view, katz_centralities, adj_matrix_row_katz_centralities.begin());

    copy_v_transform_reduce_in_nbr(
      handle,
      graph_device_view,
      adj_matrix_row_katz_centralities.begin(),
      thrust::make_constant_iterator(0) /* dummy */,
      [alpha] __device__(auto src_val, auto dst_val, weight_t w) {
        return static_cast<result_t>(alpha * src_val * w);
      },
      betas != nullptr ? result_t{0.0} : beta,
      katz_centralities);

    if (betas != nullptr) {
      auto val_first = thrust::make_zip_iterator(thrust::make_tuple(katz_centralities, betas));
      thrust::transform(thrust::cuda::par.on(handle.get_stream()),
                        val_first,
                        val_first + graph_device_view.get_number_of_local_vertices(),
                        katz_centralities,
                        [] __device__(auto val) {
                          auto const katz_centrality = thrust::get<0>(val);
                          auto const beta            = thrust::get<1>(val);
                          return katz_centrality + beta;
                        });
    }

    auto diff_sum = transform_reduce_v_with_adj_matrix_row(
      handle,
      graph_device_view,
      katz_centralities,
      adj_matrix_row_katz_centralities.begin(),
      [] __device__(auto v_val, auto row_val) { return std::abs(v_val - row_val); },
      result_t{0.0});

    iter++;

    if (diff_sum < static_cast<result_t>(num_vertices) * epsilon) {
      break;
    } else if (iter >= max_iterations) {
      CUGRAPH_FAIL("Katz Centrality failed to converge.");
    }
  }

  if (normalize) {
    auto l2_norm = transform_reduce_v(
      handle,
      graph_device_view,
      katz_centralities,
      [] __device__(auto val) { return val * val; },
      result_t{0.0});
    l2_norm = std::sqrt(l2_norm);
    CUGRAPH_EXPECTS(l2_norm > 0.0,
                    "L2 norm of the computed Katz Centrality values should be positive.");
    thrust::transform(thrust::cuda::par.on(handle.get_stream()),
                      katz_centralities,
                      katz_centralities + graph_device_view.get_number_of_local_vertices(),
                      katz_centralities,
                      [l2_norm] __device__(auto val) { return val / l2_norm; });
  }

  return;
}

}  // namespace detail

template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
void katz_centrality(raft::handle_t &handle,
                     GraphCSCView<vertex_t, edge_t, weight_t> const &graph,
                     result_t *betas,
                     result_t *katz_centralities,
                     result_t alpha,
                     result_t beta,  // relevant only if beta == nullptr
                     result_t epsilon,
                     size_t max_iterations,
                     bool has_initial_guess,
                     bool normalize,
                     bool do_expensive_check)
{
  detail::katz_centrality(handle,
                          graph,
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

// explicit instantiation

template void katz_centrality(raft::handle_t &handle,
                              GraphCSCView<int32_t, int32_t, float> const &graph,
                              float *betas,
                              float *katz_centralities,
                              float alpha,
                              float beta,
                              float epsilon,
                              size_t max_iterations,
                              bool has_initial_guess,
                              bool normalize,
                              bool do_expensive_check);

}  // namespace experimental
}  // namespace cugraph
