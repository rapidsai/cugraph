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
#include <experimental/graph_view.hpp>
#include <patterns/copy_to_adj_matrix_row_col.cuh>
#include <patterns/copy_v_transform_reduce_in_out_nbr.cuh>
#include <patterns/count_if_v.cuh>
#include <patterns/transform_reduce_v.cuh>
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

template <typename GraphViewType, typename result_t>
void katz_centrality(raft::handle_t const &handle,
                     GraphViewType const &pull_graph_view,
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
  using vertex_t = typename GraphViewType::vertex_type;
  using weight_t = typename GraphViewType::weight_type;

  static_assert(std::is_integral<vertex_t>::value,
                "GraphViewType::vertex_type should be integral.");
  static_assert(std::is_floating_point<result_t>::value,
                "result_t should be a floating-point type.");
  static_assert(GraphViewType::is_adj_matrix_transposed,
                "GraphViewType should support the pull model.");

  auto const num_vertices = pull_graph_view.get_number_of_vertices();
  if (num_vertices == 0) { return; }

  // 1. check input arguments

  CUGRAPH_EXPECTS((alpha >= 0.0) && (alpha <= 1.0),
                  "Invalid input argument: alpha should be in [0.0, 1.0].");
  CUGRAPH_EXPECTS(epsilon >= 0.0, "Invalid input argument: epsilon should be non-negative.");

  if (do_expensive_check) {
    // FIXME: should I check for betas?

    if (has_initial_guess) {
      auto num_negative_values = count_if_v(
        handle, pull_graph_view, katz_centralities, [] __device__(auto val) { return val < 0.0; });
      CUGRAPH_EXPECTS(num_negative_values == 0,
                      "Invalid input argument: initial guess values should be non-negative.");
    }
  }

  // 2. initialize katz centrality values

  if (!has_initial_guess) {
    thrust::fill(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                 katz_centralities,
                 katz_centralities + pull_graph_view.get_number_of_local_vertices(),
                 result_t{0.0});
  }

  // 3. katz centrality iteration

  // old katz centrality values
  rmm::device_uvector<result_t> tmp_katz_centralities(
    pull_graph_view.get_number_of_local_vertices(), handle.get_stream());
  rmm::device_uvector<result_t> adj_matrix_row_katz_centralities(
    pull_graph_view.get_number_of_local_adj_matrix_partition_rows(), handle.get_stream());
  auto new_katz_centralities = katz_centralities;
  auto old_katz_centralities = tmp_katz_centralities.data();
  size_t iter{0};
  while (true) {
    std::swap(new_katz_centralities, old_katz_centralities);

    copy_to_adj_matrix_row(
      handle, pull_graph_view, old_katz_centralities, adj_matrix_row_katz_centralities.begin());

    copy_v_transform_reduce_in_nbr(
      handle,
      pull_graph_view,
      adj_matrix_row_katz_centralities.begin(),
      thrust::make_constant_iterator(0) /* dummy */,
      [alpha] __device__(vertex_t src, vertex_t dst, weight_t w, auto src_val, auto dst_val) {
        return static_cast<result_t>(alpha * src_val * w);
      },
      betas != nullptr ? result_t{0.0} : beta,
      new_katz_centralities);

    if (betas != nullptr) {
      auto val_first = thrust::make_zip_iterator(thrust::make_tuple(new_katz_centralities, betas));
      thrust::transform(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                        val_first,
                        val_first + pull_graph_view.get_number_of_local_vertices(),
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
      [] __device__(auto val) { return std::abs(thrust::get<0>(val) - thrust::get<1>(val)); },
      result_t{0.0});

    iter++;

    if (diff_sum < epsilon) {
      break;
    } else if (iter >= max_iterations) {
      CUGRAPH_FAIL("Katz Centrality failed to converge.");
    }
  }

  if (new_katz_centralities != katz_centralities) {
    thrust::copy(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                 new_katz_centralities,
                 new_katz_centralities + pull_graph_view.get_number_of_local_vertices(),
                 katz_centralities);
  }

  if (normalize) {
    auto l2_norm = transform_reduce_v(
      handle,
      pull_graph_view,
      katz_centralities,
      [] __device__(auto val) { return val * val; },
      result_t{0.0});
    l2_norm = std::sqrt(l2_norm);
    CUGRAPH_EXPECTS(l2_norm > 0.0,
                    "L2 norm of the computed Katz Centrality values should be positive.");
    thrust::transform(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                      katz_centralities,
                      katz_centralities + pull_graph_view.get_number_of_local_vertices(),
                      katz_centralities,
                      [l2_norm] __device__(auto val) { return val / l2_norm; });
  }

  return;
}

}  // namespace detail

template <typename vertex_t, typename edge_t, typename weight_t, typename result_t, bool multi_gpu>
void katz_centrality(raft::handle_t const &handle,
                     graph_view_t<vertex_t, edge_t, weight_t, true, multi_gpu> const &graph_view,
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
                          graph_view,
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

template void katz_centrality(raft::handle_t const &handle,
                              graph_view_t<int32_t, int32_t, float, true, true> const &graph_view,
                              float *betas,
                              float *katz_centralities,
                              float alpha,
                              float beta,
                              float epsilon,
                              size_t max_iterations,
                              bool has_initial_guess,
                              bool normalize,
                              bool do_expensive_check);

template void katz_centrality(raft::handle_t const &handle,
                              graph_view_t<int32_t, int32_t, double, true, true> const &graph_view,
                              double *betas,
                              double *katz_centralities,
                              double alpha,
                              double beta,
                              double epsilon,
                              size_t max_iterations,
                              bool has_initial_guess,
                              bool normalize,
                              bool do_expensive_check);

template void katz_centrality(raft::handle_t const &handle,
                              graph_view_t<int32_t, int64_t, float, true, true> const &graph_view,
                              float *betas,
                              float *katz_centralities,
                              float alpha,
                              float beta,
                              float epsilon,
                              size_t max_iterations,
                              bool has_initial_guess,
                              bool normalize,
                              bool do_expensive_check);

template void katz_centrality(raft::handle_t const &handle,
                              graph_view_t<int32_t, int64_t, double, true, true> const &graph_view,
                              double *betas,
                              double *katz_centralities,
                              double alpha,
                              double beta,
                              double epsilon,
                              size_t max_iterations,
                              bool has_initial_guess,
                              bool normalize,
                              bool do_expensive_check);

template void katz_centrality(raft::handle_t const &handle,
                              graph_view_t<int64_t, int64_t, float, true, true> const &graph_view,
                              float *betas,
                              float *katz_centralities,
                              float alpha,
                              float beta,
                              float epsilon,
                              size_t max_iterations,
                              bool has_initial_guess,
                              bool normalize,
                              bool do_expensive_check);

template void katz_centrality(raft::handle_t const &handle,
                              graph_view_t<int64_t, int64_t, double, true, true> const &graph_view,
                              double *betas,
                              double *katz_centralities,
                              double alpha,
                              double beta,
                              double epsilon,
                              size_t max_iterations,
                              bool has_initial_guess,
                              bool normalize,
                              bool do_expensive_check);

template void katz_centrality(raft::handle_t const &handle,
                              graph_view_t<int32_t, int32_t, float, true, false> const &graph_view,
                              float *betas,
                              float *katz_centralities,
                              float alpha,
                              float beta,
                              float epsilon,
                              size_t max_iterations,
                              bool has_initial_guess,
                              bool normalize,
                              bool do_expensive_check);

template void katz_centrality(raft::handle_t const &handle,
                              graph_view_t<int32_t, int32_t, double, true, false> const &graph_view,
                              double *betas,
                              double *katz_centralities,
                              double alpha,
                              double beta,
                              double epsilon,
                              size_t max_iterations,
                              bool has_initial_guess,
                              bool normalize,
                              bool do_expensive_check);

template void katz_centrality(raft::handle_t const &handle,
                              graph_view_t<int32_t, int64_t, float, true, false> const &graph_view,
                              float *betas,
                              float *katz_centralities,
                              float alpha,
                              float beta,
                              float epsilon,
                              size_t max_iterations,
                              bool has_initial_guess,
                              bool normalize,
                              bool do_expensive_check);

template void katz_centrality(raft::handle_t const &handle,
                              graph_view_t<int32_t, int64_t, double, true, false> const &graph_view,
                              double *betas,
                              double *katz_centralities,
                              double alpha,
                              double beta,
                              double epsilon,
                              size_t max_iterations,
                              bool has_initial_guess,
                              bool normalize,
                              bool do_expensive_check);

template void katz_centrality(raft::handle_t const &handle,
                              graph_view_t<int64_t, int64_t, float, true, false> const &graph_view,
                              float *betas,
                              float *katz_centralities,
                              float alpha,
                              float beta,
                              float epsilon,
                              size_t max_iterations,
                              bool has_initial_guess,
                              bool normalize,
                              bool do_expensive_check);

template void katz_centrality(raft::handle_t const &handle,
                              graph_view_t<int64_t, int64_t, double, true, false> const &graph_view,
                              double *betas,
                              double *katz_centralities,
                              double alpha,
                              double beta,
                              double epsilon,
                              size_t max_iterations,
                              bool has_initial_guess,
                              bool normalize,
                              bool do_expensive_check);

}  // namespace experimental
}  // namespace cugraph
