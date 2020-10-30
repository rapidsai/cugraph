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
#include <patterns/any_of_adj_matrix_row.cuh>
#include <patterns/copy_to_adj_matrix_row_col.cuh>
#include <patterns/copy_v_transform_reduce_in_out_nbr.cuh>
#include <patterns/count_if_e.cuh>
#include <patterns/count_if_v.cuh>
#include <patterns/reduce_v.cuh>
#include <patterns/transform_reduce_v.cuh>
#include <utilities/error.hpp>
#include <vertex_partition_device.cuh>

#include <rmm/thrust_rmm_allocator.h>
#include <raft/handle.hpp>

#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

namespace cugraph {
namespace experimental {
namespace detail {

// FIXME: personalization_vector_size is confusing in OPG (local or aggregate?)
template <typename GraphViewType, typename result_t>
void pagerank(raft::handle_t const& handle,
              GraphViewType const& pull_graph_view,
              typename GraphViewType::weight_type* precomputed_vertex_out_weight_sums,
              typename GraphViewType::vertex_type* personalization_vertices,
              result_t* personalization_values,
              typename GraphViewType::vertex_type personalization_vector_size,
              result_t* pageranks,
              result_t alpha,
              result_t epsilon,
              size_t max_iterations,
              bool has_initial_guess,
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

  auto aggregate_personalization_vector_size =
    GraphViewType::is_multi_gpu
      ? host_scalar_allreduce(handle.get_comms(), personalization_vector_size, handle.get_stream())
      : personalization_vector_size;

  // 1. check input arguments

  CUGRAPH_EXPECTS(
    ((personalization_vector_size > 0) && (personalization_vertices != nullptr) &&
     (personalization_values != nullptr)) ||
      ((personalization_vector_size == 0) && (personalization_vertices == nullptr) &&
       (personalization_values == nullptr)),
    "Invalid input argument: if personalization_vector_size is non-zero, personalization verties "
    "and personalization values should be provided. Otherwise, they should not be provided.");
  CUGRAPH_EXPECTS((alpha >= 0.0) && (alpha <= 1.0),
                  "Invalid input argument: alpha should be in [0.0, 1.0].");
  CUGRAPH_EXPECTS(epsilon >= 0.0, "Invalid input argument: epsilon should be non-negative.");

  if (do_expensive_check) {
    if (precomputed_vertex_out_weight_sums != nullptr) {
      auto num_negative_precomputed_vertex_out_weight_sums = count_if_v(
        handle, pull_graph_view, precomputed_vertex_out_weight_sums, [] __device__(auto val) {
          return val < result_t{0.0};
        });
      CUGRAPH_EXPECTS(
        num_negative_precomputed_vertex_out_weight_sums == 0,
        "Invalid input argument: outgoing edge weight sum values should be non-negative.");
    }

    if (pull_graph_view.is_weighted()) {
      auto num_nonpositive_edge_weights = count_if_e(
        handle,
        pull_graph_view,
        thrust::make_constant_iterator(0) /* dummy */,
        thrust::make_constant_iterator(0) /* dummy */,
        [] __device__(vertex_t src, vertex_t dst, weight_t w, auto src_val, auto dst_val) {
          return w <= 0.0;
        });
      CUGRAPH_EXPECTS(num_nonpositive_edge_weights == 0,
                      "Invalid input argument: input graph should have postive edge weights.");
    }

    if (has_initial_guess) {
      auto num_negative_values = count_if_v(
        handle, pull_graph_view, pageranks, [] __device__(auto val) { return val < 0.0; });
      CUGRAPH_EXPECTS(num_negative_values == 0,
                      "Invalid input argument: initial guess values should be non-negative.");
    }

    if (aggregate_personalization_vector_size > 0) {
      vertex_partition_device_t<GraphViewType> vertex_partition(pull_graph_view);
      auto num_invalid_vertices =
        count_if_v(handle,
                   pull_graph_view,
                   personalization_vertices,
                   personalization_vertices + personalization_vector_size,
                   [vertex_partition] __device__(auto val) {
                     return !(vertex_partition.is_valid_vertex(val) &&
                              vertex_partition.is_local_vertex_nocheck(val));
                   });
      CUGRAPH_EXPECTS(num_invalid_vertices == 0,
                      "Invalid input argument: peresonalization vertices have invalid vertex IDs.");
      auto num_negative_values = count_if_v(handle,
                                            pull_graph_view,
                                            personalization_values,
                                            personalization_values + personalization_vector_size,
                                            [] __device__(auto val) { return val < 0.0; });
      CUGRAPH_EXPECTS(num_negative_values == 0,
                      "Invalid input argument: peresonalization values should be non-negative.");
    }
  }

  // 2. compute the sums of the out-going edge weights (if not provided)

  rmm::device_uvector<weight_t> tmp_vertex_out_weight_sums(0, handle.get_stream());
  if (precomputed_vertex_out_weight_sums == nullptr) {
    tmp_vertex_out_weight_sums.resize(pull_graph_view.get_number_of_local_vertices(),
                                      handle.get_stream());
    // FIXME: better refactor this out (computing out-degree).
    copy_v_transform_reduce_out_nbr(
      handle,
      pull_graph_view,
      thrust::make_constant_iterator(0) /* dummy */,
      thrust::make_constant_iterator(0) /* dummy */,
      [alpha] __device__(vertex_t src, vertex_t dst, weight_t w, auto src_val, auto dst_val) {
        return w;
      },
      weight_t{0.0},
      tmp_vertex_out_weight_sums.data());
  }

  auto vertex_out_weight_sums = precomputed_vertex_out_weight_sums != nullptr
                                  ? precomputed_vertex_out_weight_sums
                                  : tmp_vertex_out_weight_sums.data();

  // 3. initialize pagerank values

  if (has_initial_guess) {
    auto sum = reduce_v(handle, pull_graph_view, pageranks, result_t{0.0});
    CUGRAPH_EXPECTS(
      sum > 0.0,
      "Invalid input argument: sum of the PageRank initial guess values should be positive.");
    thrust::transform(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                      pageranks,
                      pageranks + pull_graph_view.get_number_of_local_vertices(),
                      pageranks,
                      [sum] __device__(auto val) { return val / sum; });
  } else {
    thrust::fill(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                 pageranks,
                 pageranks + pull_graph_view.get_number_of_local_vertices(),
                 result_t{1.0} / static_cast<result_t>(num_vertices));
  }

  // 4. sum the personalization values

  result_t personalization_sum{0.0};
  if (aggregate_personalization_vector_size > 0) {
    personalization_sum = reduce_v(handle,
                                   pull_graph_view,
                                   personalization_values,
                                   personalization_values + personalization_vector_size,
                                   result_t{0.0});
    CUGRAPH_EXPECTS(personalization_sum > 0.0,
                    "Invalid input argument: sum of personalization valuese should be positive.");
  }

  // 5. pagerank iteration

  // old PageRank values
  rmm::device_uvector<result_t> old_pageranks(pull_graph_view.get_number_of_local_vertices(),
                                              handle.get_stream());
  rmm::device_uvector<result_t> adj_matrix_row_pageranks(
    pull_graph_view.get_number_of_local_adj_matrix_partition_rows(), handle.get_stream());
  size_t iter{0};
  while (true) {
    thrust::copy(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                 pageranks,
                 pageranks + pull_graph_view.get_number_of_local_vertices(),
                 old_pageranks.data());

    auto vertex_val_first =
      thrust::make_zip_iterator(thrust::make_tuple(pageranks, vertex_out_weight_sums));

    auto dangling_sum = transform_reduce_v(
      handle,
      pull_graph_view,
      vertex_val_first,
      [] __device__(auto val) {
        auto const pagerank       = thrust::get<0>(val);
        auto const out_weight_sum = thrust::get<1>(val);
        return out_weight_sum == result_t{0.0} ? pagerank : result_t{0.0};
      },
      result_t{0.0});

    thrust::transform(rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
                      vertex_val_first,
                      vertex_val_first + pull_graph_view.get_number_of_local_vertices(),
                      pageranks,
                      [] __device__(auto val) {
                        auto const pagerank       = thrust::get<0>(val);
                        auto const out_weight_sum = thrust::get<1>(val);
                        auto const divisor =
                          out_weight_sum == result_t{0.0} ? result_t{1.0} : out_weight_sum;
                        return pagerank / divisor;
                      });

    copy_to_adj_matrix_row(handle, pull_graph_view, pageranks, adj_matrix_row_pageranks.begin());

    auto unvarying_part = aggregate_personalization_vector_size == 0
                            ? (dangling_sum * alpha + static_cast<result_t>(1.0 - alpha)) /
                                static_cast<result_t>(num_vertices)
                            : result_t{0.0};

    copy_v_transform_reduce_in_nbr(
      handle,
      pull_graph_view,
      adj_matrix_row_pageranks.begin(),
      thrust::make_constant_iterator(0) /* dummy */,
      [alpha] __device__(vertex_t src, vertex_t dst, weight_t w, auto src_val, auto dst_val) {
        return src_val * w * alpha;
      },
      unvarying_part,
      pageranks);

    if (aggregate_personalization_vector_size > 0) {
      vertex_partition_device_t<GraphViewType> vertex_partition(pull_graph_view);
      auto val_first = thrust::make_zip_iterator(
        thrust::make_tuple(personalization_vertices, personalization_values));
      thrust::for_each(
        rmm::exec_policy(handle.get_stream())->on(handle.get_stream()),
        val_first,
        val_first + personalization_vector_size,
        [vertex_partition, pageranks, dangling_sum, personalization_sum, alpha] __device__(
          auto val) {
          auto v     = thrust::get<0>(val);
          auto value = thrust::get<1>(val);
          *(pageranks + vertex_partition.get_local_vertex_offset_from_vertex_nocheck(v)) +=
            (dangling_sum * alpha + static_cast<result_t>(1.0 - alpha)) *
            (value / personalization_sum);
        });
    }

    auto diff_sum = transform_reduce_v(
      handle,
      pull_graph_view,
      thrust::make_zip_iterator(thrust::make_tuple(pageranks, old_pageranks.data())),
      [] __device__(auto val) { return std::abs(thrust::get<0>(val) - thrust::get<1>(val)); },
      result_t{0.0});

    iter++;

    if (diff_sum < epsilon) {
      break;
    } else if (iter >= max_iterations) {
      CUGRAPH_FAIL("PageRank failed to converge.");
    }
  }

  return;
}

}  // namespace detail

template <typename vertex_t, typename edge_t, typename weight_t, typename result_t, bool multi_gpu>
void pagerank(raft::handle_t const& handle,
              graph_view_t<vertex_t, edge_t, weight_t, true, multi_gpu> const& graph_view,
              weight_t* precomputed_vertex_out_weight_sums,
              vertex_t* personalization_vertices,
              result_t* personalization_values,
              vertex_t personalization_vector_size,
              result_t* pageranks,
              result_t alpha,
              result_t epsilon,
              size_t max_iterations,
              bool has_initial_guess,
              bool do_expensive_check)
{
  detail::pagerank(handle,
                   graph_view,
                   precomputed_vertex_out_weight_sums,
                   personalization_vertices,
                   personalization_values,
                   personalization_vector_size,
                   pageranks,
                   alpha,
                   epsilon,
                   max_iterations,
                   has_initial_guess,
                   do_expensive_check);
}

// explicit instantiation

template void pagerank(raft::handle_t const& handle,
                       graph_view_t<int32_t, int32_t, float, true, true> const& graph_view,
                       float* precomputed_vertex_out_weight_sums,
                       int32_t* personalization_vertices,
                       float* personalization_values,
                       int32_t personalization_vector_size,
                       float* pageranks,
                       float alpha,
                       float epsilon,
                       size_t max_iterations,
                       bool has_initial_guess,
                       bool do_expensive_check);

template void pagerank(raft::handle_t const& handle,
                       graph_view_t<int32_t, int32_t, double, true, true> const& graph_view,
                       double* precomputed_vertex_out_weight_sums,
                       int32_t* personalization_vertices,
                       double* personalization_values,
                       int32_t personalization_vector_size,
                       double* pageranks,
                       double alpha,
                       double epsilon,
                       size_t max_iterations,
                       bool has_initial_guess,
                       bool do_expensive_check);

template void pagerank(raft::handle_t const& handle,
                       graph_view_t<int32_t, int64_t, float, true, true> const& graph_view,
                       float* precomputed_vertex_out_weight_sums,
                       int32_t* personalization_vertices,
                       float* personalization_values,
                       int32_t personalization_vector_size,
                       float* pageranks,
                       float alpha,
                       float epsilon,
                       size_t max_iterations,
                       bool has_initial_guess,
                       bool do_expensive_check);

template void pagerank(raft::handle_t const& handle,
                       graph_view_t<int32_t, int64_t, double, true, true> const& graph_view,
                       double* precomputed_vertex_out_weight_sums,
                       int32_t* personalization_vertices,
                       double* personalization_values,
                       int32_t personalization_vector_size,
                       double* pageranks,
                       double alpha,
                       double epsilon,
                       size_t max_iterations,
                       bool has_initial_guess,
                       bool do_expensive_check);

template void pagerank(raft::handle_t const& handle,
                       graph_view_t<int64_t, int64_t, float, true, true> const& graph_view,
                       float* precomputed_vertex_out_weight_sums,
                       int64_t* personalization_vertices,
                       float* personalization_values,
                       int64_t personalization_vector_size,
                       float* pageranks,
                       float alpha,
                       float epsilon,
                       size_t max_iterations,
                       bool has_initial_guess,
                       bool do_expensive_check);

template void pagerank(raft::handle_t const& handle,
                       graph_view_t<int64_t, int64_t, double, true, true> const& graph_view,
                       double* precomputed_vertex_out_weight_sums,
                       int64_t* personalization_vertices,
                       double* personalization_values,
                       int64_t personalization_vector_size,
                       double* pageranks,
                       double alpha,
                       double epsilon,
                       size_t max_iterations,
                       bool has_initial_guess,
                       bool do_expensive_check);

template void pagerank(raft::handle_t const& handle,
                       graph_view_t<int32_t, int32_t, float, true, false> const& graph_view,
                       float* precomputed_vertex_out_weight_sums,
                       int32_t* personalization_vertices,
                       float* personalization_values,
                       int32_t personalization_vector_size,
                       float* pageranks,
                       float alpha,
                       float epsilon,
                       size_t max_iterations,
                       bool has_initial_guess,
                       bool do_expensive_check);

template void pagerank(raft::handle_t const& handle,
                       graph_view_t<int32_t, int32_t, double, true, false> const& graph_view,
                       double* precomputed_vertex_out_weight_sums,
                       int32_t* personalization_vertices,
                       double* personalization_values,
                       int32_t personalization_vector_size,
                       double* pageranks,
                       double alpha,
                       double epsilon,
                       size_t max_iterations,
                       bool has_initial_guess,
                       bool do_expensive_check);

template void pagerank(raft::handle_t const& handle,
                       graph_view_t<int32_t, int64_t, float, true, false> const& graph_view,
                       float* precomputed_vertex_out_weight_sums,
                       int32_t* personalization_vertices,
                       float* personalization_values,
                       int32_t personalization_vector_size,
                       float* pageranks,
                       float alpha,
                       float epsilon,
                       size_t max_iterations,
                       bool has_initial_guess,
                       bool do_expensive_check);

template void pagerank(raft::handle_t const& handle,
                       graph_view_t<int32_t, int64_t, double, true, false> const& graph_view,
                       double* precomputed_vertex_out_weight_sums,
                       int32_t* personalization_vertices,
                       double* personalization_values,
                       int32_t personalization_vector_size,
                       double* pageranks,
                       double alpha,
                       double epsilon,
                       size_t max_iterations,
                       bool has_initial_guess,
                       bool do_expensive_check);

template void pagerank(raft::handle_t const& handle,
                       graph_view_t<int64_t, int64_t, float, true, false> const& graph_view,
                       float* precomputed_vertex_out_weight_sums,
                       int64_t* personalization_vertices,
                       float* personalization_values,
                       int64_t personalization_vector_size,
                       float* pageranks,
                       float alpha,
                       float epsilon,
                       size_t max_iterations,
                       bool has_initial_guess,
                       bool do_expensive_check);

template void pagerank(raft::handle_t const& handle,
                       graph_view_t<int64_t, int64_t, double, true, false> const& graph_view,
                       double* precomputed_vertex_out_weight_sums,
                       int64_t* personalization_vertices,
                       double* personalization_values,
                       int64_t personalization_vector_size,
                       double* pageranks,
                       double alpha,
                       double epsilon,
                       size_t max_iterations,
                       bool has_initial_guess,
                       bool do_expensive_check);

}  // namespace experimental
}  // namespace cugraph
