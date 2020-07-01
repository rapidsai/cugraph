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
#include <detail/graph_device_view.cuh>
#include <detail/patterns/copy_to_adj_matrix_row.cuh>
#include <detail/patterns/copy_v_transform_reduce_e.cuh>
#include <detail/patterns/count_if_adj_matrix_row.cuh>
#include <detail/patterns/count_if_e.cuh>
#include <detail/patterns/count_if_v.cuh>
#include <detail/patterns/reduce_v.cuh>
#include <detail/patterns/transform_reduce_v_with_adj_matrix_row.cuh>
#include <graph.hpp>
#include <utilities/error.hpp>

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
template <typename GraphType, typename result_t>
void pagerank(raft::handle_t& handle,
              GraphType const& pull_graph,
              typename GraphType::weight_type* adj_matrix_row_out_weight_sums,
              typename GraphType::vertex_type* personalization_vertices,
              result_t* personalization_values,
              typename GraphType::vertex_type personalization_vector_size,
              result_t* pageranks,
              result_t alpha,
              result_t epsilon,
              size_t max_iterations,
              bool has_initial_guess,
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

  CUGRAPH_EXPECTS(
    (personalization_vertices == nullptr) || (personalization_values != nullptr),
    "Invalid input argument: if personalization verties are provided, personalization "
    "values should be provided as well.");
  CUGRAPH_EXPECTS((alpha >= 0.0) && (alpha <= 1.0),
                  "Invalid input argument: alpha should be in [0.0, 1.0].");
  CUGRAPH_EXPECTS(epsilon >= 0.0, "Invalid input argument: epsilon should be non-negative.");

  if (do_expensive_check) {
    if (adj_matrix_row_out_weight_sums != nullptr) {
      auto num_negative_weight_sums = count_if_adj_matrix_row(
        handle, graph_device_view, adj_matrix_row_out_weight_sums, [] __device__(auto val) {
          return val < static_cast<result_t>(0.0);
        });
      CUGRAPH_EXPECTS(
        num_negative_weight_sums == 0,
        "Invalid input argument: outgoing edge weight sum values should be non-negative.");
    }

    if (graph_device_view.is_weighted()) {
      auto num_nonpositive_edge_weights =
        count_if_e(handle,
                   graph_device_view,
                   thrust::make_constant_iterator(0) /* dummy */,
                   thrust::make_constant_iterator(0) /* dummy */,
                   [] __device__(auto src_val, auto dst_val, weight_t w) { return w <= 0.0; });
      CUGRAPH_EXPECTS(num_nonpositive_edge_weights == 0,
                      "Invalid input argument: input graph should have postive edge weights.");
    }

    if (has_initial_guess) {
      auto num_negative_values = count_if_v(
        handle, graph_device_view, pageranks, [] __device__(auto val) { return val < 0.0; });
      CUGRAPH_EXPECTS(num_negative_values == 0,
                      "Invalid input argument: initial guess values should be non-negative.");
    }

    if (personalization_vertices != nullptr) {
      auto num_invalid_vertices =
        count_if_v(handle,
                   graph_device_view,
                   personalization_vertices,
                   personalization_vertices + personalization_vector_size,
                   [graph_device_view] __device__(auto val) {
                     return !(graph_device_view.is_valid_vertex(val) &&
                              graph_device_view.is_local_vertex_nocheck(val));
                   });
      CUGRAPH_EXPECTS(num_invalid_vertices == 0,
                      "Invalid input argument: peresonalization vertices have invalid vertex IDs.");
      auto num_negative_values = count_if_v(handle,
                                            graph_device_view,
                                            personalization_values,
                                            personalization_values + personalization_vector_size,
                                            [] __device__(auto val) { return val < 0.0; });
      CUGRAPH_EXPECTS(num_negative_values == 0,
                      "Invalid input argument: peresonalization values should be non-negative.");
    }
  }

  // 2. compute the sums of the out-going edge weights (if not provided)

  rmm::device_vector<weight_t> tmp_adj_matrix_row_out_weight_sums{};
  if (adj_matrix_row_out_weight_sums == nullptr) {
    rmm::device_vector<weight_t> tmp_out_weight_sums(
      graph_device_view.get_number_of_local_vertices(), weight_t{0.0});
    // FIXME: better refactor this out (computing out-degree).
    copy_v_transform_reduce_out_nbr(
      handle,
      graph_device_view,
      thrust::make_constant_iterator(0) /* dummy */,
      thrust::make_constant_iterator(0) /* dummy */,
      tmp_out_weight_sums.data().get(),
      [alpha] __device__(auto src_val, auto dst_val, weight_t w) { return w; },
      weight_t{0.0});

    tmp_adj_matrix_row_out_weight_sums.assign(
      graph_device_view.get_number_of_adj_matrix_local_rows(), weight_t{0.0});
    copy_to_adj_matrix_row(handle,
                           graph_device_view,
                           tmp_out_weight_sums.data().get(),
                           tmp_adj_matrix_row_out_weight_sums.begin());
  }

  auto row_out_weight_sums = adj_matrix_row_out_weight_sums != nullptr
                               ? adj_matrix_row_out_weight_sums
                               : tmp_adj_matrix_row_out_weight_sums.data().get();

  // 3. initialize pagerank values

  if (has_initial_guess) {
    auto sum = reduce_v(handle, graph_device_view, pageranks, static_cast<result_t>(0.0));
    CUGRAPH_EXPECTS(
      sum > 0.0,
      "Invalid input argument: sum of the PageRank initial guess values should be positive.");
    thrust::transform(thrust::cuda::par.on(handle.get_stream()),
                      pageranks,
                      pageranks + graph_device_view.get_number_of_local_vertices(),
                      pageranks,
                      [sum] __device__(auto val) { return val / sum; });
  } else {
    thrust::fill(thrust::cuda::par.on(handle.get_stream()),
                 pageranks,
                 pageranks + graph_device_view.get_number_of_local_vertices(),
                 static_cast<result_t>(1.0) / static_cast<result_t>(num_vertices));
  }

  // 4. sum the personalization values

  result_t personalization_sum{0.0};
  if (personalization_vertices != nullptr) {
    personalization_sum = reduce_v(handle,
                                   graph_device_view,
                                   personalization_values,
                                   personalization_values + personalization_vector_size,
                                   static_cast<result_t>(0.0));
    CUGRAPH_EXPECTS(personalization_sum > 0.0,
                    "Invalid input argument: sum of personalization valuese should be positive.");
  }

  // 5. pagerank iteration

  // old PageRank values
  rmm::device_vector<result_t> adj_matrix_row_pageranks(
    graph_device_view.get_number_of_adj_matrix_local_rows(), static_cast<result_t>(0.0));
  size_t iter{0};
  while (true) {
    copy_to_adj_matrix_row(handle, graph_device_view, pageranks, adj_matrix_row_pageranks.begin());

    auto row_val_first = thrust::make_zip_iterator(
      thrust::make_tuple(adj_matrix_row_pageranks.begin(), row_out_weight_sums));
    thrust::transform(thrust::cuda::par.on(handle.get_stream()),
                      row_val_first,
                      row_val_first + graph_device_view.get_number_of_adj_matrix_local_rows(),
                      adj_matrix_row_pageranks.begin(),
                      [] __device__(auto val) {
                        auto const row_pagerank       = thrust::get<0>(val);
                        auto const row_out_weight_sum = thrust::get<1>(val);
                        auto const divisor = row_out_weight_sum == static_cast<result_t>(0.0)
                                               ? static_cast<result_t>(1.0)
                                               : row_out_weight_sum;
                        return row_pagerank / divisor;
                      });

    auto dangling_sum = transform_reduce_v_with_adj_matrix_row(
      handle,
      graph_device_view,
      thrust::make_constant_iterator(0) /* dummy */,
      row_val_first,
      [] __device__(auto v_val, auto row_val) {
        auto const row_pagerank       = thrust::get<0>(row_val);
        auto const row_out_weight_sum = thrust::get<1>(row_val);
        return row_out_weight_sum == static_cast<result_t>(0.0) ? row_pagerank
                                                                : static_cast<result_t>(0.0);
      },
      static_cast<result_t>(0.0));

    auto unvarying_part =
      personalization_vertices == nullptr
        ? (static_cast<result_t>(1.0 - alpha)) / static_cast<result_t>(num_vertices) +
            static_cast<result_t>(alpha) * (dangling_sum / static_cast<result_t>(num_vertices))
        : static_cast<result_t>(0.0);

    copy_v_transform_reduce_in_nbr(
      handle,
      graph_device_view,
      adj_matrix_row_pageranks.begin(),
      thrust::make_constant_iterator(0) /* dummy */,
      pageranks,
      [alpha] __device__(auto src_val, auto dst_val, weight_t w) { return src_val * alpha; },
      unvarying_part);

    if (personalization_vertices != nullptr) {
      auto val_first = thrust::make_zip_iterator(
        thrust::make_tuple(personalization_vertices, personalization_values));
      thrust::for_each(
        val_first,
        val_first + personalization_vector_size,
        [graph_device_view, pageranks, dangling_sum, personalization_sum, alpha] __device__(
          auto val) {
          auto v     = thrust::get<0>(val);
          auto value = thrust::get<1>(val);
          *(pageranks + graph_device_view.get_local_vertex_offset_from_vertex_nocheck(v)) +=
            (dangling_sum + static_cast<result_t>(1.0 - alpha)) * (value / personalization_sum);
        });
    }

    auto diff_sum = transform_reduce_v_with_adj_matrix_row(
      handle,
      graph_device_view,
      pageranks,
      adj_matrix_row_pageranks.begin(),
      [] __device__(auto v_val, auto col_val) { return std::abs(v_val - col_val); },
      static_cast<result_t>(0.0));

    iter++;

    if (diff_sum < static_cast<result_t>(num_vertices) * static_cast<result_t>(epsilon)) {
      break;
    } else if (iter >= max_iterations) {
      CUGRAPH_FAIL("PageRank failed to converge.");
    }
  }

  return;
}

}  // namespace detail

template <typename vertex_t, typename edge_t, typename weight_t, typename result_t>
void pagerank(raft::handle_t& handle,
              GraphCSCView<vertex_t, edge_t, weight_t> const& graph,
              weight_t* adj_matrix_row_out_weight_sums,
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
                   graph,
                   adj_matrix_row_out_weight_sums,
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

template void pagerank(raft::handle_t& handle,
                       GraphCSCView<int32_t, int32_t, float> const& graph,
                       float* adj_matrix_row_out_weight_sums,
                       int32_t* personalization_vertices,
                       float* personalization_values,
                       int32_t personalization_vector_size,
                       float* pageranks,
                       float alpha,
                       float epsilon,
                       size_t max_iterations,
                       bool has_initial_guess,
                       bool do_expensive_check);

}  // namespace experimental
}  // namespace cugraph
