/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

#include "prims/count_if_e.cuh"
#include "prims/count_if_v.cuh"
#include "prims/per_v_transform_reduce_incoming_outgoing_e.cuh"
#include "prims/reduce_op.cuh"
#include "prims/reduce_v.cuh"
#include "prims/transform_reduce_v.cuh"
#include "prims/update_edge_src_dst_property.cuh"

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/error.hpp>

#include <raft/core/handle.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

namespace cugraph {
namespace detail {

template <typename GraphViewType, typename weight_t, typename result_t>
centrality_algorithm_metadata_t pagerank(
  raft::handle_t const& handle,
  GraphViewType const& pull_graph_view,
  std::optional<edge_property_view_t<typename GraphViewType::edge_type, weight_t const*>>
    edge_weight_view,
  std::optional<raft::device_span<weight_t const>> precomputed_vertex_out_weight_sums,
  std::optional<std::tuple<raft::device_span<typename GraphViewType::vertex_type const>,
                           raft::device_span<result_t const>>> personalization,
  raft::device_span<result_t> pageranks,
  result_t alpha,
  result_t epsilon,
  size_t max_iterations,
  bool do_expensive_check)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  static_assert(std::is_integral<vertex_t>::value,
                "GraphViewType::vertex_type should be integral.");
  static_assert(std::is_floating_point<result_t>::value,
                "result_t should be a floating-point type.");
  static_assert(GraphViewType::is_storage_transposed,
                "GraphViewType should support the pull model.");

  auto const num_vertices = pull_graph_view.number_of_vertices();
  if (num_vertices == 0) { return centrality_algorithm_metadata_t{0, true}; }

  auto aggregate_personalization_vector_size =
    personalization ? GraphViewType::is_multi_gpu
                        ? host_scalar_allreduce(handle.get_comms(),
                                                std::get<0>(*personalization).size(),
                                                raft::comms::op_t::SUM,
                                                handle.get_stream())
                        : std::get<0>(*personalization).size()
                    : vertex_t{0};

  // 1. check input arguments

  CUGRAPH_EXPECTS((personalization.has_value() == false) ||
                    (std::get<0>(*personalization).size() == std::get<1>(*personalization).size()),
                  "Invalid input argument: if personalization.has_value() is true, the size of "
                  "vertices and values should match");
  CUGRAPH_EXPECTS(
    (personalization.has_value() == false) || (aggregate_personalization_vector_size > 0),
    "Invalid input argument: if personalizations.has_value() is true, the input "
    "personalization vector size should not be 0.");
  CUGRAPH_EXPECTS((alpha >= 0.0) && (alpha <= 1.0),
                  "Invalid input argument: alpha should be in [0.0, 1.0].");
  CUGRAPH_EXPECTS(epsilon >= 0.0, "Invalid input argument: epsilon should be non-negative.");

  if (do_expensive_check) {
    if (precomputed_vertex_out_weight_sums) {
      auto num_negative_precomputed_vertex_out_weight_sums =
        count_if_v(handle,
                   pull_graph_view,
                   precomputed_vertex_out_weight_sums->data(),
                   [] __device__(auto, auto val) { return val < result_t{0.0}; });
      CUGRAPH_EXPECTS(
        num_negative_precomputed_vertex_out_weight_sums == 0,
        "Invalid input argument: outgoing edge weight sum values should be non-negative.");
    }

    if (edge_weight_view) {
      auto num_negative_edge_weights =
        count_if_e(handle,
                   pull_graph_view,
                   edge_src_dummy_property_t{}.view(),
                   edge_dst_dummy_property_t{}.view(),
                   *edge_weight_view,
                   [] __device__(vertex_t, vertex_t, auto, auto, weight_t w) { return w < 0.0; });
      CUGRAPH_EXPECTS(
        num_negative_edge_weights == 0,
        "Invalid input argument: input edge weights should have non-negative values.");
    }

    if constexpr (GraphViewType::is_multi_gpu) {
      auto num_gpus_with_valid_personalization_vector =
        host_scalar_allreduce(handle.get_comms(),
                              personalization ? int{1} : int{0},
                              raft::comms::op_t::SUM,
                              handle.get_stream());
      CUGRAPH_EXPECTS(
        (num_gpus_with_valid_personalization_vector == 0) ||
          (num_gpus_with_valid_personalization_vector == handle.get_comms().get_size()),
        "Invalid input argument: personalization vectors should be provided in either no ranks or "
        "all ranks.");
    }

    if (aggregate_personalization_vector_size > 0) {
      auto vertex_partition = vertex_partition_device_view_t<vertex_t, GraphViewType::is_multi_gpu>(
        pull_graph_view.local_vertex_partition_view());
      auto num_invalid_vertices =
        thrust::count_if(handle.get_thrust_policy(),
                         std::get<0>(*personalization).begin(),
                         std::get<0>(*personalization).end(),
                         [vertex_partition] __device__(auto val) {
                           return !(vertex_partition.is_valid_vertex(val) &&
                                    vertex_partition.in_local_vertex_partition_range_nocheck(val));
                         });
      if constexpr (GraphViewType::is_multi_gpu) {
        num_invalid_vertices = host_scalar_allreduce(
          handle.get_comms(), num_invalid_vertices, raft::comms::op_t::SUM, handle.get_stream());
      }
      CUGRAPH_EXPECTS(num_invalid_vertices == 0,
                      "Invalid input argument: peresonalization vertices have invalid vertex IDs.");
      auto num_negative_values = thrust::count_if(handle.get_thrust_policy(),
                                                  std::get<1>(*personalization).begin(),
                                                  std::get<1>(*personalization).end(),
                                                  [] __device__(auto val) { return val < 0.0; });
      if constexpr (GraphViewType::is_multi_gpu) {
        num_negative_values = host_scalar_allreduce(
          handle.get_comms(), num_negative_values, raft::comms::op_t::SUM, handle.get_stream());
      }
      CUGRAPH_EXPECTS(num_negative_values == 0,
                      "Invalid input argument: peresonalization values should be non-negative.");

      rmm::device_uvector<vertex_t> check_for_duplicates(std::get<0>(*personalization).size(),
                                                         handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   std::get<0>(*personalization).begin(),
                   std::get<0>(*personalization).end(),
                   check_for_duplicates.begin());

      thrust::sort(
        handle.get_thrust_policy(), check_for_duplicates.begin(), check_for_duplicates.end());

      auto num_uniques =
        thrust::count_if(handle.get_thrust_policy(),
                         thrust::make_counting_iterator(size_t{0}),
                         thrust::make_counting_iterator(check_for_duplicates.size()),
                         detail::is_first_in_run_t<vertex_t const*>{check_for_duplicates.data()});

      CUGRAPH_EXPECTS(
        static_cast<size_t>(num_uniques) == check_for_duplicates.size(),
        "Invalid input argument: personalization vertices should not contain duplicate entries.");
    }
  }

  // 2. compute the sums of the out-going edge weights (if not provided)

  std::optional<rmm::device_uvector<weight_t>> tmp_vertex_out_weight_sums{std::nullopt};
  if (!precomputed_vertex_out_weight_sums) {
    if (edge_weight_view) {
      tmp_vertex_out_weight_sums =
        compute_out_weight_sums(handle, pull_graph_view, *edge_weight_view);
    } else {
      auto tmp_vertex_out_degrees = pull_graph_view.compute_out_degrees(handle);
      tmp_vertex_out_weight_sums =
        rmm::device_uvector<weight_t>(tmp_vertex_out_degrees.size(), handle.get_stream());
      thrust::transform(handle.get_thrust_policy(),
                        tmp_vertex_out_degrees.begin(),
                        tmp_vertex_out_degrees.end(),
                        (*tmp_vertex_out_weight_sums).begin(),
                        detail::typecast_t<edge_t, weight_t>{});
    }
  }
  auto vertex_out_weight_sums = precomputed_vertex_out_weight_sums
                                  ? (*precomputed_vertex_out_weight_sums).data()
                                  : (*tmp_vertex_out_weight_sums).data();

  // 3. sum the personalization values

  result_t personalization_sum{0.0};
  if (aggregate_personalization_vector_size > 0) {
    personalization_sum = thrust::reduce(handle.get_thrust_policy(),
                                         std::get<1>(*personalization).begin(),
                                         std::get<1>(*personalization).end(),
                                         result_t{0.0});
    if constexpr (GraphViewType::is_multi_gpu) {
      personalization_sum = host_scalar_allreduce(
        handle.get_comms(), personalization_sum, raft::comms::op_t::SUM, handle.get_stream());
    }
    CUGRAPH_EXPECTS(personalization_sum > 0.0,
                    "Invalid input argument: sum of personalization valuese "
                    "should be positive.");
  }

  // 5. pagerank iteration

  // old PageRank values
  rmm::device_uvector<result_t> old_pageranks(pull_graph_view.local_vertex_partition_range_size(),
                                              handle.get_stream());
  edge_src_property_t<vertex_t, result_t> edge_src_pageranks(handle, pull_graph_view);
  size_t iter{0};
  while (true) {
    thrust::copy(
      handle.get_thrust_policy(), pageranks.begin(), pageranks.end(), old_pageranks.data());

    auto dangling_sum = transform_reduce_v(
      handle,
      pull_graph_view,
      thrust::make_zip_iterator(pageranks.begin(), vertex_out_weight_sums),
      [] __device__(auto, auto val) {
        auto const pagerank       = thrust::get<0>(val);
        auto const out_weight_sum = thrust::get<1>(val);
        return out_weight_sum == result_t{0.0} ? pagerank : result_t{0.0};
      },
      result_t{0.0});

    thrust::transform(
      handle.get_thrust_policy(),
      thrust::make_zip_iterator(pageranks.begin(), vertex_out_weight_sums),
      thrust::make_zip_iterator(
        pageranks.end(),
        vertex_out_weight_sums + pull_graph_view.local_vertex_partition_range_size()),
      pageranks.begin(),
      [] __device__(auto val) {
        auto const pagerank       = thrust::get<0>(val);
        auto const out_weight_sum = thrust::get<1>(val);
        auto const divisor = out_weight_sum == result_t{0.0} ? result_t{1.0} : out_weight_sum;
        return pagerank / divisor;
      });

    update_edge_src_property(
      handle, pull_graph_view, pageranks.data(), edge_src_pageranks.mutable_view());

    auto unvarying_part = aggregate_personalization_vector_size == 0
                            ? (dangling_sum * alpha + static_cast<result_t>(1.0 - alpha)) /
                                static_cast<result_t>(num_vertices)
                            : result_t{0.0};

    if (edge_weight_view) {
      per_v_transform_reduce_incoming_e(
        handle,
        pull_graph_view,
        edge_src_pageranks.view(),
        edge_dst_dummy_property_t{}.view(),
        *edge_weight_view,
        [alpha] __device__(vertex_t, vertex_t, auto src_val, auto, weight_t w) {
          return src_val * w * alpha;
        },
        unvarying_part,
        reduce_op::plus<result_t>{},
        pageranks.begin());
    } else {
      per_v_transform_reduce_incoming_e(
        handle,
        pull_graph_view,
        edge_src_pageranks.view(),
        edge_dst_dummy_property_t{}.view(),
        edge_dummy_property_t{}.view(),
        [alpha] __device__(vertex_t, vertex_t, auto src_val, auto, auto) {
          return src_val * alpha;
        },
        unvarying_part,
        reduce_op::plus<result_t>{},
        pageranks.begin());
    }

    if (aggregate_personalization_vector_size > 0) {
      auto vertex_partition = vertex_partition_device_view_t<vertex_t, GraphViewType::is_multi_gpu>(
        pull_graph_view.local_vertex_partition_view());
      thrust::for_each(
        handle.get_thrust_policy(),
        thrust::make_zip_iterator(thrust::make_tuple(std::get<0>(*personalization).begin(),
                                                     std::get<1>(*personalization).begin())),
        thrust::make_zip_iterator(thrust::make_tuple(std::get<0>(*personalization).end(),
                                                     std::get<1>(*personalization).end())),
        [vertex_partition,
         pageranks = pageranks.data(),
         dangling_sum,
         personalization_sum,
         alpha] __device__(auto val) {
          auto v     = thrust::get<0>(val);
          auto value = thrust::get<1>(val);
          *(pageranks + vertex_partition.local_vertex_partition_offset_from_vertex_nocheck(v)) +=
            (dangling_sum * alpha + static_cast<result_t>(1.0 - alpha)) *
            (value / personalization_sum);
        });
    }

    auto diff_sum = transform_reduce_v(
      handle,
      pull_graph_view,
      thrust::make_zip_iterator(thrust::make_tuple(pageranks.begin(), old_pageranks.begin())),
      [] __device__(auto, auto val) { return std::abs(thrust::get<0>(val) - thrust::get<1>(val)); },
      result_t{0.0});

    iter++;

    if (diff_sum < epsilon) {
      break;
    } else if (iter >= max_iterations) {
      break;
    }
  }

  return centrality_algorithm_metadata_t{iter, (iter < max_iterations)};
}

}  // namespace detail

template <typename vertex_t, typename edge_t, typename weight_t, typename result_t, bool multi_gpu>
void pagerank(raft::handle_t const& handle,
              graph_view_t<vertex_t, edge_t, true, multi_gpu> const& graph_view,
              std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
              std::optional<weight_t const*> precomputed_vertex_out_weight_sums,
              std::optional<vertex_t const*> personalization_vertices,
              std::optional<result_t const*> personalization_values,
              std::optional<vertex_t> personalization_vector_size,
              result_t* pageranks,
              result_t alpha,
              result_t epsilon,
              size_t max_iterations,
              bool has_initial_guess,
              bool do_expensive_check)
{
  CUGRAPH_EXPECTS(!graph_view.has_edge_mask(), "unimplemented.");

  CUGRAPH_EXPECTS((personalization_vertices.has_value() == false) ||
                    (personalization_values.has_value() && personalization_vector_size.has_value()),
                  "Invalid input argument: if personalization_vertices.has_value() is true, ");

  // initialize pagerank values
  if (has_initial_guess) {
    if (do_expensive_check) {
      auto num_negative_values = count_if_v(
        handle, graph_view, pageranks, [] __device__(auto, auto val) { return val < 0.0; });
      CUGRAPH_EXPECTS(num_negative_values == 0,
                      "Invalid input argument: initial guess values should be non-negative.");
    }

    auto sum = reduce_v(handle, graph_view, pageranks, result_t{0.0});
    CUGRAPH_EXPECTS(sum > 0.0,
                    "Invalid input argument: sum of the PageRank initial "
                    "guess values should be positive.");
    thrust::transform(handle.get_thrust_policy(),
                      pageranks,
                      pageranks + graph_view.local_vertex_partition_range_size(),
                      pageranks,
                      [sum] __device__(auto val) { return val / sum; });
  } else {
    thrust::fill(handle.get_thrust_policy(),
                 pageranks,
                 pageranks + graph_view.local_vertex_partition_range_size(),
                 result_t{1.0} / static_cast<result_t>(graph_view.number_of_vertices()));
  }

  auto metadata = detail::pagerank(
    handle,
    graph_view,
    edge_weight_view,
    precomputed_vertex_out_weight_sums
      ? std::make_optional(raft::device_span<weight_t const>{
          *precomputed_vertex_out_weight_sums,
          static_cast<size_t>(graph_view.local_vertex_partition_range_size())})
      : std::nullopt,
    personalization_vertices
      ? std::make_optional(std::make_tuple(
          raft::device_span<vertex_t const>{*personalization_vertices,
                                            static_cast<size_t>(*personalization_vector_size)},
          raft::device_span<result_t const>{*personalization_values,
                                            static_cast<size_t>(*personalization_vector_size)}))
      : std::nullopt,
    raft::device_span<result_t>{
      pageranks, static_cast<size_t>(graph_view.local_vertex_partition_range_size())},
    alpha,
    epsilon,
    max_iterations,
    do_expensive_check);

  CUGRAPH_EXPECTS(metadata.converged_, "PageRank failed to converge.");
}

template <typename vertex_t, typename edge_t, typename weight_t, typename result_t, bool multi_gpu>
std::tuple<rmm::device_uvector<result_t>, centrality_algorithm_metadata_t> pagerank(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, true, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::optional<raft::device_span<weight_t const>> precomputed_vertex_out_weight_sums,
  std::optional<std::tuple<raft::device_span<vertex_t const>, raft::device_span<result_t const>>>
    personalization,
  std::optional<raft::device_span<result_t const>> initial_pageranks,
  result_t alpha,
  result_t epsilon,
  size_t max_iterations,
  bool do_expensive_check)
{
  rmm::device_uvector<result_t> local_pageranks(graph_view.local_vertex_partition_range_size(),
                                                handle.get_stream());
  if (!initial_pageranks) {
    thrust::fill(handle.get_thrust_policy(),
                 local_pageranks.begin(),
                 local_pageranks.end(),
                 result_t{1.0} / graph_view.number_of_vertices());
  } else {
    thrust::copy(handle.get_thrust_policy(),
                 initial_pageranks->begin(),
                 initial_pageranks->end(),
                 local_pageranks.begin());
  }

  auto metadata =
    detail::pagerank(handle,
                     graph_view,
                     edge_weight_view,
                     precomputed_vertex_out_weight_sums,
                     personalization,
                     raft::device_span<result_t>{local_pageranks.data(), local_pageranks.size()},
                     alpha,
                     epsilon,
                     max_iterations,
                     do_expensive_check);

  return std::make_tuple(std::move(local_pageranks), metadata);
}

}  // namespace cugraph
