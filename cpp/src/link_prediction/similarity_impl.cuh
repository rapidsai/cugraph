/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <prims/count_if_e.cuh>
#include <prims/per_v_pair_transform_dst_nbr_intersection.cuh>
#include <prims/update_edge_src_dst_property.cuh>
#include <utilities/error_check_utils.cuh>

#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <optional>
#include <tuple>

namespace cugraph {
namespace detail {

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu, typename functor_t>
rmm::device_uvector<weight_t> similarity(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::tuple<raft::device_span<vertex_t const>, raft::device_span<vertex_t const>> vertex_pairs,
  functor_t functor,
  bool do_expensive_check = false)
{
  using GraphViewType = graph_view_t<vertex_t, edge_t, false, multi_gpu>;

  CUGRAPH_EXPECTS(std::get<0>(vertex_pairs).size() == std::get<1>(vertex_pairs).size(),
                  "vertex pairs have mismatched sizes");
  CUGRAPH_EXPECTS(graph_view.is_symmetric(),
                  "similarity algorithms require an undirected(symmetric) graph");

  size_t num_vertex_pairs = std::get<0>(vertex_pairs).size();
  auto vertex_pairs_begin =
    thrust::make_zip_iterator(std::get<0>(vertex_pairs).data(), std::get<1>(vertex_pairs).data());

  if (do_expensive_check) {
    auto num_invalids = detail::count_invalid_vertex_pairs(
      handle, graph_view, vertex_pairs_begin, vertex_pairs_begin + num_vertex_pairs);
    CUGRAPH_EXPECTS(num_invalids == 0,
                    "Invalid input arguments: there are invalid input vertex pairs.");

    if (edge_weight_view) {
      auto num_negative_edge_weights =
        count_if_e(handle,
                   graph_view,
                   edge_src_dummy_property_t{}.view(),
                   edge_dst_dummy_property_t{}.view(),
                   *edge_weight_view,
                   [] __device__(vertex_t, vertex_t, auto, auto, weight_t w) { return w < 0.0; });
      CUGRAPH_EXPECTS(
        num_negative_edge_weights == 0,
        "Invalid input argument: input edge weights should have non-negative values.");
    }
  }

  if (edge_weight_view) {
    rmm::device_uvector<weight_t> similarity_score(num_vertex_pairs, handle.get_stream());
    rmm::device_uvector<weight_t> weighted_out_degrees =
      compute_out_weight_sums(handle, graph_view, *edge_weight_view);

    per_v_pair_transform_dst_nbr_intersection(
      handle,
      graph_view,
      *edge_weight_view,
      vertex_pairs_begin,
      vertex_pairs_begin + num_vertex_pairs,
      weighted_out_degrees.begin(),
      [functor] __device__(auto a,
                           auto b,
                           auto weight_a,
                           auto weight_b,
                           auto intersection,
                           auto intersected_properties_a,
                           auto intersected_properties_b) {
        weight_t sum_of_min_weight_a_intersect_b = weight_t{0};
        weight_t sum_of_max_weight_a_intersect_b = weight_t{0};
        weight_t sum_of_intersected_a            = weight_t{0};
        weight_t sum_of_intersected_b            = weight_t{0};

        auto pair_first = thrust::make_zip_iterator(intersected_properties_a.data(),
                                                    intersected_properties_b.data());
        thrust::tie(sum_of_min_weight_a_intersect_b,
                    sum_of_max_weight_a_intersect_b,
                    sum_of_intersected_a,
                    sum_of_intersected_b) =
          thrust::transform_reduce(
            thrust::seq,
            pair_first,
            pair_first + intersected_properties_a.size(),
            [] __device__(auto property_pair) {
              auto prop_a = thrust::get<0>(property_pair);
              auto prop_b = thrust::get<1>(property_pair);
              return thrust::make_tuple(min(prop_a, prop_b), max(prop_a, prop_b), prop_a, prop_b);
            },
            thrust::make_tuple(weight_t{0}, weight_t{0}, weight_t{0}, weight_t{0}),
            [] __device__(auto lhs, auto rhs) {
              return thrust::make_tuple(thrust::get<0>(lhs) + thrust::get<0>(rhs),
                                        thrust::get<1>(lhs) + thrust::get<1>(rhs),
                                        thrust::get<2>(lhs) + thrust::get<2>(rhs),
                                        thrust::get<3>(lhs) + thrust::get<3>(rhs));
            });

        weight_t sum_of_uniq_a = weight_a - sum_of_intersected_a;
        weight_t sum_of_uniq_b = weight_b - sum_of_intersected_b;

        sum_of_max_weight_a_intersect_b += sum_of_uniq_a + sum_of_uniq_b;

        return functor.compute_score(static_cast<weight_t>(weight_a),
                                     static_cast<weight_t>(weight_b),
                                     static_cast<weight_t>(sum_of_min_weight_a_intersect_b),
                                     static_cast<weight_t>(sum_of_max_weight_a_intersect_b));
      },
      similarity_score.begin(),
      do_expensive_check);

    return similarity_score;
  } else {
    rmm::device_uvector<weight_t> similarity_score(num_vertex_pairs, handle.get_stream());

    auto out_degrees = graph_view.compute_out_degrees(handle);

    per_v_pair_transform_dst_nbr_intersection(
      handle,
      graph_view,
      cugraph::edge_dummy_property_t{}.view(),
      vertex_pairs_begin,
      vertex_pairs_begin + num_vertex_pairs,
      out_degrees.begin(),
      [functor] __device__(
        auto v1, auto v2, auto v1_degree, auto v2_degree, auto intersection, auto, auto) {
        return functor.compute_score(
          static_cast<weight_t>(v1_degree),
          static_cast<weight_t>(v2_degree),
          static_cast<weight_t>(intersection.size()),
          static_cast<weight_t>(v1_degree + v2_degree - intersection.size()));
      },
      similarity_score.begin(),
      do_expensive_check);

    return similarity_score;
  }
}

}  // namespace detail
}  // namespace cugraph
