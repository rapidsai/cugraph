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
#include <prims/per_v_transform_reduce_incoming_outgoing_e.cuh>
#include <prims/update_edge_src_dst_property.cuh>

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

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu, typename functor_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           rmm::device_uvector<weight_t>>
all_pairs_similarity(raft::handle_t const& handle,
                     graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
                     std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
                     std::optional<raft::device_span<vertex_t const>> source_vertices,
                     std::optional<size_t> topk,
                     functor_t functor,
                     bool do_expensive_check = false)
{
  using GraphViewType = graph_view_t<vertex_t, edge_t, false, multi_gpu>;

  CUGRAPH_EXPECTS(graph_view.is_symmetric(),
                  "similarity algorithms require an undirected(symmetric) graph");

  if (do_expensive_check) {
    if (source_vertices) {
      auto vertex_partition = vertex_partition_device_view_t<vertex_t, GraphViewType::is_multi_gpu>(
        graph_view.local_vertex_partition_view());
      auto num_invalid_vertices =
        thrust::count_if(handle.get_thrust_policy(),
                         source_vertices->begin(),
                         source_vertices->end(),
                         [vertex_partition] __device__(auto val) {
                           return !(vertex_partition.is_valid_vertex(val) &&
                                    vertex_partition.in_local_vertex_partition_range_nocheck(val));
                         });

      if constexpr (multi_gpu) {
        num_invalid_vertices = cugraph::host_scalar_allreduce(
          handle.get_comms(), num_invalid_vertices, raft::comms::op_t::SUM, handle.get_stream());
      }

      CUGRAPH_EXPECTS(num_invalid_vertices == 0,
                      "Invalid input arguments: there are invalid input vertices.");
    }

    if (edge_weight_view) {
      auto num_negative_edge_weights =
        count_if_e(handle,
                   graph_view,
                   edge_src_dummy_property_t{}.view(),
                   edge_dst_dummy_property_t{}.view(),
                   *edge_weight_view,
                   [] __device__(vertex_t, vertex_t, auto, auto, weight_t w) { return w < 0.0; });

      if constexpr (multi_gpu) {
        num_negative_edge_weights = cugraph::host_scalar_allreduce(handle.get_comms(),
                                                                   num_negative_edge_weights,
                                                                   raft::comms::op_t::SUM,
                                                                   handle.get_stream());
      }

      CUGRAPH_EXPECTS(
        num_negative_edge_weights == 0,
        "Invalid input argument: input edge weights should have non-negative values.");
    }
  }

  rmm::device_uvector<vertex_t> sources(0, handle.get_stream());

  if (source_vertices) {
    sources.resize(source_vertices->size(), handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 source_vertices->begin(),
                 source_vertices->end(),
                 sources.begin());
  } else {
    sources.resize(graph_view.local_vertex_partition_range_size(), handle.get_stream());
    thrust::sequence(handle.get_thrust_policy(),
                     sources.begin(),
                     sources.end(),
                     graph_view.local_vertex_partition_range_first());
  }

  if (topk) {
    std::cout << "topk = " << *topk << std::endl;
    //  We can reduce memory footprint by doing work in batches and
    //  computing/updating topk with each batch
    rmm::device_uvector<vertex_t> top_v1(*topk, handle.get_stream());
    rmm::device_uvector<vertex_t> top_v2(*topk, handle.get_stream());
    rmm::device_uvector<weight_t> top_score(*topk, handle.get_stream());

    //   FIXME: Think about what this should be
    // edge_t const MAX_PAIRS{2 << 20};
    edge_t const MAX_PAIRS{32768};

    rmm::device_uvector<edge_t> degrees = graph_view.compute_out_degrees(handle);
    rmm::device_uvector<edge_t> two_hop_degrees(degrees.size(), handle.get_stream());

    // Let's compute the maximum size of the 2-hop neighborhood of each vertex
    // FIXME: If sources is specified, this could be done on a subset of the vertices
    //
    edge_dst_property_t<GraphViewType, edge_t> edge_dst_degrees(handle, graph_view);
    per_v_transform_reduce_incoming_e(
      handle,
      graph_view,
      edge_src_dummy_property_t{}.view(),
      edge_dst_degrees.view(),
      edge_dummy_property_t{}.view(),
      [] __device__(vertex_t, vertex_t, auto, auto dst_degree, auto) { return dst_degree; },
      edge_t{0},
      reduce_op::plus<edge_t>{},
      two_hop_degrees.begin());

    thrust::sort_by_key(handle.get_thrust_policy(),
                        two_hop_degrees.begin(),
                        two_hop_degrees.end(),
                        sources.begin(),
                        thrust::greater<edge_t>{});

    thrust::inclusive_scan(handle.get_thrust_policy(),
                           two_hop_degrees.begin(),
                           two_hop_degrees.end(),
                           two_hop_degrees.begin());

    size_t current_pos{0};
    size_t next_pos{0};
    edge_t next_boundary{MAX_PAIRS};

    while (true) {
      std::cout << "processing a batch, current_pos = " << current_pos << std::endl;
      if (current_pos < two_hop_degrees.size()) {
        next_pos = current_pos + thrust::distance(two_hop_degrees.begin() + current_pos,
                                                  thrust::upper_bound(handle.get_thrust_policy(),
                                                                      two_hop_degrees.begin(),
                                                                      two_hop_degrees.end(),
                                                                      next_boundary));

        if (next_pos == current_pos) next_pos++;
      }

      size_t batch_size = next_pos - current_pos;

      if constexpr (multi_gpu) {
        batch_size = cugraph::host_scalar_allreduce(
          handle.get_comms(), batch_size, raft::comms::op_t::SUM, handle.get_stream());
      }

      if (batch_size == 0) break;

      auto [offsets, v2] = k_hop_nbrs(
        handle,
        graph_view,
        raft::device_span<vertex_t const>{sources.begin() + current_pos, next_pos - current_pos},
        2,
        do_expensive_check);

      auto v1 = cugraph::detail::expand_sparse_offsets(
        raft::device_span<size_t const>{offsets.data(), offsets.size()},
        vertex_t{0},
        handle.get_stream());

      cugraph::unrenumber_local_int_vertices(handle,
                                             v1.data(),
                                             v1.size(),
                                             sources.data() + current_pos,
                                             vertex_t{0},
                                             static_cast<vertex_t>(next_pos - current_pos),
                                             do_expensive_check);

      auto new_size = thrust::distance(
        thrust::make_zip_iterator(v1.begin(), v2.begin()),
        thrust::remove_if(
          handle.get_thrust_policy(),
          thrust::make_zip_iterator(v1.begin(), v2.begin()),
          thrust::make_zip_iterator(v1.end(), v2.end()),
          [] __device__(auto tuple) { return thrust::get<0>(tuple) == thrust::get<1>(tuple); }));

      v1.resize(new_size, handle.get_stream());
      v2.resize(new_size, handle.get_stream());

      auto score =
        similarity(handle,
                   graph_view,
                   edge_weight_view,
                   std::make_tuple(raft::device_span<vertex_t const>{v1.data(), v1.size()},
                                   raft::device_span<vertex_t const>{v2.data(), v2.size()}),
                   functor,
                   do_expensive_check);

      thrust::sort_by_key(handle.get_thrust_policy(),
                          score.begin(),
                          score.end(),
                          thrust::make_zip_iterator(v1.begin(), v2.begin()),
                          thrust::greater<weight_t>{});

      if (score.size() < (2 * (*topk))) {
        score.resize(2 * (*topk), handle.get_stream());
        v1.resize(2 * (*topk), handle.get_stream());
        v2.resize(2 * (*topk), handle.get_stream());
      }

      thrust::copy(handle.get_thrust_policy(), top_v1.begin(), top_v1.end(), v1.begin() + *topk);
      thrust::copy(handle.get_thrust_policy(), top_v2.begin(), top_v2.end(), v2.begin() + *topk);
      thrust::copy(
        handle.get_thrust_policy(), top_score.begin(), top_score.end(), score.begin() + *topk);

      thrust::sort_by_key(handle.get_thrust_policy(),
                          score.begin(),
                          score.end(),
                          thrust::make_zip_iterator(v1.begin(), v2.begin()),
                          thrust::greater<weight_t>{});

      thrust::copy(handle.get_thrust_policy(), v1.begin(), v1.end(), top_v1.begin());
      thrust::copy(handle.get_thrust_policy(), v2.begin(), v2.end(), top_v2.begin());
      thrust::copy(handle.get_thrust_policy(), score.begin(), score.end(), top_score.begin());

      current_pos = next_pos;
    }

    return std::make_tuple(std::move(top_v1), std::move(top_v2), std::move(top_score));
  } else {
    std::cout << "topk not specified " << std::endl;

    auto [offsets, v2] =
      k_hop_nbrs(handle,
                 graph_view,
                 raft::device_span<vertex_t const>{sources.data(), sources.size()},
                 2,
                 do_expensive_check);

    auto v1 = cugraph::detail::expand_sparse_offsets(
      raft::device_span<size_t const>{offsets.data(), offsets.size()},
      vertex_t{0},
      handle.get_stream());

    cugraph::unrenumber_local_int_vertices(handle,
                                           v1.data(),
                                           v1.size(),
                                           sources.data(),
                                           vertex_t{0},
                                           static_cast<vertex_t>(sources.size()),
                                           do_expensive_check);

    auto new_size = thrust::distance(
      thrust::make_zip_iterator(v1.begin(), v2.begin()),
      thrust::remove_if(
        handle.get_thrust_policy(),
        thrust::make_zip_iterator(v1.begin(), v2.begin()),
        thrust::make_zip_iterator(v1.end(), v2.end()),
        [] __device__(auto tuple) { return thrust::get<0>(tuple) == thrust::get<1>(tuple); }));

    v1.resize(new_size, handle.get_stream());
    v2.resize(new_size, handle.get_stream());

    auto score =
      similarity(handle,
                 graph_view,
                 edge_weight_view,
                 std::make_tuple(raft::device_span<vertex_t const>{v1.data(), v1.size()},
                                 raft::device_span<vertex_t const>{v2.data(), v2.size()}),
                 functor,
                 do_expensive_check);

    return std::make_tuple(std::move(v1), std::move(v2), std::move(score));
  }
}

}  // namespace detail
}  // namespace cugraph
