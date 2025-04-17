/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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
#include "prims/per_v_pair_transform_dst_nbr_intersection.cuh"
#include "prims/per_v_transform_reduce_incoming_outgoing_e.cuh"
#include "prims/update_edge_src_dst_property.cuh"
#include "utilities/error_check_utils.cuh"

#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/std/iterator>

#include <optional>
#include <tuple>

namespace cugraph {
namespace detail {

enum class coefficient_t { JACCARD, SORENSEN, OVERLAP, COSINE };

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu, typename functor_t>
rmm::device_uvector<weight_t> similarity(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
  std::tuple<raft::device_span<vertex_t const>, raft::device_span<vertex_t const>> vertex_pairs,
  functor_t functor,
  coefficient_t coeff,
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
      [functor, coeff] __device__(auto a,
                                  auto b,
                                  auto weight_a,
                                  auto weight_b,
                                  auto intersection,
                                  auto intersected_properties_a,
                                  auto intersected_properties_b) {
        if (coeff == coefficient_t::COSINE) {
          weight_t norm_a                    = weight_t{0};
          weight_t norm_b                    = weight_t{0};
          weight_t sum_of_product_of_a_and_b = weight_t{0};

          auto pair_first = thrust::make_zip_iterator(intersected_properties_a.data(),
                                                      intersected_properties_b.data());
          thrust::tie(norm_a, norm_b, sum_of_product_of_a_and_b) = thrust::transform_reduce(
            thrust::seq,
            pair_first,
            pair_first + intersected_properties_a.size(),
            [] __device__(auto property_pair) {
              auto prop_a = thrust::get<0>(property_pair);
              auto prop_b = thrust::get<1>(property_pair);
              return thrust::make_tuple(prop_a * prop_a, prop_b * prop_b, prop_a * prop_b);
            },
            thrust::make_tuple(weight_t{0}, weight_t{0}, weight_t{0}),
            [] __device__(auto lhs, auto rhs) {
              return thrust::make_tuple(thrust::get<0>(lhs) + thrust::get<0>(rhs),
                                        thrust::get<1>(lhs) + thrust::get<1>(rhs),
                                        thrust::get<2>(lhs) + thrust::get<2>(rhs));
            });

          return functor.compute_score(static_cast<weight_t>(sqrt(norm_a)),
                                       static_cast<weight_t>(sqrt(norm_b)),
                                       static_cast<weight_t>(sum_of_product_of_a_and_b),
                                       weight_t{1.0});

        } else {
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
        }
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
      [functor, coeff] __device__(
        auto v1, auto v2, auto v1_degree, auto v2_degree, auto intersection, auto, auto) {
        if (coeff == coefficient_t::COSINE) {
          return functor.compute_score(weight_t{1},
                                       weight_t{1},
                                       intersection.size() >= 1 ? weight_t{1} : weight_t{0},
                                       weight_t{1});
        } else {
          return functor.compute_score(
            static_cast<weight_t>(v1_degree),
            static_cast<weight_t>(v2_degree),
            static_cast<weight_t>(intersection.size()),
            static_cast<weight_t>(v1_degree + v2_degree - intersection.size()));
        }
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
                     std::optional<raft::device_span<vertex_t const>> vertices,
                     std::optional<size_t> topk,
                     functor_t functor,
                     coefficient_t coeff,
                     bool do_expensive_check = false)
{
  using GraphViewType = graph_view_t<vertex_t, edge_t, false, multi_gpu>;

  CUGRAPH_EXPECTS(graph_view.is_symmetric(),
                  "similarity algorithms require an undirected(symmetric) graph");

  // FIXME: See https://github.com/rapidsai/cugraph/issues/4132
  //   Once that issue is resolved we can drop this check
  CUGRAPH_EXPECTS(!graph_view.is_multigraph() || !edge_weight_view,
                  "Weighted implementation currently fails on multi-graph");

  if (do_expensive_check) {
    if (vertices) {
      auto vertex_partition = vertex_partition_device_view_t<vertex_t, GraphViewType::is_multi_gpu>(
        graph_view.local_vertex_partition_view());
      auto num_invalid_vertices =
        thrust::count_if(handle.get_thrust_policy(),
                         vertices->begin(),
                         vertices->end(),
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

  if (topk) {
    rmm::device_uvector<vertex_t> tmp_vertices(0, handle.get_stream());

    if (vertices) {
      tmp_vertices.resize(vertices->size(), handle.get_stream());
      thrust::copy(
        handle.get_thrust_policy(), vertices->begin(), vertices->end(), tmp_vertices.begin());
    } else {
      tmp_vertices.resize(graph_view.local_vertex_partition_range_size(), handle.get_stream());
      thrust::sequence(handle.get_thrust_policy(),
                       tmp_vertices.begin(),
                       tmp_vertices.end(),
                       graph_view.local_vertex_partition_range_first());
    }

    //  We can reduce memory footprint by doing work in batches and
    //  computing/updating topk with each batch

    //   FIXME: Experiment with this and adjust as necessary
    size_t const MAX_PAIRS_PER_BATCH{
      static_cast<size_t>(handle.get_device_properties().multiProcessorCount) * (1 << 15)};

    rmm::device_uvector<edge_t> degrees = graph_view.compute_out_degrees(handle);
    rmm::device_uvector<size_t> two_hop_degrees(degrees.size() + 1, handle.get_stream());

    // Let's compute the maximum size of the 2-hop neighborhood of each vertex
    // FIXME: If vertices is specified, this could be done on a subset of the vertices
    //
    edge_dst_property_t<GraphViewType, edge_t> edge_dst_degrees(handle, graph_view);
    update_edge_dst_property(handle, graph_view, degrees.begin(), edge_dst_degrees.mutable_view());

    per_v_transform_reduce_incoming_e(
      handle,
      graph_view,
      edge_src_dummy_property_t{}.view(),
      edge_dst_degrees.view(),
      edge_dummy_property_t{}.view(),
      [] __device__(vertex_t, vertex_t, auto, auto dst_degree, auto) {
        return static_cast<size_t>(dst_degree);
      },
      size_t{0},
      reduce_op::plus<size_t>{},
      two_hop_degrees.begin());

    if (vertices) {
      rmm::device_uvector<size_t> gathered_two_hop_degrees(tmp_vertices.size() + 1,
                                                           handle.get_stream());

      thrust::gather(
        handle.get_thrust_policy(),
        thrust::make_transform_iterator(
          tmp_vertices.begin(),
          cugraph::detail::shift_left_t<vertex_t>{graph_view.local_vertex_partition_range_first()}),
        thrust::make_transform_iterator(
          tmp_vertices.end(),
          cugraph::detail::shift_left_t<vertex_t>{graph_view.local_vertex_partition_range_first()}),
        two_hop_degrees.begin(),
        gathered_two_hop_degrees.begin());

      two_hop_degrees = std::move(gathered_two_hop_degrees);
    }

    thrust::sort_by_key(handle.get_thrust_policy(),
                        two_hop_degrees.begin(),
                        two_hop_degrees.end() - 1,
                        tmp_vertices.begin(),
                        thrust::greater<size_t>{});

    thrust::exclusive_scan(handle.get_thrust_policy(),
                           two_hop_degrees.begin(),
                           two_hop_degrees.end(),
                           two_hop_degrees.begin());

    auto two_hop_degree_offsets = std::move(two_hop_degrees);

    rmm::device_uvector<vertex_t> top_v1(0, handle.get_stream());
    rmm::device_uvector<vertex_t> top_v2(0, handle.get_stream());
    rmm::device_uvector<weight_t> top_score(0, handle.get_stream());

    top_v1.reserve(*topk, handle.get_stream());
    top_v2.reserve(*topk, handle.get_stream());
    top_score.reserve(*topk, handle.get_stream());

    size_t sum_two_hop_degrees{0};
    weight_t similarity_threshold{0};
    std::vector<size_t> batch_offsets;

    raft::update_host(&sum_two_hop_degrees,
                      two_hop_degree_offsets.data() + two_hop_degree_offsets.size() - 1,
                      1,
                      handle.get_stream());

    handle.sync_stream();

    std::tie(batch_offsets, std::ignore) = compute_offset_aligned_element_chunks(
      handle,
      raft::device_span<size_t const>{two_hop_degree_offsets.data(), two_hop_degree_offsets.size()},
      sum_two_hop_degrees,
      MAX_PAIRS_PER_BATCH);

    // FIXME: compute_offset_aligned_element_chunks can return duplicates.  Should it?  Should
    // explore
    //  whether this functionality should be pushed into that function
    batch_offsets.resize(std::distance(batch_offsets.begin(),
                                       std::unique(batch_offsets.begin(), batch_offsets.end())));

    size_t num_batches = batch_offsets.size() - 1;
    if constexpr (multi_gpu) {
      num_batches = cugraph::host_scalar_allreduce(
        handle.get_comms(), num_batches, raft::comms::op_t::MAX, handle.get_stream());
    }

    for (size_t batch_number = 0; batch_number < num_batches; ++batch_number) {
      raft::device_span<vertex_t const> batch_seeds{tmp_vertices.data(), size_t{0}};

      if (((batch_number + 1) < batch_offsets.size()) &&
          (batch_offsets[batch_number + 1] > batch_offsets[batch_number])) {
        batch_seeds = raft::device_span<vertex_t const>{
          tmp_vertices.data() + batch_offsets[batch_number],
          batch_offsets[batch_number + 1] - batch_offsets[batch_number]};
      }

      auto [offsets, v2] = k_hop_nbrs(handle, graph_view, batch_seeds, 2, do_expensive_check);

      auto v1 = cugraph::detail::expand_sparse_offsets(
        raft::device_span<size_t const>{offsets.data(), offsets.size()},
        vertex_t{0},
        handle.get_stream());

      cugraph::unrenumber_local_int_vertices(
        handle,
        v1.data(),
        v1.size(),
        tmp_vertices.data() + batch_offsets[batch_number],
        vertex_t{0},
        static_cast<vertex_t>(batch_offsets[batch_number + 1] - batch_offsets[batch_number]),
        do_expensive_check);

      auto new_size = cuda::std::distance(
        thrust::make_zip_iterator(v1.begin(), v2.begin()),
        thrust::remove_if(
          handle.get_thrust_policy(),
          thrust::make_zip_iterator(v1.begin(), v2.begin()),
          thrust::make_zip_iterator(v1.end(), v2.end()),
          [] __device__(auto tuple) { return thrust::get<0>(tuple) == thrust::get<1>(tuple); }));

      v1.resize(new_size, handle.get_stream());
      v2.resize(new_size, handle.get_stream());

      if constexpr (multi_gpu) {
        // shuffle vertex pairs
        auto vertex_partition_range_lasts = graph_view.vertex_partition_range_lasts();

        std::tie(
          v1, v2, std::ignore, std::ignore, std::ignore, std::ignore, std::ignore, std::ignore) =
          detail::shuffle_int_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning<vertex_t,
                                                                                         edge_t,
                                                                                         weight_t,
                                                                                         int32_t,
                                                                                         int32_t>(
            handle,
            std::move(v1),
            std::move(v2),
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            vertex_partition_range_lasts);
      }

      auto score =
        similarity(handle,
                   graph_view,
                   edge_weight_view,
                   std::make_tuple(raft::device_span<vertex_t const>{v1.data(), v1.size()},
                                   raft::device_span<vertex_t const>{v2.data(), v2.size()}),
                   functor,
                   coeff,
                   do_expensive_check);

      // Add a remove_if to remove items that are less than the last topk element
      new_size = cuda::std::distance(
        thrust::make_zip_iterator(score.begin(), v1.begin(), v2.begin()),
        thrust::remove_if(handle.get_thrust_policy(),
                          thrust::make_zip_iterator(score.begin(), v1.begin(), v2.begin()),
                          thrust::make_zip_iterator(score.end(), v1.end(), v2.end()),
                          [similarity_threshold] __device__(auto tuple) {
                            return thrust::get<0>(tuple) < similarity_threshold;
                          }));

      score.resize(new_size, handle.get_stream());
      v1.resize(new_size, handle.get_stream());
      v2.resize(new_size, handle.get_stream());

      thrust::sort_by_key(handle.get_thrust_policy(),
                          score.begin(),
                          score.end(),
                          thrust::make_zip_iterator(v1.begin(), v2.begin()),
                          thrust::greater<weight_t>{});

      size_t v1_keep = std::min(*topk, v1.size());

      if (score.size() < (top_v1.size() + v1_keep)) {
        score.resize(top_v1.size() + v1_keep, handle.get_stream());
        v1.resize(score.size(), handle.get_stream());
        v2.resize(score.size(), handle.get_stream());
      }

      thrust::copy(handle.get_thrust_policy(), top_v1.begin(), top_v1.end(), v1.begin() + v1_keep);
      thrust::copy(handle.get_thrust_policy(), top_v2.begin(), top_v2.end(), v2.begin() + v1_keep);
      thrust::copy(
        handle.get_thrust_policy(), top_score.begin(), top_score.end(), score.begin() + v1_keep);

      thrust::sort_by_key(handle.get_thrust_policy(),
                          score.begin(),
                          score.end(),
                          thrust::make_zip_iterator(v1.begin(), v2.begin()),
                          thrust::greater<weight_t>{});

      if (top_v1.size() < std::min(*topk, v1.size())) {
        top_v1.resize(std::min(*topk, v1.size()), handle.get_stream());
        top_v2.resize(top_v1.size(), handle.get_stream());
        top_score.resize(top_v1.size(), handle.get_stream());
      }

      thrust::copy(
        handle.get_thrust_policy(), v1.begin(), v1.begin() + top_v1.size(), top_v1.begin());
      thrust::copy(
        handle.get_thrust_policy(), v2.begin(), v2.begin() + top_v1.size(), top_v2.begin());
      thrust::copy(handle.get_thrust_policy(),
                   score.begin(),
                   score.begin() + top_v1.size(),
                   top_score.begin());

      if constexpr (multi_gpu) {
        bool is_root  = handle.get_comms().get_rank() == int{0};
        auto rx_sizes = cugraph::host_scalar_gather(
          handle.get_comms(), top_v1.size(), int{0}, handle.get_stream());
        std::vector<size_t> rx_displs;
        size_t gathered_size{0};

        if (is_root) {
          rx_displs.resize(handle.get_comms().get_size());
          rx_displs[0] = 0;
          std::partial_sum(rx_sizes.begin(), rx_sizes.end() - 1, rx_displs.begin() + 1);
          gathered_size = std::reduce(rx_sizes.begin(), rx_sizes.end());
        }

        rmm::device_uvector<vertex_t> gathered_v1(gathered_size, handle.get_stream());
        rmm::device_uvector<vertex_t> gathered_v2(gathered_size, handle.get_stream());
        rmm::device_uvector<weight_t> gathered_score(gathered_size, handle.get_stream());

        cugraph::device_gatherv(
          handle.get_comms(),
          thrust::make_zip_iterator(top_v1.begin(), top_v2.begin(), top_score.begin()),
          thrust::make_zip_iterator(
            gathered_v1.begin(), gathered_v2.begin(), gathered_score.begin()),
          top_v1.size(),
          raft::host_span<size_t const>(rx_sizes.data(), rx_sizes.size()),
          raft::host_span<size_t const>(rx_displs.data(), rx_displs.size()),
          int{0},
          handle.get_stream());

        if (is_root) {
          thrust::sort_by_key(handle.get_thrust_policy(),
                              gathered_score.begin(),
                              gathered_score.end(),
                              thrust::make_zip_iterator(gathered_v1.begin(), gathered_v2.begin()),
                              thrust::greater<weight_t>{});

          if (gathered_v1.size() > *topk) {
            gathered_v1.resize(*topk, handle.get_stream());
            gathered_v2.resize(*topk, handle.get_stream());
            gathered_score.resize(*topk, handle.get_stream());
          }

          top_v1    = std::move(gathered_v1);
          top_v2    = std::move(gathered_v2);
          top_score = std::move(gathered_score);
        } else {
          top_v1.resize(0, handle.get_stream());
          top_v2.resize(0, handle.get_stream());
          top_score.resize(0, handle.get_stream());
        }
      }

      if (top_score.size() == *topk) {
        raft::update_host(
          &similarity_threshold, top_score.data() + *topk - 1, 1, handle.get_stream());
      }
      if constexpr (multi_gpu) {
        similarity_threshold =
          host_scalar_bcast(handle.get_comms(), similarity_threshold, int{0}, handle.get_stream());
      }
    }

    return std::make_tuple(std::move(top_v1), std::move(top_v2), std::move(top_score));
  } else {
    rmm::device_uvector<vertex_t> tmp_vertices(0, handle.get_stream());
    raft::device_span<vertex_t const> vertices_span{nullptr, size_t{0}};

    if (vertices) {
      vertices_span = raft::device_span<vertex_t const>{vertices->data(), vertices->size()};
    } else {
      tmp_vertices.resize(graph_view.local_vertex_partition_range_size(), handle.get_stream());
      thrust::sequence(handle.get_thrust_policy(),
                       tmp_vertices.begin(),
                       tmp_vertices.end(),
                       graph_view.local_vertex_partition_range_first());
      vertices_span = raft::device_span<vertex_t const>{tmp_vertices.data(), tmp_vertices.size()};
    }

    auto [offsets, v2] = k_hop_nbrs(handle, graph_view, vertices_span, 2, do_expensive_check);

    auto v1 = cugraph::detail::expand_sparse_offsets(
      raft::device_span<size_t const>{offsets.data(), offsets.size()},
      vertex_t{0},
      handle.get_stream());

    cugraph::unrenumber_local_int_vertices(handle,
                                           v1.data(),
                                           v1.size(),
                                           vertices_span.data(),
                                           vertex_t{0},
                                           static_cast<vertex_t>(vertices_span.size()),
                                           do_expensive_check);

    auto new_size = cuda::std::distance(
      thrust::make_zip_iterator(v1.begin(), v2.begin()),
      thrust::remove_if(
        handle.get_thrust_policy(),
        thrust::make_zip_iterator(v1.begin(), v2.begin()),
        thrust::make_zip_iterator(v1.end(), v2.end()),
        [] __device__(auto tuple) { return thrust::get<0>(tuple) == thrust::get<1>(tuple); }));

    v1.resize(new_size, handle.get_stream());
    v2.resize(new_size, handle.get_stream());

    if constexpr (multi_gpu) {
      // shuffle vertex pairs
      auto vertex_partition_range_lasts = graph_view.vertex_partition_range_lasts();

      std::tie(
        v1, v2, std::ignore, std::ignore, std::ignore, std::ignore, std::ignore, std::ignore) =
        detail::shuffle_int_vertex_pairs_with_values_to_local_gpu_by_edge_partitioning<vertex_t,
                                                                                       edge_t,
                                                                                       weight_t,
                                                                                       int32_t,
                                                                                       int32_t>(
          handle,
          std::move(v1),
          std::move(v2),
          std::nullopt,
          std::nullopt,
          std::nullopt,
          std::nullopt,
          std::nullopt,
          vertex_partition_range_lasts);
    }

    auto score =
      similarity(handle,
                 graph_view,
                 edge_weight_view,
                 std::make_tuple(raft::device_span<vertex_t const>{v1.data(), v1.size()},
                                 raft::device_span<vertex_t const>{v2.data(), v2.size()}),
                 functor,
                 coeff,
                 do_expensive_check);

    return std::make_tuple(std::move(v1), std::move(v2), std::move(score));
  }
}

}  // namespace detail
}  // namespace cugraph
