/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "prims/fill_edge_property.cuh"
#include "prims/fill_edge_src_dst_property.cuh"
#include "prims/make_initialized_edge_property.cuh"
#include "prims/transform_e.cuh"
#include "prims/transform_reduce_if_v_frontier_outgoing_e_by_dst.cuh"
#include "prims/update_edge_src_dst_property.cuh"
#include "prims/update_v_frontier.cuh"
#include "prims/vertex_frontier.cuh"
#include "utilities/collect_comm.cuh"

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/shuffle_functions.hpp>
#include <cugraph/utilities/device_comm.hpp>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/shuffle_comm.cuh>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/functional>
#include <cuda/std/iterator>
#include <cuda/std/optional>
#include <cuda/std/tuple>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/merge.h>
#include <thrust/partition.h>
#include <thrust/random.h>
#include <thrust/scan.h>
#include <thrust/set_operations.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/unique.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

namespace cugraph {

namespace {

// Iteratively peel vertices with zero in-degree or zero out-degree. Each iteration partitions
// candidates into (non-zero degree, zero degree), peels the zero-degree vertices, and propagates
// degree decrements to neighbors via both forward and inverse graph traversals.
template <typename GraphViewType>
rmm::device_uvector<typename GraphViewType::vertex_type> find_trivial_singleton_scc_vertices(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  GraphViewType const& inverse_graph_view,
  raft::device_span<typename GraphViewType::vertex_type const> inverse_renumber_map,
  raft::device_span<typename GraphViewType::vertex_type const> candidate_vertices,
  rmm::device_uvector<typename GraphViewType::vertex_type>&& in_degrees,
  rmm::device_uvector<typename GraphViewType::vertex_type>&& out_degrees)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  constexpr bool multi_gpu = GraphViewType::is_multi_gpu;

  auto num_aggregated_candidate_vertices = candidate_vertices.size();
  if constexpr (multi_gpu) {
    num_aggregated_candidate_vertices = host_scalar_allreduce(
      handle.get_comms(), num_aggregated_candidate_vertices, raft::comms::op_t::SUM, handle.get_stream());
  }

  if (num_aggregated_candidate_vertices == 0) {
    return rmm::device_uvector<vertex_t>(0, handle.get_stream());
  }

  rmm::device_uvector<vertex_t> candidates(candidate_vertices.size(), handle.get_stream());
  thrust::copy(handle.get_thrust_policy(),
               candidate_vertices.begin(),
               candidate_vertices.end(),
               candidates.begin());

  rmm::device_uvector<vertex_t> cur_in_degrees(in_degrees.size(), handle.get_stream());
  thrust::copy(
    handle.get_thrust_policy(), in_degrees.begin(), in_degrees.end(), cur_in_degrees.begin());

  rmm::device_uvector<vertex_t> cur_out_degrees(out_degrees.size(), handle.get_stream());
  thrust::copy(
    handle.get_thrust_policy(), out_degrees.begin(), out_degrees.end(), cur_out_degrees.begin());

  rmm::device_uvector<vertex_t> peeled_vertices(0, handle.get_stream());
  auto v_first = graph_view.local_vertex_partition_range_first();

  while (true) {
    // 1. Partition: first half = non-zero in AND out degree; second half = zero in OR out degree
    auto partition_point = thrust::partition(
      handle.get_thrust_policy(),
      candidates.begin(),
      candidates.end(),
      [cur_in_degrees  = raft::device_span<vertex_t const>(cur_in_degrees.data(), cur_in_degrees.size()),
       cur_out_degrees = raft::device_span<vertex_t const>(cur_out_degrees.data(), cur_out_degrees.size()),
       v_first] __device__(auto v) {
        return cur_in_degrees[v - v_first] > 0 && cur_out_degrees[v - v_first] > 0;
      });

    auto num_surviving  = static_cast<size_t>(cuda::std::distance(candidates.begin(), partition_point));
    auto num_zero       = static_cast<size_t>(cuda::std::distance(partition_point, candidates.end()));
    auto aggregate_zero = num_zero;
    if constexpr (multi_gpu) {
      aggregate_zero = host_scalar_allreduce(
        handle.get_comms(), num_zero, raft::comms::op_t::SUM, handle.get_stream());
    }
    if (aggregate_zero == 0) break;

    // 2. Append zero-degree vertices to result
    auto old_size = peeled_vertices.size();
    peeled_vertices.resize(old_size + num_zero, handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 partition_point,
                 candidates.end(),
                 peeled_vertices.begin() + old_size);

    // Sort the frontier (partition does not preserve order) for vertex_frontier_t insert
    thrust::sort(handle.get_thrust_policy(), partition_point, candidates.end());

    // 3. Forward pass: traverse outgoing edges from frontier, decrement in_degrees of destinations
    {
      vertex_frontier_t<vertex_t, void, multi_gpu, true> frontier(handle, 1);
      frontier.bucket(0).insert(partition_point, candidates.end());

      auto [dst_vertices, decrement_counts] =
        cugraph::transform_reduce_if_v_frontier_outgoing_e_by_dst(
          handle,
          graph_view,
          frontier.bucket(0),
          edge_src_dummy_property_t{}.view(),
          edge_dst_dummy_property_t{}.view(),
          edge_dummy_property_t{}.view(),
          cuda::proclaim_return_type<edge_t>(
            [] __device__(auto, auto, auto, auto, auto) { return edge_t{1}; }),
          reduce_op::plus<edge_t>(),
          cuda::proclaim_return_type<bool>(
            [] __device__(auto, auto, auto, auto, auto) { return true; }));

      thrust::for_each(
        handle.get_thrust_policy(),
        thrust::make_counting_iterator(size_t{0}),
        thrust::make_counting_iterator(dst_vertices.size()),
        [dst_vertices     = raft::device_span<vertex_t const>(dst_vertices.data(), dst_vertices.size()),
         decrement_counts = raft::device_span<edge_t const>(decrement_counts.data(), decrement_counts.size()),
         cur_in_degrees = raft::device_span<vertex_t>(cur_in_degrees.data(), cur_in_degrees.size()),
         v_first] __device__(size_t i) {
          auto d              = cur_in_degrees[dst_vertices[i] - v_first];
          cur_in_degrees[dst_vertices[i] - v_first] = d >= decrement_counts[i] ? d - decrement_counts[i] : vertex_t{0};
        });
    }

    // 4. Inverse pass: traverse incoming edges (via inverse graph), decrement out_degrees of sources
    {
      rmm::device_uvector<vertex_t> inv_frontier(num_zero, handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   partition_point,
                   candidates.end(),
                   inv_frontier.begin());

      if constexpr (multi_gpu) {
        std::tie(inv_frontier, std::ignore) = shuffle_ext_vertices(
          handle,
          std::move(inv_frontier),
          std::vector<cugraph::arithmetic_device_uvector_t>{});
      }

      renumber_local_ext_vertices<vertex_t, multi_gpu>(
        handle,
        inv_frontier.data(),
        inv_frontier.size(),
        inverse_renumber_map.data(),
        inverse_graph_view.local_vertex_partition_range_first(),
        inverse_graph_view.local_vertex_partition_range_last(),
        false);

      thrust::sort(handle.get_thrust_policy(), inv_frontier.begin(), inv_frontier.end());

      vertex_frontier_t<vertex_t, void, multi_gpu, true> inv_vertex_frontier(handle, 1);
      inv_vertex_frontier.bucket(0).insert(inv_frontier.begin(), inv_frontier.end());

      auto [inv_dst_vertices, inv_decrement_counts] =
        cugraph::transform_reduce_if_v_frontier_outgoing_e_by_dst(
          handle,
          inverse_graph_view,
          inv_vertex_frontier.bucket(0),
          edge_src_dummy_property_t{}.view(),
          edge_dst_dummy_property_t{}.view(),
          edge_dummy_property_t{}.view(),
          cuda::proclaim_return_type<edge_t>(
            [] __device__(auto, auto, auto, auto, auto) { return edge_t{1}; }),
          reduce_op::plus<edge_t>(),
          cuda::proclaim_return_type<bool>(
            [] __device__(auto, auto, auto, auto, auto) { return true; }));

      unrenumber_local_int_vertices(handle,
                                    inv_dst_vertices.data(),
                                    inv_dst_vertices.size(),
                                    inverse_renumber_map.data(),
                                    inverse_graph_view.local_vertex_partition_range_first(),
                                    inverse_graph_view.local_vertex_partition_range_last());

      if constexpr (multi_gpu) {
        std::vector<arithmetic_device_uvector_t> properties{};
        properties.push_back(std::move(inv_decrement_counts));
        std::tie(inv_dst_vertices, properties) = shuffle_ext_vertices(
          handle, std::move(inv_dst_vertices), std::move(properties));
        inv_decrement_counts =
          std::move(std::get<rmm::device_uvector<edge_t>>(properties[0]));
      }

      thrust::for_each(
        handle.get_thrust_policy(),
        thrust::make_counting_iterator(size_t{0}),
        thrust::make_counting_iterator(inv_dst_vertices.size()),
        [inv_dst_vertices     = raft::device_span<vertex_t const>(inv_dst_vertices.data(), inv_dst_vertices.size()),
         inv_decrement_counts = raft::device_span<edge_t const>(inv_decrement_counts.data(), inv_decrement_counts.size()),
         cur_out_degrees = raft::device_span<vertex_t>(cur_out_degrees.data(), cur_out_degrees.size()),
         v_first] __device__(size_t i) {
          auto d              = cur_out_degrees[inv_dst_vertices[i] - v_first];
          cur_out_degrees[inv_dst_vertices[i] - v_first] = d >= inv_decrement_counts[i] ? d - inv_decrement_counts[i] : vertex_t{0};
        });
    }

    // 5. Shrink candidates to surviving (non-zero degree) vertices only
    candidates.resize(num_surviving, handle.get_stream());
    candidates.shrink_to_fit(handle.get_stream());
  }

  thrust::sort(handle.get_thrust_policy(), peeled_vertices.begin(), peeled_vertices.end());
  peeled_vertices.shrink_to_fit(handle.get_stream());

  return peeled_vertices;
}

// find pivots;returns (pivot vertices, pivot unresolved component indexes) pairs; the returned
// pairs should be sorted by pivot vertex. Currently, we return one pivot per unresolved component
// (or zero if all the vertices in the unresolved component are excluded). We may update this code
// to return more than one pivot per unresolved component to extract additional parallelism for
// large diameter graphs (but this will complicate the reachable_sets function implementation).
template <typename GraphViewType>
std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>>
find_pivots(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  raft::device_span<typename GraphViewType::vertex_type const> unresolved_component_offsets,
  raft::device_span<typename GraphViewType::vertex_type const> unresolved_component_vertices,
  raft::device_span<typename GraphViewType::edge_type const> unresolved_component_vertex_in_degrees,
  raft::device_span<typename GraphViewType::edge_type const>
    unresolved_component_vertex_out_degrees,
  raft::device_span<typename GraphViewType::vertex_type const>
    sorted_excluded_vertices /* should not be selected as pivots */)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  // reduce key (component index) value (pivot, priority) pairs (find highest priority (pivot,
  // priority) pairs for each component ID)

  rmm::device_uvector<vertex_t> component_idxs(unresolved_component_offsets.size() - 1,
                                               handle.get_stream());
  rmm::device_uvector<vertex_t> pivots(component_idxs.size(), handle.get_stream());
  rmm::device_uvector<edge_t> priorities(component_idxs.size(), handle.get_stream());
  {
    auto component_idx_first = cuda::make_transform_iterator(
      thrust::make_counting_iterator(vertex_t{0}),
      cuda::proclaim_return_type<vertex_t>(
        [component_offsets = raft::device_span<vertex_t const>(
           unresolved_component_offsets.data(),
           unresolved_component_offsets.size())] __device__(vertex_t i) {
          return static_cast<vertex_t>(cuda::std::distance(
            component_offsets.begin() + 1,
            thrust::upper_bound(
              thrust::seq, component_offsets.begin() + 1, component_offsets.end(), i)));
        }));
    auto priority_first = cuda::make_transform_iterator(
      thrust::make_zip_iterator(unresolved_component_vertices.begin(),
                                unresolved_component_vertex_in_degrees.begin(),
                                unresolved_component_vertex_out_degrees.begin()),
      cuda::proclaim_return_type<edge_t>(
        [excluded_vertices = raft::device_span<vertex_t const>(
           sorted_excluded_vertices.data(),
           sorted_excluded_vertices.size())] __device__(auto triplet) {
          auto excluded = thrust::binary_search(thrust::seq,
                                                excluded_vertices.begin(),
                                                excluded_vertices.end(),
                                                cuda::std::get<0>(triplet));
          if (excluded)
            return edge_t{0};
          else
            return cuda::std::min(cuda::std::get<1>(triplet), cuda::std::get<2>(triplet));
        }));
    auto ret = thrust::reduce_by_key(
      handle.get_thrust_policy(),
      component_idx_first,
      component_idx_first + unresolved_component_vertices.size(),
      thrust::make_zip_iterator(unresolved_component_vertices.begin(), priority_first),
      component_idxs.begin(),
      thrust::make_zip_iterator(pivots.begin(), priorities.begin()),
      cuda::std::equal_to<size_t>{},
      cuda::proclaim_return_type<cuda::std::tuple<vertex_t, edge_t>>(
        [] __device__(auto lhs, auto rhs) {
          return cuda::std::get<1>(lhs) >= cuda::std::get<1>(rhs) ? lhs : rhs;
        }));
    component_idxs.resize(cuda::std::distance(component_idxs.begin(), ret.first),
                          handle.get_stream());
    pivots.resize(component_idxs.size(), handle.get_stream());
    priorities.resize(component_idxs.size(), handle.get_stream());
    component_idxs.shrink_to_fit(handle.get_stream());
    pivots.shrink_to_fit(handle.get_stream());
    priorities.shrink_to_fit(handle.get_stream());
  }

  // remove 0 priority key value pairs (0 priority means excluded)

  {
    auto triplet_first =
      thrust::make_zip_iterator(component_idxs.begin(), pivots.begin(), priorities.begin());
    component_idxs.resize(
      cuda::std::distance(
        triplet_first,
        thrust::remove_if(handle.get_thrust_policy(),
                          triplet_first,
                          triplet_first + component_idxs.size(),
                          cuda::proclaim_return_type<bool>([] __device__(auto triplet) {
                            return cuda::std::get<2>(triplet) == 0;
                          }))),
      handle.get_stream());
    pivots.resize(component_idxs.size(), handle.get_stream());
    priorities.resize(component_idxs.size(), handle.get_stream());
    component_idxs.shrink_to_fit(handle.get_stream());
    pivots.shrink_to_fit(handle.get_stream());
    priorities.shrink_to_fit(handle.get_stream());
  }

  if constexpr (GraphViewType::is_multi_gpu) {
    // multi-GPU reduction

    std::vector<arithmetic_device_uvector_t> vertex_properties{};
    vertex_properties.push_back(std::move(pivots));
    vertex_properties.push_back(std::move(priorities));
    std::tie(component_idxs, vertex_properties) =
      shuffle_ext_vertices(handle, std::move(component_idxs), std::move(vertex_properties));
    pivots     = std::move(std::get<rmm::device_uvector<vertex_t>>(vertex_properties[0]));
    priorities = std::move(std::get<rmm::device_uvector<edge_t>>(vertex_properties[1]));
    thrust::sort_by_key(handle.get_thrust_policy(),
                        component_idxs.begin(),
                        component_idxs.end(),
                        thrust::make_zip_iterator(pivots.begin(), priorities.begin()));
    rmm::device_uvector<vertex_t> tmp_component_idxs(unresolved_component_offsets.size() - 1,
                                                     handle.get_stream());
    rmm::device_uvector<vertex_t> tmp_pivots(tmp_component_idxs.size(), handle.get_stream());
    rmm::device_uvector<edge_t> tmp_priorities(tmp_component_idxs.size(), handle.get_stream());
    auto ret =
      thrust::reduce_by_key(handle.get_thrust_policy(),
                            component_idxs.begin(),
                            component_idxs.end(),
                            thrust::make_zip_iterator(pivots.begin(), priorities.begin()),
                            tmp_component_idxs.begin(),
                            thrust::make_zip_iterator(tmp_pivots.begin(), tmp_priorities.begin()),
                            cuda::std::equal_to<size_t>{},
                            cuda::proclaim_return_type<cuda::std::tuple<vertex_t, edge_t>>(
                              [] __device__(auto lhs, auto rhs) {
                                return cuda::std::get<1>(lhs) >= cuda::std::get<1>(rhs) ? lhs : rhs;
                              }));
    tmp_priorities.resize(0, handle.get_stream());
    tmp_priorities.shrink_to_fit(handle.get_stream());
    tmp_component_idxs.resize(cuda::std::distance(tmp_component_idxs.begin(), ret.first),
                              handle.get_stream());
    tmp_pivots.resize(tmp_component_idxs.size(), handle.get_stream());
    tmp_component_idxs.shrink_to_fit(handle.get_stream());
    tmp_pivots.shrink_to_fit(handle.get_stream());

    vertex_properties.clear();
    vertex_properties.push_back(std::move(tmp_component_idxs));
    std::tie(pivots, vertex_properties) =
      shuffle_int_vertices(handle,
                           std::move(tmp_pivots),
                           std::move(vertex_properties),
                           graph_view.vertex_partition_range_lasts());
    component_idxs = std::move(std::get<rmm::device_uvector<vertex_t>>(vertex_properties[0]));
  }

  thrust::sort_by_key(
    handle.get_thrust_policy(), pivots.begin(), pivots.end(), component_idxs.begin());

  return std::make_tuple(std::move(pivots), std::move(component_idxs));
}

// return (offsets, reachable vertices) paris (offsets.size() = num_unresolved_components + 1)
template <typename GraphViewType>
std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>>
reachable_sets(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  raft::device_span<typename GraphViewType::vertex_type const> sorted_starting_vertices,
  raft::device_span<typename GraphViewType::vertex_type const>
    sorted_starting_vertex_unresolved_component_idxs,
  typename GraphViewType::vertex_type num_unresolved_components)
{
  using vertex_t = typename GraphViewType::vertex_type;

  rmm::device_uvector<vertex_t> idxs(graph_view.local_vertex_partition_range_size(),
                                     handle.get_stream());
  {
    thrust::fill(
      handle.get_thrust_policy(), idxs.begin(), idxs.end(), invalid_component_id_v<vertex_t>);

    // run multi-source BFS

    rmm::device_uvector<vertex_t> predecessors(graph_view.local_vertex_partition_range_size(),
                                               handle.get_stream());
    {
      rmm::device_uvector<vertex_t> distances(graph_view.local_vertex_partition_range_size(),
                                              handle.get_stream());
      bfs(handle,
          graph_view,
          distances.data(),
          predecessors.data(),
          sorted_starting_vertices.data(),
          sorted_starting_vertices.size());
      thrust::scatter(
        handle.get_thrust_policy(),
        sorted_starting_vertices.begin(),
        sorted_starting_vertices.end(),
        thrust::make_transform_iterator(
          sorted_starting_vertices.begin(),
          cugraph::detail::shift_left_t<vertex_t>{graph_view.local_vertex_partition_range_first()}),
        predecessors
          .begin());  // bfs sets predecessors of starting vertices to invalid_vertex_id_v<vertex_t>
    }

    // back-track to the starting vertices

    cugraph::kv_store_t<vertex_t, vertex_t, true> vertex_pred_store(
      thrust::make_counting_iterator(graph_view.local_vertex_partition_range_first()),
      thrust::make_counting_iterator(graph_view.local_vertex_partition_range_last()),
      predecessors.begin(),
      invalid_vertex_id_v<vertex_t>,
      true /* key_sorted */,
      handle.get_stream());
    auto vertex_pred_store_view = vertex_pred_store.view();

    cugraph::kv_store_t<vertex_t, vertex_t, true> starting_vertex_unresolved_component_idx_store(
      sorted_starting_vertices.begin(),
      sorted_starting_vertices.end(),
      sorted_starting_vertex_unresolved_component_idxs.begin(),
      invalid_component_id_v<vertex_t>,
      true /* key_sorted */,
      handle.get_stream());
    auto starting_vertex_unresolved_component_idx_store_view =
      starting_vertex_unresolved_component_idx_store.view();

    rmm::device_uvector<vertex_t> remaining_vertices(graph_view.local_vertex_partition_range_size(),
                                                     handle.get_stream());
    rmm::device_uvector<vertex_t> remaining_vertex_ancestors(
      graph_view.local_vertex_partition_range_size(), handle.get_stream());
    auto input_pair_first = thrust::make_zip_iterator(
      thrust::make_counting_iterator(graph_view.local_vertex_partition_range_first()),
      predecessors.begin());
    auto output_pair_first =
      thrust::make_zip_iterator(remaining_vertices.begin(), remaining_vertex_ancestors.begin());
    remaining_vertices.resize(
      cuda::std::distance(
        output_pair_first,
        thrust::copy_if(handle.get_thrust_policy(),
                        input_pair_first,
                        input_pair_first + graph_view.local_vertex_partition_range_size(),
                        output_pair_first,
                        [invalid_vertex = invalid_vertex_id_v<vertex_t>] __device__(auto pair) {
                          return cuda::std::get<1>(pair) != invalid_vertex;
                        })),
      handle.get_stream());
    remaining_vertex_ancestors.resize(remaining_vertices.size(), handle.get_stream());

    vertex_t num_remainings = static_cast<vertex_t>(remaining_vertices.size());
    if constexpr (GraphViewType::is_multi_gpu) {
      num_remainings = host_scalar_allreduce(
        handle.get_comms(), num_remainings, raft::comms::op_t::SUM, handle.get_stream());
    }
    while (num_remainings > vertex_t{0}) {
      {
        rmm::device_uvector<vertex_t> remaining_vertex_unresolved_component_idxs(
          0, handle.get_stream());
        if constexpr (GraphViewType::is_multi_gpu) {
          auto h_vertex_partition_range_lasts = graph_view.vertex_partition_range_lasts();
          rmm::device_uvector<vertex_t> d_vertex_partition_range_lasts(
            h_vertex_partition_range_lasts.size(), handle.get_stream());
          raft::update_device(d_vertex_partition_range_lasts.data(),
                              h_vertex_partition_range_lasts.data(),
                              h_vertex_partition_range_lasts.size(),
                              handle.get_stream());
          auto& major_comm = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
          auto const major_comm_size = major_comm.get_size();
          auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
          auto const minor_comm_size                 = minor_comm.get_size();
          remaining_vertex_unresolved_component_idxs = collect_values_for_keys(
            handle.get_comms(),
            starting_vertex_unresolved_component_idx_store_view,
            remaining_vertex_ancestors.begin(),
            remaining_vertex_ancestors.end(),
            cugraph::detail::compute_gpu_id_from_int_vertex_t<vertex_t>{
              raft::device_span<vertex_t const>(d_vertex_partition_range_lasts.data(),
                                                d_vertex_partition_range_lasts.size()),
              major_comm_size,
              minor_comm_size},
            handle.get_stream());
        } else {
          remaining_vertex_unresolved_component_idxs.resize(remaining_vertex_ancestors.size(),
                                                            handle.get_stream());
          starting_vertex_unresolved_component_idx_store_view.find(
            remaining_vertex_ancestors.begin(),
            remaining_vertex_ancestors.end(),
            remaining_vertex_unresolved_component_idxs.begin(),
            handle.get_stream());
        }

        auto triplet_first =
          thrust::make_zip_iterator(remaining_vertices.begin(),
                                    remaining_vertex_ancestors.begin(),
                                    remaining_vertex_unresolved_component_idxs.begin());
        auto triplet_last = thrust::partition(
          handle.get_thrust_policy(),
          triplet_first,
          triplet_first + remaining_vertices.size(),
          cuda::proclaim_return_type<bool>(
            [invalid_component_id = invalid_component_id_v<vertex_t>] __device__(auto triplet) {
              return cuda::std::get<2>(triplet) == invalid_component_id;
            }));
        thrust::scatter(
          handle.get_thrust_policy(),
          remaining_vertex_unresolved_component_idxs.begin() +
            cuda::std::distance(triplet_first, triplet_last),
          remaining_vertex_unresolved_component_idxs.end(),
          thrust::make_transform_iterator(
            remaining_vertices.begin() + cuda::std::distance(triplet_first, triplet_last),
            cugraph::detail::shift_left_t<vertex_t>{
              graph_view.local_vertex_partition_range_first()}),
          idxs.begin());

        remaining_vertices.resize(cuda::std::distance(triplet_first, triplet_last),
                                  handle.get_stream());
        remaining_vertex_ancestors.resize(remaining_vertices.size(), handle.get_stream());
        remaining_vertices.shrink_to_fit(handle.get_stream());
        remaining_vertex_ancestors.shrink_to_fit(handle.get_stream());
      }

      if constexpr (GraphViewType::is_multi_gpu) {
        auto h_vertex_partition_range_lasts = graph_view.vertex_partition_range_lasts();
        rmm::device_uvector<vertex_t> d_vertex_partition_range_lasts(
          h_vertex_partition_range_lasts.size(), handle.get_stream());
        raft::update_device(d_vertex_partition_range_lasts.data(),
                            h_vertex_partition_range_lasts.data(),
                            h_vertex_partition_range_lasts.size(),
                            handle.get_stream());
        auto& major_comm = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
        auto const major_comm_size = major_comm.get_size();
        auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
        auto const minor_comm_size = minor_comm.get_size();
        remaining_vertex_ancestors = collect_values_for_keys(
          handle.get_comms(),
          vertex_pred_store_view,
          remaining_vertex_ancestors.begin(),
          remaining_vertex_ancestors.end(),
          cugraph::detail::compute_gpu_id_from_int_vertex_t<vertex_t>{
            raft::device_span<vertex_t const>(d_vertex_partition_range_lasts.data(),
                                              d_vertex_partition_range_lasts.size()),
            major_comm_size,
            minor_comm_size},
          handle.get_stream());
      } else {
        vertex_pred_store_view.find(remaining_vertex_ancestors.begin(),
                                    remaining_vertex_ancestors.end(),
                                    remaining_vertex_ancestors.begin(),
                                    handle.get_stream());
      }

      num_remainings = remaining_vertices.size();
      if constexpr (GraphViewType::is_multi_gpu) {
        num_remainings = host_scalar_allreduce(
          handle.get_comms(), num_remainings, raft::comms::op_t::SUM, handle.get_stream());
      }
    }
  }

  rmm::device_uvector<vertex_t> vertices(graph_view.local_vertex_partition_range_size(),
                                         handle.get_stream());
  {
    thrust::sequence(handle.get_thrust_policy(),
                     vertices.begin(),
                     vertices.end(),
                     graph_view.local_vertex_partition_range_first());
    auto pair_first = thrust::make_zip_iterator(vertices.begin(), idxs.begin());
    vertices.resize(cuda::std::distance(
                      pair_first,
                      thrust::remove_if(
                        handle.get_thrust_policy(),
                        pair_first,
                        pair_first + graph_view.local_vertex_partition_range_size(),
                        [invalid_component_id = invalid_component_id_v<vertex_t>] __device__(
                          auto pair) { return cuda::std::get<1>(pair) == invalid_component_id; })),
                    handle.get_stream());
    idxs.resize(vertices.size(), handle.get_stream());
    vertices.shrink_to_fit(handle.get_stream());
    idxs.shrink_to_fit(handle.get_stream());
  }

  {
    auto pair_first = thrust::make_zip_iterator(idxs.begin(), vertices.begin());
    thrust::sort(handle.get_thrust_policy(), pair_first, pair_first + idxs.size());
  }

  // starting_vertices => component indices

  rmm::device_uvector<vertex_t> offsets(num_unresolved_components + 1, handle.get_stream());
  offsets.set_element_to_zero_async(0, handle.get_stream());
  thrust::upper_bound(handle.get_thrust_policy(),
                      idxs.begin(),
                      idxs.end(),
                      thrust::make_counting_iterator(vertex_t{0}),
                      thrust::make_counting_iterator(num_unresolved_components),
                      offsets.begin() + 1);

  return std::make_tuple(std::move(offsets), std::move(vertices));
}

// return component_ids, component_offsets, component_vertices, num_unresolved_components (last
// component_ids.size() - num_unresolved_components are fully resolved SCCs). in multi-GPU,
// component IDs for unresolved components in every GPU should follow one global ordering (this will
// enable setting unresolved_component_offsets without re-sorting).
template <typename GraphViewType>
std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>,
           size_t>
intersect_reachable_sets(raft::handle_t const& handle,
                         rmm::device_uvector<typename GraphViewType::vertex_type>&& unresolved_component_offsets,
                         rmm::device_uvector<typename GraphViewType::vertex_type>&& unresolved_component_vertices,
                         rmm::device_uvector<typename GraphViewType::vertex_type>&& forward_set_offsets,
                         rmm::device_uvector<typename GraphViewType::vertex_type>&& forward_set_vertices,
                         rmm::device_uvector<typename GraphViewType::vertex_type>&& backward_set_offsets,
                         rmm::device_uvector<typename GraphViewType::vertex_type>&& backward_set_vertices,
                         rmm::device_uvector<typename GraphViewType::vertex_type>&& trivial_singleton_scc_vertices)
                         {
  using vertex_t = typename GraphViewType::vertex_type;

  // Find the component_idx for each vertex based on CSR offsets
  auto unresolved_component_idxs_first = thrust::make_transform_iterator(
    thrust::make_counting_iterator(size_t{0}),
    cuda::proclaim_return_type<vertex_t>(
      [offsets = raft::device_span<vertex_t const>(
         unresolved_component_offsets.data(), unresolved_component_offsets.size())] __device__(size_t i) {
        auto component_pos = thrust::upper_bound(
          thrust::seq, offsets.begin() + 1, offsets.end(), static_cast<vertex_t>(i));
        return static_cast<vertex_t>(cuda::std::distance(offsets.begin() + 1, component_pos));
      }));
  auto forward_set_component_idxs_first = thrust::make_transform_iterator(
    thrust::make_counting_iterator(size_t{0}),
    cuda::proclaim_return_type<vertex_t>(
      [offsets = raft::device_span<vertex_t const>(
         forward_set_offsets.data(), forward_set_offsets.size())] __device__(size_t i) {
        auto component_pos = thrust::upper_bound(
          thrust::seq, offsets.begin() + 1, offsets.end(), static_cast<vertex_t>(i));
        return static_cast<vertex_t>(cuda::std::distance(offsets.begin() + 1, component_pos));
      }));
  auto backward_set_component_idxs_first = thrust::make_transform_iterator(
    thrust::make_counting_iterator(size_t{0}),
    cuda::proclaim_return_type<vertex_t>(
      [offsets = raft::device_span<vertex_t const>(
         backward_set_offsets.data(), backward_set_offsets.size())] __device__(size_t i) {
        auto component_pos = thrust::upper_bound(
          thrust::seq, offsets.begin() + 1, offsets.end(), static_cast<vertex_t>(i));
        return static_cast<vertex_t>(cuda::std::distance(offsets.begin() + 1, component_pos));
      }));

  auto unresolved_component_pairs_first = thrust::make_zip_iterator(unresolved_component_idxs_first, unresolved_component_vertices.begin());
  auto forward_set_pairs_first = thrust::make_zip_iterator(forward_set_component_idxs_first, forward_set_vertices.begin());
  auto backward_set_pairs_first = thrust::make_zip_iterator(backward_set_component_idxs_first, backward_set_vertices.begin());

  // Precondition: vertices within each CSR segment are already sorted,
  // so (comp_idx, vertex) pairs are fully sorted and ready for set operations.

  // 1. SCC = FWD ∩ BWD
  rmm::device_uvector<vertex_t> scc_component_idxs(
    std::min(forward_set_vertices.size(), backward_set_vertices.size()), handle.get_stream());
  rmm::device_uvector<vertex_t> scc_vertices(scc_component_idxs.size(), handle.get_stream());
  {
    auto out_first = thrust::make_zip_iterator(scc_component_idxs.begin(), scc_vertices.begin());
    auto out_end   = thrust::set_intersection(
      handle.get_thrust_policy(),
      forward_set_pairs_first,
      forward_set_pairs_first + forward_set_vertices.size(),
      backward_set_pairs_first,
      backward_set_pairs_first + backward_set_vertices.size(),
      out_first);
    auto count = static_cast<size_t>(cuda::std::distance(out_first, out_end));
    scc_component_idxs.resize(count, handle.get_stream());
    scc_component_idxs.shrink_to_fit(handle.get_stream());
    scc_vertices.resize(count, handle.get_stream());
    scc_vertices.shrink_to_fit(handle.get_stream());
  }

  // 2. FWD_ONLY = FWD \ BWD
  rmm::device_uvector<vertex_t> fwd_only_component_idxs(forward_set_vertices.size(), handle.get_stream());
  rmm::device_uvector<vertex_t> fwd_only_vertices(forward_set_vertices.size(), handle.get_stream());
  {
    auto out_first = thrust::make_zip_iterator(fwd_only_component_idxs.begin(), fwd_only_vertices.begin());
    auto out_end   = thrust::set_difference(
      handle.get_thrust_policy(),
      forward_set_pairs_first,
      forward_set_pairs_first + forward_set_vertices.size(),
      backward_set_pairs_first,
      backward_set_pairs_first + backward_set_vertices.size(),
      out_first);
    auto count = static_cast<size_t>(cuda::std::distance(out_first, out_end));
    fwd_only_component_idxs.resize(count, handle.get_stream());
    fwd_only_vertices.resize(count, handle.get_stream());
    fwd_only_component_idxs.shrink_to_fit(handle.get_stream());
    fwd_only_vertices.shrink_to_fit(handle.get_stream());
  }

  // 3. BWD_ONLY = BWD \ FWD
  rmm::device_uvector<vertex_t> bwd_only_component_idxs(backward_set_vertices.size(), handle.get_stream());
  rmm::device_uvector<vertex_t> bwd_only_vertices(backward_set_vertices.size(), handle.get_stream());
  {
    auto out_first = thrust::make_zip_iterator(bwd_only_component_idxs.begin(), bwd_only_vertices.begin());
    auto out_end   = thrust::set_difference(
      handle.get_thrust_policy(),
      backward_set_pairs_first,
      backward_set_pairs_first + backward_set_vertices.size(),
      forward_set_pairs_first,
      forward_set_pairs_first + forward_set_vertices.size(),
      out_first);
    auto count = static_cast<size_t>(cuda::std::distance(out_first, out_end));
    bwd_only_component_idxs.resize(count, handle.get_stream());
    bwd_only_vertices.resize(count, handle.get_stream());
    bwd_only_component_idxs.shrink_to_fit(handle.get_stream());
    bwd_only_vertices.shrink_to_fit(handle.get_stream());
  }

  // 4. REMAINDER = UC \ (FWD ∪ BWD)
  rmm::device_uvector<vertex_t> remaining_component_idxs(
    unresolved_component_vertices.size(), handle.get_stream());
  rmm::device_uvector<vertex_t> remaining_vertices(remaining_component_idxs.size(), handle.get_stream());
  {
    rmm::device_uvector<vertex_t> union_component_idxs(
      forward_set_vertices.size() + backward_set_vertices.size(), handle.get_stream());
    rmm::device_uvector<vertex_t> union_vertices(union_component_idxs.size(), handle.get_stream());
    {
      auto out_first = thrust::make_zip_iterator(union_component_idxs.begin(), union_vertices.begin());
      auto out_end   = thrust::set_union(
        handle.get_thrust_policy(),
        forward_set_pairs_first,
        forward_set_pairs_first + forward_set_vertices.size(),
        backward_set_pairs_first,
        backward_set_pairs_first + backward_set_vertices.size(),
        out_first);
      auto count = static_cast<size_t>(cuda::std::distance(out_first, out_end));
      union_component_idxs.resize(count, handle.get_stream());
      union_vertices.resize(count, handle.get_stream());
      union_component_idxs.shrink_to_fit(handle.get_stream());
      union_vertices.shrink_to_fit(handle.get_stream());
    }

    auto union_pairs_first =
      thrust::make_zip_iterator(union_component_idxs.begin(), union_vertices.begin());
    auto out_first = thrust::make_zip_iterator(remaining_component_idxs.begin(), remaining_vertices.begin());
    auto out_end   = thrust::set_difference(
      handle.get_thrust_policy(),
      unresolved_component_pairs_first,
      unresolved_component_pairs_first + unresolved_component_vertices.size(),
      union_pairs_first,
      union_pairs_first + union_vertices.size(),
      out_first);
    auto count = static_cast<size_t>(cuda::std::distance(out_first, out_end));
    remaining_component_idxs.resize(count, handle.get_stream());
    remaining_vertices.resize(count, handle.get_stream());
    remaining_component_idxs.shrink_to_fit(handle.get_stream());
    remaining_vertices.shrink_to_fit(handle.get_stream());
  }

  // Remove singletons from fwd_only, bwd_only, remainder.
  // Singletons can't appear in SCC (FWD ∩ BWD) so that set is already clean.

  {
    auto remove_singletons = [&](rmm::device_uvector<vertex_t>& component_idxs,
                                 rmm::device_uvector<vertex_t>& component_vertices) {
      auto zip_first = thrust::make_zip_iterator(component_idxs.begin(), component_vertices.begin());
      auto zip_end   = thrust::remove_if(
        handle.get_thrust_policy(),
        zip_first,
        zip_first + component_vertices.size(),
        [trivial_singleton_scc_vertices = raft::device_span<vertex_t const>(trivial_singleton_scc_vertices.data(), trivial_singleton_scc_vertices.size())] __device__(auto pair) {
          auto v = cuda::std::get<1>(pair);
          return thrust::binary_search(thrust::seq, trivial_singleton_scc_vertices.begin(), trivial_singleton_scc_vertices.end(), v);
        });
      auto count = static_cast<size_t>(cuda::std::distance(zip_first, zip_end));
      component_idxs.resize(count, handle.get_stream());
      component_vertices.resize(count, handle.get_stream());
    };

    remove_singletons(fwd_only_component_idxs, fwd_only_vertices);
    remove_singletons(bwd_only_component_idxs, bwd_only_vertices);
    remove_singletons(remaining_component_idxs, remaining_vertices);
  }

  // For each subset (scc, fwd, bwd, remainder), find the local min vertex id per component idx
  // on the current GPU and perform global reduction.

  auto find_min_vertex_id_per_component = [&](rmm::device_uvector<vertex_t>& component_idxs,
                                             rmm::device_uvector<vertex_t>& vertices) {
    rmm::device_uvector<vertex_t> unique_idxs(component_idxs.size(), handle.get_stream());
    rmm::device_uvector<vertex_t> min_vertex_ids(component_idxs.size(), handle.get_stream());

    auto [unique_end, min_end] = thrust::reduce_by_key(handle.get_thrust_policy(),
                                                       component_idxs.begin(),
                                                       component_idxs.end(),
                                                       vertices.begin(),
                                                       unique_idxs.begin(),
                                                       min_vertex_ids.begin(),
                                                       thrust::equal_to<vertex_t>{},
                                                       thrust::minimum<vertex_t>());

    unique_idxs.resize(cuda::std::distance(unique_idxs.begin(), unique_end), handle.get_stream());
    min_vertex_ids.resize(cuda::std::distance(min_vertex_ids.begin(), min_end), handle.get_stream());
    unique_idxs.shrink_to_fit(handle.get_stream());
    min_vertex_ids.shrink_to_fit(handle.get_stream());

    return std::make_pair(std::move(unique_idxs), std::move(min_vertex_ids));
  };

  auto [unique_scc_component_idxs, scc_component_min_vertex_ids] =
  find_min_vertex_id_per_component(scc_component_idxs, scc_vertices);
  auto [unique_fwd_component_idxs, fwd_component_min_vertex_ids] =
  find_min_vertex_id_per_component(fwd_only_component_idxs, fwd_only_vertices);
  auto [unique_bwd_component_idxs, bwd_component_min_vertex_ids] =
  find_min_vertex_id_per_component(bwd_only_component_idxs, bwd_only_vertices);
  auto [unique_remaining_component_idxs, remaining_component_min_vertex_ids] =
  find_min_vertex_id_per_component(remaining_component_idxs, remaining_vertices);

  // Build a flat array of 4 * num_unresolved_components to hold the min vertex id for each
  // (component, subset) pair [scc_0, fwd_0, bwd_0, rem_0, scc_1, fwd_1, ...]

  auto num_unresolved_components = unresolved_component_offsets.size() - 1;
  rmm::device_uvector<vertex_t> component_local_min_vertex_ids(
    4 * num_unresolved_components, handle.get_stream());
  thrust::fill(handle.get_thrust_policy(),
               component_local_min_vertex_ids.begin(),
               component_local_min_vertex_ids.end(),
               std::numeric_limits<vertex_t>::max());

  // Scatter each subset's local min vertex ids into the interleaved array.
  auto scatter_mins = [&](rmm::device_uvector<vertex_t> const& unique_component_idxs,
                          rmm::device_uvector<vertex_t> const& min_vertex_ids,
                          size_t subset_offset) {
    thrust::scatter(
      handle.get_thrust_policy(),
      min_vertex_ids.begin(),
      min_vertex_ids.end(),
      thrust::make_transform_iterator(
        unique_component_idxs.begin(),
        cuda::proclaim_return_type<size_t>(
          [subset_offset] __device__(vertex_t idx) {
            return 4 * static_cast<size_t>(idx) + subset_offset;
          })),
      component_local_min_vertex_ids.begin());
  };

  scatter_mins(unique_scc_component_idxs, scc_component_min_vertex_ids, 0);
  scatter_mins(unique_fwd_component_idxs, fwd_component_min_vertex_ids, 1);
  scatter_mins(unique_bwd_component_idxs, bwd_component_min_vertex_ids, 2);
  scatter_mins(unique_remaining_component_idxs, remaining_component_min_vertex_ids, 3);

  // Multi-GPU: reduce across GPUs to get global min vertex id per sub-component.
  if constexpr (GraphViewType::is_multi_gpu) {
    device_allreduce(handle.get_comms(),
                     component_local_min_vertex_ids.begin(),
                     component_local_min_vertex_ids.begin(),
                     component_local_min_vertex_ids.size(),
                     raft::comms::op_t::MIN,
                     handle.get_stream());
  }

  // For fwd only, bwd only and remaining components, if global component size is 1
  // we can treat them as trivial SCCs.

  auto find_size_per_component = [&](rmm::device_uvector<vertex_t>& component_idxs) {
    rmm::device_uvector<size_t> component_sizes(component_idxs.size(), handle.get_stream());

    thrust::reduce_by_key(handle.get_thrust_policy(),
                          component_idxs.begin(),
                          component_idxs.end(),
                          thrust::make_constant_iterator(size_t{1}),
                          thrust::make_discard_iterator(),
                          component_sizes.begin(),
                          thrust::equal_to<vertex_t>{},
                          thrust::plus<size_t>());

    component_sizes.resize(cuda::std::distance(component_sizes.begin(), component_sizes_end), handle.get_stream());
    component_sizes.shrink_to_fit(handle.get_stream());
    return std::move(component_sizes);
  };
  
  auto fwd_component_sizes       = find_size_per_component(fwd_only_component_idxs);
  fwd_component_sizes.resize(unique_fwd_component_idxs.size(), handle.get_stream());
  auto bwd_component_sizes       = find_size_per_component(bwd_only_component_idxs);
  bwd_component_sizes.resize(unique_bwd_component_idxs.size(), handle.get_stream());
  auto remaining_component_sizes = find_size_per_component(remaining_component_idxs);
  remaining_component_sizes.resize(unique_remaining_component_idxs.size(), handle.get_stream());

  // Interleaved array: [fwd_0, bwd_0, rem_0, fwd_1, bwd_1, rem_1, ...]
  rmm::device_uvector<size_t> component_local_sizes(
    3 * num_unresolved_components, handle.get_stream());
  thrust::fill(handle.get_thrust_policy(),
               component_local_sizes.begin(),
               component_local_sizes.end(),
               size_t{0});

  auto scatter_sizes = [&](rmm::device_uvector<vertex_t> const& unique_component_idxs,
                           rmm::device_uvector<size_t> const& sizes,
                           size_t subset_offset) {
    thrust::scatter(
      handle.get_thrust_policy(),
      sizes.begin(),
      sizes.end(),
      thrust::make_transform_iterator(
        unique_component_idxs.begin(),
        cuda::proclaim_return_type<size_t>(
          [subset_offset] __device__(vertex_t idx) {
            return 3 * static_cast<size_t>(idx) + subset_offset;
          })),
      component_local_sizes.begin());
  };

  scatter_sizes(unique_fwd_component_idxs, fwd_component_sizes, 0);
  scatter_sizes(unique_bwd_component_idxs, bwd_component_sizes, 1);
  scatter_sizes(unique_remaining_component_idxs, remaining_component_sizes, 2);

  if constexpr (GraphViewType::is_multi_gpu) {
    device_allreduce(handle.get_comms(),
                     component_local_sizes.begin(),
                     component_local_sizes.begin(),
                     component_local_sizes.size(),
                     raft::comms::op_t::SUM,
                     handle.get_stream());
  }

  // Partition fwd/bwd/remaining vertices: resolved (size <= 1) left, unresolved (size > 1) right.
  // Size-0 sub-components have no local vertices, so they are naturally absent.

  auto partition_by_size = [&](rmm::device_uvector<vertex_t>& component_idxs,
                               rmm::device_uvector<vertex_t>& component_vertices,
                               size_t size_offset) {
    auto zip_first = thrust::make_zip_iterator(component_idxs.begin(), component_vertices.begin());
    auto partition_point = thrust::stable_partition(
      handle.get_thrust_policy(),
      zip_first,
      zip_first + component_vertices.size(),
      [component_local_sizes = raft::device_span<size_t const>(component_local_sizes.data(),
                                                               component_local_sizes.size()),
       size_offset] __device__(auto pair) {
        auto idx = cuda::std::get<0>(pair);
        return component_local_sizes[3 * static_cast<size_t>(idx) + size_offset] <= size_t{1};
      });
    return static_cast<size_t>(cuda::std::distance(zip_first, partition_point));
  };

  auto fwd_resolved_count       = partition_by_size(fwd_only_component_idxs, fwd_only_vertices, 0);
  auto bwd_resolved_count       = partition_by_size(bwd_only_component_idxs, bwd_only_vertices, 1);
  auto remaining_resolved_count = partition_by_size(remaining_component_idxs, remaining_vertices, 2);

  // Count unresolved components: number of sub-components with global size > 1.
  auto num_unresolved_components = static_cast<size_t>(thrust::count_if(
    handle.get_thrust_policy(),
    component_local_sizes.begin(),
    component_local_sizes.end(),
    [] __device__(size_t s) { return s > size_t{1}; }));

  // Map component_idx -> component_id via component_local_min_vertex_ids (in-place).
  // IMPORTANT: This must run after partition_by_size and find_size_per_component, which index into
  // component_local_sizes using component_idx values. After this transform, the arrays hold vertex
  // IDs (arbitrarily large) and can no longer be used as array indices.
  auto map_idxs_to_ids = [&](rmm::device_uvector<vertex_t>& component_idxs, size_t subset_idx) {
    thrust::transform(
      handle.get_thrust_policy(),
      component_idxs.begin(),
      component_idxs.end(),
      component_idxs.begin(),
      [component_local_min_vertex_ids = raft::device_span<vertex_t const>(
         component_local_min_vertex_ids.data(), component_local_min_vertex_ids.size()),
       subset_idx] __device__(vertex_t component_idx) {
        return component_local_min_vertex_ids[4 * static_cast<size_t>(component_idx) + subset_idx];
      });
  };

  map_idxs_to_ids(scc_component_idxs, 0);
  map_idxs_to_ids(fwd_only_component_idxs, 1);
  map_idxs_to_ids(bwd_only_component_idxs, 2);
  map_idxs_to_ids(remaining_component_idxs, 3);

  // Build output arrays incrementally: unresolved first, then resolved SCCs.
  // After partitioning, each fwd/bwd/rem array has [resolved | unresolved].
  // append_and_trim_source appends elements from src_offset to end, then resizes the source to src_offset
  // and shrinks.
  // Order: [fwd_unresolved | bwd_unresolved | rem_unresolved |
  //         scc | trivial_singletons | fwd_resolved | bwd_resolved | rem_resolved]

  rmm::device_uvector<vertex_t> all_component_ids(0, handle.get_stream());
  rmm::device_uvector<vertex_t> component_vertices(0, handle.get_stream());

  auto append_and_trim_source = [&](rmm::device_uvector<vertex_t>& src_component_ids,
                         rmm::device_uvector<vertex_t>& src_component_vertices,
                         size_t src_offset) {
    auto count      = src_component_ids.size() - src_offset;
    auto dst_offset = all_component_ids.size();
    all_component_ids.resize(dst_offset + count, handle.get_stream());
    component_vertices.resize(dst_offset + count, handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 src_component_ids.begin() + src_offset,
                 src_component_ids.end(),
                 all_component_ids.begin() + dst_offset);
    thrust::copy(handle.get_thrust_policy(),
                 src_component_vertices.begin() + src_offset,
                 src_component_vertices.end(),
                 component_vertices.begin() + dst_offset);
    src_component_ids.resize(src_offset, handle.get_stream());
    src_component_ids.shrink_to_fit(handle.get_stream());
    src_component_vertices.resize(src_offset, handle.get_stream());
    src_component_vertices.shrink_to_fit(handle.get_stream());
  };

  // Unresolved sub-components (right portion, then trim source to resolved only).
  append_and_trim_source(fwd_only_component_idxs, fwd_only_vertices, fwd_resolved_count);
  append_and_trim_source(bwd_only_component_idxs, bwd_only_vertices, bwd_resolved_count);
  append_and_trim_source(remaining_component_idxs, remaining_vertices, remaining_resolved_count);

  // Resolved: SCC vertices.
  append_and_trim_source(scc_component_idxs, scc_vertices, 0);

  // Resolved: trivial singleton vertices (component_id = vertex_id).
  {
    auto dst_offset = all_component_ids.size();
    auto count      = trivial_singleton_scc_vertices.size();
    all_component_ids.resize(dst_offset + count, handle.get_stream());
    component_vertices.resize(dst_offset + count, handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 trivial_singleton_scc_vertices.begin(),
                 trivial_singleton_scc_vertices.end(),
                 all_component_ids.begin() + dst_offset);
    thrust::copy(handle.get_thrust_policy(),
                 trivial_singleton_scc_vertices.begin(),
                 trivial_singleton_scc_vertices.end(),
                 component_vertices.begin() + dst_offset);
    trivial_singleton_scc_vertices.resize(0, handle.get_stream());
    trivial_singleton_scc_vertices.shrink_to_fit(handle.get_stream());
  }

  // Resolved: size-1 fwd/bwd/remaining sub-components (entire remaining source, then free).
  append_and_trim_source(fwd_only_component_idxs, fwd_only_vertices, 0);
  append_and_trim_source(bwd_only_component_idxs, bwd_only_vertices, 0);
  append_and_trim_source(remaining_component_idxs, remaining_vertices, 0);

  // Extract unique component_ids and their offsets in a single pass.
  // Data is already grouped by component_id (contiguous runs), so we just find run boundaries.
  rmm::device_uvector<vertex_t> component_ids(component_vertices.size(), handle.get_stream());
  rmm::device_uvector<vertex_t> component_offsets(component_vertices.size() + 1, handle.get_stream());

  auto input_first = thrust::make_zip_iterator(
    all_component_ids.begin(), thrust::make_counting_iterator(vertex_t{0}));
  auto output_first = thrust::make_zip_iterator(
    component_ids.begin(), component_offsets.begin());
  auto output_end = thrust::copy_if(
    handle.get_thrust_policy(),
    input_first,
    input_first + component_vertices.size(),
    thrust::make_counting_iterator(size_t{0}),
    output_first,
    detail::is_first_in_run_t<vertex_t const*>{all_component_ids.data()});

  auto num_components = static_cast<size_t>(cuda::std::distance(output_first, output_end));
  component_ids.resize(num_components, handle.get_stream());
  component_offsets.resize(num_components + 1, handle.get_stream());
  component_offsets.set_element(
    num_components, static_cast<vertex_t>(component_vertices.size()), handle.get_stream());

  return std::make_tuple(std::move(component_ids),
                         std::move(component_offsets),
                         std::move(component_vertices),
                         num_unresolved_components);
}

// return component_ids, component_offsets, component_vertices, num_unresolved_components
template <typename GraphViewType>
std::tuple<rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>,
           rmm::device_uvector<typename GraphViewType::vertex_type>,
           size_t>
forward_backward_intersect(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  GraphViewType const& inverse_graph_view,
  raft::device_span<typename GraphViewType::vertex_type const> inverse_renumber_map,
  rmm::device_uvector<typename GraphViewType::vertex_type>&& unresolved_component_offsets,
  rmm::device_uvector<typename GraphViewType::vertex_type>&& unresolved_component_vertices)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  constexpr bool multi_gpu = GraphViewType::is_multi_gpu;

  auto in_degrees = inverse_graph_view.compute_out_degrees(handle);
  {
    auto tmp_degrees = rmm::device_uvector<edge_t>(graph_view.local_vertex_partition_range_size(),
                                                   handle.get_stream());
    if constexpr (multi_gpu) {
      auto tmp_vertices =
        rmm::device_uvector<vertex_t>(inverse_renumber_map.size(), handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   inverse_renumber_map.begin(),
                   inverse_renumber_map.end(),
                   tmp_vertices.begin());
      std::vector<arithmetic_device_uvector_t> vertex_properties{};
      vertex_properties.push_back(std::move(in_degrees));
      std::tie(tmp_vertices, vertex_properties) =
        shuffle_int_vertices(handle,
                             std::move(tmp_vertices),
                             std::move(vertex_properties),
                             graph_view.vertex_partition_range_lasts());
      in_degrees = std::move(std::get<rmm::device_uvector<vertex_t>>(vertex_properties[0]));
      thrust::scatter(
        handle.get_thrust_policy(),
        in_degrees.begin(),
        in_degrees.end(),
        cuda::make_transform_iterator(
          tmp_vertices.begin(),
          detail::shift_left_t<vertex_t>{graph_view.local_vertex_partition_range_first()}),
        tmp_degrees.begin());
    } else {
      thrust::gather(handle.get_thrust_policy(),
                     inverse_renumber_map.begin(),
                     inverse_renumber_map.end(),
                     in_degrees.begin(),
                     tmp_degrees.begin());
    }
    in_degrees = std::move(tmp_degrees);
  }
  auto out_degrees = graph_view.compute_out_degrees(handle);

  rmm::device_uvector<edge_t> unresolved_component_vertex_in_degrees(
    unresolved_component_vertices.size(), handle.get_stream());
  rmm::device_uvector<edge_t> unresolved_component_vertex_out_degrees(
    unresolved_component_vertices.size(), handle.get_stream());
  {
    auto map_first = cuda::make_transform_iterator(
      unresolved_component_vertices.begin(),
      detail::shift_left_t<vertex_t>{graph_view.local_vertex_partition_range_first()});
    thrust::gather(handle.get_thrust_policy(),
                   map_first,
                   map_first + unresolved_component_vertices.size(),
                   in_degrees.begin(),
                   unresolved_component_vertex_in_degrees.begin());
    thrust::gather(handle.get_thrust_policy(),
                   map_first,
                   map_first + unresolved_component_vertices.size(),
                   out_degrees.begin(),
                   unresolved_component_vertex_out_degrees.begin());
  }

  auto trivial_singleton_scc_vertices = find_trivial_singleton_scc_vertices(
    handle,
    graph_view,
    inverse_graph_view,
    inverse_renumber_map,
    raft::device_span<vertex_t>(unresolved_component_vertices.data(),
                                unresolved_component_vertices.size()),
    std::move(in_degrees),
    std::move(out_degrees));

  auto [pivots, pivot_unresolved_component_idxs] =
    find_pivots(handle,
                graph_view,
                raft::device_span<vertex_t const>(unresolved_component_offsets.data(),
                                                  unresolved_component_offsets.size()),
                raft::device_span<vertex_t const>(unresolved_component_vertices.data(),
                                                  unresolved_component_vertices.size()),
                raft::device_span<edge_t const>(unresolved_component_vertex_in_degrees.data(),
                                                unresolved_component_vertex_in_degrees.size()),
                raft::device_span<edge_t const>(unresolved_component_vertex_out_degrees.data(),
                                                unresolved_component_vertex_out_degrees.size()),
                raft::device_span<vertex_t const>(trivial_singleton_scc_vertices.data(),
                                                  trivial_singleton_scc_vertices.size()));
  unresolved_component_vertex_in_degrees.resize(0, handle.get_stream());
  unresolved_component_vertex_in_degrees.shrink_to_fit(handle.get_stream());
  unresolved_component_vertex_out_degrees.resize(0, handle.get_stream());
  unresolved_component_vertex_out_degrees.shrink_to_fit(handle.get_stream());

  auto num_aggregate_pivots = pivots.size();
  if constexpr (GraphViewType::is_multi_gpu) {
    num_aggregate_pivots = host_scalar_allreduce(
      handle.get_comms(), num_aggregate_pivots, raft::comms::op_t::SUM, handle.get_stream());
  }

  rmm::device_uvector<vertex_t> forward_set_offsets(0, handle.get_stream());
  rmm::device_uvector<vertex_t> forward_set_vertices(0, handle.get_stream());
  if (num_aggregate_pivots > 0) {
    std::tie(forward_set_offsets, forward_set_vertices) =
      reachable_sets(handle,
                     graph_view,
                     raft::device_span<vertex_t const>(pivots.data(), pivots.size()),
                     raft::device_span<vertex_t const>(pivot_unresolved_component_idxs.data(),
                                                       pivot_unresolved_component_idxs.size()),
                     static_cast<vertex_t>(unresolved_component_offsets.size() - 1));
  } else {
    forward_set_offsets.resize(unresolved_component_offsets.size() - 1, handle.get_stream());
    thrust::fill(handle.get_thrust_policy(),
                 forward_set_offsets.begin(),
                 forward_set_offsets.end(),
                 vertex_t{0});
  }

  rmm::device_uvector<vertex_t> backward_set_offsets(0, handle.get_stream());
  rmm::device_uvector<vertex_t> backward_set_vertices(0, handle.get_stream());
  if (num_aggregate_pivots > 0) {
    if constexpr (multi_gpu) {
      std::vector<arithmetic_device_uvector_t> vertex_properties{};
      vertex_properties.push_back(std::move(pivot_unresolved_component_idxs));
      std::tie(pivots, vertex_properties) =
        shuffle_ext_vertices(handle, std::move(pivots), std::move(vertex_properties));
      pivot_unresolved_component_idxs =
        std::move(std::get<rmm::device_uvector<vertex_t>>(vertex_properties[0]));
    }
    renumber_local_ext_vertices<vertex_t, multi_gpu>(
      handle,
      pivots.data(),
      pivots.size(),
      inverse_renumber_map.data(),
      inverse_graph_view.local_vertex_partition_range_first(),
      inverse_graph_view.local_vertex_partition_range_last());
    thrust::sort_by_key(handle.get_thrust_policy(),
                        pivots.begin(),
                        pivots.end(),
                        pivot_unresolved_component_idxs.begin());

    std::tie(backward_set_offsets, backward_set_vertices) =
      reachable_sets(handle,
                     inverse_graph_view,
                     raft::device_span<vertex_t const>(pivots.data(), pivots.size()),
                     raft::device_span<vertex_t const>(pivot_unresolved_component_idxs.data(),
                                                       pivot_unresolved_component_idxs.size()),
                     static_cast<vertex_t>(unresolved_component_offsets.size() - 1));

    unrenumber_local_int_vertices(handle,
                                  backward_set_vertices.data(),
                                  backward_set_vertices.size(),
                                  inverse_renumber_map.data(),
                                  inverse_graph_view.local_vertex_partition_range_first(),
                                  inverse_graph_view.local_vertex_partition_range_last());
    rmm::device_uvector<vertex_t> tmp_unresolved_component_idxs(backward_set_vertices.size(),
                                                                handle.get_stream());
    auto component_idx_first = cuda::make_transform_iterator(
      thrust::make_counting_iterator(vertex_t{0}),
      cuda::proclaim_return_type<vertex_t>(
        [component_offsets = raft::device_span<vertex_t const>(
           backward_set_offsets.data(), backward_set_offsets.size())] __device__(vertex_t i) {
          return static_cast<vertex_t>(cuda::std::distance(
            component_offsets.begin() + 1,
            thrust::upper_bound(
              thrust::seq, component_offsets.begin() + 1, component_offsets.end(), i)));
        }));
    thrust::copy(handle.get_thrust_policy(),
                 component_idx_first,
                 component_idx_first + backward_set_vertices.size(),
                 tmp_unresolved_component_idxs.begin());
    if constexpr (multi_gpu) {
      std::vector<arithmetic_device_uvector_t> vertex_properties{};
      vertex_properties.push_back(std::move(tmp_unresolved_component_idxs));
      std::tie(backward_set_vertices, vertex_properties) =
        shuffle_int_vertices(handle,
                             std::move(backward_set_vertices),
                             std::move(vertex_properties),
                             graph_view.vertex_partition_range_lasts());
      tmp_unresolved_component_idxs =
        std::move(std::get<rmm::device_uvector<vertex_t>>(vertex_properties[0]));
      auto pair_first = thrust::make_zip_iterator(tmp_unresolved_component_idxs.begin(),
                                                  backward_set_vertices.begin());
      thrust::sort(
        handle.get_thrust_policy(), pair_first, pair_first + tmp_unresolved_component_idxs.size());
      backward_set_offsets.set_element_to_zero_async(0, handle.get_stream());
      thrust::upper_bound(
        handle.get_thrust_policy(),
        tmp_unresolved_component_idxs.begin(),
        tmp_unresolved_component_idxs.end(),
        thrust::make_counting_iterator(vertex_t{0}),
        thrust::make_counting_iterator(static_cast<vertex_t>(backward_set_offsets.size() - 1)),
        backward_set_offsets.begin() + 1);
    } else {
      auto pair_first = thrust::make_zip_iterator(tmp_unresolved_component_idxs.begin(),
                                                  backward_set_vertices.begin());
      thrust::sort(
        handle.get_thrust_policy(), pair_first, pair_first + tmp_unresolved_component_idxs.size());
    }
  } else {
    backward_set_offsets.resize(unresolved_component_offsets.size() - 1, handle.get_stream());
    thrust::fill(handle.get_thrust_policy(),
                 backward_set_offsets.begin(),
                 backward_set_offsets.end(),
                 vertex_t{0});
  }

  return intersect_reachable_sets<GraphViewType>(handle,
                                                       std::move(unresolved_component_offsets),
                                                       std::move(unresolved_component_vertices),
                                                       std::move(forward_set_offsets),
                                                       std::move(forward_set_vertices),
                                                       std::move(backward_set_offsets),
                                                       std::move(backward_set_vertices),
                                                       std::move(trivial_singleton_scc_vertices));
}

template <typename GraphViewType>
void strongly_connected_components_impl(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  raft::device_span<typename GraphViewType::vertex_type> components,
  bool do_expensive_check)
{
  using vertex_t                  = typename GraphViewType::vertex_type;
  using edge_t                    = typename GraphViewType::edge_type;
  using weight_t                  = float;    // dummy
  using edge_type_t               = int32_t;  // dummy
  constexpr bool store_transposed = GraphViewType::is_storage_transposed;
  constexpr bool multi_gpu        = GraphViewType::is_multi_gpu;

  static_assert(std::is_integral<vertex_t>::value,
                "GraphViewType::vertex_type should be integral.");
  static_assert(!GraphViewType::is_storage_transposed,
                "GraphViewType should support the push model.");

  auto const num_vertices = graph_view.number_of_vertices();
  if (num_vertices == 0) { return; }

  // 1. check input arguments

  CUGRAPH_EXPECTS(
    !graph_view.is_symmetric(),
    "Invalid input argument>: call weakly_connnected_components instead for symmetric graphs.");

  if (do_expensive_check) {
    // nothing to do
  }

  // 2. initialize component IDs (initially, every vertex belongs to one unresolved component)

  thrust::fill(handle.get_thrust_policy(), components.begin(), components.end(), vertex_t{0});

  // 3. create an edge mask and mask out self-loops & multi-edges (except for the first one); this
  // edge mask will be used to mask out edges between different components

  auto forward_graph_view = graph_view;
  edge_property_t<edge_t, bool> edge_mask(handle, forward_graph_view);
  {
    if (forward_graph_view.has_edge_mask()) { forward_graph_view.clear_edge_mask(); }
    cugraph::fill_edge_property(handle, forward_graph_view, edge_mask.mutable_view(), false);
    edge_multi_index_property_t<edge_t, vertex_t> edge_multi_indices(handle, graph_view);
    transform_e(handle,
                graph_view,
                edge_src_dummy_property_t{}.view(),
                edge_dst_dummy_property_t{}.view(),
                edge_multi_indices.view(),
                cuda::proclaim_return_type<bool>(
                  [] __device__(auto src, auto dst, auto, auto, auto multi_edge_index) {
                    return (src != dst) && (multi_edge_index == 0);
                  }),
                edge_mask.mutable_view());
    forward_graph_view.attach_edge_mask(edge_mask.view());
  }

  // 4. create an inverse graph

  graph_t<vertex_t, edge_t, store_transposed, multi_gpu> inverse_graph(handle);
  rmm::device_uvector<vertex_t> inverse_renumber_map(0, handle.get_stream());
  {  // FIXME: we may avoid this if we create a transpose_graph function that takes a const
     // reference to a graph_view_t object
    rmm::device_uvector<vertex_t> edgelist_srcs(0, handle.get_stream());
    rmm::device_uvector<vertex_t> edgelist_dsts(0, handle.get_stream());
    std::tie(edgelist_srcs, edgelist_dsts, std::ignore, std::ignore, std::ignore) =
      decompress_to_edgelist<vertex_t, edge_t, weight_t, edge_type_t, store_transposed, multi_gpu>(
        handle, forward_graph_view, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
    std::swap(edgelist_srcs, edgelist_dsts);
    if constexpr (multi_gpu) {
      std::tie(edgelist_srcs, edgelist_dsts, std::ignore, std::ignore) =
        shuffle_ext_edges(handle,
                          std::move(edgelist_srcs),
                          std::move(edgelist_dsts),
                          std::vector<cugraph::arithmetic_device_uvector_t>{},
                          store_transposed);
    }
    std::optional<rmm::device_uvector<vertex_t>> tmp_renumber_map{std::nullopt};
    std::tie(inverse_graph, std::ignore, tmp_renumber_map) =
      create_graph_from_edgelist<vertex_t, edge_t, store_transposed, multi_gpu>(
        handle,
        std::nullopt,
        std::move(edgelist_srcs),
        std::move(edgelist_dsts),
        std::vector<cugraph::arithmetic_device_uvector_t>{},
        graph_properties_t{false, false},
        true);
    inverse_renumber_map = std::move(*tmp_renumber_map);
  }
  auto inverse_graph_view = inverse_graph.view();
  edge_property_t<edge_t, bool> inverse_edge_mask(handle, inverse_graph_view);
  cugraph::fill_edge_property(handle, inverse_graph_view, inverse_edge_mask.mutable_view(), true);
  inverse_graph_view.attach_edge_mask(inverse_edge_mask.view());

  // 5. prepare for recursvie forward-backward SCC: set unresolved_component_offsets and
  // unresolved_component_vertices and initialzie edge_src|dst_components

  rmm::device_uvector<vertex_t> unresolved_component_offsets(2, handle.get_stream());
  rmm::device_uvector<vertex_t> unresolved_component_vertices(
    forward_graph_view.local_vertex_partition_range_size(), handle.get_stream());
  unresolved_component_offsets.set_element_to_zero_async(0, handle.get_stream());
  unresolved_component_offsets.set_element(
    1, forward_graph_view.local_vertex_partition_range_size(), handle.get_stream());
  thrust::sequence(handle.get_thrust_policy(),
                   unresolved_component_vertices.begin(),
                   unresolved_component_vertices.end(),
                   forward_graph_view.local_vertex_partition_range_first());

  auto edge_src_components = multi_gpu
                               ? edge_src_property_t<vertex_t, vertex_t>(handle, forward_graph_view)
                               : edge_src_property_t<vertex_t, vertex_t>(handle);
  auto edge_dst_components = multi_gpu
                               ? edge_dst_property_t<vertex_t, vertex_t>(handle, forward_graph_view)
                               : edge_dst_property_t<vertex_t, vertex_t>(handle);
  auto inverse_edge_src_components =
    multi_gpu ? edge_src_property_t<vertex_t, vertex_t>(handle, inverse_graph_view)
              : edge_src_property_t<vertex_t, vertex_t>(handle);
  auto inverse_edge_dst_components =
    multi_gpu ? edge_dst_property_t<vertex_t, vertex_t>(handle, inverse_graph_view)
              : edge_dst_property_t<vertex_t, vertex_t>(handle);
  if constexpr (multi_gpu) {
    fill_edge_src_property(
      handle, forward_graph_view, edge_src_components.mutable_view(), vertex_t{0});
    fill_edge_dst_property(
      handle, forward_graph_view, edge_dst_components.mutable_view(), vertex_t{0});
    fill_edge_src_property(
      handle, inverse_graph_view, inverse_edge_src_components.mutable_view(), vertex_t{0});
    fill_edge_dst_property(
      handle, inverse_graph_view, inverse_edge_dst_components.mutable_view(), vertex_t{0});
  }

  // 6. recursive forward-backward SCC

  while ((unresolved_component_offsets.size() - 1) > 0) {
    // 6-1. perform forward-backward SCC

    auto [component_ids, component_offsets, component_vertices, num_unresolved_components] =
      forward_backward_intersect(
        handle,
        forward_graph_view,
        inverse_graph_view,
        raft::device_span<vertex_t const>(inverse_renumber_map.data(), inverse_renumber_map.size()),
        std::move(unresolved_component_offsets),
        std::move(unresolved_component_vertices));

    // 6-2. update components

    auto component_id_first = cuda::make_transform_iterator(
      thrust::make_counting_iterator(size_t{0}),
      cuda::proclaim_return_type<vertex_t>(
        [component_offsets =
           raft::device_span<vertex_t const>(component_offsets.data(), component_offsets.size()),
         component_ids = raft::device_span<vertex_t const>(
           component_ids.data(), component_ids.size())] __device__(size_t i) {
          auto idx = cuda::std::distance(
            component_offsets.begin() + 1,
            thrust::upper_bound(
              thrust::seq, component_offsets.begin() + 1, component_offsets.end(), i));
          return component_ids[idx];
        }));
    auto map_first = cuda::make_transform_iterator(
      component_vertices.begin(),
      detail::shift_left_t<vertex_t>{forward_graph_view.local_vertex_partition_range_first()});
    thrust::scatter(handle.get_thrust_policy(),
                    component_id_first,
                    component_id_first + component_vertices.size(),
                    map_first,
                    components.begin());

    // 6-3. mask out edges between different components

    if constexpr (multi_gpu) {
      rmm::device_uvector<vertex_t> tmp_vertices(component_vertices.size(), handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   component_vertices.begin(),
                   component_vertices.end(),
                   tmp_vertices.begin());
      rmm::device_uvector<vertex_t> tmp_components(component_vertices.size(), handle.get_stream());
      thrust::gather(handle.get_thrust_policy(),
                     map_first,
                     map_first + component_vertices.size(),
                     components.begin(),
                     tmp_components.begin());
      thrust::sort_by_key(handle.get_thrust_policy(),
                          tmp_vertices.begin(),
                          tmp_vertices.end(),
                          tmp_components.begin());

      update_edge_src_property(handle,
                               forward_graph_view,
                               tmp_vertices.begin(),
                               tmp_vertices.end(),
                               tmp_components.begin(),
                               edge_src_components.mutable_view());
      update_edge_dst_property(handle,
                               forward_graph_view,
                               tmp_vertices.begin(),
                               tmp_vertices.end(),
                               tmp_components.begin(),
                               edge_dst_components.mutable_view());

      std::vector<arithmetic_device_uvector_t> vertex_properties{};
      vertex_properties.push_back(arithmetic_device_uvector_t{std::move(tmp_components)});
      std::tie(tmp_vertices, vertex_properties) =
        shuffle_ext_vertices(handle, std::move(tmp_vertices), std::move(vertex_properties));
      tmp_components = std::move(std::get<rmm::device_uvector<vertex_t>>(vertex_properties[0]));
      renumber_local_ext_vertices<vertex_t, multi_gpu>(
        handle,
        tmp_vertices.data(),
        tmp_vertices.size(),
        inverse_renumber_map.data(),
        inverse_graph_view.local_vertex_partition_range_first(),
        inverse_graph_view.local_vertex_partition_range_last());
      thrust::sort_by_key(handle.get_thrust_policy(),
                          tmp_vertices.begin(),
                          tmp_vertices.end(),
                          tmp_components.begin());

      update_edge_src_property(handle,
                               inverse_graph_view,
                               tmp_vertices.begin(),
                               tmp_vertices.end(),
                               tmp_components.begin(),
                               inverse_edge_src_components.mutable_view());
      update_edge_dst_property(handle,
                               inverse_graph_view,
                               tmp_vertices.begin(),
                               tmp_vertices.end(),
                               tmp_components.begin(),
                               inverse_edge_dst_components.mutable_view());
    }

    auto edge_src_component_view = multi_gpu
                                     ? edge_src_components.view()
                                     : make_edge_src_property_view<vertex_t, vertex_t>(
                                         forward_graph_view,
                                         components.begin(),
                                         forward_graph_view.local_vertex_partition_range_size());
    auto edge_dst_component_view = multi_gpu
                                     ? edge_dst_components.view()
                                     : make_edge_dst_property_view<vertex_t, vertex_t>(
                                         forward_graph_view,
                                         components.begin(),
                                         forward_graph_view.local_vertex_partition_range_size());

    auto new_edge_mask = make_initialized_edge_property(handle, forward_graph_view, false);
    transform_e(handle,
                forward_graph_view,
                edge_src_component_view,
                edge_dst_component_view,
                edge_dummy_property_t{}.view(),
                cuda::proclaim_return_type<bool>(
                  [] __device__(auto, auto, auto src_component, auto dst_component, auto) {
                    return src_component == dst_component;
                  }),
                new_edge_mask.mutable_view());
    if (forward_graph_view.has_edge_mask()) { forward_graph_view.clear_edge_mask(); }
    edge_mask = std::move(new_edge_mask);
    forward_graph_view.attach_edge_mask(edge_mask.view());

    {
      std::optional<rmm::device_uvector<vertex_t>> inverse_components{std::nullopt};
      if constexpr (!multi_gpu) {
        inverse_components = rmm::device_uvector<vertex_t>(
          inverse_graph_view.local_vertex_partition_range_size(), handle.get_stream());
        thrust::gather(handle.get_thrust_policy(),
                       inverse_renumber_map.begin(),
                       inverse_renumber_map.end(),
                       components.begin(),
                       inverse_components->begin());
      }

      auto inverse_edge_src_component_view =
        multi_gpu ? inverse_edge_src_components.view()
                  : make_edge_src_property_view<vertex_t, vertex_t>(
                      inverse_graph_view,
                      inverse_components->begin(),
                      inverse_graph_view.local_vertex_partition_range_size());
      auto inverse_edge_dst_component_view =
        multi_gpu ? inverse_edge_dst_components.view()
                  : make_edge_dst_property_view<vertex_t, vertex_t>(
                      inverse_graph_view,
                      inverse_components->begin(),
                      inverse_graph_view.local_vertex_partition_range_size());

      auto new_inverse_edge_mask =
        make_initialized_edge_property(handle, inverse_graph_view, false);
      transform_e(handle,
                  inverse_graph_view,
                  inverse_edge_src_component_view,
                  inverse_edge_dst_component_view,
                  edge_dummy_property_t{}.view(),
                  cuda::proclaim_return_type<bool>(
                    [] __device__(auto, auto, auto src_component, auto dst_component, auto) {
                      return src_component == dst_component;
                    }),
                  new_inverse_edge_mask.mutable_view());
      if (inverse_graph_view.has_edge_mask()) { inverse_graph_view.clear_edge_mask(); }
      inverse_edge_mask = std::move(new_inverse_edge_mask);
      inverse_graph_view.attach_edge_mask(inverse_edge_mask.view());
    }

    // 6-4. update unresolved_component_offsets and unresolved_component_vertices

    component_ids.resize(num_unresolved_components, handle.get_stream());
    component_offsets.resize(num_unresolved_components + 1, handle.get_stream());
    component_vertices.resize(component_offsets.back_element(handle.get_stream()),
                              handle.get_stream());
    component_ids.shrink_to_fit(handle.get_stream());
    component_offsets.shrink_to_fit(handle.get_stream());
    component_vertices.shrink_to_fit(handle.get_stream());

    unresolved_component_offsets  = std::move(component_offsets);
    unresolved_component_vertices = std::move(component_vertices);
  }

  return;
}

}  // namespace

template <typename vertex_t, typename edge_t, bool multi_gpu>
rmm::device_uvector<vertex_t> strongly_connected_components(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  bool do_expensive_check)
{
  rmm::device_uvector<vertex_t> components(graph_view.local_vertex_partition_range_size(),
                                           handle.get_stream());
  strongly_connected_components_impl(
    handle,
    graph_view,
    raft::device_span<vertex_t>(components.data(), components.size()),
    do_expensive_check);

  return std::move(components);
}

}  // namespace cugraph
