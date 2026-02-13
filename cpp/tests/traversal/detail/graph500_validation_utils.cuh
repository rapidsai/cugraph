/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "detail/shuffle_wrappers.hpp"
#include "graph500_misc.cuh"
#include "prims/count_if_e.cuh"
#include "prims/extract_transform_if_e.cuh"
#include "prims/fill_edge_src_dst_property.cuh"
#include "prims/transform_e.cuh"
#include "prims/update_edge_src_dst_property.cuh"
#include "utilities/collect_comm.cuh"

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_property.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/large_buffer_manager.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/shuffle_functions.hpp>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/host_scalar_comm.hpp>

#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/std/iterator>
#include <cuda/std/tuple>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <gtest/gtest.h>

#include <random>

template <typename vertex_t, typename edge_t, typename distance_t>
edge_t compute_number_of_visited_undirected_edges(
  raft::handle_t const& handle,
  raft::device_span<distance_t const> mg_distances,
  cugraph::graph_view_t<vertex_t, edge_t, false, true> const& mg_pruned_graph_view,
  std::optional<raft::device_span<distance_t const>> mg_pruned_graph_distances,
  raft::device_span<vertex_t const> mg_graph_to_pruned_graph_map,
  vertex_t invalid_vertex,
  distance_t invalid_distance)
{
  auto& comm = handle.get_comms();

  edge_t visited_undirected_edges{};
  if (mg_pruned_graph_distances) {
    rmm::device_uvector<bool> visited(mg_pruned_graph_view.local_vertex_partition_range_size(),
                                      handle.get_stream());
    thrust::transform(handle.get_thrust_policy(),
                      mg_pruned_graph_distances->begin(),
                      mg_pruned_graph_distances->end(),
                      visited.begin(),
                      cuda::proclaim_return_type<bool>(
                        [invalid_distance] __device__(auto d) { return d != invalid_distance; }));
    cugraph::edge_src_property_t<vertex_t, bool> edge_src_visited(handle, mg_pruned_graph_view);
    cugraph::update_edge_src_property(
      handle, mg_pruned_graph_view, visited.begin(), edge_src_visited.mutable_view());
    visited_undirected_edges =
      cugraph::count_if_e(
        handle,
        mg_pruned_graph_view,
        edge_src_visited.view(),
        cugraph::edge_dst_dummy_property_t{}.view(),
        cugraph::edge_dummy_property_t{}.view(),
        [] __device__(auto, auto, auto src_visited, auto, auto) { return src_visited; }) /
      edge_t{2} /* undirected */;
    auto forest_edge_count = thrust::count_if(
      handle.get_thrust_policy(),
      thrust::make_zip_iterator(mg_distances.begin(), mg_graph_to_pruned_graph_map.begin()),
      thrust::make_zip_iterator(mg_distances.end(), mg_graph_to_pruned_graph_map.end()),
      [invalid_distance, invalid_vertex] __device__(auto pair) {
        return (cuda::std::get<0>(pair) != invalid_distance /* reachable */) &&
               (cuda::std::get<1>(pair) == invalid_vertex /* not in the pruned graph */);
      });  // # vertices reachable from 2-cores but not in 2-cores
#if 1      // FIXME: we should add host_allreduce to raft
    forest_edge_count = cugraph::host_scalar_allreduce(
      comm, forest_edge_count, raft::comms::op_t::SUM, handle.get_stream());
#else
    comm.host_allreduce(std::addressof(forest_edge_count),
                        std::addressof(forest_edge_count),
                        size_t{1},
                        raft::comms::op_t::SUM);
#endif
    visited_undirected_edges += forest_edge_count;
  } else {  // isolated trees
    auto num_visited =
      thrust::count_if(handle.get_thrust_policy(),
                       mg_distances.begin(),
                       mg_distances.end(),
                       [invalid_distance] __device__(auto d) { return d != invalid_distance; });
#if 1  // FIXME: we should add host_allreduce to raft
    num_visited = cugraph::host_scalar_allreduce(
      comm, num_visited, raft::comms::op_t::SUM, handle.get_stream());
#else
    comm.host_allreduce(
      std::addressof(num_visited), std::addressof(num_visited), size_t{1}, raft::comms::op_t::SUM);
#endif
    visited_undirected_edges = num_visited - 1;  // # edges in a tree is # vertices - 1
  }
  return visited_undirected_edges;
}

// no cycle and backtrace to the starting vertex
template <typename vertex_t>
bool is_valid_predecessor_tree(raft::handle_t const& handle,
                               raft::device_span<vertex_t const> mg_predecessors,
                               raft::host_span<vertex_t const> vertex_partition_range_offsets,
                               vertex_t starting_vertex,
                               vertex_t local_vertex_partition_range_first,
                               vertex_t invalid_vertex)
{
  auto& comm = handle.get_comms();

  rmm::device_uvector<vertex_t> ancestors(mg_predecessors.size(), handle.get_stream());
  ancestors.resize(
    cuda::std::distance(
      ancestors.begin(),
      thrust::copy_if(
        handle.get_thrust_policy(),
        mg_predecessors.begin(),
        mg_predecessors.end(),
        ancestors.begin(),
        cuda::proclaim_return_type<bool>([starting_vertex, invalid_vertex] __device__(auto pred) {
          return (pred != starting_vertex) && (pred != invalid_vertex);
        }))),
    handle.get_stream());

  size_t level{0};
  auto aggregate_size = ancestors.size();
#if 1  // FIXME: we should add host_allreduce to raft
  aggregate_size = cugraph::host_scalar_allreduce(
    comm, aggregate_size, raft::comms::op_t::SUM, handle.get_stream());
#else
  comm.host_allreduce(std::addressof(aggregate_size),
                      std::addressof(aggregate_size),
                      size_t{1},
                      raft::comms::op_t::SUM);
#endif
  while (aggregate_size > size_t{0}) {
    if (level >= static_cast<size_t>(vertex_partition_range_offsets.back() - 1)) { return false; }
    auto num_invalids =
      thrust::count(handle.get_thrust_policy(), ancestors.begin(), ancestors.end(), invalid_vertex);
#if 1  // FIXME: we should add host_allreduce to raft
    num_invalids = cugraph::host_scalar_allreduce(
      comm, num_invalids, raft::comms::op_t::SUM, handle.get_stream());
#else
    comm.host_allreduce(std::addressof(num_invalids),
                        std::addressof(num_invalids),
                        size_t{1},
                        raft::comms::op_t::SUM);
#endif
    if (num_invalids > 0) { return false; }
    ancestors = cugraph::collect_values_for_int_vertices(
      handle,
      ancestors.begin(),
      ancestors.end(),
      mg_predecessors.begin(),
      raft::host_span<vertex_t const>(vertex_partition_range_offsets.data() + 1,
                                      vertex_partition_range_offsets.size() - 1),
      local_vertex_partition_range_first);
    ancestors.resize(cuda::std::distance(
                       ancestors.begin(),
                       thrust::remove_if(handle.get_thrust_policy(),
                                         ancestors.begin(),
                                         ancestors.end(),
                                         cugraph::detail::is_equal_t<vertex_t>{starting_vertex})),
                     handle.get_stream());
    aggregate_size = ancestors.size();
#if 1  // FIXME: we should add host_allreduce to raft
    aggregate_size = cugraph::host_scalar_allreduce(
      comm, aggregate_size, raft::comms::op_t::SUM, handle.get_stream());
#else
    comm.host_allreduce(std::addressof(aggregate_size),
                        std::addressof(aggregate_size),
                        size_t{1},
                        raft::comms::op_t::SUM);
#endif
    ++level;
  }
  return true;
}

// check that
// BFS: distance(v) = distance(parents(v)) + 1
// SSSP: distance(v) = distance(parents(v)) + w
template <typename vertex_t, typename distance_t>
bool check_distance_from_parents(
  raft::handle_t const& handle,
  raft::device_span<vertex_t const> mg_predecessors,
  raft::device_span<distance_t const> mg_distances,
  std::optional<raft::device_span<distance_t const>> mg_w_to_predecessors,
  raft::host_span<vertex_t const> vertex_partition_range_offsets,
  vertex_t starting_vertex,
  vertex_t local_vertex_partition_range_first,
  vertex_t invalid_vertex)
{
  auto& comm = handle.get_comms();

  rmm::device_uvector<vertex_t> tree_srcs(mg_predecessors.size(), handle.get_stream());
  rmm::device_uvector<vertex_t> tree_dsts(tree_srcs.size(), handle.get_stream());
  auto input_pair_first = thrust::make_zip_iterator(
    mg_predecessors.begin(), thrust::make_counting_iterator(local_vertex_partition_range_first));
  auto output_pair_first = thrust::make_zip_iterator(tree_srcs.begin(), tree_dsts.begin());
  tree_srcs.resize(cuda::std::distance(
                     output_pair_first,
                     thrust::copy_if(handle.get_thrust_policy(),
                                     input_pair_first,
                                     input_pair_first + mg_predecessors.size(),
                                     output_pair_first,
                                     cuda::proclaim_return_type<bool>(
                                       [starting_vertex, invalid_vertex] __device__(auto pair) {
                                         auto pred = cuda::std::get<0>(pair);
                                         auto v    = cuda::std::get<1>(pair);
                                         return (pred != invalid_vertex) && (v != starting_vertex);
                                       }))),
                   handle.get_stream());
  tree_dsts.resize(tree_srcs.size(), handle.get_stream());

  auto tree_src_dists = cugraph::collect_values_for_int_vertices(
    handle,
    tree_srcs.begin(),
    tree_srcs.end(),
    mg_distances.begin(),
    raft::host_span<vertex_t const>(vertex_partition_range_offsets.data() + 1,
                                    vertex_partition_range_offsets.size() - 1),
    local_vertex_partition_range_first);

  rmm::device_uvector<distance_t> tree_dst_dists(tree_dsts.size(), handle.get_stream());
  thrust::transform(
    handle.get_thrust_policy(),
    tree_dsts.begin(),
    tree_dsts.end(),
    tree_dst_dists.begin(),
    cuda::proclaim_return_type<distance_t>(
      [distances = raft::device_span<distance_t const>(mg_distances.data(), mg_distances.size()),
       v_first   = local_vertex_partition_range_first] __device__(auto v) {
        return distances[v - v_first];
      }));

  std::optional<rmm::device_uvector<distance_t>> tree_edge_weights{
    std::nullopt};  // this assumes that the distance to the parent is identical to the distance
                    // from the parent (this is true as Graph 500 assumes an undirected graph)

  if (mg_w_to_predecessors) {
    tree_edge_weights = rmm::device_uvector<distance_t>(tree_dsts.size(), handle.get_stream());
    thrust::transform(handle.get_thrust_policy(),
                      tree_dsts.begin(),
                      tree_dsts.end(),
                      tree_edge_weights->begin(),
                      cuda::proclaim_return_type<distance_t>(
                        [weights = raft::device_span<distance_t const>(
                           mg_w_to_predecessors->data(), mg_w_to_predecessors->size()),
                         v_first = local_vertex_partition_range_first] __device__(auto v) {
                          return weights[v - v_first];
                        }));
  }

  if (tree_src_dists.size() != tree_dst_dists.size()) { return false; }

  size_t num_invalids{0};
  if constexpr (std::is_floating_point_v<distance_t>) {  // SSSP
    auto triplet_first = thrust::make_zip_iterator(
      tree_src_dists.begin(), tree_dst_dists.begin(), tree_edge_weights->begin());
    num_invalids = static_cast<size_t>(thrust::count_if(
      handle.get_thrust_policy(),
      triplet_first,
      triplet_first + tree_src_dists.size(),
      cuda::proclaim_return_type<bool>([] __device__(auto triplet) {
        auto src_dist = cuda::std::get<0>(triplet);
        auto dst_dist = cuda::std::get<1>(triplet);
        auto w        = cuda::std::get<2>(triplet);
        auto diff     = cuda::std::abs((src_dist + w) - dst_dist);
        return diff >
               cuda::std::max(
                 diff * 1e-4,
                 1e-6);  // 1e-4 & 1e-6 to consider limited floating point arithmetic resolution
      })));
  } else {  // BFS
    auto pair_first = thrust::make_zip_iterator(tree_src_dists.begin(), tree_dst_dists.begin());
    num_invalids    = static_cast<size_t>(
      thrust::count_if(handle.get_thrust_policy(),
                       pair_first,
                       pair_first + tree_src_dists.size(),
                       cuda::proclaim_return_type<bool>([] __device__(auto pair) {
                         auto src_dist = cuda::std::get<0>(pair);
                         auto dst_dist = cuda::std::get<1>(pair);
                         return (src_dist + 1) != dst_dist;
                       })));
  }
#if 1  // FIXME: we should add host_allreduce to raft
  num_invalids =
    cugraph::host_scalar_allreduce(comm, num_invalids, raft::comms::op_t::SUM, handle.get_stream());
#else
  comm.host_allreduce(
    std::addressof(num_invalids), std::addressof(num_invalids), size_t{1}, raft::comms::op_t::SUM);
#endif
  if (num_invalids > 0) { return false; }

  return true;
}

// check that for every edge e = (u, v),
// BFS: abs(dist(u) - dist(v)) <= 1 or dist(u) == dist(v) == invalid_distance
// SSSP: abs(dist(u) - dist(v)) <= w or dist(u) == dist(v) == invalid_distance
template <typename vertex_t, typename edge_t, typename distance_t>
bool check_edge_endpoint_distances(
  raft::handle_t const& handle,
  raft::device_span<vertex_t const> parents /* found in forest pruning */,
  std::optional<raft::device_span<distance_t const>> w_to_parents,
  raft::device_span<distance_t const> mg_distances,
  cugraph::graph_view_t<vertex_t, edge_t, false, true> const& mg_pruned_graph_view,
  std::optional<cugraph::edge_property_view_t<edge_t, distance_t const*>> const&
    mg_pruned_graph_edge_weight_view,
  raft::device_span<vertex_t const> mg_graph_to_pruned_graph_map,
  cugraph::graph_view_t<vertex_t, edge_t, false, true> const& mg_isolated_trees_view,
  std::optional<cugraph::edge_property_view_t<edge_t, distance_t const*>> const&
    mg_isolated_trees_edge_weight_view,
  raft::device_span<vertex_t const> mg_graph_to_isolated_trees_map,
  raft::host_span<vertex_t const> vertex_partition_range_offsets,
  vertex_t local_vertex_partition_range_first,
  vertex_t invalid_vertex,
  distance_t invalid_distance,
  bool reachable_from_2cores)
{
  using edge_type_t = int32_t;  // dummy

  assert(mg_pruned_graph_edge_weight_view.has_value() ==
         mg_isolated_trees_edge_weight_view.has_value());

  auto& comm = handle.get_comms();

  auto const& mg_subgraph_view =
    reachable_from_2cores ? mg_pruned_graph_view : mg_isolated_trees_view;
  cugraph::edge_src_property_t<vertex_t, bool> edge_src_flags(handle, mg_subgraph_view);
  cugraph::edge_dst_property_t<vertex_t, bool> edge_dst_flags(handle, mg_subgraph_view);

  // first, validate the traversed edges in the subgraph

  if constexpr (std::is_floating_point_v<distance_t>) {  // SSSP
    auto tmp_mg_subgraph_view = mg_subgraph_view;
    auto const& mg_subgraph_edge_weight_view =
      reachable_from_2cores ? mg_pruned_graph_edge_weight_view : mg_isolated_trees_edge_weight_view;

    rmm::device_uvector<distance_t> mg_subgraph_distances(
      tmp_mg_subgraph_view.local_vertex_partition_range_size(), handle.get_stream());
    auto pair_first =
      thrust::make_zip_iterator(reachable_from_2cores ? mg_graph_to_pruned_graph_map.begin()
                                                      : mg_graph_to_isolated_trees_map.begin(),
                                mg_distances.begin());
    thrust::for_each(handle.get_thrust_policy(),
                     pair_first,
                     pair_first + mg_distances.size(),
                     [mg_subgraph_distances = raft::device_span<distance_t>(
                        mg_subgraph_distances.data(), mg_subgraph_distances.size()),
                      invalid_vertex,
                      invalid_distance] __device__(auto pair) {
                       auto idx  = cuda::std::get<0>(pair);
                       auto dist = cuda::std::get<1>(pair);
                       if (idx != invalid_vertex) {  // in the subgraph
                         mg_subgraph_distances[idx] = dist;
                       }
                     });

    constexpr size_t num_rounds = 16;  // validate in multiple rounds to cut peak memory usage
    for (size_t i = 0; i < num_rounds; ++i) {
      cugraph::edge_property_t<edge_t, bool> edge_mask(handle, tmp_mg_subgraph_view);
      cugraph::transform_e(
        handle,
        tmp_mg_subgraph_view,
        cugraph::edge_src_dummy_property_t{}.view(),
        cugraph::edge_dst_dummy_property_t{}.view(),
        cugraph::edge_dummy_property_t{}.view(),
        cuda::proclaim_return_type<bool>(
          [i, num_rounds, hash_func = hash_vertex_pair_t<vertex_t>{}] __device__(
            auto src, auto dst, auto, auto, auto) {
            return (static_cast<size_t>(hash_func(cuda::std::make_tuple(src, dst)) % num_rounds) ==
                    i);
          }),
        edge_mask.mutable_view());
      tmp_mg_subgraph_view.attach_edge_mask(edge_mask.view());
      auto subgraph_edges =
        cugraph::decompress_to_edgelist<vertex_t, edge_t, distance_t, edge_type_t, false, true>(
          handle,
          tmp_mg_subgraph_view,
          mg_subgraph_edge_weight_view,
          std::nullopt,
          std::nullopt,
          std::nullopt);
      tmp_mg_subgraph_view.clear_edge_mask();

      auto srcs      = std::move(std::get<0>(subgraph_edges));
      auto dsts      = std::move(std::get<1>(subgraph_edges));
      auto weights   = std::move(*(std::get<2>(subgraph_edges)));
      auto src_dists = cugraph::collect_values_for_int_vertices(
        handle,
        srcs.begin(),
        srcs.end(),
        mg_subgraph_distances.begin(),
        tmp_mg_subgraph_view.vertex_partition_range_lasts(),
        tmp_mg_subgraph_view.local_vertex_partition_range_first());
      auto dst_dists = cugraph::collect_values_for_int_vertices(
        handle,
        dsts.begin(),
        dsts.end(),
        mg_subgraph_distances.begin(),
        tmp_mg_subgraph_view.vertex_partition_range_lasts(),
        tmp_mg_subgraph_view.local_vertex_partition_range_first());
      auto triplet_first =
        thrust::make_zip_iterator(src_dists.begin(), dst_dists.begin(), weights.begin());
      auto num_invalids = thrust::count_if(
        handle.get_thrust_policy(),
        triplet_first,
        triplet_first + src_dists.size(),
        cuda::proclaim_return_type<bool>([invalid_distance] __device__(auto triplet) {
          auto src_dist = cuda::std::get<0>(triplet);
          auto dst_dist = cuda::std::get<1>(triplet);
          auto w        = cuda::std::get<2>(triplet);
          if (src_dist == invalid_distance) {
            return dst_dist != invalid_distance;
          } else {
            auto diff = cuda::std::abs(src_dist - dst_dist);
            return (diff > w + 1e-6) &&
                   (diff > w * (1.0 + 1e-4));  // 1e-4 & 1e-6 to consider limited floating point
                                               // arithmetic resolution
          }
        }));
      if (num_invalids > 0) { return false; }
    }
  } else {  // BFS
    auto max_distance = thrust::transform_reduce(
      handle.get_thrust_policy(),
      mg_distances.begin(),
      mg_distances.end(),
      cuda::proclaim_return_type<distance_t>([invalid_distance] __device__(auto d) {
        return d == invalid_distance ? distance_t{0} : d;
      }),
      distance_t{0},
      thrust::maximum<distance_t>{});
#if 1  // FIXME: we should add host_allreduce to raft
    max_distance = cugraph::host_scalar_allreduce(
      comm, max_distance, raft::comms::op_t::MAX, handle.get_stream());
#else
    comm.host_allreduce(std::addressof(max_distance),
                        std::addressof(max_distance),
                        size_t{1},
                        raft::comms::op_t::MAX);
#endif
    auto pair_first =
      thrust::make_zip_iterator(reachable_from_2cores ? mg_graph_to_pruned_graph_map.begin()
                                                      : mg_graph_to_isolated_trees_map.begin(),
                                mg_distances.begin());
    for (vertex_t level = 0; level <= max_distance;
         ++level) {  // validate in multple rounds to cut peak memory usage (to store
                     // source|destination vertex properties using 1 bit per vertex)
      rmm::device_uvector<vertex_t> subgraph_level_v_offsets(
        mg_subgraph_view.local_vertex_partition_range_size(),
        handle.get_stream());  // vertices with mg_distances[] == level
      rmm::device_uvector<vertex_t> subgraph_adjacent_level_v_offsets(
        mg_subgraph_view.local_vertex_partition_range_size(),
        handle.get_stream());  // vertices with mg_distances[] = level - 1, level, or level + 1
      subgraph_level_v_offsets.resize(
        cuda::std::distance(
          subgraph_level_v_offsets.begin(),
          thrust::copy_if(handle.get_thrust_policy(),
                          reachable_from_2cores ? mg_graph_to_pruned_graph_map.begin()
                                                : mg_graph_to_isolated_trees_map.begin(),
                          reachable_from_2cores ? mg_graph_to_pruned_graph_map.end()
                                                : mg_graph_to_isolated_trees_map.end(),
                          pair_first,
                          subgraph_level_v_offsets.begin(),
                          [level, invalid_vertex, invalid_distance] __device__(auto pair) {
                            auto d = cuda::std::get<1>(pair);
                            return (cuda::std::get<0>(pair) !=
                                    invalid_vertex /* in the subgraph */) &&
                                   (d == level);
                          })),
        handle.get_stream());
      subgraph_level_v_offsets.shrink_to_fit(handle.get_stream());
      subgraph_adjacent_level_v_offsets.resize(
        cuda::std::distance(
          subgraph_adjacent_level_v_offsets.begin(),
          thrust::copy_if(handle.get_thrust_policy(),
                          reachable_from_2cores ? mg_graph_to_pruned_graph_map.begin()
                                                : mg_graph_to_isolated_trees_map.begin(),
                          reachable_from_2cores ? mg_graph_to_pruned_graph_map.end()
                                                : mg_graph_to_isolated_trees_map.end(),
                          pair_first,
                          subgraph_adjacent_level_v_offsets.begin(),
                          cuda::proclaim_return_type<bool>(
                            [level, invalid_vertex, invalid_distance] __device__(auto pair) {
                              auto d = cuda::std::get<1>(pair);
                              return (cuda::std::get<0>(pair) !=
                                      invalid_vertex /* in the subgraph */) &&
                                     (((d >= level) ? (d - level) : (level - d)) <= 1);
                            }))),
        handle.get_stream());
      subgraph_adjacent_level_v_offsets.shrink_to_fit(handle.get_stream());

      auto subgraph_level_vs = std::move(subgraph_level_v_offsets);
      thrust::transform(
        handle.get_thrust_policy(),
        subgraph_level_vs.begin(),
        subgraph_level_vs.end(),
        subgraph_level_vs.begin(),
        cuda::proclaim_return_type<vertex_t>(
          [v_first = mg_subgraph_view.local_vertex_partition_range_first()] __device__(
            auto v_offset) { return v_first + v_offset; }));
      thrust::sort(handle.get_thrust_policy(), subgraph_level_vs.begin(), subgraph_level_vs.end());
      cugraph::fill_edge_src_property(
        handle, mg_subgraph_view, edge_src_flags.mutable_view(), false);
      cugraph::fill_edge_src_property(handle,
                                      mg_subgraph_view,
                                      subgraph_level_vs.begin(),
                                      subgraph_level_vs.end(),
                                      edge_src_flags.mutable_view(),
                                      true);  // true if the distance is level
      auto subgraph_adjacent_level_vs = std::move(subgraph_adjacent_level_v_offsets);
      thrust::transform(
        handle.get_thrust_policy(),
        subgraph_adjacent_level_vs.begin(),
        subgraph_adjacent_level_vs.end(),
        subgraph_adjacent_level_vs.begin(),
        cuda::proclaim_return_type<vertex_t>(
          [v_first = mg_subgraph_view.local_vertex_partition_range_first()] __device__(
            auto v_offset) { return v_first + v_offset; }));
      thrust::sort(handle.get_thrust_policy(),
                   subgraph_adjacent_level_vs.begin(),
                   subgraph_adjacent_level_vs.end());
      cugraph::fill_edge_dst_property(
        handle, mg_subgraph_view, edge_dst_flags.mutable_view(), false);
      cugraph::fill_edge_dst_property(handle,
                                      mg_subgraph_view,
                                      subgraph_adjacent_level_vs.begin(),
                                      subgraph_adjacent_level_vs.end(),
                                      edge_dst_flags.mutable_view(),
                                      true);  // true if the abs(distance - level) <= 1
      auto num_invalids =
        cugraph::count_if_e(handle,
                            mg_subgraph_view,
                            edge_src_flags.view(),
                            edge_dst_flags.view(),
                            cugraph::edge_dummy_property_t{}.view(),
                            cuda::proclaim_return_type<bool>(
                              [level, invalid_distance] __device__(
                                auto src, auto dst, bool level_src, bool adjacent_level_dst, auto) {
                                return level_src && !adjacent_level_dst;
                              }));
      if (num_invalids > 0) {
        return false;  // only one of the two connected vertices is reachable from the starting
                       // vertex or the distances from the starting vertex differ by more than one
      }
    }
  }

  // second, validate the edges from/to unvisited vertices

  {
    auto pair_first =
      thrust::make_zip_iterator(reachable_from_2cores ? mg_graph_to_pruned_graph_map.begin()
                                                      : mg_graph_to_isolated_trees_map.begin(),
                                mg_distances.begin());

    rmm::device_uvector<vertex_t> unreachable_v_offsets(
      mg_subgraph_view.local_vertex_partition_range_size(), handle.get_stream());
    unreachable_v_offsets.resize(
      cuda::std::distance(
        unreachable_v_offsets.begin(),
        thrust::copy_if(handle.get_thrust_policy(),
                        reachable_from_2cores ? mg_graph_to_pruned_graph_map.begin()
                                              : mg_graph_to_isolated_trees_map.begin(),
                        reachable_from_2cores ? mg_graph_to_pruned_graph_map.end()
                                              : mg_graph_to_isolated_trees_map.end(),
                        pair_first,
                        unreachable_v_offsets.begin(),
                        cuda::proclaim_return_type<bool>(
                          [invalid_vertex, invalid_distance] __device__(auto pair) {
                            return (cuda::std::get<0>(pair) !=
                                    invalid_vertex /* in the subgraph */) &&
                                   (cuda::std::get<1>(pair) == invalid_distance /* unreachable */);
                          }))),
      handle.get_stream());
    auto unreachable_vs = std::move(unreachable_v_offsets);
    thrust::transform(
      handle.get_thrust_policy(),
      unreachable_vs.begin(),
      unreachable_vs.end(),
      unreachable_vs.begin(),
      cuda::proclaim_return_type<vertex_t>(
        [v_first = mg_subgraph_view.local_vertex_partition_range_first()] __device__(
          auto v_offset) { return v_first + v_offset; }));
    cugraph::fill_edge_src_property(handle, mg_subgraph_view, edge_src_flags.mutable_view(), false);
    cugraph::fill_edge_src_property(handle,
                                    mg_subgraph_view,
                                    unreachable_vs.begin(),
                                    unreachable_vs.end(),
                                    edge_src_flags.mutable_view(),
                                    true);  // true if the distance is invalid_distance
    cugraph::fill_edge_dst_property(handle, mg_subgraph_view, edge_dst_flags.mutable_view(), false);
    cugraph::fill_edge_dst_property(handle,
                                    mg_subgraph_view,
                                    unreachable_vs.begin(),
                                    unreachable_vs.end(),
                                    edge_dst_flags.mutable_view(),
                                    true);  // true if the distance is invalid_distance
    auto num_invalids = cugraph::count_if_e(
      handle,
      mg_subgraph_view,
      edge_src_flags.view(),
      edge_dst_flags.view(),
      cugraph::edge_dummy_property_t{}.view(),
      cuda::proclaim_return_type<bool>(
        [] __device__(auto src, auto dst, bool src_unreachable, bool dst_unreachable, auto) {
          return src_unreachable != dst_unreachable;
        }));
    if (num_invalids > 0) {
      return false;  // only one of the two connected vertices is reachable from the starting vertex
    }
  }

  // thrid, validate the edges in the pruned forest (if reachble_from_2cores is true)

  if (reachable_from_2cores) {
    rmm::device_uvector<vertex_t> forest_edge_parents(mg_distances.size(), handle.get_stream());
    rmm::device_uvector<vertex_t> forest_edge_vertices(forest_edge_parents.size(),
                                                       handle.get_stream());
    std::optional<rmm::device_uvector<distance_t>> forest_edge_weights{std::nullopt};
    if constexpr (std::is_floating_point_v<distance_t>) {  // SSSP
      forest_edge_weights =
        rmm::device_uvector<distance_t>(forest_edge_parents.size(), handle.get_stream());
      auto input_first = thrust::make_zip_iterator(
        parents.begin(),
        thrust::make_counting_iterator(local_vertex_partition_range_first),
        w_to_parents->begin());
      auto output_first = thrust::make_zip_iterator(
        forest_edge_parents.begin(), forest_edge_vertices.begin(), forest_edge_weights->begin());
      forest_edge_parents.resize(
        cuda::std::distance(output_first,
                            thrust::copy_if(handle.get_thrust_policy(),
                                            input_first,
                                            input_first + forest_edge_parents.size(),
                                            output_first,
                                            cuda::proclaim_return_type<bool>(
                                              [invalid_vertex] __device__(auto triplet) {
                                                auto p = cuda::std::get<0>(triplet);
                                                auto v = cuda::std::get<1>(triplet);
                                                return (p != invalid_vertex /* reachable from 2-cores */) &&
                                 (p != v /* not in a 2-core */);
                                              }))),
        handle.get_stream());
    } else {
      auto input_first = thrust::make_zip_iterator(
        parents.begin(), thrust::make_counting_iterator(local_vertex_partition_range_first));
      auto output_first =
        thrust::make_zip_iterator(forest_edge_parents.begin(), forest_edge_vertices.begin());
      forest_edge_parents.resize(
        cuda::std::distance(
          output_first,
          thrust::copy_if(handle.get_thrust_policy(),
                          input_first,
                          input_first + forest_edge_parents.size(),
                          output_first,
                          cuda::proclaim_return_type<bool>([invalid_vertex] __device__(auto pair) {
                            auto p = cuda::std::get<0>(pair);
                            auto v = cuda::std::get<1>(pair);
                            return (p != invalid_vertex /* reachable from 2-cores */) &&
                                   (p != v /* not in a 2-core */);
                          }))),
        handle.get_stream());
    }
    forest_edge_vertices.resize(forest_edge_parents.size(), handle.get_stream());
    forest_edge_parents.shrink_to_fit(handle.get_stream());
    forest_edge_vertices.shrink_to_fit(handle.get_stream());
    if (w_to_parents) {
      forest_edge_weights->resize(forest_edge_parents.size(), handle.get_stream());
      forest_edge_weights->shrink_to_fit(handle.get_stream());
    }

    auto forest_edge_src_dists = cugraph::collect_values_for_int_vertices(
      handle,
      forest_edge_parents.begin(),
      forest_edge_parents.end(),
      mg_distances.begin(),
      raft::host_span<vertex_t const>(vertex_partition_range_offsets.data() + 1,
                                      vertex_partition_range_offsets.size() - 1),
      local_vertex_partition_range_first);
    auto forest_edge_dst_dists = cugraph::collect_values_for_int_vertices(
      handle,
      forest_edge_vertices.begin(),
      forest_edge_vertices.end(),
      mg_distances.begin(),
      raft::host_span<vertex_t const>(vertex_partition_range_offsets.data() + 1,
                                      vertex_partition_range_offsets.size() - 1),
      local_vertex_partition_range_first);
    size_t num_invalids{};
    if constexpr (std::is_floating_point_v<distance_t>) {  // SSSP
      auto triplet_first = thrust::make_zip_iterator(
        forest_edge_src_dists.begin(), forest_edge_dst_dists.begin(), forest_edge_weights->begin());
      num_invalids = static_cast<size_t>(thrust::count_if(
        handle.get_thrust_policy(),
        triplet_first,
        triplet_first + forest_edge_src_dists.size(),
        cuda::proclaim_return_type<bool>([invalid_distance] __device__(auto triplet) {
          auto src_dist = cuda::std::get<0>(triplet);
          auto dst_dist = cuda::std::get<1>(triplet);
          auto w        = cuda::std::get<2>(triplet);
          if (src_dist == invalid_distance) {
            return dst_dist != invalid_distance;
          } else {
            if (dst_dist == invalid_distance) {
              return true;
            } else {
              auto diff = cuda::std::abs(src_dist - dst_dist);
              return (diff > w + 1e-6) &&
                     (diff > w * (1.0 + 1e-4));  // 1e-4 & 1e-6 to consider limited floating point
                                                 // arithmetic resolution
            }
            return (dst_dist == invalid_distance) || (cuda::std::abs(src_dist - dst_dist) > w);
          }
        })));
    } else {  // BFS
      auto pair_first =
        thrust::make_zip_iterator(forest_edge_src_dists.begin(), forest_edge_dst_dists.begin());
      num_invalids = static_cast<size_t>(thrust::count_if(
        handle.get_thrust_policy(),
        pair_first,
        pair_first + forest_edge_src_dists.size(),
        cuda::proclaim_return_type<bool>([invalid_distance] __device__(auto pair) {
          auto src_dist = cuda::std::get<0>(pair);
          auto dst_dist = cuda::std::get<1>(pair);
          if (src_dist == invalid_distance) {
            return dst_dist != invalid_distance;
          } else {
            return (dst_dist == invalid_distance) ||
                   (((src_dist >= dst_dist) ? (src_dist - dst_dist) : (dst_dist - src_dist)) > 1);
          }
        })));
    }
#if 1  // FIXME: we should add host_allreduce to raft
    num_invalids = cugraph::host_scalar_allreduce(
      comm, num_invalids, raft::comms::op_t::SUM, handle.get_stream());
#else
    comm.host_allreduce(std::addressof(num_invalids),
                        std::addressof(num_invalids),
                        size_t{1},
                        raft::comms::op_t::SUM);
#endif
    if (num_invalids > 0) {
      return false;  // the distances from the starting vertex differ by more than one
    }
  }

  return true;
}

// all the visited vertices are in the same connected component, all the unreachable vertices are in
// different connected components
template <typename vertex_t>
bool check_connected_components(raft::handle_t const& handle,
                                raft::device_span<vertex_t const> components,
                                raft::device_span<vertex_t const> mg_predecessors,
                                vertex_t starting_vertex_component,
                                vertex_t invalid_vertex)
{
  auto& comm = handle.get_comms();

  auto pair_first = thrust::make_zip_iterator(components.begin(), mg_predecessors.begin());
  auto num_invalids =
    thrust::count_if(handle.get_thrust_policy(),
                     pair_first,
                     pair_first + components.size(),
                     cuda::proclaim_return_type<bool>(
                       [starting_vertex_component, invalid_vertex] __device__(auto pair) {
                         auto c    = cuda::std::get<0>(pair);
                         auto pred = cuda::std::get<1>(pair);
                         if (c == starting_vertex_component) {
                           return pred == invalid_vertex;
                         } else {
                           return pred != invalid_vertex;
                         }
                       }));
#if 1  // FIXME: we should add host_allreduce to raft
  num_invalids =
    cugraph::host_scalar_allreduce(comm, num_invalids, raft::comms::op_t::SUM, handle.get_stream());
#else
  comm.host_allreduce(
    std::addressof(num_invalids), std::addressof(num_invalids), size_t{1}, raft::comms::op_t::SUM);
#endif
  if (num_invalids > 0) {
    return false;  // the BFS tree does not span the entire connected component of the starting
                   // vertex
  } else {
    return true;
  }
}

// check that parents(v)->v edge exists in the input graph
template <typename vertex_t, typename edge_t>
bool check_has_edge_from_parents(
  raft::handle_t const& handle,
  raft::device_span<vertex_t const> parents,
  raft::device_span<vertex_t const> mg_predecessors,
  cugraph::graph_view_t<vertex_t, edge_t, false, true> const& mg_pruned_graph_view,
  raft::device_span<vertex_t const> mg_graph_to_pruned_graph_map,
  cugraph::graph_view_t<vertex_t, edge_t, false, true> const& mg_isolated_trees_view,
  raft::device_span<vertex_t const> mg_graph_to_isolated_trees_map,
  raft::host_span<vertex_t const> vertex_partition_range_offsets,
  vertex_t starting_vertex,
  vertex_t local_vertex_partition_range_first,
  vertex_t invalid_vertex,
  bool reachable_from_2cores,
  bool in_2cores)
{
  auto& comm = handle.get_comms();

  rmm::device_uvector<vertex_t> query_preds(mg_predecessors.size(), handle.get_stream());
  rmm::device_uvector<vertex_t> query_vertices(query_preds.size(), handle.get_stream());
  auto input_edge_first = thrust::make_zip_iterator(
    mg_predecessors.begin(), thrust::make_counting_iterator(local_vertex_partition_range_first));
  auto output_edge_first = thrust::make_zip_iterator(query_preds.begin(), query_vertices.begin());
  query_preds.resize(
    cuda::std::distance(
      output_edge_first,
      thrust::copy_if(
        handle.get_thrust_policy(),
        input_edge_first,
        input_edge_first + mg_predecessors.size(),
        output_edge_first,
        cuda::proclaim_return_type<bool>([invalid_vertex, starting_vertex] __device__(auto pair) {
          auto pred = cuda::std::get<0>(pair);
          auto v    = cuda::std::get<1>(pair);
          return (pred != invalid_vertex /* reachable */) && (v != starting_vertex);
        }))),
    handle.get_stream());
  query_vertices.resize(query_preds.size(), handle.get_stream());
  if (reachable_from_2cores) {  // exclude the edges in the forest (parents[v] -> v)
    auto query_edge_first = thrust::make_zip_iterator(query_preds.begin(), query_vertices.begin());
    query_preds.resize(
      cuda::std::distance(
        query_edge_first,
        thrust::remove_if(
          handle.get_thrust_policy(),
          query_edge_first,
          query_edge_first + query_preds.size(),
          cuda::proclaim_return_type<bool>(
            [parents = raft::device_span<vertex_t const>(parents.data(), parents.size()),
             v_first = local_vertex_partition_range_first] __device__(auto pair) {
              auto pred   = cuda::std::get<0>(pair);
              auto v      = cuda::std::get<1>(pair);
              auto parent = parents[v - v_first];
              return parent == pred;  // the query edge exists in the forest
            }))),
      handle.get_stream());
    query_vertices.resize(query_preds.size(), handle.get_stream());
    if (!in_2cores) {  // found BFS predecessor tree may contain edges from v ->
                       // parents[v] (instead of parents[v] -> v)
      rmm::device_uvector<vertex_t> forest_edge_vertices(parents.size(), handle.get_stream());
      rmm::device_uvector<vertex_t> forest_edge_parents(forest_edge_vertices.size(),
                                                        handle.get_stream());
      auto input_first = thrust::make_zip_iterator(
        thrust::make_counting_iterator(local_vertex_partition_range_first), parents.begin());
      auto output_first =
        thrust::make_zip_iterator(forest_edge_vertices.begin(), forest_edge_parents.begin());
      forest_edge_vertices.resize(
        cuda::std::distance(
          output_first,
          thrust::copy_if(handle.get_thrust_policy(),
                          input_first,
                          input_first + mg_predecessors.size(),
                          output_first,
                          cuda::proclaim_return_type<bool>([invalid_vertex] __device__(auto pair) {
                            auto v      = cuda::std::get<0>(pair);
                            auto parent = cuda::std::get<1>(pair);
                            return (parent != invalid_vertex /* reachable */) &&
                                   (parent != v /* v is not in 2-cores */);
                          }))),
        handle.get_stream());
      forest_edge_parents.resize(forest_edge_vertices.size(), handle.get_stream());
      {
        std::vector<cugraph::arithmetic_device_uvector_t> vertex_properties{};
        vertex_properties.push_back(std::move(forest_edge_vertices));
        std::tie(forest_edge_parents, vertex_properties) = cugraph::shuffle_int_vertices<vertex_t>(
          handle,
          std::move(forest_edge_parents),
          std::move(vertex_properties),
          raft::host_span<vertex_t const>(vertex_partition_range_offsets.data() + 1,
                                          vertex_partition_range_offsets.size() - 1));
        forest_edge_vertices =
          std::move(std::get<rmm::device_uvector<vertex_t>>(vertex_properties[0]));
      }
      auto forest_edge_first =
        thrust::make_zip_iterator(forest_edge_vertices.begin(), forest_edge_parents.begin());
      thrust::sort(handle.get_thrust_policy(),
                   forest_edge_first,
                   forest_edge_first + forest_edge_vertices.size());
      query_edge_first = thrust::make_zip_iterator(query_preds.begin(), query_vertices.begin());
      query_preds.resize(
        cuda::std::distance(
          query_edge_first,
          thrust::remove_if(
            handle.get_thrust_policy(),
            query_edge_first,
            query_edge_first + query_preds.size(),
            cuda::proclaim_return_type<bool>([forest_edge_first,
                                              forest_edge_last =
                                                forest_edge_first +
                                                forest_edge_vertices.size()] __device__(auto pair) {
              auto pred = cuda::std::get<0>(pair);
              auto v    = cuda::std::get<1>(pair);
              auto key  = cuda::std::make_tuple(pred, v);
              auto it = thrust::lower_bound(thrust::seq, forest_edge_first, forest_edge_last, key);
              return (it != forest_edge_last) && (*it == key);
            }))),
        handle.get_stream());
      query_vertices.resize(query_preds.size(), handle.get_stream());
    }
  }

  auto mg_graph_to_subgraph_map = raft::device_span<vertex_t const>(
    reachable_from_2cores ? mg_graph_to_pruned_graph_map.data()
                          : mg_graph_to_isolated_trees_map.data(),
    reachable_from_2cores ? mg_graph_to_pruned_graph_map.size()
                          : mg_graph_to_isolated_trees_map.size());
  auto mg_subgraph_view = reachable_from_2cores ? mg_pruned_graph_view : mg_isolated_trees_view;

  thrust::transform(handle.get_thrust_policy(),
                    query_vertices.begin(),
                    query_vertices.end(),
                    query_vertices.begin(),
                    cuda::proclaim_return_type<vertex_t>(
                      [mg_graph_to_subgraph_map,
                       subgraph_v_first = mg_subgraph_view.local_vertex_partition_range_first(),
                       v_first          = local_vertex_partition_range_first,
                       invalid_vertex] __device__(auto v) {
                        auto v_offset = mg_graph_to_subgraph_map[v - v_first];
                        return (v_offset != invalid_vertex) ? (subgraph_v_first + v_offset)
                                                            : invalid_vertex;
                      }));
  {
    std::vector<cugraph::arithmetic_device_uvector_t> vertex_properties{};
    vertex_properties.push_back(std::move(query_vertices));
    std::tie(query_preds, vertex_properties) = cugraph::shuffle_int_vertices<vertex_t>(
      handle,
      std::move(query_preds),
      std::move(vertex_properties),
      raft::host_span<vertex_t const>(vertex_partition_range_offsets.data() + 1,
                                      vertex_partition_range_offsets.size() - 1));
    query_vertices = std::move(std::get<rmm::device_uvector<vertex_t>>(vertex_properties[0]));
  }
  thrust::transform(handle.get_thrust_policy(),
                    query_preds.begin(),
                    query_preds.end(),
                    query_preds.begin(),
                    [mg_graph_to_subgraph_map,
                     subgraph_v_first = mg_subgraph_view.local_vertex_partition_range_first(),
                     v_first          = local_vertex_partition_range_first,
                     invalid_vertex] __device__(auto v) {
                      auto v_offset = mg_graph_to_subgraph_map[v - v_first];
                      return (v_offset != invalid_vertex) ? (subgraph_v_first + v_offset)
                                                          : invalid_vertex;
                    });
  auto num_invalids =
    thrust::count_if(handle.get_thrust_policy(),
                     query_preds.begin(),
                     query_preds.end(),
                     [invalid_vertex] __device__(auto pred) { return pred == invalid_vertex; });
  num_invalids +=
    thrust::count_if(handle.get_thrust_policy(),
                     query_vertices.begin(),
                     query_vertices.end(),
                     [invalid_vertex] __device__(auto v) { return v == invalid_vertex; });
#if 1  // FIXME: we should add host_allreduce to raft
  num_invalids =
    cugraph::host_scalar_allreduce(comm, num_invalids, raft::comms::op_t::SUM, handle.get_stream());
#else
  comm.host_allreduce(
    std::addressof(num_invalids), std::addressof(num_invalids), size_t{1}, raft::comms::op_t::SUM);
#endif
  if (num_invalids > 0) {
    return false;  // predecessor->v missing in the input graph
  }

  std::vector<cugraph::arithmetic_device_uvector_t> edge_properties{};

  std::tie(query_preds, query_vertices, std::ignore) =
    cugraph::shuffle_int_edges(handle,
                               std::move(query_preds),
                               std::move(query_vertices),
                               std::move(edge_properties),
                               false /* store_transposed */,
                               mg_subgraph_view.vertex_partition_range_lasts());

  auto flags = mg_subgraph_view.has_edge(
    handle,
    raft::device_span<vertex_t const>(query_preds.data(), query_preds.size()),
    raft::device_span<vertex_t const>(query_vertices.data(), query_vertices.size()));
  num_invalids = thrust::count(handle.get_thrust_policy(), flags.begin(), flags.end(), false);
#if 1  // FIXME: we should add host_allreduce to raft
  num_invalids =
    cugraph::host_scalar_allreduce(comm, num_invalids, raft::comms::op_t::SUM, handle.get_stream());
#else
  comm.host_allreduce(
    std::addressof(num_invalids), std::addressof(num_invalids), size_t{1}, raft::comms::op_t::SUM);
#endif
  if (num_invalids > 0) {
    return false;  // predecessor->v missing in the input graph
  }

  return true;
}
