/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "graph500_misc.cuh"
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

#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <gtest/gtest.h>

#include <random>

// for vertices that belong to a 2-core, parents[v] = v
// for vertices that do not belong to any 2-core but reachable from 2-cores, parents[v] is updated
// to the parent of v in the tree spanning from a vertex in a 2-core. for vertices unreachable
// from any 2-core, parents[v] = invalid_vertex
// return valid w_to_weights if mg_edge_weights.has_value() is true.
// w_to_weights[] = 0.0 if parents[v] = v;
// w_to_weights[] = invalid_distance if parents[v] = invalid_vertex
// w_to_weights[] = weight to the parent, otherwise
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>, std::optional<rmm::device_uvector<weight_t>>>
find_trees_from_2cores(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& mg_graph_view,
  std::optional<cugraph::edge_property_view_t<edge_t, weight_t const*>> mg_edge_weight_view,
  vertex_t invalid_vertex,
  std::optional<weight_t> invalid_distance)
{
  assert(mg_edge_weight_view.has_value() == invalid_distance.has_value());

  auto& comm = handle.get_comms();

  rmm::device_uvector<vertex_t> parents(mg_graph_view.local_vertex_partition_range_size(),
                                        handle.get_stream());
  thrust::fill(handle.get_thrust_policy(), parents.begin(), parents.end(), invalid_vertex);
  std::optional<rmm::device_uvector<weight_t>> w_to_parents{std::nullopt};
  if (mg_edge_weight_view) {
    w_to_parents = rmm::device_uvector<weight_t>(parents.size(), handle.get_stream());
    thrust::fill(
      handle.get_thrust_policy(), w_to_parents->begin(), w_to_parents->end(), *invalid_distance);
  }

  rmm::device_uvector<bool> in_2cores(mg_graph_view.local_vertex_partition_range_size(),
                                      handle.get_stream());
  {
    rmm::device_uvector<edge_t> core_numbers(mg_graph_view.local_vertex_partition_range_size(),
                                             handle.get_stream());
    cugraph::core_number(handle,
                         mg_graph_view,
                         core_numbers.data(),
                         cugraph::k_core_degree_type_t::OUT,
                         size_t{2},
                         size_t{2});
    thrust::transform(handle.get_thrust_policy(),
                      core_numbers.begin(),
                      core_numbers.end(),
                      in_2cores.begin(),
                      cuda::proclaim_return_type<bool>(
                        [] __device__(auto core_number) { return core_number >= 2; }));
  }
  if (w_to_parents) {
    thrust::transform_if(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator(mg_graph_view.local_vertex_partition_range_first()),
      thrust::make_counting_iterator(mg_graph_view.local_vertex_partition_range_last()),
      in_2cores.begin(),
      thrust::make_zip_iterator(parents.begin(), w_to_parents->begin()),
      cuda::proclaim_return_type<thrust::tuple<vertex_t, weight_t>>(
        [] __device__(auto v) { return thrust::make_tuple(v, weight_t{0.0}); }),
      cuda::proclaim_return_type<bool>([] __device__(auto in_2core) { return in_2core; }));
  } else {
    thrust::transform_if(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator(mg_graph_view.local_vertex_partition_range_first()),
      thrust::make_counting_iterator(mg_graph_view.local_vertex_partition_range_last()),
      in_2cores.begin(),
      parents.begin(),
      cuda::proclaim_return_type<vertex_t>([] __device__(auto v) { return v; }),
      cuda::proclaim_return_type<bool>([] __device__(auto in_2core) { return in_2core; }));
  }

  assert(mg_graph_view.has_edge_mask() == false);
  cugraph::edge_src_property_t<vertex_t, bool> edge_src_reachable_from_2cores(handle,
                                                                              mg_graph_view);
  cugraph::update_edge_src_property(
    handle, mg_graph_view, in_2cores.begin(), edge_src_reachable_from_2cores.mutable_view());
  cugraph::edge_dst_property_t<vertex_t, bool> edge_dst_reachable_from_2cores(handle,
                                                                              mg_graph_view);
  cugraph::update_edge_dst_property(
    handle, mg_graph_view, in_2cores.begin(), edge_dst_reachable_from_2cores.mutable_view());
  in_2cores.resize(0, handle.get_stream());
  in_2cores.shrink_to_fit(handle.get_stream());

  cugraph::edge_property_t<edge_t, bool> edge_mask(handle, mg_graph_view);
  cugraph::transform_e(
    handle,
    mg_graph_view,
    edge_src_reachable_from_2cores.view(),
    edge_dst_reachable_from_2cores.view(),
    cugraph::edge_dummy_property_t{}.view(),
    cuda::proclaim_return_type<bool>(
      [] __device__(auto, auto, auto src_reachable, auto dst_reachable, auto) {
        return !src_reachable ||
               !dst_reachable;  // mask-out the edges in 2-cores (for faster iteration)
      }),
    edge_mask.mutable_view());
  auto tmp_graph_view = mg_graph_view;
  tmp_graph_view.attach_edge_mask(edge_mask.view());

  while (true) {
    rmm::device_uvector<vertex_t> srcs(0, handle.get_stream());
    rmm::device_uvector<vertex_t> dsts(0, handle.get_stream());
    std::optional<rmm::device_uvector<weight_t>> weights{std::nullopt};
    if (mg_edge_weight_view) {
      std::tie(srcs, dsts, weights) = cugraph::extract_transform_if_e(
        handle,
        tmp_graph_view,
        edge_src_reachable_from_2cores.view(),
        edge_dst_reachable_from_2cores.view(),
        *mg_edge_weight_view,
        cuda::proclaim_return_type<thrust::tuple<vertex_t, vertex_t, weight_t>>(
          [] __device__(auto src, auto dst, auto, auto, auto w) {
            return thrust::make_tuple(src, dst, w);
          }),
        cuda::proclaim_return_type<bool>(
          [] __device__(auto, auto, auto src_reachable, auto dst_reachable, auto) {
            return (src_reachable == false) && (dst_reachable == true);
          }));
    } else {
      std::tie(srcs, dsts) = cugraph::extract_transform_if_e(
        handle,
        tmp_graph_view,
        edge_src_reachable_from_2cores.view(),
        edge_dst_reachable_from_2cores.view(),
        cugraph::edge_dummy_property_t{}.view(),
        cuda::proclaim_return_type<thrust::tuple<vertex_t, vertex_t>>(
          [] __device__(auto src, auto dst, auto, auto, auto) {
            return thrust::make_tuple(src, dst);
          }),
        cuda::proclaim_return_type<bool>(
          [] __device__(auto, auto, auto src_reachable, auto dst_reachable, auto) {
            return (src_reachable == false) && (dst_reachable == true);
          }));
    }
    auto tot_count = cugraph::host_scalar_allreduce(
      comm, srcs.size(), raft::comms::op_t::SUM, handle.get_stream());
    if (tot_count > 0) {
      if (mg_edge_weight_view) {
        std::vector<cugraph::arithmetic_device_uvector_t> src_properties{};
        src_properties.push_back(std::move(dsts));
        src_properties.push_back(std::move(*weights));
        std::tie(srcs, src_properties) = cugraph::shuffle_local_edge_srcs<vertex_t>(
          handle,
          std::move(srcs),
          std::move(src_properties),
          tmp_graph_view.vertex_partition_range_lasts(),
          store_transposed);  // note srcs can't have duplicates as these vertices belong to a
                              // forest
        dsts    = std::move(std::get<rmm::device_uvector<vertex_t>>(src_properties[0]));
        weights = std::move(std::get<rmm::device_uvector<weight_t>>(src_properties[1]));
        auto triplet_first =
          thrust::make_zip_iterator(srcs.begin(), dsts.begin(), weights->begin());
        thrust::for_each(
          handle.get_thrust_policy(),
          triplet_first,
          triplet_first + srcs.size(),
          cuda::proclaim_return_type<void>(
            [parents      = raft::device_span<vertex_t>(parents.data(), parents.size()),
             w_to_parents = raft::device_span<weight_t>(w_to_parents->data(), w_to_parents->size()),
             v_first =
               tmp_graph_view.local_vertex_partition_range_first()] __device__(auto triplet) {
              auto v_offset          = thrust::get<0>(triplet) - v_first;
              parents[v_offset]      = thrust::get<1>(triplet);
              w_to_parents[v_offset] = thrust::get<2>(triplet);
            }));
      } else {
        std::vector<cugraph::arithmetic_device_uvector_t> src_properties{};
        src_properties.push_back(std::move(dsts));
        std::tie(srcs, src_properties) = cugraph::shuffle_local_edge_srcs<vertex_t>(
          handle,
          std::move(srcs),
          std::move(src_properties),
          tmp_graph_view.vertex_partition_range_lasts(),
          store_transposed);  // note srcs can't have duplicates as these vertices belong to a
                              // forest
        dsts            = std::move(std::get<rmm::device_uvector<vertex_t>>(src_properties[0]));
        auto pair_first = thrust::make_zip_iterator(srcs.begin(), dsts.begin());
        thrust::for_each(
          handle.get_thrust_policy(),
          pair_first,
          pair_first + srcs.size(),
          cuda::proclaim_return_type<void>(
            [parents = raft::device_span<vertex_t>(parents.data(), parents.size()),
             v_first = tmp_graph_view.local_vertex_partition_range_first()] __device__(auto pair) {
              auto v_offset     = thrust::get<0>(pair) - v_first;
              parents[v_offset] = thrust::get<1>(pair);
            }));
      }
      dsts.resize(0, handle.get_stream());
      dsts.shrink_to_fit(handle.get_stream());
      weights = std::nullopt;

      auto new_reachable_vertices = std::move(srcs);
      thrust::sort(
        handle.get_thrust_policy(), new_reachable_vertices.begin(), new_reachable_vertices.end());
      fill_edge_src_property(handle,
                             tmp_graph_view,
                             new_reachable_vertices.begin(),
                             new_reachable_vertices.end(),
                             edge_src_reachable_from_2cores.mutable_view(),
                             true);
      fill_edge_dst_property(handle,
                             tmp_graph_view,
                             new_reachable_vertices.begin(),
                             new_reachable_vertices.end(),
                             edge_dst_reachable_from_2cores.mutable_view(),
                             true);
    } else {
      break;
    }
  }

  return std::make_tuple(std::move(parents), std::move(w_to_parents));
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename edge_time_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<
  cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>,  // mg_pruned_graph
  std::optional<cugraph::edge_property_t<edge_t, weight_t>>,        // mg_pruned_graph_edge_weights
  rmm::device_uvector<vertex_t>,                                    // mg_pruned_graph_renumber_map
  rmm::device_uvector<vertex_t>,  // mg_graph_to_pruned_graph_map (mg_graph v_offset to
                                  // mg_pruned_graph v_offset)
  rmm::device_uvector<vertex_t>,  // mg_pruned_graph_to_graph_map (mg_pruned_graph
                                  // v_offset to mg_graph v_offset)
  cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>,  // mg_isolated_trees
  std::optional<cugraph::edge_property_t<edge_t, weight_t>>,  // mg_isolated_trees_edge_weights
  rmm::device_uvector<vertex_t>,                              // mg_isolated_trees_renumber_map
  rmm::device_uvector<vertex_t>,  // mg_graph_to_isolated_trees_map (mg_graph v_offset to
                                  // mg_isolated_trees v_offset)
  rmm::device_uvector<
    vertex_t>>  // mg_isolated_trees_to_graph_map (mg_isolated_trees v_offset to mg_graph
                // v_offset)
extract_forest_pruned_graph_and_isolated_trees(
  raft::handle_t const& handle,
  cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>&& mg_graph,
  std::optional<cugraph::edge_property_t<edge_t, weight_t>>&& mg_edge_weights,
  raft::device_span<vertex_t const> mg_renumber_map,
  raft::device_span<vertex_t const> parents /* found in forest pruning */,
  vertex_t invalid_vertex,
  std::optional<cugraph::large_buffer_type_t> large_buffer_type)
{
  auto mg_graph_view = mg_graph.view();
  auto mg_edge_weight_view =
    mg_edge_weights ? std::make_optional(mg_edge_weights->view()) : std::nullopt;

  // extract pruned graph edges

  std::vector<rmm::device_uvector<vertex_t>> pruned_graph_src_chunks{};
  std::vector<rmm::device_uvector<vertex_t>> pruned_graph_dst_chunks{};
  std::optional<std::vector<rmm::device_uvector<weight_t>>> pruned_graph_weight_chunks =
    mg_edge_weights ? std::make_optional(std::vector<rmm::device_uvector<weight_t>>{})
                    : std::nullopt;
  {
    size_t constexpr num_chunks{
      8};  // extract in multiple chunks to reduce peak memory usage (temporaraily store the edge
           // list in large memory buffer if bfs_usecase.use_large_buffer is set to true)
    for (size_t i = 0; i < num_chunks; ++i) {
      pruned_graph_src_chunks.emplace_back(0, handle.get_stream());
      pruned_graph_dst_chunks.emplace_back(0, handle.get_stream());
      if (pruned_graph_weight_chunks) {
        pruned_graph_weight_chunks->emplace_back(0, handle.get_stream());
      }
    }
    cugraph::edge_src_property_t<vertex_t, bool> edge_src_in_2cores(handle, mg_graph_view);
    cugraph::edge_dst_property_t<vertex_t, bool> edge_dst_in_2cores(handle, mg_graph_view);
    {
      rmm::device_uvector<bool> in_2cores(mg_graph_view.local_vertex_partition_range_size(),
                                          handle.get_stream());
      auto pair_first = thrust::make_zip_iterator(
        thrust::make_counting_iterator(mg_graph_view.local_vertex_partition_range_first()),
        parents.begin());
      thrust::transform(handle.get_thrust_policy(),
                        pair_first,
                        pair_first + mg_graph_view.local_vertex_partition_range_size(),
                        in_2cores.begin(),
                        cuda::proclaim_return_type<bool>([] __device__(auto pair) {
                          return thrust::get<0>(pair) == thrust::get<1>(pair);
                        }));
      cugraph::update_edge_src_property(
        handle, mg_graph_view, in_2cores.begin(), edge_src_in_2cores.mutable_view());
      cugraph::update_edge_dst_property(
        handle, mg_graph_view, in_2cores.begin(), edge_dst_in_2cores.mutable_view());
    }
    for (size_t i = 0; i < num_chunks; ++i) {
      cugraph::edge_property_t<edge_t, bool> edge_mask(handle, mg_graph_view);
      cugraph::transform_e(
        handle,
        mg_graph_view,
        edge_src_in_2cores.view(),
        edge_dst_in_2cores.view(),
        cugraph::edge_dummy_property_t{}.view(),
        cuda::proclaim_return_type<bool>(
          [i, num_chunks, hash_func = hash_vertex_pair_t<vertex_t>{}] __device__(
            auto src, auto dst, auto src_in_2cores, auto dst_in_2cores, auto) {
            return (src_in_2cores && dst_in_2cores) &&
                   (static_cast<size_t>(hash_func(thrust::make_tuple(src, dst)) % num_chunks) == i);
          }),
        edge_mask.mutable_view());
      mg_graph_view.attach_edge_mask(edge_mask.view());
      auto pruned_graph_edges = cugraph::decompress_to_edgelist<vertex_t,
                                                                edge_t,
                                                                weight_t,
                                                                edge_type_t,
                                                                store_transposed,
                                                                multi_gpu>(
        handle,
        mg_graph_view,
        mg_edge_weight_view,
        std::nullopt,
        std::nullopt,
        std::make_optional<raft::device_span<vertex_t const>>(mg_renumber_map.data(),
                                                              mg_renumber_map.size()),
        large_buffer_type);
      mg_graph_view.clear_edge_mask();
      pruned_graph_src_chunks[i] = std::move(std::get<0>(pruned_graph_edges));
      pruned_graph_dst_chunks[i] = std::move(std::get<1>(pruned_graph_edges));
      if (pruned_graph_weight_chunks) {
        (*pruned_graph_weight_chunks)[i] = std::move(*(std::get<2>(pruned_graph_edges)));
      }
    }
  }

  // extract isolated trees

  rmm::device_uvector<vertex_t> isolated_tree_edge_srcs(0, handle.get_stream());
  rmm::device_uvector<vertex_t> isolated_tree_edge_dsts(0, handle.get_stream());
  std::optional<rmm::device_uvector<weight_t>> isolated_tree_edge_weights =
    mg_edge_weights ? std::make_optional(rmm::device_uvector<weight_t>(0, handle.get_stream()))
                    : std::nullopt;
  {
    rmm::device_uvector<bool> reachable_from_2cores(
      mg_graph_view.local_vertex_partition_range_size(), handle.get_stream());
    thrust::transform(handle.get_thrust_policy(),
                      parents.begin(),
                      parents.end(),
                      reachable_from_2cores.begin(),
                      cuda::proclaim_return_type<bool>([invalid_vertex] __device__(auto parent) {
                        return parent != invalid_vertex;
                      }));
    cugraph::edge_src_property_t<vertex_t, bool> edge_src_reachable_from_2cores(handle,
                                                                                mg_graph_view);
    cugraph::update_edge_src_property(handle,
                                      mg_graph_view,
                                      reachable_from_2cores.begin(),
                                      edge_src_reachable_from_2cores.mutable_view());
    cugraph::edge_property_t<edge_t, bool> edge_mask(handle, mg_graph_view);
    cugraph::transform_e(
      handle,
      mg_graph_view,
      edge_src_reachable_from_2cores.view(),
      cugraph::edge_dst_dummy_property_t{}.view(),
      cugraph::edge_dummy_property_t{}.view(),
      cuda::proclaim_return_type<bool>(
        [] __device__(auto, auto, auto src_reachable, auto, auto) { return !src_reachable; }),
      edge_mask.mutable_view());
    mg_graph_view.attach_edge_mask(edge_mask.view());
    auto isolated_tree_edges = cugraph::
      decompress_to_edgelist<vertex_t, edge_t, weight_t, edge_type_t, store_transposed, multi_gpu>(
        handle,
        mg_graph_view,
        mg_edge_weight_view,
        std::nullopt,
        std::nullopt,
        std::make_optional<raft::device_span<vertex_t const>>(mg_renumber_map.data(),
                                                              mg_renumber_map.size()),
        large_buffer_type);
    isolated_tree_edge_srcs = std::move(std::get<0>(isolated_tree_edges));
    isolated_tree_edge_dsts = std::move(std::get<1>(isolated_tree_edges));
    if (isolated_tree_edge_weights) {
      isolated_tree_edge_weights = std::move(*(std::get<2>(isolated_tree_edges)));
    }
  }

  // clear mg_graph & mg_edge_weights

  mg_graph = cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>(handle);
  if (mg_edge_weights) { mg_edge_weights = std::nullopt; }

  // create the forest pruned graph

  cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu> mg_pruned_graph(handle);
  std::optional<cugraph::edge_property_t<edge_t, weight_t>> mg_pruned_graph_edge_weights{
    std::nullopt};
  rmm::device_uvector<vertex_t> mg_pruned_graph_renumber_map(0, handle.get_stream());
  {
    std::optional<rmm::device_uvector<vertex_t>> tmp_map{};
    std::tie(mg_pruned_graph,
             mg_pruned_graph_edge_weights,
             std::ignore,
             std::ignore,
             std::ignore,
             std::ignore,
             tmp_map) = cugraph::create_graph_from_edgelist<vertex_t,
                                                            edge_t,
                                                            weight_t,
                                                            edge_type_t,
                                                            edge_time_t,
                                                            store_transposed,
                                                            multi_gpu>(
      handle,
      std::nullopt,
      std::move(pruned_graph_src_chunks),
      std::move(pruned_graph_dst_chunks),
      std::move(pruned_graph_weight_chunks),
      std::nullopt,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      cugraph::graph_properties_t{true /* symmetric */, false /* multi-graph */},
      true);
    mg_pruned_graph_renumber_map = std::move(*tmp_map);
  }

  // create the isolated trees graph

  cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu> mg_isolated_trees(handle);
  std::optional<cugraph::edge_property_t<edge_t, weight_t>> mg_isolated_trees_edge_weights{
    std::nullopt};
  rmm::device_uvector<vertex_t> mg_isolated_trees_renumber_map(0, handle.get_stream());
  {
    std::optional<rmm::device_uvector<vertex_t>> tmp_map{};
    std::tie(mg_isolated_trees,
             mg_isolated_trees_edge_weights,
             std::ignore,
             std::ignore,
             std::ignore,
             std::ignore,
             tmp_map) = cugraph::create_graph_from_edgelist<vertex_t,
                                                            edge_t,
                                                            weight_t,
                                                            edge_type_t,
                                                            edge_time_t,
                                                            store_transposed,
                                                            multi_gpu>(
      handle,
      std::nullopt,
      std::move(isolated_tree_edge_srcs),
      std::move(isolated_tree_edge_dsts),
      std::move(isolated_tree_edge_weights),
      std::nullopt,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      cugraph::graph_properties_t{true /* symmetric */, false /* multi-graph */},
      true);
    mg_isolated_trees_renumber_map = std::move(*tmp_map);
  }

  // update v_offset mappings between mg_graph and mg_isolated_trees & mg_pruned_graph

  rmm::device_uvector<vertex_t> mg_graph_to_isolated_trees_map(0, handle.get_stream());
  rmm::device_uvector<vertex_t> mg_isolated_trees_to_graph_map(0, handle.get_stream());
  {
    // mg_graph_to_isolated_trees_map

    rmm::device_uvector<vertex_t> sorted_vertices(mg_isolated_trees_renumber_map.size(),
                                                  handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 mg_isolated_trees_renumber_map.begin(),
                 mg_isolated_trees_renumber_map.end(),
                 sorted_vertices.begin());
    rmm::device_uvector<vertex_t> indices(sorted_vertices.size(), handle.get_stream());
    thrust::sequence(handle.get_thrust_policy(), indices.begin(), indices.end(), vertex_t{0});
    thrust::sort_by_key(
      handle.get_thrust_policy(), sorted_vertices.begin(), sorted_vertices.end(), indices.begin());
    mg_graph_to_isolated_trees_map.resize(mg_renumber_map.size(), handle.get_stream());
    thrust::transform(handle.get_thrust_policy(),
                      mg_renumber_map.begin(),
                      mg_renumber_map.end(),
                      mg_graph_to_isolated_trees_map.begin(),
                      [sorted_vertices = raft::device_span<vertex_t const>(sorted_vertices.data(),
                                                                           sorted_vertices.size()),
                       indices = raft::device_span<vertex_t const>(indices.data(), indices.size()),
                       invalid_vertex] __device__(auto v) {
                        auto it = thrust::lower_bound(
                          thrust::seq, sorted_vertices.begin(), sorted_vertices.end(), v);
                        if ((it == sorted_vertices.end()) || (*it != v)) {
                          return invalid_vertex;
                        } else {
                          return indices[cuda::std::distance(sorted_vertices.begin(), it)];
                        }
                      });

    // mg_isolated_trees_to_graph_map

    sorted_vertices.resize(mg_renumber_map.size(), handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 mg_renumber_map.begin(),
                 mg_renumber_map.end(),
                 sorted_vertices.begin());
    indices.resize(sorted_vertices.size(), handle.get_stream());
    thrust::sequence(handle.get_thrust_policy(), indices.begin(), indices.end(), vertex_t{0});
    thrust::sort_by_key(
      handle.get_thrust_policy(), sorted_vertices.begin(), sorted_vertices.end(), indices.begin());
    mg_isolated_trees_to_graph_map.resize(mg_isolated_trees_renumber_map.size(),
                                          handle.get_stream());
    thrust::transform(
      handle.get_thrust_policy(),
      mg_isolated_trees_renumber_map.begin(),
      mg_isolated_trees_renumber_map.end(),
      mg_isolated_trees_to_graph_map.begin(),
      [sorted_vertices =
         raft::device_span<vertex_t const>(sorted_vertices.data(), sorted_vertices.size()),
       indices =
         raft::device_span<vertex_t const>(indices.data(), indices.size())] __device__(auto v) {
        auto it =
          thrust::lower_bound(thrust::seq, sorted_vertices.begin(), sorted_vertices.end(), v);
        assert((it != sorted_vertices.end()) && (*it == v));
        return indices[cuda::std::distance(sorted_vertices.begin(), it)];
      });
  }

  rmm::device_uvector<vertex_t> mg_graph_to_pruned_graph_map(0, handle.get_stream());
  rmm::device_uvector<vertex_t> mg_pruned_graph_to_graph_map(0, handle.get_stream());
  {
    // mg_graph_to_pruned_graph_map

    rmm::device_uvector<vertex_t> sorted_vertices(mg_pruned_graph_renumber_map.size(),
                                                  handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 mg_pruned_graph_renumber_map.begin(),
                 mg_pruned_graph_renumber_map.end(),
                 sorted_vertices.begin());
    rmm::device_uvector<vertex_t> indices(sorted_vertices.size(), handle.get_stream());
    thrust::sequence(handle.get_thrust_policy(), indices.begin(), indices.end(), vertex_t{0});
    thrust::sort_by_key(
      handle.get_thrust_policy(), sorted_vertices.begin(), sorted_vertices.end(), indices.begin());
    mg_graph_to_pruned_graph_map.resize(mg_renumber_map.size(), handle.get_stream());
    thrust::transform(handle.get_thrust_policy(),
                      mg_renumber_map.begin(),
                      mg_renumber_map.end(),
                      mg_graph_to_pruned_graph_map.begin(),
                      [sorted_vertices = raft::device_span<vertex_t const>(sorted_vertices.data(),
                                                                           sorted_vertices.size()),
                       indices = raft::device_span<vertex_t const>(indices.data(), indices.size()),
                       invalid_vertex] __device__(auto v) {
                        auto it = thrust::lower_bound(
                          thrust::seq, sorted_vertices.begin(), sorted_vertices.end(), v);
                        if ((it == sorted_vertices.end()) || (*it != v)) {
                          return invalid_vertex;
                        } else {
                          return indices[cuda::std::distance(sorted_vertices.begin(), it)];
                        }
                      });

    // mg_pruned_graph_to_graph_map

    sorted_vertices.resize(mg_renumber_map.size(), handle.get_stream());
    thrust::copy(handle.get_thrust_policy(),
                 mg_renumber_map.begin(),
                 mg_renumber_map.end(),
                 sorted_vertices.begin());
    indices.resize(sorted_vertices.size(), handle.get_stream());
    thrust::sequence(handle.get_thrust_policy(), indices.begin(), indices.end(), vertex_t{0});
    thrust::sort_by_key(
      handle.get_thrust_policy(), sorted_vertices.begin(), sorted_vertices.end(), indices.begin());
    mg_pruned_graph_to_graph_map.resize(mg_pruned_graph_renumber_map.size(), handle.get_stream());
    thrust::transform(
      handle.get_thrust_policy(),
      mg_pruned_graph_renumber_map.begin(),
      mg_pruned_graph_renumber_map.end(),
      mg_pruned_graph_to_graph_map.begin(),
      [sorted_vertices =
         raft::device_span<vertex_t const>(sorted_vertices.data(), sorted_vertices.size()),
       indices =
         raft::device_span<vertex_t const>(indices.data(), indices.size())] __device__(auto v) {
        auto it =
          thrust::lower_bound(thrust::seq, sorted_vertices.begin(), sorted_vertices.end(), v);
        assert((it != sorted_vertices.end()) && (*it == v));
        return indices[cuda::std::distance(sorted_vertices.begin(), it)];
      });
  }

  return std::make_tuple(std::move(mg_pruned_graph),
                         std::move(mg_pruned_graph_edge_weights),
                         std::move(mg_pruned_graph_renumber_map),
                         std::move(mg_graph_to_pruned_graph_map),
                         std::move(mg_pruned_graph_to_graph_map),
                         std::move(mg_isolated_trees),
                         std::move(mg_isolated_trees_edge_weights),
                         std::move(mg_isolated_trees_renumber_map),
                         std::move(mg_graph_to_isolated_trees_map),
                         std::move(mg_isolated_trees_to_graph_map));
}

template <typename vertex_t, typename distance_t>
std::tuple<vertex_t, int, distance_t, vertex_t, std::optional<distance_t>> traverse_to_pruned_graph(
  raft::handle_t const handle,
  raft::device_span<vertex_t const> parents /* found in forest pruning */,
  std::optional<raft::device_span<distance_t const>> w_to_parents,
  raft::device_span<vertex_t const> mg_renumber_map,
  raft::host_span<vertex_t const> vertex_partition_range_offsets,
  raft::device_span<vertex_t> mg_unrenumbered_predecessors /* [INOUT] */,
  raft::device_span<distance_t> mg_distances /* [INOUT] */,
  std::optional<raft::device_span<distance_t>> mg_w_to_predecessors /* [INOUT] */,
  vertex_t starting_vertex,
  vertex_t unrenumbered_starting_vertex,
  int starting_vertex_vertex_partition_id,
  vertex_t starting_vertex_parent,
  std::optional<distance_t> w_to_starting_vertex_parent,
  vertex_t local_vertex_partition_range_first,
  int vertex_partition_id)
{
  // note that updaing mg_w_to_predecessors can happen outside the timed part (as this is just used
  // for validation), but we are performing this in this function for simplicity (and this doesn't
  // noticeably increase the execution time).

  assert(w_to_parents.has_value() == mg_w_to_predecessors.has_value());
  assert(w_to_parents.has_value() == w_to_starting_vertex_parent.has_value());
  auto& comm                 = handle.get_comms();
  auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
  auto const major_comm_size = major_comm.get_size();
  auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
  auto const minor_comm_size = minor_comm.get_size();

  vertex_t subgraph_starting_vertex{starting_vertex};
  int subgraph_starting_vertex_vertex_partition_id{starting_vertex_vertex_partition_id};
  distance_t subgraph_starting_vertex_distance{0};
  if (starting_vertex_vertex_partition_id == vertex_partition_id) {
    distance_t zero{0};
    raft::update_device(
      mg_unrenumbered_predecessors.data() + (starting_vertex - local_vertex_partition_range_first),
      std::addressof(unrenumbered_starting_vertex),
      size_t{1},
      handle.get_stream());  // Graph 500 requires the predecessor of a starting vertex to be itself
    raft::update_device(
      mg_distances.data() + (starting_vertex - local_vertex_partition_range_first),
      std::addressof(subgraph_starting_vertex_distance /* zero */),
      size_t{1},
      handle.get_stream());
    if constexpr (std::is_floating_point_v<distance_t>) {  // SSSP
      raft::update_device(
        mg_w_to_predecessors->data() + (starting_vertex - local_vertex_partition_range_first),
        std::addressof(subgraph_starting_vertex_distance /* zero */),
        size_t{1},
        handle.get_stream());
    }
    handle.sync_stream();
  }

  // reverse the parent child relationship till we reach a 2-core

  auto unrenumbered_v = unrenumbered_starting_vertex;
  auto n              = starting_vertex_parent;
  auto w_to_v = w_to_starting_vertex_parent;  // w(n->v), assumes that the input graph is symmetric
                                              // w(v->n) == w(n->v)
  vertex_t unrenumbered_subgraph_starting_vertex_parent{};
  std::optional<distance_t> w_to_subgraph_starting_vertex_parent{};
  while (true) {
    if constexpr (std::is_floating_point_v<distance_t>) {  // SSSP
      subgraph_starting_vertex_distance += *w_to_v;
    } else {  // BFS
      ++subgraph_starting_vertex_distance;
    }
    auto n_vertex_partition_id = static_cast<int>(std::distance(
      vertex_partition_range_offsets.begin() + 1,
      std::upper_bound(
        vertex_partition_range_offsets.begin() + 1, vertex_partition_range_offsets.end(), n)));
    vertex_t unrenumbered_n{};
    vertex_t nn{};
    auto w_to_n = w_to_v;  // w(nn->n)
    if (n_vertex_partition_id == vertex_partition_id) {
      raft::update_host(std::addressof(unrenumbered_n),
                        mg_renumber_map.data() + (n - local_vertex_partition_range_first),
                        size_t{1},
                        handle.get_stream());
      raft::update_host(std::addressof(nn),
                        parents.data() + (n - local_vertex_partition_range_first),
                        size_t{1},
                        handle.get_stream());
      if constexpr (std::is_floating_point_v<distance_t>) {  // SSSP
        raft::update_host(
          std::addressof(*w_to_n),
          w_to_parents->data() + (n - local_vertex_partition_range_first),
          size_t{1},
          handle.get_stream());  // assumes that the input graph is symmetric w(nn->n) == w(n->nn)
      }
      handle.sync_stream();
    }

    if constexpr (std::is_floating_point_v<distance_t>) {  // SSSP
      thrust::tie(unrenumbered_n, nn, w_to_n) = cugraph::host_scalar_bcast(
        comm,
        thrust::make_tuple(unrenumbered_n, nn, *w_to_n),
        cugraph::partition_manager::compute_global_comm_rank_from_vertex_partition_id(
          major_comm_size, minor_comm_size, n_vertex_partition_id),
        handle.get_stream());
    } else {  // BFS
      thrust::tie(unrenumbered_n, nn) = cugraph::host_scalar_bcast(
        comm,
        thrust::make_tuple(unrenumbered_n, nn),
        cugraph::partition_manager::compute_global_comm_rank_from_vertex_partition_id(
          major_comm_size, minor_comm_size, n_vertex_partition_id),
        handle.get_stream());
    }

    if (n == nn) {  // reached a 2-core
      subgraph_starting_vertex                     = n;
      subgraph_starting_vertex_vertex_partition_id = n_vertex_partition_id;
      unrenumbered_subgraph_starting_vertex_parent = unrenumbered_v;
      w_to_subgraph_starting_vertex_parent         = w_to_v;
      break;
    }

    if (n_vertex_partition_id == vertex_partition_id) {
      raft::update_device(
        mg_unrenumbered_predecessors.data() + (n - local_vertex_partition_range_first),
        std::addressof(unrenumbered_v),
        size_t{1},
        handle.get_stream());
      raft::update_device(mg_distances.data() + (n - local_vertex_partition_range_first),
                          std::addressof(subgraph_starting_vertex_distance),
                          size_t{1},
                          handle.get_stream());
      if (w_to_v) {
        raft::update_device(mg_w_to_predecessors->data() + (n - local_vertex_partition_range_first),
                            std::addressof(*w_to_v),
                            size_t{1},
                            handle.get_stream());
      }
      handle.sync_stream();
    }

    unrenumbered_v = unrenumbered_n;
    n              = nn;
    w_to_v         = w_to_n;
  }

  return std::make_tuple(subgraph_starting_vertex,
                         subgraph_starting_vertex_vertex_partition_id,
                         subgraph_starting_vertex_distance,
                         unrenumbered_subgraph_starting_vertex_parent,
                         w_to_subgraph_starting_vertex_parent);
}

template <typename vertex_t, typename distance_t>
void update_unvisited_vertex_distances(
  raft::handle_t const& handle,
  raft::device_span<vertex_t const> parents /* found in forest pruning */,
  std::optional<raft::device_span<distance_t const>> w_to_parents,
  raft::device_span<vertex_t const> components,
  raft::host_span<vertex_t const> vertex_partition_range_offsets,
  raft::device_span<distance_t> mg_distances /* [INOUT] */,
  vertex_t starting_vertex_component,
  vertex_t local_vertex_partition_range_first,
  vertex_t local_vertex_partition_range_last,
  distance_t invalid_distance)
{
  auto& comm = handle.get_comms();

  rmm::device_uvector<vertex_t> remaining_vertices(
    local_vertex_partition_range_last - local_vertex_partition_range_first,
    handle.get_stream());  // unvisited vertices in the forest (but in the same connected
                           // component with the starting vertex)
  remaining_vertices.resize(
    cuda::std::distance(
      remaining_vertices.begin(),
      thrust::copy_if(handle.get_thrust_policy(),
                      thrust::make_counting_iterator(local_vertex_partition_range_first),
                      thrust::make_counting_iterator(local_vertex_partition_range_last),
                      thrust::make_zip_iterator(components.begin(), mg_distances.begin()),
                      remaining_vertices.begin(),
                      [starting_vertex_component, invalid_distance] __device__(auto pair) {
                        return thrust::get<0>(pair) == starting_vertex_component &&
                               thrust::get<1>(pair) == invalid_distance;
                      })),
    handle.get_stream());
  while (true) {
    auto tot_remaining_vertex_count = cugraph::host_scalar_allreduce(
      comm, remaining_vertices.size(), raft::comms::op_t::SUM, handle.get_stream());
    if (tot_remaining_vertex_count == 0) { break; }
    rmm::device_uvector<vertex_t> remaining_vertex_parents(remaining_vertices.size(),
                                                           handle.get_stream());
    auto gather_offset_first = thrust::make_transform_iterator(
      remaining_vertices.begin(),
      cugraph::detail::shift_left_t<vertex_t>{local_vertex_partition_range_first});
    thrust::gather(handle.get_thrust_policy(),
                   gather_offset_first,
                   gather_offset_first + remaining_vertices.size(),
                   parents.begin(),
                   remaining_vertex_parents.begin());
    auto remaing_vertex_parent_dists = cugraph::collect_values_for_int_vertices(
      handle,
      remaining_vertex_parents.begin(),
      remaining_vertex_parents.end(),
      mg_distances.begin(),
      raft::host_span<vertex_t const>(vertex_partition_range_offsets.data() + 1,
                                      vertex_partition_range_offsets.size() - 1),
      local_vertex_partition_range_first);
    auto pair_first =
      thrust::make_zip_iterator(remaining_vertices.begin(), remaing_vertex_parent_dists.begin());
    auto remaining_last = thrust::partition(handle.get_thrust_policy(),
                                            pair_first,
                                            pair_first + remaining_vertices.size(),
                                            [invalid_distance] __device__(auto pair) {
                                              return thrust::get<1>(pair) == invalid_distance;
                                            });
    auto new_size       = thrust::distance(pair_first, remaining_last);
    auto scatter_offset_first =
      thrust::make_transform_iterator(
        remaining_vertices.begin(),
        cugraph::detail::shift_left_t<vertex_t>{local_vertex_partition_range_first}) +
      new_size;
    if constexpr (std::is_floating_point_v<distance_t>) {  // SSSP
      auto dist_first = thrust::make_transform_iterator(
        pair_first + new_size,
        cuda::proclaim_return_type<distance_t>(
          [weights =
             raft::device_span<distance_t const>(w_to_parents->data(), w_to_parents->size()),
           local_vertex_partition_range_first] __device__(auto pair) {
            auto v_offset = thrust::get<0>(pair) - local_vertex_partition_range_first;
            auto w        = weights[v_offset];  // this assumes that the distance to the parent is
                                         // identical to the distance from the parent (this is true
                                         // as Graph 500 assumes an undirected graph)
            return thrust::get<1>(pair) + w;
          }));
      thrust::scatter(handle.get_thrust_policy(),
                      dist_first,
                      dist_first + (remaining_vertices.size() - new_size),
                      scatter_offset_first,
                      mg_distances.begin());
    } else {  // BFS
      auto dist_first =
        thrust::make_transform_iterator(remaing_vertex_parent_dists.begin() + new_size,
                                        cuda::proclaim_return_type<distance_t>(
                                          [] __device__(auto d) { return d + distance_t{1}; }));
      thrust::scatter(handle.get_thrust_policy(),
                      dist_first,
                      dist_first + (remaining_vertices.size() - new_size),
                      scatter_offset_first,
                      mg_distances.begin());
    }
    remaining_vertices.resize(new_size, handle.get_stream());
  }
}
