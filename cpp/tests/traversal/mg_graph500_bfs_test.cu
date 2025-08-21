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

#include "detail/graph500_forest_pruning_utils.cuh"
#include "detail/graph500_nbr_unrenumber_cache.cuh"
#include "detail/graph500_validation_utils.cuh"
#include "utilities/base_fixture.hpp"
#include "utilities/collect_comm.cuh"
#include "utilities/conversion_utilities.hpp"
#include "utilities/device_comm_wrapper.hpp"
#include "utilities/mg_utilities.hpp"
#include "utilities/property_generator_utilities.hpp"
#include "utilities/test_graphs.hpp"
#include "utilities/thrust_wrapper.hpp"

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_partition_endpoint_property_device_view.cuh>
#include <cugraph/edge_property.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/large_buffer_manager.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/shuffle_functions.hpp>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/high_res_timer.hpp>
#include <cugraph/utilities/misc_utils.cuh>
#include <cugraph/utilities/shuffle_comm.cuh>

#include <raft/comms/mpi_comms.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/merge.h>
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

#include <gtest/gtest.h>

#include <random>

struct Graph500_BFS_Usecase {
  bool use_pruned_graph_unrenumber_cache{
    false};  // use cache to locally unrenumber (at the expense of additional memory usage)
  bool use_large_buffer{false};
  bool validate{true};
};

void init_nccl_env_variables() {}

// for vertices that belong to a 2-core, parents[v] = v
// for vertices that do not belong to any 2-core but reachable from 2-cores, parents[v] is updated
// to the parent of v in the tree spanning from a vertex in a 2-core. for vertices unreachable
// from any 2-core, parents[v] = invalid_vertex
template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
rmm::device_uvector<vertex_t> find_trees_from_2cores(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& mg_graph_view,
  vertex_t invalid_vertex)
{
  auto& comm = handle.get_comms();

  rmm::device_uvector<vertex_t> parents(mg_graph_view.local_vertex_partition_range_size(),
                                        handle.get_stream());
  thrust::fill(handle.get_thrust_policy(), parents.begin(), parents.end(), invalid_vertex);

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
  thrust::transform_if(
    handle.get_thrust_policy(),
    thrust::make_counting_iterator(mg_graph_view.local_vertex_partition_range_first()),
    thrust::make_counting_iterator(mg_graph_view.local_vertex_partition_range_last()),
    in_2cores.begin(),
    parents.begin(),
    cuda::proclaim_return_type<vertex_t>([] __device__(auto v) { return v; }),
    cuda::proclaim_return_type<bool>([] __device__(auto in_2core) { return in_2core; }));

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
    auto vertex_pairs = cugraph::extract_transform_if_e(
      handle,
      tmp_graph_view,
      edge_src_reachable_from_2cores.view(),
      edge_dst_reachable_from_2cores.view(),
      cugraph::edge_dummy_property_t{}.view(),
      cuda::proclaim_return_type<cuda::std::tuple<vertex_t, vertex_t>>(
        [] __device__(auto src, auto dst, auto, auto, auto) {
          return cuda::std::make_tuple(src, dst);
        }),
      cuda::proclaim_return_type<bool>(
        [] __device__(auto, auto, auto src_reachable, auto dst_reachable, auto) {
          return (src_reachable == false) && (dst_reachable == true);
        }));
    auto tot_vertex_pair_count =
      cugraph::host_scalar_allreduce(comm,
                                     cugraph::size_dataframe_buffer(vertex_pairs),
                                     raft::comms::op_t::SUM,
                                     handle.get_stream());
    if (tot_vertex_pair_count > 0) {
      auto srcs = std::move(std::get<0>(vertex_pairs));
      std::vector<cugraph::arithmetic_device_uvector_t> src_properties{};
      src_properties.push_back(std::move(std::get<1>(vertex_pairs)));
      std::tie(srcs, src_properties) = cugraph::shuffle_local_edge_srcs<vertex_t>(
        handle,
        std::move(std::get<0>(vertex_pairs)),
        std::move(src_properties),
        tmp_graph_view.vertex_partition_range_lasts(),
        store_transposed);  // note std::get<0>(vertex_pairs) can't have duplicates as these
                            // vertices belong to a forest
      vertex_pairs = std::make_tuple(
        std::move(srcs), std::move(std::get<rmm::device_uvector<vertex_t>>(src_properties[0])));
      thrust::for_each(
        handle.get_thrust_policy(),
        cugraph::get_dataframe_buffer_begin(vertex_pairs),
        cugraph::get_dataframe_buffer_end(vertex_pairs),
        cuda::proclaim_return_type<void>(
          [parents = raft::device_span<vertex_t>(parents.data(), parents.size()),
           v_first = tmp_graph_view.local_vertex_partition_range_first()] __device__(auto pair) {
            parents[cuda::std::get<0>(pair) - v_first] = cuda::std::get<1>(pair);
          }));
      std::get<1>(vertex_pairs).resize(0, handle.get_stream());
      std::get<1>(vertex_pairs).shrink_to_fit(handle.get_stream());

      auto new_reachable_vertices = std::move(std::get<0>(vertex_pairs));
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

  return parents;
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          typename edge_type_t,
          typename edge_time_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>,
           rmm::device_uvector<vertex_t>,  // mg_pruned_graph_renumber_map
           rmm::device_uvector<vertex_t>,  // mg_graph_to_pruned_graph_map (mg_graph v_offset to
                                           // mg_pruned_graph v_offset)
           rmm::device_uvector<vertex_t>,  // mg_pruned_graph_to_graph_map (mg_pruned_graph
                                           // v_offset to mg_graph v_offset)
           cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>,
           rmm::device_uvector<vertex_t>,  // mg_isolated_trees_renumber_map
           rmm::device_uvector<vertex_t>,  // mg_graph_to_isolated_trees_map (mg_graph v_offset to
                                           // mg_isolated_trees v_offset)
           rmm::device_uvector<
             vertex_t>>  // mg_isolated_trees_to_graph_map (mg_isolated_trees v_offset to mg_graph
                         // v_offset)
extract_forest_pruned_graph_and_isolated_trees(
  raft::handle_t const& handle,
  cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>&& mg_graph,
  raft::device_span<vertex_t const> mg_renumber_map,
  raft::device_span<vertex_t const> parents,
  vertex_t invalid_vertex,
  std::optional<cugraph::large_buffer_type_t> large_buffer_type)
{
  auto mg_graph_view = mg_graph.view();

  // extract pruned graph edges

  std::vector<rmm::device_uvector<vertex_t>> pruned_graph_src_chunks{};
  std::vector<rmm::device_uvector<vertex_t>> pruned_graph_dst_chunks{};
  {
    size_t constexpr num_chunks{
      8};  // extract in multiple chunks to reduce peak memory usage (temporaraily store the edge
           // list in large memory buffer if bfs_usecase.use_large_buffer is set to true)
    pruned_graph_src_chunks.reserve(num_chunks);
    pruned_graph_dst_chunks.reserve(num_chunks);
    for (size_t i = 0; i < num_chunks; ++i) {
      pruned_graph_src_chunks.emplace_back(0, handle.get_stream());
      pruned_graph_dst_chunks.emplace_back(0, handle.get_stream());
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
                          return cuda::std::get<0>(pair) == cuda::std::get<1>(pair);
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
                   (static_cast<size_t>(hash_func(cuda::std::make_tuple(src, dst)) % num_chunks) ==
                    i);
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
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::make_optional<raft::device_span<vertex_t const>>(mg_renumber_map.data(),
                                                              mg_renumber_map.size()));
      mg_graph_view.clear_edge_mask();
      if (large_buffer_type) {
        CUGRAPH_EXPECTS(cugraph::large_buffer_manager::memory_buffer_initialized(),
                        "Large memory buffer is not initialized.");
        auto tmp_srcs = cugraph::large_buffer_manager::allocate_memory_buffer<vertex_t>(
          std::get<0>(pruned_graph_edges).size(), handle.get_stream());
        auto tmp_dsts = cugraph::large_buffer_manager::allocate_memory_buffer<vertex_t>(
          std::get<1>(pruned_graph_edges).size(), handle.get_stream());
        thrust::copy(handle.get_thrust_policy(),
                     std::get<0>(pruned_graph_edges).begin(),
                     std::get<0>(pruned_graph_edges).end(),
                     tmp_srcs.begin());
        thrust::copy(handle.get_thrust_policy(),
                     std::get<1>(pruned_graph_edges).begin(),
                     std::get<1>(pruned_graph_edges).end(),
                     tmp_dsts.begin());
        pruned_graph_src_chunks[i] = std::move(tmp_srcs);
        pruned_graph_dst_chunks[i] = std::move(tmp_dsts);
      } else {
        pruned_graph_src_chunks[i] = std::move(std::get<0>(pruned_graph_edges));
        pruned_graph_dst_chunks[i] = std::move(std::get<1>(pruned_graph_edges));
      }
    }
  }

  // extract isolated trees

  rmm::device_uvector<vertex_t> isolated_tree_edge_srcs(0, handle.get_stream());
  rmm::device_uvector<vertex_t> isolated_tree_edge_dsts(0, handle.get_stream());
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
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::make_optional<raft::device_span<vertex_t const>>(mg_renumber_map.data(),
                                                              mg_renumber_map.size()));
    if (large_buffer_type) {
      CUGRAPH_EXPECTS(cugraph::large_buffer_manager::memory_buffer_initialized(),
                      "Large memory buffer is not initialized.");
      auto tmp_srcs = cugraph::large_buffer_manager::allocate_memory_buffer<vertex_t>(
        std::get<0>(isolated_tree_edges).size(), handle.get_stream());
      auto tmp_dsts = cugraph::large_buffer_manager::allocate_memory_buffer<vertex_t>(
        std::get<1>(isolated_tree_edges).size(), handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   std::get<0>(isolated_tree_edges).begin(),
                   std::get<0>(isolated_tree_edges).end(),
                   tmp_srcs.begin());
      thrust::copy(handle.get_thrust_policy(),
                   std::get<1>(isolated_tree_edges).begin(),
                   std::get<1>(isolated_tree_edges).end(),
                   tmp_dsts.begin());
      isolated_tree_edge_srcs = std::move(tmp_srcs);
      isolated_tree_edge_dsts = std::move(tmp_dsts);
    } else {
      isolated_tree_edge_srcs = std::move(std::get<0>(isolated_tree_edges));
      isolated_tree_edge_dsts = std::move(std::get<1>(isolated_tree_edges));
    }
  }

  // clear mg_graph

  mg_graph = cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>(handle);

  // create the forest pruned graph

  cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu> mg_pruned_graph(handle);
  rmm::device_uvector<vertex_t> mg_pruned_graph_renumber_map(0, handle.get_stream());
  {
    std::optional<rmm::device_uvector<vertex_t>> tmp{};
    std::tie(
      mg_pruned_graph, std::ignore, std::ignore, std::ignore, std::ignore, std::ignore, tmp) =
      cugraph::create_graph_from_edgelist<vertex_t,
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
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        cugraph::graph_properties_t{true /* symmetric */, false /* multi-graph */},
        true);
    mg_pruned_graph_renumber_map = std::move(*tmp);
  }

  // create the isolated trees graph

  cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu> mg_isolated_trees(handle);
  rmm::device_uvector<vertex_t> mg_isolated_trees_renumber_map(0, handle.get_stream());
  {
    std::optional<rmm::device_uvector<vertex_t>> tmp{};
    std::tie(
      mg_isolated_trees, std::ignore, std::ignore, std::ignore, std::ignore, std::ignore, tmp) =
      cugraph::create_graph_from_edgelist<vertex_t,
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
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt,
        cugraph::graph_properties_t{true /* symmetric */, false /* multi-graph */},
        true);
    mg_graph_view.clear_edge_mask();
    mg_isolated_trees_renumber_map = std::move(*tmp);
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
                          return indices[thrust::distance(sorted_vertices.begin(), it)];
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
        return indices[thrust::distance(sorted_vertices.begin(), it)];
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
                          return indices[thrust::distance(sorted_vertices.begin(), it)];
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
        return indices[thrust::distance(sorted_vertices.begin(), it)];
      });
  }

  return std::make_tuple(std::move(mg_pruned_graph),
                         std::move(mg_pruned_graph_renumber_map),
                         std::move(mg_graph_to_pruned_graph_map),
                         std::move(mg_pruned_graph_to_graph_map),
                         std::move(mg_isolated_trees),
                         std::move(mg_isolated_trees_renumber_map),
                         std::move(mg_graph_to_isolated_trees_map),
                         std::move(mg_isolated_trees_to_graph_map));
}

>>>>>>> 989bade5147ca1d849bc39c28613f537c1eb5f1e
template <typename input_usecase_t>
class Tests_GRAPH500_MGBFS
  : public ::testing::TestWithParam<std::tuple<Graph500_BFS_Usecase, input_usecase_t>> {
 public:
  Tests_GRAPH500_MGBFS() {}

  static void SetUpTestCase()
  {
    init_nccl_env_variables();

    size_t pool_size =
      12;  // note that CUDA_DEVICE_MAX_CONNECTIONS (default: 8) should be set to a value larger
           // than pool_size to avoid false dependency among different streams
    handle_ = cugraph::test::initialize_mg_handle(pool_size);

    cugraph::large_buffer_manager::init(
      *handle_, cugraph::large_buffer_manager::create_memory_buffer_resource(), std::nullopt);
  }

  static void TearDownTestCase() { handle_.reset(); }

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t>
  void run_current_test(Graph500_BFS_Usecase const& bfs_usecase,
                        input_usecase_t const& input_usecase)
  {
    using weight_t    = float;    // dummy
    using edge_type_t = int32_t;  // dummy
    using edge_time_t = int32_t;  // dummy

    bool constexpr store_transposed = false;
    bool constexpr multi_gpu        = true;
    bool constexpr renumber         = true;
    bool constexpr test_weighted    = false;
    bool constexpr shuffle = false;  // Graph 500 requirement (edges can't be pre-shuffled, edges
                                     // should be shuffled in Kernel 1)
    size_t num_warmup_starting_vertices = 1;   // to enforce all CUDA & NCCL initializations
    size_t num_timed_starting_vertices  = 64;  // Graph 500 requirement (64)

    HighResTimer hr_timer{};
    raft::random::RngState rng_state{0};

    auto& comm           = handle_->get_comms();
    auto const comm_rank = comm.get_rank();
    auto const comm_size = comm.get_size();
    auto& major_comm     = handle_->get_subcomm(cugraph::partition_manager::major_comm_name());
    auto const major_comm_rank = major_comm.get_rank();
    auto const major_comm_size = major_comm.get_size();
    auto& minor_comm = handle_->get_subcomm(cugraph::partition_manager::minor_comm_name());
    auto const minor_comm_rank = minor_comm.get_rank();
    auto const minor_comm_size = minor_comm.get_size();
    auto vertex_partition_id =
      cugraph::partition_manager::compute_vertex_partition_id_from_graph_subcomm_ranks(
        major_comm_size, minor_comm_size, major_comm_rank, minor_comm_rank);

    std::cout << "comm_size=" << comm_size << " major_comm_size=" << major_comm_size
              << " minor_comm_size=" << minor_comm_size << std::endl;

    constexpr auto invalid_distance = std::numeric_limits<vertex_t>::max();
    constexpr auto invalid_vertex   = cugraph::invalid_vertex_id<vertex_t>::value;

    // 1. force NCCL P2P initialization

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      comm.barrier();
      hr_timer.start("NCCL P2P buffer initialization");
    }

    cugraph::test::enforce_p2p_initialization(comm, handle_->get_stream());
    cugraph::test::enforce_p2p_initialization(major_comm, handle_->get_stream());
    cugraph::test::enforce_p2p_initialization(minor_comm, handle_->get_stream());

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      comm.barrier();
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    // 2. create a graph & meta data

    rmm::device_uvector<vertex_t> mg_renumber_map(0, handle_->get_stream());
    rmm::device_uvector<vertex_t> components(0, handle_->get_stream());
    rmm::device_uvector<vertex_t> parents(0, handle_->get_stream());
    rmm::device_uvector<vertex_t> unrenumbered_parents(0, handle_->get_stream());
    std::vector<vertex_t> vertex_partition_range_offsets(comm_size + 1);
    vertex_t local_vertex_partition_range_first{};
    vertex_t local_vertex_partition_range_last{};

    cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu> mg_pruned_graph(*handle_);
    rmm::device_uvector<vertex_t> mg_pruned_graph_renumber_map(0, handle_->get_stream());
    rmm::device_uvector<vertex_t> mg_graph_to_pruned_graph_map(
      0, handle_->get_stream());  // we may store this in host buffer to save HBM
    rmm::device_uvector<vertex_t> mg_pruned_graph_to_graph_map(0, handle_->get_stream());

    cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu> mg_isolated_trees(*handle_);
    rmm::device_uvector<vertex_t> mg_isolated_trees_renumber_map(0, handle_->get_stream());
    rmm::device_uvector<vertex_t> mg_graph_to_isolated_trees_map(
      0, handle_->get_stream());  // we may store this in host buffer to save HBM
    rmm::device_uvector<vertex_t> mg_isolated_trees_to_graph_map(0, handle_->get_stream());

    std::optional<cugraph::test::nbr_unrenumber_cache_t<vertex_t>>
      mg_pruned_graph_pred_unrenumber_cache{std::nullopt};
    {
      edge_t num_input_edges{};
      edge_t num_edges{};  // after removing self-loops and multi-edges

      // 2-1. create an edge list

      if (cugraph::test::g_perf) {
        RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
        comm.barrier();
        hr_timer.start("MG Construct edge list");
      }

      std::vector<rmm::device_uvector<vertex_t>> src_chunks{};
      std::vector<rmm::device_uvector<vertex_t>> dst_chunks{};
      std::tie(src_chunks, dst_chunks, std::ignore, std::ignore, std::ignore) =
        input_usecase.template construct_edgelist<vertex_t, weight_t>(
          *handle_, test_weighted, store_transposed, multi_gpu, shuffle);
      ASSERT_TRUE(input_usecase.undirected());

      if (cugraph::test::g_perf) {
        RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
        comm.barrier();
        hr_timer.stop();
        hr_timer.display_and_clear(std::cout);
      }

      num_input_edges = 0;
      for (size_t i = 0; i < src_chunks.size(); ++i) {
        num_input_edges += static_cast<edge_t>(src_chunks[i].size());
      }
      num_input_edges = cugraph::host_scalar_allreduce(
        comm, num_input_edges, raft::comms::op_t::SUM, handle_->get_stream());

      // 2-2. create an MG graph

      if (cugraph::test::g_perf) {
        RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
        comm.barrier();
        hr_timer.start("MG Construct graph (Kernel 1)");
      }

      for (size_t i = 0; i < src_chunks.size(); ++i) {
        std::tie(src_chunks[i],
                 dst_chunks[i],
                 std::ignore,
                 std::ignore,
                 std::ignore,
                 std::ignore,
                 std::ignore) =
          cugraph::remove_self_loops<vertex_t, edge_t, weight_t, edge_type_t, edge_time_t>(
            *handle_,
            std::move(src_chunks[i]),
            std::move(dst_chunks[i]),
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt);

        std::vector<cugraph::arithmetic_device_uvector_t> dummy_edge_property_chunk{};

        std::tie(src_chunks[i], dst_chunks[i], dummy_edge_property_chunk, std::ignore) =
          cugraph::shuffle_ext_edges(*handle_,
                                     std::move(src_chunks[i]),
                                     std::move(dst_chunks[i]),
                                     std::move(dummy_edge_property_chunk),
                                     store_transposed);
      }

      std::tie(
        src_chunks, dst_chunks, std::ignore, std::ignore, std::ignore, std::ignore, std::ignore) =
        cugraph::remove_multi_edges<vertex_t, edge_t, weight_t, edge_type_t, edge_time_t>(
          *handle_,
          std::move(src_chunks),
          std::move(dst_chunks),
          std::nullopt,
          std::nullopt,
          std::nullopt,
          std::nullopt,
          std::nullopt,
          true /* keep_min_value_edge */);

      num_edges = 0;
      for (size_t i = 0; i < src_chunks.size(); ++i) {
        num_edges += static_cast<edge_t>(src_chunks[i].size());
      }
      num_edges = cugraph::host_scalar_allreduce(
        comm, num_edges, raft::comms::op_t::SUM, handle_->get_stream());

      cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu> mg_graph(*handle_);
      std::optional<rmm::device_uvector<vertex_t>> tmp_map{};
      std::tie(mg_graph, std::ignore, std::ignore, std::ignore, std::ignore, std::ignore, tmp_map) =
        cugraph::create_graph_from_edgelist<vertex_t,
                                            edge_t,
                                            weight_t,
                                            edge_type_t,
                                            edge_time_t,
                                            store_transposed,
                                            multi_gpu>(
          *handle_,
          std::nullopt,
          std::move(src_chunks),
          std::move(dst_chunks),
          std::nullopt,
          std::nullopt,
          std::nullopt,
          std::nullopt,
          std::nullopt,
          cugraph::graph_properties_t{true /* symmetric */, false /* multi-graph */},
          renumber);
      mg_renumber_map = std::move(*tmp_map);
      {
        auto mg_graph_view = mg_graph.view();
        auto offsets       = mg_graph_view.vertex_partition_range_offsets();
        std::copy(offsets.begin(), offsets.end(), vertex_partition_range_offsets.begin());
        local_vertex_partition_range_first = mg_graph_view.local_vertex_partition_range_first();
        local_vertex_partition_range_last  = mg_graph_view.local_vertex_partition_range_last();
      }

      // 2-3. Forest pruning

      {
        auto mg_graph_view = mg_graph.view();

        components.resize(mg_graph_view.local_vertex_partition_range_size(), handle_->get_stream());
        cugraph::weakly_connected_components(
          *handle_, mg_graph_view, components.data(), components.size());
        std::tie(parents, std::ignore) =
          find_trees_from_2cores<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
            *handle_, mg_graph_view, std::nullopt, invalid_vertex, std::nullopt);
      }

      std::optional<rmm::device_uvector<vertex_t>> tmp_components{std::nullopt};
      if (bfs_usecase
            .use_large_buffer) {  // temporarily store components in host buffer to free up HBM
                                  // before extracting sub-graphs (which uses a lot of HBM)
        tmp_components = cugraph::large_buffer_manager::allocate_memory_buffer<vertex_t>(
          components.size(), handle_->get_stream());
        thrust::copy(handle_->get_thrust_policy(),
                     components.begin(),
                     components.end(),
                     tmp_components->begin());
        components.resize(0, handle_->get_stream());
        components.shrink_to_fit(handle_->get_stream());
      }

      std::tie(mg_pruned_graph,
               std::ignore,
               mg_pruned_graph_renumber_map,
               mg_graph_to_pruned_graph_map,
               mg_pruned_graph_to_graph_map,
               mg_isolated_trees,
               std::ignore,
               mg_isolated_trees_renumber_map,
               mg_graph_to_isolated_trees_map,
               mg_isolated_trees_to_graph_map) =
        extract_forest_pruned_graph_and_isolated_trees<vertex_t,
                                                       edge_t,
                                                       weight_t,
                                                       edge_type_t,
                                                       edge_time_t,
                                                       store_transposed,
                                                       multi_gpu>(
          *handle_,
          std::move(mg_graph),
          std::nullopt,
          raft::device_span<vertex_t const>(mg_renumber_map.data(), mg_renumber_map.size()),
          raft::device_span<vertex_t const>(parents.data(), parents.size()),
          invalid_vertex,
          bfs_usecase.use_large_buffer ? std::make_optional(cugraph::large_buffer_type_t::MEMORY)
                                       : std::nullopt);

      if (bfs_usecase.use_large_buffer) {
        components.resize(tmp_components->size(), handle_->get_stream());
        thrust::copy(handle_->get_thrust_policy(),
                     tmp_components->begin(),
                     tmp_components->end(),
                     components.begin());
        tmp_components = std::nullopt;
      }

      unrenumbered_parents.resize(parents.size(), handle_->get_stream());
      thrust::copy(
        handle_->get_thrust_policy(), parents.begin(), parents.end(), unrenumbered_parents.begin());
      cugraph::unrenumber_int_vertices<vertex_t, multi_gpu>(
        *handle_,
        unrenumbered_parents.data(),
        unrenumbered_parents.size(),
        mg_renumber_map.data(),
        raft::host_span<vertex_t const>(vertex_partition_range_offsets.data() + 1,
                                        vertex_partition_range_offsets.size() - 1));

      if (bfs_usecase.use_pruned_graph_unrenumber_cache) {
        mg_pruned_graph_pred_unrenumber_cache = cugraph::test::build_nbr_unrenumber_cache(
          *handle_,
          mg_pruned_graph.view(),
          raft::device_span<vertex_t const>(mg_pruned_graph_renumber_map.data(),
                                            mg_pruned_graph_renumber_map.size()),
          invalid_vertex,
          bfs_usecase.use_large_buffer ? std::make_optional(cugraph::large_buffer_type_t::MEMORY)
                                       : std::nullopt);
      }

      if (cugraph::test::g_perf) {
        RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
        comm.barrier();
        hr_timer.stop();
        hr_timer.display_and_clear(std::cout);
      }

      std::cout << "num_input_edges=" << num_input_edges
                << " V=" << vertex_partition_range_offsets.back() << " E=" << num_edges
                << " undirected E=" << (num_edges / 2) << std::endl;
    }
    auto mg_pruned_graph_view   = mg_pruned_graph.view();
    auto mg_isolated_trees_view = mg_isolated_trees.view();
    std::cout << "mg_pruned_graph V=" << mg_pruned_graph_view.number_of_vertices()
              << " E=" << mg_pruned_graph_view.compute_number_of_edges(*handle_)
              << " mg_isolated_trees_view V=" << mg_isolated_trees_view.number_of_vertices()
              << " E=" << mg_isolated_trees_view.compute_number_of_edges(*handle_) << std::endl;

    // 3. randomly select starting vertices

    std::vector<vertex_t> starting_vertices{};
    {
      ASSERT_TRUE(vertex_partition_range_offsets.back() > 0)
        << "Invalid input graph, the input graph should have at least one vertex";
      rmm::device_uvector<vertex_t> d_starting_vertices(
        num_warmup_starting_vertices + num_timed_starting_vertices, handle_->get_stream());
      if (comm_rank == 0) {
        cugraph::detail::uniform_random_fill(handle_->get_stream(),
                                             d_starting_vertices.data(),
                                             d_starting_vertices.size(),
                                             vertex_partition_range_offsets[0],
                                             vertex_partition_range_offsets.back(),
                                             rng_state);
        raft::print_device_vector(
          "d_starting_vertices", d_starting_vertices.data(), d_starting_vertices.size(), std::cout);
      }
      cugraph::device_bcast(comm,
                            d_starting_vertices.data(),
                            d_starting_vertices.data(),
                            d_starting_vertices.size(),
                            int{0},
                            handle_->get_stream());
      starting_vertices = cugraph::test::to_host(*handle_, d_starting_vertices);
    }

    // 4. run MG BFS

    rmm::device_uvector<vertex_t> d_mg_distances(
      mg_renumber_map.size(),
      handle_->get_stream());  // Graph500 BFS doesn't require computing distances (so we can update
                               // this outside the timed region)
    rmm::device_uvector<vertex_t> d_mg_unrenumbered_predecessors(mg_renumber_map.size(),
                                                                 handle_->get_stream());

    double total_elapsed{0.0};
    double tteps_sum{0.0};
    double one_over_tteps_sum{0.0};  // to compute harmonic mean
    for (size_t i = 0; i < (num_warmup_starting_vertices + num_timed_starting_vertices); ++i) {
      double elapsed{0.0};

      thrust::fill(handle_->get_thrust_policy(),
                   d_mg_distances.begin(),
                   d_mg_distances.end(),
                   invalid_distance);

      if (cugraph::test::g_perf) {
        RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
        comm.barrier();
        hr_timer.start("MG BFS (Kernel 2)");
      }

      auto starting_vertex = starting_vertices[i];
      auto starting_vertex_vertex_partition_id =
        static_cast<int>(std::distance(vertex_partition_range_offsets.begin() + 1,
                                       std::upper_bound(vertex_partition_range_offsets.begin() + 1,
                                                        vertex_partition_range_offsets.end(),
                                                        starting_vertex)));

      vertex_t unrenumbered_starting_vertex{};
      vertex_t starting_vertex_parent{starting_vertex};
      vertex_t starting_vertex_component{};
      if (starting_vertex_vertex_partition_id == vertex_partition_id) {
        unrenumbered_starting_vertex = mg_renumber_map.element(
          starting_vertex - local_vertex_partition_range_first, handle_->get_stream());
        starting_vertex_parent = parents.element(
          starting_vertex - local_vertex_partition_range_first, handle_->get_stream());
        starting_vertex_component = components.element(
          starting_vertex - local_vertex_partition_range_first, handle_->get_stream());
      }
      thrust::tie(unrenumbered_starting_vertex, starting_vertex_parent, starting_vertex_component) =
        cugraph::host_scalar_bcast(
          comm,
          cuda::std::make_tuple(
            unrenumbered_starting_vertex, starting_vertex_parent, starting_vertex_component),
          cugraph::partition_manager::compute_global_comm_rank_from_vertex_partition_id(
            major_comm_size, minor_comm_size, starting_vertex_vertex_partition_id),
          handle_->get_stream());
      bool reachable_from_2cores{starting_vertex_parent != invalid_vertex};
      bool in_2cores{starting_vertex == starting_vertex_parent};

      if (reachable_from_2cores) {
        thrust::transform(handle_->get_thrust_policy(),
                          unrenumbered_parents.begin(),
                          unrenumbered_parents.end(),
                          components.begin(),
                          d_mg_unrenumbered_predecessors.begin(),
                          cuda::proclaim_return_type<vertex_t>(
                            [starting_vertex_component, invalid_vertex] __device__(auto p, auto c) {
                              return (c == starting_vertex_component)
                                       ? p /* for the vertices in 2-cores (or the vertices in the
                                              path from the starting vertex to the first reachable
                                              2-core vertex), this will be over-written */
                                       : invalid_vertex;
                            }));
      } else {
        thrust::fill(handle_->get_thrust_policy(),
                     d_mg_unrenumbered_predecessors.begin(),
                     d_mg_unrenumbered_predecessors.end(),
                     invalid_vertex);
      }

      vertex_t subgraph_starting_vertex{starting_vertex};
      int subgraph_starting_vertex_vertex_partition_id{starting_vertex_vertex_partition_id};
      vertex_t subgraph_starting_vertex_distance{0};
      vertex_t unrenumbered_subgraph_starting_vertex_parent{};
      if (reachable_from_2cores && !in_2cores) {  // find the path from starting_vertex to a 2-core
        std::tie(subgraph_starting_vertex,
                 subgraph_starting_vertex_vertex_partition_id,
                 subgraph_starting_vertex_distance,
                 unrenumbered_subgraph_starting_vertex_parent,
                 std::ignore) =
          traverse_to_pruned_graph<vertex_t, vertex_t>(
            *handle_,
            raft::device_span<vertex_t const>(parents.data(), parents.size()),
            std::nullopt,
            raft::device_span<vertex_t const>(mg_renumber_map.data(), mg_renumber_map.size()),
            raft::host_span<vertex_t const>(vertex_partition_range_offsets.data(),
                                            vertex_partition_range_offsets.size()),
            raft::device_span<vertex_t>(d_mg_unrenumbered_predecessors.data(),
                                        d_mg_unrenumbered_predecessors.size()),
            raft::device_span<vertex_t>(d_mg_distances.data(), d_mg_distances.size()),
            std::nullopt,
            starting_vertex,
            unrenumbered_starting_vertex,
            starting_vertex_vertex_partition_id,
            starting_vertex_parent,
            std::nullopt,
            local_vertex_partition_range_first,
            vertex_partition_id);
      }

      std::optional<rmm::device_scalar<vertex_t>> d_bfs_starting_vertex{std::nullopt};
      if (subgraph_starting_vertex_vertex_partition_id == vertex_partition_id) {
        auto bfs_starting_vertex =
          reachable_from_2cores ? mg_pruned_graph_view.local_vertex_partition_range_first() +
                                    mg_graph_to_pruned_graph_map.element(
                                      subgraph_starting_vertex - local_vertex_partition_range_first,
                                      handle_->get_stream())
                                : mg_isolated_trees_view.local_vertex_partition_range_first() +
                                    mg_graph_to_isolated_trees_map.element(
                                      subgraph_starting_vertex - local_vertex_partition_range_first,
                                      handle_->get_stream());
        d_bfs_starting_vertex =
          rmm::device_scalar<vertex_t>(bfs_starting_vertex, handle_->get_stream());
      }

      rmm::device_uvector<vertex_t> d_mg_bfs_predecessors(
        reachable_from_2cores ? mg_pruned_graph_view.local_vertex_partition_range_size()
                              : mg_isolated_trees_view.local_vertex_partition_range_size(),
        handle_->get_stream());
      rmm::device_uvector<vertex_t> d_mg_bfs_distances(d_mg_bfs_predecessors.size(),
                                                       handle_->get_stream());

      cugraph::bfs(*handle_,
                   reachable_from_2cores ? mg_pruned_graph_view : mg_isolated_trees_view,
                   d_mg_bfs_distances.data(),
                   d_mg_bfs_predecessors.data(),
                   d_bfs_starting_vertex ? d_bfs_starting_vertex->data()
                                         : static_cast<vertex_t const*>(nullptr),
                   d_bfs_starting_vertex ? size_t{1} : size_t{0},
                   true /* direction_optimizing */,
                   std::numeric_limits<vertex_t>::max() /* depth limit */);

      if (reachable_from_2cores && mg_pruned_graph_pred_unrenumber_cache) {
        mg_pruned_graph_pred_unrenumber_cache->unrenumber(
          *handle_,
          raft::device_span<vertex_t>(d_mg_bfs_predecessors.data(), d_mg_bfs_predecessors.size()));
      } else {
        cugraph::unrenumber_int_vertices<vertex_t, multi_gpu>(
          *handle_,
          d_mg_bfs_predecessors.data(),
          d_mg_bfs_predecessors.size(),
          reachable_from_2cores ? mg_pruned_graph_renumber_map.data()
                                : mg_isolated_trees_renumber_map.data(),
          reachable_from_2cores ? mg_pruned_graph_view.vertex_partition_range_lasts()
                                : mg_isolated_trees_view.vertex_partition_range_lasts());
      }

      thrust::scatter(handle_->get_thrust_policy(),
                      d_mg_bfs_predecessors.begin(),
                      d_mg_bfs_predecessors.end(),
                      reachable_from_2cores ? mg_pruned_graph_to_graph_map.begin()
                                            : mg_isolated_trees_to_graph_map.begin(),
                      d_mg_unrenumbered_predecessors.begin());

      {  // update the starting vertex's parent
        if (subgraph_starting_vertex_vertex_partition_id ==
            vertex_partition_id) {  // cugraph::bfs sets the predecessor of the starting vertex to
                                    // invalid_vertex
          if (subgraph_starting_vertex_distance > vertex_t{0}) {
            d_mg_unrenumbered_predecessors.set_element_async(
              subgraph_starting_vertex - local_vertex_partition_range_first,
              unrenumbered_subgraph_starting_vertex_parent,
              handle_->get_stream());
          } else {
            assert(starting_vertex == subgraph_starting_vertex);
            d_mg_unrenumbered_predecessors.set_element_async(
              starting_vertex - local_vertex_partition_range_first,
              unrenumbered_starting_vertex,
              handle_->get_stream());
          }
          handle_->sync_stream();
        }
      }

      if (cugraph::test::g_perf) {
        RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
        comm.barrier();
        elapsed = hr_timer.stop();
        if (i >= num_warmup_starting_vertices) { total_elapsed += elapsed; }
        hr_timer.display_and_clear(std::cout);
      }

      // update d_mg_distances (for validation, Graph 500 doesn't require computing distances)

      if (reachable_from_2cores) {
        if (subgraph_starting_vertex_distance > vertex_t{0}) {
          thrust::transform(handle_->get_thrust_policy(),
                            d_mg_bfs_distances.begin(),
                            d_mg_bfs_distances.end(),
                            d_mg_bfs_distances.begin(),
                            cuda::proclaim_return_type<vertex_t>(
                              [delta = subgraph_starting_vertex_distance] __device__(auto d) {
                                if (d != invalid_distance) {
                                  return d + delta;
                                } else {
                                  return invalid_distance;
                                }
                              }));
        }
        thrust::scatter(handle_->get_thrust_policy(),
                        d_mg_bfs_distances.begin(),
                        d_mg_bfs_distances.end(),
                        mg_pruned_graph_to_graph_map.begin(),
                        d_mg_distances.begin());
        update_unvisited_vertex_distances<vertex_t, vertex_t>(
          *handle_,
          raft::device_span<vertex_t const>(parents.data(), parents.size()),
          std::nullopt,
          raft::device_span<vertex_t const>(components.data(), components.size()),
          raft::host_span<vertex_t const>(vertex_partition_range_offsets.data(),
                                          vertex_partition_range_offsets.size()),
          raft::device_span<vertex_t>(d_mg_distances.data(), d_mg_distances.size()),
          starting_vertex_component,
          local_vertex_partition_range_first,
          local_vertex_partition_range_last,
          invalid_distance);
      } else {
        assert(subgraph_starting_vertex_distance == vertex_t{0});
        thrust::scatter(handle_->get_thrust_policy(),
                        d_mg_bfs_distances.begin(),
                        d_mg_bfs_distances.end(),
                        mg_isolated_trees_to_graph_map.begin(),
                        d_mg_distances.begin());
      }

      /* compute the number of visisted edges */

      {
        edge_t visited_edge_count = compute_number_of_visited_undirected_edges(
          *handle_,
          raft::device_span<vertex_t const>(d_mg_distances.data(), d_mg_distances.size()),
          mg_pruned_graph_view,
          reachable_from_2cores ? std::make_optional(raft::device_span<vertex_t const>(
                                    d_mg_bfs_distances.data(), d_mg_bfs_distances.size()))
                                : std::nullopt,
          raft::device_span<vertex_t const>(mg_graph_to_pruned_graph_map.data(),
                                            mg_graph_to_pruned_graph_map.size()),
          invalid_vertex,
          invalid_distance);
        auto tteps = (static_cast<double>(visited_edge_count) / 1e12) / elapsed;
        if (i >= num_warmup_starting_vertices) {
          tteps_sum += tteps;
          one_over_tteps_sum +=
            (tteps > 0.0) ? 1.0 / tteps : std::numeric_limits<double>::infinity();
        }
        std::cout << "# visited undirected edges=" << visited_edge_count
                  << " TTEPS=" << (static_cast<double>(visited_edge_count) / 1e12) / elapsed
                  << std::endl;
      }

      if (bfs_usecase.validate) {
        /* renumber for validation */

        rmm::device_uvector<vertex_t> d_mg_predecessors(d_mg_unrenumbered_predecessors.size(),
                                                        handle_->get_stream());
        thrust::copy(handle_->get_thrust_policy(),
                     d_mg_unrenumbered_predecessors.begin(),
                     d_mg_unrenumbered_predecessors.end(),
                     d_mg_predecessors.begin());
        cugraph::renumber_ext_vertices<vertex_t, multi_gpu>(*handle_,
                                                            d_mg_predecessors.data(),
                                                            d_mg_predecessors.size(),
                                                            mg_renumber_map.data(),
                                                            local_vertex_partition_range_first,
                                                            local_vertex_partition_range_last);

        /* check starting vertex's predecessor */

        if (cugraph::test::g_perf) {
          RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
          comm.barrier();
          hr_timer.start("validate (starting vertex's predecessor)");
        }

        {
          size_t num_invalids{0};
          if (starting_vertex_vertex_partition_id == vertex_partition_id) {
            auto starting_vertex_predecessor = d_mg_predecessors.element(
              starting_vertex - local_vertex_partition_range_first, handle_->get_stream());
            if (starting_vertex_predecessor != starting_vertex) { ++num_invalids; }
          }
          num_invalids = cugraph::host_scalar_allreduce(
            comm, num_invalids, raft::comms::op_t::SUM, handle_->get_stream());
          ASSERT_EQ(num_invalids, 0) << "predecessor of a starting vertex should be itself";
        }

        if (cugraph::test::g_perf) {
          RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
          comm.barrier();
          hr_timer.stop();
          hr_timer.display_and_clear(std::cout);
        }

        /* check for cycles (update predecessor to predecessor's predecessor till reaching the
         * starting vertex, if there exists a cycle, this won't finish) */

        if (cugraph::test::g_perf) {
          RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
          comm.barrier();
          hr_timer.start("validate (cycle)");
        }

        {
          bool test_passed = is_valid_predecessor_tree(
            *handle_,
            raft::device_span<vertex_t const>(d_mg_predecessors.data(), d_mg_predecessors.size()),
            raft::host_span<vertex_t const>(vertex_partition_range_offsets.data(),
                                            vertex_partition_range_offsets.size()),
            starting_vertex,
            local_vertex_partition_range_first,
            invalid_vertex);
          ASSERT_TRUE(test_passed) << "BFS predecessor tree is invalid (failed to backtrace to the "
                                      "starting vertex) or has a cycle.";
        }

        if (cugraph::test::g_perf) {
          RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
          comm.barrier();
          hr_timer.stop();
          hr_timer.display_and_clear(std::cout);
        }

        /* check that distance(v) = distance(predecssor(v)) + 1 */

        if (cugraph::test::g_perf) {
          RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
          comm.barrier();
          hr_timer.start("validate (predecessor tree distances)");
        }

        {
          bool test_passed = check_distance_from_parents<vertex_t, vertex_t>(
            *handle_,
            raft::device_span<vertex_t const>(d_mg_predecessors.data(), d_mg_predecessors.size()),
            raft::device_span<vertex_t const>(d_mg_distances.data(), d_mg_distances.size()),
            std::nullopt,
            raft::host_span<vertex_t const>(vertex_partition_range_offsets.data(),
                                            vertex_partition_range_offsets.size()),
            starting_vertex,
            local_vertex_partition_range_first,
            invalid_vertex);
          ASSERT_TRUE(test_passed)
            << " source and destination vertices in the BFS predecessor tree ar not one hop away";
        }

        if (cugraph::test::g_perf) {
          RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
          comm.barrier();
          hr_timer.stop();
          hr_timer.display_and_clear(std::cout);
        }

        /* for every edge e = (u, v), abs(dist(u) - dist(v)) <= 1 or dist(u) == dist(v) ==
         * invalid_distance */

        if (cugraph::test::g_perf) {
          RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
          comm.barrier();
          hr_timer.start("validate (graph distances)");
        }

        {
          bool test_passed = check_edge_endpoint_distances<vertex_t, edge_t, vertex_t>(
            *handle_,
            raft::device_span<vertex_t const>(parents.data(), parents.size()),
            std::nullopt,
            raft::device_span<vertex_t const>(d_mg_distances.data(), d_mg_distances.size()),
            mg_pruned_graph_view,
            std::nullopt,
            raft::device_span<vertex_t const>(mg_graph_to_pruned_graph_map.data(),
                                              mg_graph_to_pruned_graph_map.size()),
            mg_isolated_trees_view,
            std::nullopt,
            raft::device_span<vertex_t const>(mg_graph_to_isolated_trees_map.data(),
                                              mg_graph_to_isolated_trees_map.size()),
            raft::host_span<vertex_t const>(vertex_partition_range_offsets.data(),
                                            vertex_partition_range_offsets.size()),
            local_vertex_partition_range_first,
            invalid_vertex,
            invalid_distance,
            reachable_from_2cores);
          ASSERT_TRUE(test_passed)
            << " only one of the two connected vertices are reachable from the starting vertex or "
               "the distance from the starting vertex differ by more than one.";
        }

        if (cugraph::test::g_perf) {
          RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
          comm.barrier();
          hr_timer.stop();
          hr_timer.display_and_clear(std::cout);
        }

        /* all the reachable vertices are in the same connected component, all the unreachable
         * vertices in different connected components */

        if (cugraph::test::g_perf) {
          RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
          comm.barrier();
          hr_timer.start("validate (connected components)");
        }

        {
          bool test_passed = check_connected_components(
            *handle_,
            raft::device_span<vertex_t const>(components.data(), components.size()),
            raft::device_span<vertex_t const>(d_mg_predecessors.data(), d_mg_predecessors.size()),
            starting_vertex_component,
            invalid_vertex);
          ASSERT_TRUE(test_passed)
            << "the BFS tree does not span the entire connected component of the starting vertex.";
        }

        if (cugraph::test::g_perf) {
          RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
          comm.barrier();
          hr_timer.stop();
          hr_timer.display_and_clear(std::cout);
        }

        /* check that predecessor->v edges exist in the input graph */

        if (cugraph::test::g_perf) {
          RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
          comm.barrier();
          hr_timer.start("validate (predecessor->v edge existence)");
        }

        {
          bool test_passed = check_has_edge_from_parents(
            *handle_,
            raft::device_span<vertex_t const>(parents.data(), parents.size()),
            raft::device_span<vertex_t const>(d_mg_predecessors.data(), d_mg_predecessors.size()),
            mg_pruned_graph_view,
            raft::device_span<vertex_t const>(mg_graph_to_pruned_graph_map.data(),
                                              mg_graph_to_pruned_graph_map.size()),
            mg_isolated_trees_view,
            raft::device_span<vertex_t const>(mg_graph_to_isolated_trees_map.data(),
                                              mg_graph_to_isolated_trees_map.size()),
            raft::host_span<vertex_t const>(vertex_partition_range_offsets.data(),
                                            vertex_partition_range_offsets.size()),
            starting_vertex,
            local_vertex_partition_range_first,
            invalid_vertex,
            reachable_from_2cores,
            in_2cores);
          ASSERT_TRUE(test_passed) << "predecessor->v missing in the input graph.";
        }

        if (cugraph::test::g_perf) {
          RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
          comm.barrier();
          hr_timer.stop();
          hr_timer.display_and_clear(std::cout);
        }
      }
    }

    std::cout << "average MG BFS (Kernel 2) time: " << (total_elapsed / num_timed_starting_vertices)
              << " TTEPS (arithmetic mean)=" << tteps_sum / num_timed_starting_vertices
              << " TTEPS (harmonic_mean)="
              << (one_over_tteps_sum > 0.0
                    ? static_cast<double>(num_timed_starting_vertices) / one_over_tteps_sum
                    : std::numeric_limits<double>::infinity())
              << std::endl;
  }

 private:
  static std::unique_ptr<raft::handle_t> handle_;
};

template <typename input_usecase_t>
std::unique_ptr<raft::handle_t> Tests_GRAPH500_MGBFS<input_usecase_t>::handle_ = nullptr;

using Tests_GRAPH500_MGBFS_Rmat = Tests_GRAPH500_MGBFS<cugraph::test::Rmat_Usecase>;

TEST_P(Tests_GRAPH500_MGBFS_Rmat, CheckInt64Int64)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_GRAPH500_MGBFS_Rmat,
  ::testing::Values(
    // enable correctness checks
    std::make_tuple(Graph500_BFS_Usecase{true, true, true},
                    cugraph::test::Rmat_Usecase(10,
                                                16,
                                                0.57,
                                                0.19,
                                                0.19,
                                                0 /* base RNG seed */,
                                                true /* undirected */,
                                                true /* scramble vertex ID */))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_GRAPH500_MGBFS_Rmat,
  ::testing::Values(
    // disable correctness checks for large graphs
    std::make_tuple(Graph500_BFS_Usecase{true, true, false},
                    cugraph::test::Rmat_Usecase(20,
                                                16,
                                                0.57,
                                                0.19,
                                                0.19,
                                                0 /* base RNG seed */,
                                                true /* undirected */,
                                                true /* scramble vertex IDs */))));

CUGRAPH_MG_TEST_PROGRAM_MAIN()
