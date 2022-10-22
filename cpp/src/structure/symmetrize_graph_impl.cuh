/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <detail/graph_utils.cuh>
#include <structure/detail/structure_utils.cuh>

#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/misc_utils.cuh>

#include <raft/handle.hpp>
#include <raft/util/device_atomics.cuh>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/cub.cuh>
#include <thrust/adjacent_difference.h>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/distance.h>
#include <thrust/equal.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include <algorithm>
#include <tuple>

namespace cugraph {

namespace {

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::enable_if_t<multi_gpu,
                 std::tuple<graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>,
                            std::optional<rmm::device_uvector<vertex_t>>>>
symmetrize_graph_impl(raft::handle_t const& handle,
                      graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>&& graph,
                      std::optional<rmm::device_uvector<vertex_t>>&& renumber_map,
                      bool reciprocal,
                      bool do_expensive_check)
{
  auto graph_view = graph.view();

  CUGRAPH_EXPECTS(renumber_map.has_value(),
                  "Invalid input arguments: renumber_map.has_value() should be true if multi-GPU.");
  CUGRAPH_EXPECTS(
    (*renumber_map).size() == static_cast<size_t>(graph_view.local_vertex_partition_range_size()),
    "Invalid input arguments: (*renumber_map).size() should match with the local "
    "vertex partition range size.");

  if (do_expensive_check) { /* currently, nothing to do */
  }

  if (graph.is_symmetric()) { return std::make_tuple(std::move(graph), std::move(renumber_map)); }

  auto is_multigraph = graph.is_multigraph();

  auto [edgelist_srcs, edgelist_dsts, edgelist_weights] =
    decompress_to_edgelist(handle,
                           graph_view,
                           std::make_optional<raft::device_span<vertex_t const>>(
                             (*renumber_map).data(), (*renumber_map).size()));
  graph = graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(handle);

  std::tie(edgelist_srcs, edgelist_dsts, edgelist_weights) =
    symmetrize_edgelist<vertex_t, weight_t, store_transposed, multi_gpu>(
      handle,
      std::move(edgelist_srcs),
      std::move(edgelist_dsts),
      std::move(edgelist_weights),
      reciprocal);

  graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu> symmetrized_graph(handle);
  std::optional<rmm::device_uvector<vertex_t>> new_renumber_map{std::nullopt};
  std::tie(symmetrized_graph, std::ignore, new_renumber_map) =
    create_graph_from_edgelist<vertex_t, edge_t, weight_t, int32_t, store_transposed, multi_gpu>(
      handle,
      std::move(renumber_map),
      std::move(edgelist_srcs),
      std::move(edgelist_dsts),
      std::move(edgelist_weights),
      std::nullopt,
      graph_properties_t{is_multigraph, true},
      true);

  return std::make_tuple(std::move(symmetrized_graph), std::move(new_renumber_map));
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::enable_if_t<!multi_gpu,
                 std::tuple<graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>,
                            std::optional<rmm::device_uvector<vertex_t>>>>
symmetrize_graph_impl(raft::handle_t const& handle,
                      graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>&& graph,
                      std::optional<rmm::device_uvector<vertex_t>>&& renumber_map,
                      bool reciprocal,
                      bool do_expensive_check)
{
  auto graph_view = graph.view();

  CUGRAPH_EXPECTS(
    !renumber_map.has_value() ||
      (*renumber_map).size() == static_cast<size_t>(graph_view.local_vertex_partition_range_size()),
    "Invalid input arguments: if renumber_map.has_value() == true, (*renumber_map).size() should "
    "match with the local vertex partition range size.");

  if (do_expensive_check) { /* currently, nothing to do */
  }

  if (graph.is_symmetric()) { return std::make_tuple(std::move(graph), std::move(renumber_map)); }

  auto number_of_vertices = graph.number_of_vertices();
  auto is_multigraph      = graph.is_multigraph();
  bool renumber           = renumber_map.has_value();

  auto [edgelist_srcs, edgelist_dsts, edgelist_weights] =
    decompress_to_edgelist(handle,
                           graph_view,
                           renumber_map ? std::make_optional<raft::device_span<vertex_t const>>(
                                            (*renumber_map).data(), (*renumber_map).size())
                                        : std::nullopt);
  graph = graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(handle);

  std::tie(edgelist_srcs, edgelist_dsts, edgelist_weights) =
    symmetrize_edgelist<vertex_t, weight_t, store_transposed, multi_gpu>(
      handle,
      std::move(edgelist_srcs),
      std::move(edgelist_dsts),
      std::move(edgelist_weights),
      reciprocal);

  auto vertices = renumber ? std::move(renumber_map)
                           : std::make_optional<rmm::device_uvector<vertex_t>>(number_of_vertices,
                                                                               handle.get_stream());
  if (!renumber) {
    thrust::sequence(
      handle.get_thrust_policy(), (*vertices).begin(), (*vertices).end(), vertex_t{0});
  }

  graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu> symmetrized_graph(handle);
  std::optional<rmm::device_uvector<vertex_t>> new_renumber_map{std::nullopt};
  std::tie(symmetrized_graph, std::ignore, new_renumber_map) =
    create_graph_from_edgelist<vertex_t, edge_t, weight_t, int32_t, store_transposed, multi_gpu>(
      handle,
      std::move(vertices),
      std::move(edgelist_srcs),
      std::move(edgelist_dsts),
      std::move(edgelist_weights),
      std::nullopt,
      graph_properties_t{is_multigraph, true},
      renumber);

  return std::make_tuple(std::move(symmetrized_graph), std::move(new_renumber_map));
}

}  // namespace

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>,
           std::optional<rmm::device_uvector<vertex_t>>>
symmetrize_graph(raft::handle_t const& handle,
                 graph_t<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>&& graph,
                 std::optional<rmm::device_uvector<vertex_t>>&& renumber_map,
                 bool reciprocal,
                 bool do_expensive_check)
{
  return symmetrize_graph_impl(
    handle, std::move(graph), std::move(renumber_map), reciprocal, do_expensive_check);
}

}  // namespace cugraph
