/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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
#include "detail/graph_partition_utils.cuh"

#include <cugraph/detail/shuffle_wrappers.hpp>
#include <cugraph/edge_partition_view.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/shuffle_comm.cuh>

#include <raft/comms/mpi_comms.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>
#include <raft/random/rng_state.hpp>

#include <thrust/for_each.h>
#include <thrust/tuple.h>

#include <iostream>
#include <string>
#include <tuple>

namespace cugraph {
namespace detail {

/**
 * @brief This function prints vertex and edge partitions.
 */
template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
std::
  tuple<rmm::device_uvector<edge_t>, rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>>
  lookup_edge_ids_impl(
    raft::handle_t const& handle,
    cugraph::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
    std::optional<cugraph::edge_property_view_t<edge_t, edge_t const*>> edge_id_view,
    raft::device_span<edge_t const> edge_ids_to_lookup)
{
  rmm::device_uvector<edge_t> sorted_edge_ids_to_lookup(edge_ids_to_lookup.size(),
                                                        handle.get_stream());
  rmm::device_uvector<vertex_t> output_srcs(sorted_edge_ids_to_lookup.size(), handle.get_stream());
  rmm::device_uvector<vertex_t> output_dsts(sorted_edge_ids_to_lookup.size(), handle.get_stream());

  return std::make_tuple(
    std::move(sorted_edge_ids_to_lookup), std::move(output_srcs), std::move(output_dsts));
}

}  // namespace detail

template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
std::
  tuple<rmm::device_uvector<edge_t>, rmm::device_uvector<vertex_t>, rmm::device_uvector<vertex_t>>
  lookup_edge_ids(
    raft::handle_t const& handle,
    cugraph::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
    std::optional<cugraph::edge_property_view_t<edge_t, edge_t const*>> edge_id_view,
    raft::device_span<edge_t const> edge_ids_to_lookup)
{
  return detail::lookup_edge_ids_impl(handle, graph_view, edge_id_view, edge_ids_to_lookup);
}

template std::
  tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>>
  lookup_edge_ids(raft::handle_t const& handle,
                  graph_view_t<int32_t, int32_t, false, true> const& graph_view,
                  std::optional<edge_property_view_t<int32_t, int32_t const*>> edge_id_view,
                  raft::device_span<int32_t const> edge_ids_to_lookup);
}  // namespace cugraph
