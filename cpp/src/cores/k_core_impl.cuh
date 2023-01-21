/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cugraph/algorithms.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/error.hpp>

#include <thrust/copy.h>
#include <thrust/iterator/discard_iterator.h>

namespace cugraph {

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>>
k_core(raft::handle_t const& handle,
       graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
       std::optional<edge_property_view_t<edge_t, weight_t const*>> edge_weight_view,
       size_t k,
       std::optional<k_core_degree_type_t> degree_type,
       std::optional<raft::device_span<edge_t const>> core_numbers,
       bool do_expensive_check)
{
  rmm::device_uvector<edge_t> computed_core_numbers(0, handle.get_stream());

  if (!core_numbers) {
    CUGRAPH_EXPECTS(degree_type.has_value(),
                    "If core_numbers is not specified then degree_type must be specified");

    computed_core_numbers.resize(graph_view.local_vertex_partition_range_size(),
                                 handle.get_stream());
    core_number(handle,
                graph_view,
                computed_core_numbers.data(),
                *degree_type,
                size_t{0},
                std::numeric_limits<size_t>::max(),
                do_expensive_check);

    core_numbers = std::make_optional(
      raft::device_span<edge_t const>{computed_core_numbers.data(), computed_core_numbers.size()});
  }

  rmm::device_uvector<vertex_t> subgraph_vertices(core_numbers->size(), handle.get_stream());
  auto iter_end = thrust::copy_if(
    handle.get_thrust_policy(),
    thrust::make_zip_iterator(
      thrust::make_counting_iterator(graph_view.local_vertex_partition_range_first()),
      core_numbers->begin()),
    thrust::make_zip_iterator(
      thrust::make_counting_iterator(graph_view.local_vertex_partition_range_last()),
      core_numbers->end()),
    thrust::make_zip_iterator(subgraph_vertices.begin(), thrust::make_discard_iterator()),
    [k] __device__(auto tuple) { return (k <= thrust::get<1>(tuple)); });

  subgraph_vertices.resize(
    thrust::distance(
      thrust::make_zip_iterator(subgraph_vertices.begin(), thrust::make_discard_iterator()),
      iter_end),
    handle.get_stream());

  rmm::device_uvector<size_t> subgraph_offsets(2, handle.get_stream());
  std::vector<size_t> h_subgraph_offsets{{0, subgraph_vertices.size()}};
  raft::update_device(subgraph_offsets.data(),
                      h_subgraph_offsets.data(),
                      h_subgraph_offsets.size(),
                      handle.get_stream());
  handle.sync_stream();

  auto [src, dst, wgt, offsets] = extract_induced_subgraphs(
    handle,
    graph_view,
    edge_weight_view,
    raft::device_span<size_t const>{subgraph_offsets.data(), subgraph_offsets.size()},
    raft::device_span<vertex_t const>{subgraph_vertices.data(), subgraph_vertices.size()},
    do_expensive_check);

  return std::make_tuple(std::move(src), std::move(dst), std::move(wgt));
}

}  // namespace cugraph
