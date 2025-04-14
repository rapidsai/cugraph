/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include <cugraph/graph.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/std/iterator>
#include <thrust/binary_search.h>
#include <thrust/count.h>

#include <vector>

namespace cugraph {
namespace detail {

// check vertices in the pair are in [0, num_vertices) and belongs to one of the local edge
// partitions.
template <typename vertex_t>
struct is_invalid_input_vertex_pair_t {
  vertex_t num_vertices{};
  raft::device_span<vertex_t const> edge_partition_major_range_firsts{};
  raft::device_span<vertex_t const> edge_partition_major_range_lasts{};
  vertex_t edge_partition_minor_range_first{};
  vertex_t edge_partition_minor_range_last{};

  __device__ bool operator()(thrust::tuple<vertex_t, vertex_t> pair) const
  {
    auto major = thrust::get<0>(pair);
    auto minor = thrust::get<1>(pair);
    if (!is_valid_vertex(num_vertices, major) || !is_valid_vertex(num_vertices, minor)) {
      return true;
    }
    auto it = thrust::upper_bound(thrust::seq,
                                  edge_partition_major_range_lasts.begin(),
                                  edge_partition_major_range_lasts.end(),
                                  major);
    if (it == edge_partition_major_range_lasts.end()) { return true; }
    auto edge_partition_idx =
      static_cast<size_t>(cuda::std::distance(edge_partition_major_range_lasts.begin(), it));
    if (major < edge_partition_major_range_firsts[edge_partition_idx]) { return true; }
    return (minor < edge_partition_minor_range_first) || (minor >= edge_partition_minor_range_last);
  }
};

template <typename GraphViewType, typename VertexPairIterator>
size_t count_invalid_vertex_pairs(raft::handle_t const& handle,
                                  GraphViewType const& graph_view,
                                  VertexPairIterator vertex_pair_first,
                                  VertexPairIterator vertex_pair_last)
{
  using vertex_t = typename GraphViewType::vertex_type;

  std::vector<vertex_t> h_edge_partition_major_range_firsts(
    graph_view.number_of_local_edge_partitions());
  std::vector<vertex_t> h_edge_partition_major_range_lasts(
    h_edge_partition_major_range_firsts.size());
  vertex_t edge_partition_minor_range_first{};
  vertex_t edge_partition_minor_range_last{};
  if constexpr (GraphViewType::is_multi_gpu) {
    for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); i++) {
      if constexpr (GraphViewType::is_storage_transposed) {
        h_edge_partition_major_range_firsts[i] = graph_view.local_edge_partition_dst_range_first(i);
        h_edge_partition_major_range_lasts[i]  = graph_view.local_edge_partition_dst_range_last(i);
      } else {
        h_edge_partition_major_range_firsts[i] = graph_view.local_edge_partition_src_range_first(i);
        h_edge_partition_major_range_lasts[i]  = graph_view.local_edge_partition_src_range_last(i);
      }
    }
    if constexpr (GraphViewType::is_storage_transposed) {
      edge_partition_minor_range_first = graph_view.local_edge_partition_src_range_first();
      edge_partition_minor_range_last  = graph_view.local_edge_partition_src_range_last();
    } else {
      edge_partition_minor_range_first = graph_view.local_edge_partition_dst_range_first();
      edge_partition_minor_range_last  = graph_view.local_edge_partition_dst_range_last();
    }
  } else {
    h_edge_partition_major_range_firsts[0] = vertex_t{0};
    h_edge_partition_major_range_lasts[0]  = graph_view.number_of_vertices();
    edge_partition_minor_range_first       = vertex_t{0};
    edge_partition_minor_range_last        = graph_view.number_of_vertices();
  }
  rmm::device_uvector<vertex_t> d_edge_partition_major_range_firsts(
    h_edge_partition_major_range_firsts.size(), handle.get_stream());
  rmm::device_uvector<vertex_t> d_edge_partition_major_range_lasts(
    h_edge_partition_major_range_lasts.size(), handle.get_stream());
  raft::update_device(d_edge_partition_major_range_firsts.data(),
                      h_edge_partition_major_range_firsts.data(),
                      h_edge_partition_major_range_firsts.size(),
                      handle.get_stream());
  raft::update_device(d_edge_partition_major_range_lasts.data(),
                      h_edge_partition_major_range_lasts.data(),
                      h_edge_partition_major_range_lasts.size(),
                      handle.get_stream());

  auto num_invalid_pairs = thrust::count_if(
    handle.get_thrust_policy(),
    vertex_pair_first,
    vertex_pair_last,
    is_invalid_input_vertex_pair_t<vertex_t>{
      graph_view.number_of_vertices(),
      raft::device_span<vertex_t const>(d_edge_partition_major_range_firsts.begin(),
                                        d_edge_partition_major_range_firsts.end()),
      raft::device_span<vertex_t const>(d_edge_partition_major_range_lasts.begin(),
                                        d_edge_partition_major_range_lasts.end()),
      edge_partition_minor_range_first,
      edge_partition_minor_range_last});
  if constexpr (GraphViewType::is_multi_gpu) {
    auto& comm = handle.get_comms();
    num_invalid_pairs =
      host_scalar_allreduce(comm, num_invalid_pairs, raft::comms::op_t::SUM, handle.get_stream());
  }

  return num_invalid_pairs;
}

}  // namespace detail
}  // namespace cugraph
