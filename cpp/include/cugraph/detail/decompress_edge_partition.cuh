/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/edge_partition_edge_property_device_view.cuh>
#include <cugraph/graph_view.hpp>

#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/optional.h>
#include <thrust/sequence.h>
#include <thrust/tuple.h>

#include <optional>
#include <tuple>
#include <vector>

namespace cugraph {
namespace detail {

// FIXME: block size requires tuning
int32_t constexpr decompress_edge_partition_block_size = 1024;

template <typename vertex_t, typename edge_t, bool multi_gpu>
__global__ void decompress_to_edgelist_mid_degree(
  edge_partition_device_view_t<vertex_t, edge_t, multi_gpu> edge_partition,
  vertex_t major_range_first,
  vertex_t major_range_last,
  vertex_t* majors)
{
  auto const tid = threadIdx.x + blockIdx.x * blockDim.x;
  static_assert(decompress_edge_partition_block_size % raft::warp_size() == 0);
  auto const lane_id = tid % raft::warp_size();
  auto major_start_offset =
    static_cast<size_t>(major_range_first - edge_partition.major_range_first());
  size_t idx = static_cast<size_t>(tid / raft::warp_size());

  while (idx < static_cast<size_t>(major_range_last - major_range_first)) {
    auto major_offset = major_start_offset + idx;
    auto major =
      edge_partition.major_from_major_offset_nocheck(static_cast<vertex_t>(major_offset));
    vertex_t const* indices{nullptr};
    [[maybe_unused]] edge_t edge_offset{};
    edge_t local_degree{};
    thrust::tie(indices, edge_offset, local_degree) = edge_partition.local_edges(major_offset);
    auto local_offset                               = edge_partition.local_offset(major_offset);
    for (edge_t i = lane_id; i < local_degree; i += raft::warp_size()) {
      majors[local_offset + i] = major;
    }
    idx += gridDim.x * (blockDim.x / raft::warp_size());
  }
}

template <typename vertex_t, typename edge_t, bool multi_gpu>
__global__ void decompress_to_edgelist_high_degree(
  edge_partition_device_view_t<vertex_t, edge_t, multi_gpu> edge_partition,
  vertex_t major_range_first,
  vertex_t major_range_last,
  vertex_t* majors)
{
  auto major_start_offset =
    static_cast<size_t>(major_range_first - edge_partition.major_range_first());
  size_t idx = static_cast<size_t>(blockIdx.x);

  while (idx < static_cast<size_t>(major_range_last - major_range_first)) {
    auto major_offset = major_start_offset + idx;
    auto major =
      edge_partition.major_from_major_offset_nocheck(static_cast<vertex_t>(major_offset));
    vertex_t const* indices{nullptr};
    [[maybe_unused]] edge_t edge_offset{};
    edge_t local_degree{};
    thrust::tie(indices, edge_offset, local_degree) =
      edge_partition.local_edges(static_cast<vertex_t>(major_offset));
    auto local_offset = edge_partition.local_offset(major_offset);
    for (edge_t i = threadIdx.x; i < local_degree; i += blockDim.x) {
      majors[local_offset + i] = major;
    }
    idx += gridDim.x;
  }
}

template <typename vertex_t, typename edge_t, bool multi_gpu>
void decompress_edge_partition_to_fill_edgelist_majors(
  raft::handle_t const& handle,
  edge_partition_device_view_t<vertex_t, edge_t, multi_gpu> edge_partition,
  vertex_t* majors,
  std::optional<std::vector<vertex_t>> const& segment_offsets)
{
  auto execution_policy = handle.get_thrust_policy();
  if (segment_offsets) {
    // FIXME: we may further improve performance by 1) concurrently running kernels on different
    // segments; 2) individually tuning block sizes for different segments; and 3) adding one more
    // segment for very high degree vertices and running segmented reduction
    static_assert(detail::num_sparse_segments_per_vertex_partition == 3);
    if ((*segment_offsets)[1] > 0) {
      raft::grid_1d_block_t update_grid((*segment_offsets)[1],
                                        detail::decompress_edge_partition_block_size,
                                        handle.get_device_properties().maxGridSize[0]);

      detail::decompress_to_edgelist_high_degree<<<update_grid.num_blocks,
                                                   update_grid.block_size,
                                                   0,
                                                   handle.get_stream()>>>(
        edge_partition,
        edge_partition.major_range_first(),
        edge_partition.major_range_first() + (*segment_offsets)[1],
        majors);
    }
    if ((*segment_offsets)[2] - (*segment_offsets)[1] > 0) {
      raft::grid_1d_warp_t update_grid((*segment_offsets)[2] - (*segment_offsets)[1],
                                       detail::decompress_edge_partition_block_size,
                                       handle.get_device_properties().maxGridSize[0]);

      detail::decompress_to_edgelist_mid_degree<<<update_grid.num_blocks,
                                                  update_grid.block_size,
                                                  0,
                                                  handle.get_stream()>>>(
        edge_partition,
        edge_partition.major_range_first() + (*segment_offsets)[1],
        edge_partition.major_range_first() + (*segment_offsets)[2],
        majors);
    }
    if ((*segment_offsets)[3] - (*segment_offsets)[2] > 0) {
      thrust::for_each(
        execution_policy,
        thrust::make_counting_iterator(edge_partition.major_range_first()) + (*segment_offsets)[2],
        thrust::make_counting_iterator(edge_partition.major_range_first()) + (*segment_offsets)[3],
        [edge_partition, majors] __device__(auto major) {
          auto major_offset = edge_partition.major_offset_from_major_nocheck(major);
          auto local_degree = edge_partition.local_degree(major_offset);
          auto local_offset = edge_partition.local_offset(major_offset);
          thrust::fill(
            thrust::seq, majors + local_offset, majors + local_offset + local_degree, major);
        });
    }
    if (edge_partition.dcs_nzd_vertex_count() && (*(edge_partition.dcs_nzd_vertex_count()) > 0)) {
      thrust::for_each(
        execution_policy,
        thrust::make_counting_iterator(vertex_t{0}),
        thrust::make_counting_iterator(*(edge_partition.dcs_nzd_vertex_count())),
        [edge_partition, major_start_offset = (*segment_offsets)[3], majors] __device__(auto idx) {
          auto major = *(edge_partition.major_from_major_hypersparse_idx_nocheck(idx));
          auto major_idx =
            major_start_offset + idx;  // major_offset != major_idx in the hypersparse region
          auto local_degree = edge_partition.local_degree(major_idx);
          auto local_offset = edge_partition.local_offset(major_idx);
          thrust::fill(
            thrust::seq, majors + local_offset, majors + local_offset + local_degree, major);
        });
    }
  } else {
    thrust::for_each(
      execution_policy,
      thrust::make_counting_iterator(edge_partition.major_range_first()),
      thrust::make_counting_iterator(edge_partition.major_range_first()) +
        edge_partition.major_range_size(),
      [edge_partition, majors] __device__(auto major) {
        auto major_offset = edge_partition.major_offset_from_major_nocheck(major);
        auto local_degree = edge_partition.local_degree(major_offset);
        auto local_offset = edge_partition.local_offset(major_offset);
        thrust::fill(
          thrust::seq, majors + local_offset, majors + local_offset + local_degree, major);
      });
  }
}

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
void decompress_edge_partition_to_edgelist(
  raft::handle_t const& handle,
  edge_partition_device_view_t<vertex_t, edge_t, multi_gpu> edge_partition,
  std::optional<edge_partition_edge_property_device_view_t<edge_t, weight_t const*>>
    edge_partition_weight_view,
  vertex_t* edgelist_majors /* [OUT] */,
  vertex_t* edgelist_minors /* [OUT] */,
  std::optional<weight_t*> edgelist_weights /* [OUT] */,
  std::optional<std::vector<vertex_t>> const& segment_offsets)
{
  auto number_of_edges = edge_partition.number_of_edges();

  decompress_edge_partition_to_fill_edgelist_majors(
    handle, edge_partition, edgelist_majors, segment_offsets);
  thrust::copy(handle.get_thrust_policy(),
               edge_partition.indices(),
               edge_partition.indices() + number_of_edges,
               edgelist_minors);
  if (edge_partition_weight_view) {
    assert(edgelist_weights.has_value());
    thrust::copy(handle.get_thrust_policy(),
                 (*edge_partition_weight_view).value_first(),
                 (*edge_partition_weight_view).value_first() + number_of_edges,
                 (*edgelist_weights));
  }
}

}  // namespace detail
}  // namespace cugraph
