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

#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/optional.h>

#include <optional>
#include <tuple>
#include <vector>

namespace cugraph {
namespace detail {

// FIXME: block size requires tuning
int32_t constexpr decompress_edge_partition_block_size = 1024;

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
__global__ void decompress_to_edgelist_mid_degree(
  edge_partition_device_view_t<vertex_t, edge_t, weight_t, multi_gpu> edge_partition,
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
    thrust::optional<weight_t const*> weights{thrust::nullopt};
    edge_t local_degree{};
    thrust::tie(indices, weights, local_degree) = edge_partition.local_edges(major_offset);
    auto local_offset                           = edge_partition.local_offset(major_offset);
    for (edge_t i = lane_id; i < local_degree; i += raft::warp_size()) {
      majors[local_offset + i] = major;
    }
    idx += gridDim.x * (blockDim.x / raft::warp_size());
  }
}

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
__global__ void decompress_to_edgelist_high_degree(
  edge_partition_device_view_t<vertex_t, edge_t, weight_t, multi_gpu> edge_partition,
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
    thrust::optional<weight_t const*> weights{thrust::nullopt};
    edge_t local_degree{};
    thrust::tie(indices, weights, local_degree) =
      edge_partition.local_edges(static_cast<vertex_t>(major_offset));
    auto local_offset = edge_partition.local_offset(major_offset);
    for (edge_t i = threadIdx.x; i < local_degree; i += blockDim.x) {
      majors[local_offset + i] = major;
    }
    idx += gridDim.x;
  }
}

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
void decompress_edge_partition_to_fill_edgelist_majors(
  raft::handle_t const& handle,
  edge_partition_device_view_t<vertex_t, edge_t, weight_t, multi_gpu> edge_partition,
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

template <typename vertex_t, typename edge_t, typename weight_t, typename prop_t, bool multi_gpu>
__global__ void partially_decompress_to_edgelist_high_degree(
  edge_partition_device_view_t<vertex_t, edge_t, weight_t, multi_gpu> edge_partition,
  vertex_t const* input_majors,
  edge_t const* input_major_start_offsets,
  vertex_t input_major_count,
  vertex_t* output_majors,
  vertex_t* output_minors,
  thrust::optional<thrust::tuple<prop_t const*, prop_t*>> property,
  thrust::optional<thrust::tuple<edge_t const*, edge_t*>> global_edge_index)
{
  size_t idx = static_cast<size_t>(blockIdx.x);
  while (idx < static_cast<size_t>(input_major_count)) {
    auto major                  = input_majors[idx];
    auto major_partition_offset = static_cast<size_t>(major - edge_partition.major_range_first());
    vertex_t const* indices{nullptr};
    thrust::optional<weight_t const*> weights{thrust::nullopt};
    edge_t local_degree{};
    thrust::tie(indices, weights, local_degree) =
      edge_partition.local_edges(static_cast<vertex_t>(major_partition_offset));
    auto major_offset = input_major_start_offsets[idx];
    for (edge_t i = threadIdx.x; i < local_degree; i += blockDim.x) {
      output_majors[major_offset + i] = major;
      output_minors[major_offset + i] = indices[i];
    }
    if (property) {
      auto input_property     = thrust::get<0>(*property)[idx];
      prop_t* output_property = thrust::get<1>(*property);
      for (edge_t i = threadIdx.x; i < local_degree; i += blockDim.x) {
        output_property[major_offset + i] = input_property;
      }
    }
    if (global_edge_index) {
      auto adjacency_list_offset = thrust::get<0>(*global_edge_index)[major_partition_offset];
      auto minor_map             = thrust::get<1>(*global_edge_index);
      for (edge_t i = threadIdx.x; i < local_degree; i += blockDim.x) {
        minor_map[major_offset + i] = adjacency_list_offset + i;
      }
    }
    idx += gridDim.x;
  }
}

template <typename vertex_t, typename edge_t, typename weight_t, typename prop_t, bool multi_gpu>
__global__ void partially_decompress_to_edgelist_mid_degree(
  edge_partition_device_view_t<vertex_t, edge_t, weight_t, multi_gpu> edge_partition,
  vertex_t const* input_majors,
  edge_t const* input_major_start_offsets,
  vertex_t input_major_count,
  vertex_t* output_majors,
  vertex_t* output_minors,
  thrust::optional<thrust::tuple<prop_t const*, prop_t*>> property,
  thrust::optional<thrust::tuple<edge_t const*, edge_t*>> global_edge_index)
{
  auto const tid = threadIdx.x + blockIdx.x * blockDim.x;
  static_assert(decompress_edge_partition_block_size % raft::warp_size() == 0);
  auto const lane_id = tid % raft::warp_size();
  size_t idx         = static_cast<size_t>(tid / raft::warp_size());
  while (idx < static_cast<size_t>(input_major_count)) {
    auto major                  = input_majors[idx];
    auto major_partition_offset = static_cast<size_t>(major - edge_partition.major_range_first());
    vertex_t const* indices{nullptr};
    edge_t local_degree{};
    auto major_offset = input_major_start_offsets[idx];
    for (edge_t i = threadIdx.x; i < local_degree; i += blockDim.x) {
      output_majors[major_offset + i] = major;
      output_minors[major_offset + i] = indices[i];
    }
    if (property) {
      auto input_property     = thrust::get<0>(*property)[idx];
      prop_t* output_property = thrust::get<1>(*property);
      for (edge_t i = lane_id; i < local_degree; i += raft::warp_size()) {
        output_property[major_offset + i] = input_property;
      }
    }
    if (global_edge_index) {
      auto adjacency_list_offset = thrust::get<0>(*global_edge_index)[major_partition_offset];
      auto minor_map             = thrust::get<1>(*global_edge_index);
      for (edge_t i = threadIdx.x; i < local_degree; i += blockDim.x) {
        minor_map[major_offset + i] = adjacency_list_offset + i;
      }
    }
    idx += gridDim.x * (blockDim.x / raft::warp_size());
  }
}

template <typename vertex_t, typename edge_t, typename weight_t, typename prop_t, bool multi_gpu>
void partially_decompress_edge_partition_to_fill_edgelist(
  raft::handle_t const& handle,
  edge_partition_device_view_t<vertex_t, edge_t, weight_t, multi_gpu> edge_partition,
  vertex_t const* input_majors,
  edge_t const* input_major_start_offsets,
  std::vector<vertex_t> const& segment_offsets,
  vertex_t* majors,
  vertex_t* minors,
  thrust::optional<thrust::tuple<prop_t const*, prop_t*>> property,
  thrust::optional<thrust::tuple<edge_t const*, edge_t*>> global_edge_index)
{
  auto execution_policy = handle.get_thrust_policy();
  static_assert(detail::num_sparse_segments_per_vertex_partition == 3);
  auto& comm           = handle.get_comms();
  auto const comm_rank = comm.get_rank();
  if (segment_offsets[1] - segment_offsets[0] > 0) {
    raft::grid_1d_block_t update_grid(segment_offsets[1] - segment_offsets[0],
                                      detail::decompress_edge_partition_block_size,
                                      handle.get_device_properties().maxGridSize[0]);

    detail::partially_decompress_to_edgelist_high_degree<<<update_grid.num_blocks,
                                                           update_grid.block_size,
                                                           0,
                                                           handle.get_stream()>>>(
      edge_partition,
      input_majors + segment_offsets[0],
      input_major_start_offsets,
      segment_offsets[1],
      majors,
      minors,
      property ? thrust::make_optional(thrust::make_tuple(
                   thrust::get<0>(*property) + segment_offsets[0], thrust::get<1>(*property)))
               : thrust::nullopt,
      global_edge_index);
  }
  if (segment_offsets[2] - segment_offsets[1] > 0) {
    raft::grid_1d_warp_t update_grid(segment_offsets[2] - segment_offsets[1],
                                     detail::decompress_edge_partition_block_size,
                                     handle.get_device_properties().maxGridSize[0]);

    detail::partially_decompress_to_edgelist_mid_degree<<<update_grid.num_blocks,
                                                          update_grid.block_size,
                                                          0,
                                                          handle.get_stream()>>>(
      edge_partition,
      input_majors + segment_offsets[1],
      input_major_start_offsets + segment_offsets[1] - segment_offsets[0],
      segment_offsets[2] - segment_offsets[1],
      majors,
      minors,
      property ? thrust::make_optional(thrust::make_tuple(
                   thrust::get<0>(*property) + segment_offsets[1], thrust::get<1>(*property)))
               : thrust::nullopt,
      global_edge_index);
  }
  if (segment_offsets[3] - segment_offsets[2] > 0) {
    thrust::for_each(
      execution_policy,
      thrust::make_counting_iterator(vertex_t{0}),
      thrust::make_counting_iterator(segment_offsets[3] - segment_offsets[2]),
      [edge_partition,
       input_majors = input_majors + segment_offsets[2],
       input_major_start_offsets =
         input_major_start_offsets + segment_offsets[2] - segment_offsets[0],
       majors,
       minors,
       property = property
                    ? thrust::make_optional(thrust::make_tuple(
                        thrust::get<0>(*property) + segment_offsets[2], thrust::get<1>(*property)))
                    : thrust::nullopt,
       global_edge_index] __device__(auto idx) {
        auto major        = input_majors[idx];
        auto major_offset = input_major_start_offsets[idx];
        auto major_partition_offset =
          static_cast<size_t>(major - edge_partition.major_range_first());
        vertex_t const* indices{nullptr};
        thrust::optional<weight_t const*> weights{thrust::nullopt};
        edge_t local_degree{};
        thrust::tie(indices, weights, local_degree) =
          edge_partition.local_edges(major_partition_offset);
        thrust::fill(
          thrust::seq, majors + major_offset, majors + major_offset + local_degree, major);
        thrust::copy(thrust::seq, indices, indices + local_degree, minors + major_offset);
        if (property) {
          auto major_input_property  = thrust::get<0>(*property)[idx];
          auto minor_output_property = thrust::get<1>(*property);
          thrust::fill(thrust::seq,
                       minor_output_property + major_offset,
                       minor_output_property + major_offset + local_degree,
                       major_input_property);
        }
        if (global_edge_index) {
          auto adjacency_list_offset = thrust::get<0>(*global_edge_index)[major_partition_offset];
          auto minor_map             = thrust::get<1>(*global_edge_index);
          thrust::sequence(thrust::seq,
                           minor_map + major_offset,
                           minor_map + major_offset + local_degree,
                           adjacency_list_offset);
        }
      });
  }
  if (edge_partition.dcs_nzd_vertex_count() && (*(edge_partition.dcs_nzd_vertex_count()) > 0)) {
    thrust::for_each(
      execution_policy,
      thrust::make_counting_iterator(vertex_t{0}),
      thrust::make_counting_iterator(segment_offsets[4] - segment_offsets[3]),
      [edge_partition,
       input_majors = input_majors + segment_offsets[3],
       input_major_start_offsets =
         input_major_start_offsets + segment_offsets[3] - segment_offsets[0],
       majors,
       minors,
       property = property
                    ? thrust::make_optional(thrust::make_tuple(
                        thrust::get<0>(*property) + segment_offsets[3], thrust::get<1>(*property)))
                    : thrust::nullopt,
       global_edge_index] __device__(auto idx) {
        auto major        = input_majors[idx];
        auto major_offset = input_major_start_offsets[idx];
        auto major_idx    = edge_partition.major_hypersparse_idx_from_major_nocheck(major);
        if (major_idx) {
          vertex_t const* indices{nullptr};
          thrust::optional<weight_t const*> weights{thrust::nullopt};
          edge_t local_degree{};
          thrust::tie(indices, weights, local_degree) = edge_partition.local_edges(*major_idx);
          thrust::fill(
            thrust::seq, majors + major_offset, majors + major_offset + local_degree, major);
          thrust::copy(thrust::seq, indices, indices + local_degree, minors + major_offset);
          if (property) {
            auto major_input_property  = thrust::get<0>(*property)[idx];
            auto minor_output_property = thrust::get<1>(*property);
            thrust::fill(thrust::seq,
                         minor_output_property + major_offset,
                         minor_output_property + major_offset + local_degree,
                         major_input_property);
          }
          if (global_edge_index) {
            auto major_partition_offset =
              static_cast<size_t>(*major_idx - edge_partition.major_range_first());
            auto adjacency_list_offset = thrust::get<0>(*global_edge_index)[major_partition_offset];
            auto minor_map             = thrust::get<1>(*global_edge_index);
            thrust::sequence(thrust::seq,
                             minor_map + major_offset,
                             minor_map + major_offset + local_degree,
                             adjacency_list_offset);
          }
        }
      });
  }
}

template <typename vertex_t, typename edge_t, typename weight_t, bool multi_gpu>
void decompress_edge_partition_to_edgelist(
  raft::handle_t const& handle,
  edge_partition_device_view_t<vertex_t, edge_t, weight_t, multi_gpu> const edge_partition,
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
  if (edgelist_weights) {
    thrust::copy(handle.get_thrust_policy(),
                 *(edge_partition.weights()),
                 *(edge_partition.weights()) + number_of_edges,
                 (*edgelist_weights));
  }
}

}  // namespace detail
}  // namespace cugraph
