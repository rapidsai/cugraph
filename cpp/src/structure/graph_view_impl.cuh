/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

#include "detail/graph_partition_utils.cuh"
#include "prims/count_if_e.cuh"
#include "prims/per_v_transform_reduce_incoming_outgoing_e.cuh"
#include "prims/reduce_op.cuh"
#include "utilities/error_check_utils.cuh"

#include <cugraph/edge_property.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/atomic_ops.cuh>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/mask_utils.cuh>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/std/iterator>
#include <cuda/std/optional>
#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace cugraph {

namespace {

// can't use lambda due to nvcc limitations (The enclosing parent function ("graph_view_t") for an
// extended __device__ lambda must allow its address to be taken)
template <typename vertex_t>
struct out_of_range_t {
  vertex_t min{};
  vertex_t max{};

  __device__ bool operator()(vertex_t v) const { return (v < min) || (v >= max); }
};

// compute out-degrees (if we are internally storing edges in the sparse 2D matrix using sources as
// major indices) or in-degrees (otherwise)
template <typename vertex_t, typename edge_t>
rmm::device_uvector<edge_t> compute_major_degrees(
  raft::handle_t const& handle,
  std::vector<raft::device_span<edge_t const>> const& edge_partition_offsets,
  std::optional<std::vector<raft::device_span<vertex_t const>>> const&
    edge_partition_dcs_nzd_vertices,
  std::optional<std::vector<raft::device_span<uint32_t const>>> const& edge_partition_masks,
  partition_t<vertex_t> const& partition,
  std::vector<vertex_t> const& edge_partition_segment_offsets)
{
  auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
  auto const major_comm_rank = major_comm.get_rank();
  auto const major_comm_size = major_comm.get_size();
  auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
  auto const minor_comm_rank = minor_comm.get_rank();
  auto const minor_comm_size = minor_comm.get_size();

  auto use_dcs = edge_partition_dcs_nzd_vertices.has_value();

  rmm::device_uvector<edge_t> local_degrees(
    0, handle.get_stream());  // excluding globally 0 degree vertices
  rmm::device_uvector<edge_t> degrees(0, handle.get_stream());

  vertex_t max_num_local_degrees{0};
  for (int i = 0; i < minor_comm_size; ++i) {
    auto major_range_vertex_partition_id =
      detail::compute_local_edge_partition_major_range_vertex_partition_id_t{
        major_comm_size, minor_comm_size, major_comm_rank, minor_comm_rank}(i);
    auto major_range_vertex_partition_size =
      partition.vertex_partition_range_size(major_range_vertex_partition_id);
    auto segment_offset_size_per_partition =
      edge_partition_segment_offsets.size() / static_cast<size_t>(minor_comm_size);
    auto num_local_degrees =
      edge_partition_segment_offsets[segment_offset_size_per_partition * i +
                                     (segment_offset_size_per_partition - 2)];
    max_num_local_degrees = std::max(max_num_local_degrees, num_local_degrees);
    if (i == minor_comm_rank) {
      degrees.resize(major_range_vertex_partition_size, handle.get_stream());
      thrust::fill(
        handle.get_thrust_policy(), degrees.begin() + num_local_degrees, degrees.end(), edge_t{0});
    }
  }
  local_degrees.resize(max_num_local_degrees, handle.get_stream());
  for (int i = 0; i < minor_comm_size; ++i) {
    auto major_range_vertex_partition_id =
      detail::compute_local_edge_partition_major_range_vertex_partition_id_t{
        major_comm_size, minor_comm_size, major_comm_rank, minor_comm_rank}(i);
    auto major_range_first =
      partition.vertex_partition_range_first(major_range_vertex_partition_id);

    auto offsets = edge_partition_offsets[i];
    auto masks   = edge_partition_masks ? cuda::std::make_optional((*edge_partition_masks)[i])
                                        : cuda::std::nullopt;
    auto segment_offset_size_per_partition =
      edge_partition_segment_offsets.size() / static_cast<size_t>(minor_comm_size);
    auto num_local_degrees =
      edge_partition_segment_offsets[segment_offset_size_per_partition * i +
                                     (segment_offset_size_per_partition - 2)];
    auto major_hypersparse_first =
      use_dcs ? major_range_first +
                  edge_partition_segment_offsets[segment_offset_size_per_partition * i +
                                                 detail::num_sparse_segments_per_vertex_partition]
              : major_range_first + num_local_degrees;
    auto execution_policy = handle.get_thrust_policy();
    thrust::transform(execution_policy,
                      thrust::make_counting_iterator(vertex_t{0}),
                      thrust::make_counting_iterator(major_hypersparse_first - major_range_first),
                      local_degrees.begin(),
                      cuda::proclaim_return_type<edge_t>([offsets, masks] __device__(auto i) {
                        auto local_degree = offsets[i + 1] - offsets[i];
                        if (masks) {
                          local_degree = static_cast<edge_t>(
                            detail::count_set_bits((*masks).begin(), offsets[i], local_degree));
                        }
                        return local_degree;
                      }));
    if (use_dcs) {
      auto dcs_nzd_vertices = (*edge_partition_dcs_nzd_vertices)[i];
      thrust::fill(execution_policy,
                   local_degrees.begin() + (major_hypersparse_first - major_range_first),
                   local_degrees.begin() + num_local_degrees,
                   edge_t{0});
      thrust::for_each(
        execution_policy,
        thrust::make_counting_iterator(vertex_t{0}),
        thrust::make_counting_iterator(static_cast<vertex_t>(dcs_nzd_vertices.size())),
        [offsets,
         dcs_nzd_vertices,
         masks,
         major_range_first,
         major_hypersparse_first,
         local_degrees = local_degrees.data()] __device__(auto i) {
          auto major_idx    = (major_hypersparse_first - major_range_first) + i;
          auto local_degree = offsets[major_idx + 1] - offsets[major_idx];
          if (masks) {
            local_degree = static_cast<edge_t>(
              detail::count_set_bits((*masks).begin(), offsets[major_idx], local_degree));
          }
          auto v                               = dcs_nzd_vertices[i];
          local_degrees[v - major_range_first] = local_degree;
        });
    }
    minor_comm.reduce(local_degrees.data(),
                      i == minor_comm_rank ? degrees.data() : static_cast<edge_t*>(nullptr),
                      static_cast<size_t>(num_local_degrees),
                      raft::comms::op_t::SUM,
                      i,
                      handle.get_stream());
  }

  return degrees;
}

// compute out-degrees (if we are internally storing edges in the sparse 2D matrix using sources as
// major indices) or in-degrees (otherwise)
template <typename vertex_t, typename edge_t>
rmm::device_uvector<edge_t> compute_major_degrees(
  raft::handle_t const& handle,
  raft::device_span<edge_t const> offsets,
  std::optional<raft::device_span<uint32_t const>> masks,
  vertex_t number_of_vertices)
{
  rmm::device_uvector<edge_t> degrees(number_of_vertices, handle.get_stream());
  thrust::tabulate(
    handle.get_thrust_policy(),
    degrees.begin(),
    degrees.end(),
    [offsets,
     masks = masks ? cuda::std::make_optional(*masks) : cuda::std::nullopt] __device__(auto i) {
      auto local_degree = offsets[i + 1] - offsets[i];
      if (masks) {
        local_degree =
          static_cast<edge_t>(detail::count_set_bits((*masks).begin(), offsets[i], local_degree));
      }
      return local_degree;
    });
  return degrees;
}

template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
rmm::device_uvector<edge_t> compute_minor_degrees(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view)
{
  rmm::device_uvector<edge_t> minor_degrees(graph_view.local_vertex_partition_range_size(),
                                            handle.get_stream());
  if (store_transposed) {
    per_v_transform_reduce_outgoing_e(
      handle,
      graph_view,
      edge_src_dummy_property_t{}.view(),
      edge_dst_dummy_property_t{}.view(),
      edge_dummy_property_t{}.view(),
      [] __device__(vertex_t, vertex_t, auto, auto, auto) { return edge_t{1}; },
      edge_t{0},
      reduce_op::plus<edge_t>{},
      minor_degrees.data());
  } else {
    per_v_transform_reduce_incoming_e(
      handle,
      graph_view,
      edge_src_dummy_property_t{}.view(),
      edge_dst_dummy_property_t{}.view(),
      edge_dummy_property_t{}.view(),
      [] __device__(vertex_t, vertex_t, auto, auto, auto) { return edge_t{1}; },
      edge_t{0},
      reduce_op::plus<edge_t>{},
      minor_degrees.data());
  }

  return minor_degrees;
}

// FIXME: block size requires tuning
int32_t constexpr count_edge_partition_multi_edges_block_size = 1024;

template <typename vertex_t, typename edge_t, bool multi_gpu>
__global__ static void for_all_major_for_all_nbr_mid_degree(
  edge_partition_device_view_t<vertex_t, edge_t, multi_gpu> edge_partition,
  vertex_t major_range_first,
  vertex_t major_range_last,
  edge_t* count)
{
  auto const tid = threadIdx.x + blockIdx.x * blockDim.x;
  static_assert(count_edge_partition_multi_edges_block_size % raft::warp_size() == 0);
  auto const lane_id = tid % raft::warp_size();
  auto major_start_offset =
    static_cast<size_t>(major_range_first - edge_partition.major_range_first());
  size_t idx = static_cast<size_t>(tid / raft::warp_size());

  using BlockReduce = cub::BlockReduce<edge_t, count_edge_partition_multi_edges_block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  property_op<edge_t, thrust::plus> edge_property_add{};
  edge_t count_sum{0};
  while (idx < static_cast<size_t>(major_range_last - major_range_first)) {
    auto major_offset = static_cast<vertex_t>(major_start_offset + idx);
    vertex_t const* indices{nullptr};
    [[maybe_unused]] edge_t edge_offset{};
    edge_t local_degree{};
    thrust::tie(indices, edge_offset, local_degree) = edge_partition.local_edges(major_offset);
    for (edge_t i = lane_id; i < local_degree; i += raft::warp_size()) {
      if ((i != 0) && (indices[i - 1] == indices[i])) { ++count_sum; }
    }
    idx += gridDim.x * (blockDim.x / raft::warp_size());
  }

  count_sum = BlockReduce(temp_storage).Reduce(count_sum, edge_property_add);
  if (threadIdx.x == 0) { atomic_add(count, count_sum); }
}

template <typename vertex_t, typename edge_t, bool multi_gpu>
__global__ static void for_all_major_for_all_nbr_high_degree(
  edge_partition_device_view_t<vertex_t, edge_t, multi_gpu> edge_partition,
  vertex_t major_range_first,
  vertex_t major_range_last,
  edge_t* count)
{
  auto major_start_offset =
    static_cast<size_t>(major_range_first - edge_partition.major_range_first());
  size_t idx = static_cast<size_t>(blockIdx.x);

  using BlockReduce = cub::BlockReduce<edge_t, count_edge_partition_multi_edges_block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  property_op<edge_t, thrust::plus> edge_property_add{};
  edge_t count_sum{0};
  while (idx < static_cast<size_t>(major_range_last - major_range_first)) {
    auto major_offset = major_start_offset + idx;
    vertex_t const* indices{nullptr};
    [[maybe_unused]] edge_t edge_offset{};
    edge_t local_degree{};
    thrust::tie(indices, edge_offset, local_degree) =
      edge_partition.local_edges(static_cast<vertex_t>(major_offset));
    for (edge_t i = threadIdx.x; i < local_degree; i += blockDim.x) {
      if ((i != 0) && (indices[i - 1] == indices[i])) { ++count_sum; }
    }
    idx += gridDim.x;
  }

  count_sum = BlockReduce(temp_storage).Reduce(count_sum, edge_property_add);
  if (threadIdx.x == 0) { atomic_add(count, count_sum); }
}

template <typename vertex_t, typename edge_t, bool multi_gpu>
edge_t count_edge_partition_multi_edges(
  raft::handle_t const& handle,
  edge_partition_device_view_t<vertex_t, edge_t, multi_gpu> edge_partition,
  std::optional<std::vector<vertex_t>> const& segment_offsets)
{
  auto execution_policy = handle.get_thrust_policy();
  if (segment_offsets) {
    rmm::device_scalar<edge_t> count(edge_t{0}, handle.get_stream());
    // FIXME: we may further improve performance by 1) concurrently running kernels on different
    // segments; 2) individually tuning block sizes for different segments; and 3) adding one more
    // segment for very high degree vertices and running segmented reduction
    static_assert(detail::num_sparse_segments_per_vertex_partition == 3);
    if ((*segment_offsets)[1] > 0) {
      raft::grid_1d_block_t update_grid((*segment_offsets)[1],
                                        count_edge_partition_multi_edges_block_size,
                                        handle.get_device_properties().maxGridSize[0]);

      cugraph::for_all_major_for_all_nbr_high_degree<<<update_grid.num_blocks,
                                                       update_grid.block_size,
                                                       0,
                                                       handle.get_stream()>>>(
        edge_partition,
        edge_partition.major_range_first(),
        edge_partition.major_range_first() + (*segment_offsets)[1],
        count.data());
    }
    if ((*segment_offsets)[2] - (*segment_offsets)[1] > 0) {
      raft::grid_1d_warp_t update_grid((*segment_offsets)[2] - (*segment_offsets)[1],
                                       count_edge_partition_multi_edges_block_size,
                                       handle.get_device_properties().maxGridSize[0]);

      cugraph::for_all_major_for_all_nbr_mid_degree<<<update_grid.num_blocks,
                                                      update_grid.block_size,
                                                      0,
                                                      handle.get_stream()>>>(
        edge_partition,
        edge_partition.major_range_first() + (*segment_offsets)[1],
        edge_partition.major_range_first() + (*segment_offsets)[2],
        count.data());
    }
    auto ret = count.value(handle.get_stream());
    if ((*segment_offsets)[3] - (*segment_offsets)[2] > 0) {
      ret += thrust::transform_reduce(
        execution_policy,
        thrust::make_counting_iterator(edge_partition.major_range_first()) + (*segment_offsets)[2],
        thrust::make_counting_iterator(edge_partition.major_range_first()) + (*segment_offsets)[3],
        cuda::proclaim_return_type<edge_t>([edge_partition] __device__(auto major) -> edge_t {
          auto major_offset = edge_partition.major_offset_from_major_nocheck(major);
          vertex_t const* indices{nullptr};
          [[maybe_unused]] edge_t edge_offset{};
          edge_t local_degree{};
          thrust::tie(indices, edge_offset, local_degree) =
            edge_partition.local_edges(major_offset);
          edge_t count{0};
          for (edge_t i = 1; i < local_degree; ++i) {  // assumes neighbors are sorted
            if (indices[i - 1] == indices[i]) { ++count; }
          }
          return count;
        }),
        edge_t{0},
        thrust::plus<edge_t>{});
    }
    if (edge_partition.dcs_nzd_vertex_count() && (*(edge_partition.dcs_nzd_vertex_count()) > 0)) {
      ret += thrust::transform_reduce(
        execution_policy,
        thrust::make_counting_iterator(vertex_t{0}),
        thrust::make_counting_iterator(*(edge_partition.dcs_nzd_vertex_count())),
        cuda::proclaim_return_type<edge_t>(
          [edge_partition,
           major_start_offset = (*segment_offsets)[3]] __device__(auto idx) -> edge_t {
            auto major_idx =
              major_start_offset + idx;  // major_offset != major_idx in the hypersparse region
            vertex_t const* indices{nullptr};
            [[maybe_unused]] edge_t edge_offset{};
            edge_t local_degree{};
            thrust::tie(indices, edge_offset, local_degree) = edge_partition.local_edges(major_idx);
            edge_t count{0};
            for (edge_t i = 1; i < local_degree; ++i) {  // assumes neighbors are sorted
              if (indices[i - 1] == indices[i]) { ++count; }
            }
            return count;
          }),
        edge_t{0},
        thrust::plus<edge_t>{});
    }

    return ret;
  } else {
    return thrust::transform_reduce(
      execution_policy,
      thrust::make_counting_iterator(edge_partition.major_range_first()),
      thrust::make_counting_iterator(edge_partition.major_range_first()) +
        edge_partition.major_range_size(),
      cuda::proclaim_return_type<edge_t>([edge_partition] __device__(auto major) -> edge_t {
        auto major_offset = edge_partition.major_offset_from_major_nocheck(major);
        vertex_t const* indices{nullptr};
        [[maybe_unused]] edge_t edge_offset{};
        edge_t local_degree{};
        thrust::tie(indices, edge_offset, local_degree) = edge_partition.local_edges(major_offset);
        edge_t count{0};
        for (edge_t i = 1; i < local_degree; ++i) {  // assumes neighbors are sorted
          if (indices[i - 1] == indices[i]) { ++count; }
        }
        return count;
      }),
      edge_t{0},
      thrust::plus<edge_t>{});
  }
}

template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
std::tuple<rmm::device_uvector<size_t>, std::vector<size_t>>
compute_edge_indices_and_edge_partition_offsets(
  raft::handle_t const& handle,
  graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu> const& graph_view,
  raft::device_span<vertex_t const> edge_majors,
  raft::device_span<vertex_t const> edge_minors)
{
  auto edge_first = thrust::make_zip_iterator(edge_majors.begin(), edge_minors.begin());

  rmm::device_uvector<size_t> edge_indices(edge_majors.size(), handle.get_stream());
  thrust::sequence(handle.get_thrust_policy(), edge_indices.begin(), edge_indices.end(), size_t{0});
  thrust::sort(handle.get_thrust_policy(),
               edge_indices.begin(),
               edge_indices.end(),
               [edge_first] __device__(size_t lhs, size_t rhs) {
                 return *(edge_first + lhs) < *(edge_first + rhs);
               });

  std::vector<size_t> h_major_range_lasts(graph_view.number_of_local_edge_partitions());
  for (size_t i = 0; i < h_major_range_lasts.size(); ++i) {
    if constexpr (store_transposed) {
      h_major_range_lasts[i] = graph_view.local_edge_partition_dst_range_last(i);
    } else {
      h_major_range_lasts[i] = graph_view.local_edge_partition_src_range_last(i);
    }
  }
  rmm::device_uvector<size_t> d_major_range_lasts(h_major_range_lasts.size(), handle.get_stream());
  raft::update_device(d_major_range_lasts.data(),
                      h_major_range_lasts.data(),
                      h_major_range_lasts.size(),
                      handle.get_stream());
  rmm::device_uvector<size_t> d_lower_bounds(d_major_range_lasts.size(), handle.get_stream());
  auto major_first        = edge_majors.begin();
  auto sorted_major_first = thrust::make_transform_iterator(
    edge_indices.begin(),
    cugraph::detail::indirection_t<size_t, decltype(major_first)>{major_first});
  thrust::lower_bound(handle.get_thrust_policy(),
                      sorted_major_first,
                      sorted_major_first + edge_indices.size(),
                      d_major_range_lasts.begin(),
                      d_major_range_lasts.end(),
                      d_lower_bounds.begin());
  std::vector<size_t> edge_partition_offsets(d_lower_bounds.size() + 1, 0);
  raft::update_host(edge_partition_offsets.data() + 1,
                    d_lower_bounds.data(),
                    d_lower_bounds.size(),
                    handle.get_stream());
  handle.sync_stream();

  return std::make_tuple(std::move(edge_indices), edge_partition_offsets);
}

}  // namespace

template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu, std::enable_if_t<multi_gpu>>::
  graph_view_t(std::vector<raft::device_span<edge_t const>> const& edge_partition_offsets,
               std::vector<raft::device_span<vertex_t const>> const& edge_partition_indices,
               std::optional<std::vector<raft::device_span<vertex_t const>>> const&
                 edge_partition_dcs_nzd_vertices,
               std::optional<std::vector<raft::device_span<uint32_t const>>> const&
                 edge_partition_dcs_nzd_range_bitmaps,
               graph_view_meta_t<vertex_t, edge_t, store_transposed, multi_gpu> meta)
  : detail::graph_base_t<vertex_t, edge_t>(
      meta.number_of_vertices, meta.number_of_edges, meta.properties),
    edge_partition_offsets_(edge_partition_offsets),
    edge_partition_indices_(edge_partition_indices),
    edge_partition_dcs_nzd_vertices_(edge_partition_dcs_nzd_vertices),
    edge_partition_dcs_nzd_range_bitmaps_(edge_partition_dcs_nzd_range_bitmaps),
    partition_(meta.partition),
    edge_partition_segment_offsets_(meta.edge_partition_segment_offsets),
    edge_partition_hypersparse_degree_offsets_(meta.edge_partition_hypersparse_degree_offsets),
    local_sorted_unique_edge_srcs_(meta.local_sorted_unique_edge_srcs),
    local_sorted_unique_edge_src_chunk_start_offsets_(
      meta.local_sorted_unique_edge_src_chunk_start_offsets),
    local_sorted_unique_edge_src_chunk_size_(meta.local_sorted_unique_edge_src_chunk_size),
    local_sorted_unique_edge_src_vertex_partition_offsets_(
      meta.local_sorted_unique_edge_src_vertex_partition_offsets),
    local_sorted_unique_edge_dsts_(meta.local_sorted_unique_edge_dsts),
    local_sorted_unique_edge_dst_chunk_start_offsets_(
      meta.local_sorted_unique_edge_dst_chunk_start_offsets),
    local_sorted_unique_edge_dst_chunk_size_(meta.local_sorted_unique_edge_dst_chunk_size),
    local_sorted_unique_edge_dst_vertex_partition_offsets_(
      meta.local_sorted_unique_edge_dst_vertex_partition_offsets)
{
  // cheap error checks

  auto use_dcs = edge_partition_dcs_nzd_vertices.has_value();

  CUGRAPH_EXPECTS(edge_partition_offsets.size() == edge_partition_indices.size(),
                  "Internal Error: edge_partition_offsets.size() and "
                  "edge_partition_indices.size() should coincide.");
  CUGRAPH_EXPECTS(
    !use_dcs || ((*edge_partition_dcs_nzd_vertices).size() == edge_partition_offsets.size()),
    "Internal Error: edge_partition_dcs_nzd_vertices.size() should coincide "
    "with edge_partition_offsets.size() (if used).");

  CUGRAPH_EXPECTS(meta.edge_partition_segment_offsets.size() ==
                    edge_partition_offsets.size() *
                      (detail::num_sparse_segments_per_vertex_partition + (use_dcs ? 3 : 2)),
                  "Internal Error: invalid edge_partition_segment_offsets.size().");

  // skip expensive error checks as this function is only called by graph_t
}

template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu, std::enable_if_t<!multi_gpu>>::
  graph_view_t(raft::device_span<edge_t const> offsets,
               raft::device_span<vertex_t const> indices,
               graph_view_meta_t<vertex_t, edge_t, store_transposed, multi_gpu> meta)
  : detail::graph_base_t<vertex_t, edge_t>(
      meta.number_of_vertices, meta.number_of_edges, meta.properties),
    vertex_partition_range_offsets_(std::vector<vertex_t>{vertex_t{0}, meta.number_of_vertices}),
    offsets_(offsets),
    indices_(indices),
    segment_offsets_(meta.segment_offsets),
    hypersparse_degree_offsets_(meta.hypersparse_degree_offsets)
{
  // cheap error checks

  CUGRAPH_EXPECTS(offsets.size() == static_cast<size_t>(meta.number_of_vertices + 1),
                  "Internal Error: offsets.size() returns an invalid value.");
  CUGRAPH_EXPECTS(indices.size() == static_cast<size_t>(meta.number_of_edges),
                  "Internal Error: indices.size() returns an invalid value.");

  CUGRAPH_EXPECTS(
    !(meta.segment_offsets).has_value() ||
      ((*(meta.segment_offsets)).size() == (detail::num_sparse_segments_per_vertex_partition + 2)),
    "Internal Error: (*(meta.segment_offsets)).size() returns an invalid value.");

  // skip expensive error checks as this function is only called by graph_t
}

template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
edge_t graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu, std::enable_if_t<multi_gpu>>::
  compute_number_of_edges(raft::handle_t const& handle) const
{
  if (this->has_edge_mask()) {
    edge_t ret{};
    auto value_firsts = (*(this->edge_mask_view())).value_firsts();
    auto edge_counts  = (*(this->edge_mask_view())).edge_counts();
    for (size_t i = 0; i < value_firsts.size(); ++i) {
      ret += static_cast<edge_t>(detail::count_set_bits(handle, value_firsts[i], edge_counts[i]));
    }
    ret =
      host_scalar_allreduce(handle.get_comms(), ret, raft::comms::op_t::SUM, handle.get_stream());
    return ret;
  } else {
    return this->number_of_edges_;
  }
}

template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
edge_t graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu, std::enable_if_t<!multi_gpu>>::
  compute_number_of_edges(raft::handle_t const& handle) const
{
  if (this->has_edge_mask()) {
    auto value_firsts = (*(this->edge_mask_view())).value_firsts();
    auto edge_counts  = (*(this->edge_mask_view())).edge_counts();
    assert(value_firsts.size() == 0);
    assert(edge_counts.size() == 0);
    return static_cast<edge_t>(detail::count_set_bits(handle, value_firsts[0], edge_counts[0]));
  } else {
    return this->number_of_edges_;
  }
}

template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
rmm::device_uvector<edge_t>
graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu, std::enable_if_t<multi_gpu>>::
  compute_in_degrees(raft::handle_t const& handle) const
{
  if (store_transposed) {
    std::optional<std::vector<raft::device_span<uint32_t const>>> edge_partition_masks{
      std::nullopt};
    if (this->has_edge_mask()) {
      edge_partition_masks =
        std::vector<raft::device_span<uint32_t const>>(this->edge_partition_offsets_.size());
      auto value_firsts = (*(this->edge_mask_view())).value_firsts();
      auto edge_counts  = (*(this->edge_mask_view())).edge_counts();
      for (size_t i = 0; i < (*edge_partition_masks).size(); ++i) {
        (*edge_partition_masks)[i] =
          raft::device_span<uint32_t const>(value_firsts[i], edge_counts[i]);
      }
    }
    return compute_major_degrees(handle,
                                 this->edge_partition_offsets_,
                                 this->edge_partition_dcs_nzd_vertices_,
                                 edge_partition_masks,
                                 this->partition_,
                                 this->edge_partition_segment_offsets_);
  } else {
    return compute_minor_degrees(handle, *this);
  }
}

template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
rmm::device_uvector<edge_t>
graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu, std::enable_if_t<!multi_gpu>>::
  compute_in_degrees(raft::handle_t const& handle) const
{
  if (store_transposed) {
    return compute_major_degrees(handle,
                                 this->offsets_,
                                 this->has_edge_mask()
                                   ? std::make_optional(raft::device_span<uint32_t const>(
                                       (*(this->edge_mask_view())).value_firsts()[0],
                                       (*(this->edge_mask_view())).edge_counts()[0]))
                                   : std::nullopt,
                                 this->local_vertex_partition_range_size());
  } else {
    return compute_minor_degrees(handle, *this);
  }
}

template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
rmm::device_uvector<edge_t>
graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu, std::enable_if_t<multi_gpu>>::
  compute_out_degrees(raft::handle_t const& handle) const
{
  if (store_transposed) {
    return compute_minor_degrees(handle, *this);
  } else {
    std::optional<std::vector<raft::device_span<uint32_t const>>> edge_partition_masks{
      std::nullopt};
    if (this->has_edge_mask()) {
      edge_partition_masks =
        std::vector<raft::device_span<uint32_t const>>(this->edge_partition_offsets_.size());
      auto value_firsts = (*(this->edge_mask_view())).value_firsts();
      auto edge_counts  = (*(this->edge_mask_view())).edge_counts();
      for (size_t i = 0; i < (*edge_partition_masks).size(); ++i) {
        (*edge_partition_masks)[i] =
          raft::device_span<uint32_t const>(value_firsts[i], edge_counts[i]);
      }
    }
    return compute_major_degrees(handle,
                                 this->edge_partition_offsets_,
                                 this->edge_partition_dcs_nzd_vertices_,
                                 edge_partition_masks,
                                 this->partition_,
                                 this->edge_partition_segment_offsets_);
  }
}

template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
rmm::device_uvector<edge_t>
graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu, std::enable_if_t<!multi_gpu>>::
  compute_out_degrees(raft::handle_t const& handle) const
{
  if (store_transposed) {
    return compute_minor_degrees(handle, *this);
  } else {
    return compute_major_degrees(handle,
                                 this->offsets_,
                                 this->has_edge_mask()
                                   ? std::make_optional(raft::device_span<uint32_t const>(
                                       (*(this->edge_mask_view())).value_firsts()[0],
                                       (*(this->edge_mask_view())).edge_counts()[0]))
                                   : std::nullopt,
                                 this->local_vertex_partition_range_size());
  }
}

template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
edge_t graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu, std::enable_if_t<multi_gpu>>::
  compute_max_in_degree(raft::handle_t const& handle) const
{
  auto in_degrees = compute_in_degrees(handle);
  auto it = thrust::max_element(handle.get_thrust_policy(), in_degrees.begin(), in_degrees.end());
  rmm::device_scalar<edge_t> ret(edge_t{0}, handle.get_stream());
  device_allreduce(handle.get_comms(),
                   it != in_degrees.end() ? it : ret.data(),
                   ret.data(),
                   1,
                   raft::comms::op_t::MAX,
                   handle.get_stream());
  return ret.value(handle.get_stream());
}

template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
edge_t graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu, std::enable_if_t<!multi_gpu>>::
  compute_max_in_degree(raft::handle_t const& handle) const
{
  auto in_degrees = compute_in_degrees(handle);
  auto it = thrust::max_element(handle.get_thrust_policy(), in_degrees.begin(), in_degrees.end());
  edge_t ret{0};
  if (it != in_degrees.end()) { raft::update_host(&ret, it, 1, handle.get_stream()); }
  handle.sync_stream();
  return ret;
}

template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
edge_t graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu, std::enable_if_t<multi_gpu>>::
  compute_max_out_degree(raft::handle_t const& handle) const
{
  auto out_degrees = compute_out_degrees(handle);
  auto it = thrust::max_element(handle.get_thrust_policy(), out_degrees.begin(), out_degrees.end());
  rmm::device_scalar<edge_t> ret(edge_t{0}, handle.get_stream());
  device_allreduce(handle.get_comms(),
                   it != out_degrees.end() ? it : ret.data(),
                   ret.data(),
                   1,
                   raft::comms::op_t::MAX,
                   handle.get_stream());
  return ret.value(handle.get_stream());
}

template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
edge_t graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu, std::enable_if_t<!multi_gpu>>::
  compute_max_out_degree(raft::handle_t const& handle) const
{
  auto out_degrees = compute_out_degrees(handle);
  auto it = thrust::max_element(handle.get_thrust_policy(), out_degrees.begin(), out_degrees.end());
  edge_t ret{0};
  if (it != out_degrees.end()) { raft::update_host(&ret, it, 1, handle.get_stream()); }
  handle.sync_stream();
  return ret;
}

template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
edge_t graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu, std::enable_if_t<multi_gpu>>::
  count_self_loops(raft::handle_t const& handle) const
{
  return count_if_e(
    handle,
    *this,
    edge_src_dummy_property_t{}.view(),
    edge_dst_dummy_property_t{}.view(),
    edge_dummy_property_t{}.view(),
    [] __device__(vertex_t src, vertex_t dst, auto, auto, auto) { return src == dst; });
}

template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
edge_t graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu, std::enable_if_t<!multi_gpu>>::
  count_self_loops(raft::handle_t const& handle) const
{
  return count_if_e(
    handle,
    *this,
    edge_src_dummy_property_t{}.view(),
    edge_dst_dummy_property_t{}.view(),
    edge_dummy_property_t{}.view(),
    [] __device__(vertex_t src, vertex_t dst, auto, auto, auto) { return src == dst; });
}

template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
edge_t graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu, std::enable_if_t<multi_gpu>>::
  count_multi_edges(raft::handle_t const& handle) const
{
  if (!this->is_multigraph()) { return edge_t{0}; }

  edge_t count{0};
  for (size_t i = 0; i < this->number_of_local_edge_partitions(); ++i) {
    count += count_edge_partition_multi_edges(
      handle,
      edge_partition_device_view_t<vertex_t, edge_t, multi_gpu>(this->local_edge_partition_view(i)),
      this->local_edge_partition_segment_offsets(i));
  }

  return host_scalar_allreduce(
    handle.get_comms(), count, raft::comms::op_t::SUM, handle.get_stream());
}

template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
edge_t graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu, std::enable_if_t<!multi_gpu>>::
  count_multi_edges(raft::handle_t const& handle) const
{
  if (!this->is_multigraph()) { return edge_t{0}; }

  return count_edge_partition_multi_edges(
    handle,
    edge_partition_device_view_t<vertex_t, edge_t, multi_gpu>(this->local_edge_partition_view()),
    this->local_edge_partition_segment_offsets());
}

template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
rmm::device_uvector<bool>
graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu, std::enable_if_t<multi_gpu>>::has_edge(
  raft::handle_t const& handle,
  raft::device_span<vertex_t const> edge_srcs,
  raft::device_span<vertex_t const> edge_dsts,
  bool do_expensive_check) const
{
  CUGRAPH_EXPECTS(
    edge_srcs.size() == edge_dsts.size(),
    "Invalid input arguments: edge_srcs.size() does not coincide with edge_dsts.size().");

  auto edge_first =
    thrust::make_zip_iterator(store_transposed ? edge_dsts.begin() : edge_srcs.begin(),
                              store_transposed ? edge_srcs.begin() : edge_dsts.begin());

  if (do_expensive_check) {
    auto num_invalids =
      detail::count_invalid_vertex_pairs(handle, *this, edge_first, edge_first + edge_srcs.size());
    CUGRAPH_EXPECTS(num_invalids == 0,
                    "Invalid input argument: there are invalid edge (src, dst) pairs.");
  }

  auto [edge_indices, edge_partition_offsets] =
    compute_edge_indices_and_edge_partition_offsets(handle,
                                                    *this,
                                                    store_transposed ? edge_dsts : edge_srcs,
                                                    store_transposed ? edge_srcs : edge_dsts);

  auto edge_mask_view = this->edge_mask_view();

  auto sorted_edge_first = thrust::make_transform_iterator(
    edge_indices.begin(), cugraph::detail::indirection_t<size_t, decltype(edge_first)>{edge_first});
  rmm::device_uvector<bool> ret(edge_srcs.size(), handle.get_stream());

  for (size_t i = 0; i < this->number_of_local_edge_partitions(); ++i) {
    auto edge_partition =
      edge_partition_device_view_t<vertex_t, edge_t, multi_gpu>(this->local_edge_partition_view(i));
    auto edge_partition_e_mask =
      edge_mask_view
        ? cuda::std::make_optional<
            detail::edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool>>(
            *edge_mask_view, i)
        : cuda::std::nullopt;
    thrust::transform(handle.get_thrust_policy(),
                      sorted_edge_first + edge_partition_offsets[i],
                      sorted_edge_first + edge_partition_offsets[i + 1],
                      thrust::make_permutation_iterator(
                        ret.begin(), edge_indices.begin() + edge_partition_offsets[i]),
                      [edge_partition, edge_partition_e_mask] __device__(auto e) {
                        auto major     = thrust::get<0>(e);
                        auto minor     = thrust::get<1>(e);
                        auto major_idx = edge_partition.major_idx_from_major_nocheck(major);
                        if (major_idx) {
                          vertex_t const* indices{nullptr};
                          edge_t local_edge_offset{};
                          edge_t local_degree{};
                          thrust::tie(indices, local_edge_offset, local_degree) =
                            edge_partition.local_edges(*major_idx);
                          auto it = thrust::lower_bound(
                            thrust::seq, indices, indices + local_degree, minor);
                          if ((it != indices + local_degree) && *it == minor) {
                            if (edge_partition_e_mask) {
                              return (*edge_partition_e_mask)
                                .get(local_edge_offset + cuda::std::distance(indices, it));
                            } else {
                              return true;
                            }
                          } else {
                            return false;
                          }
                        } else {
                          return false;
                        }
                      });
  }

  return ret;
}

template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
rmm::device_uvector<bool>
graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu, std::enable_if_t<!multi_gpu>>::has_edge(
  raft::handle_t const& handle,
  raft::device_span<vertex_t const> edge_srcs,
  raft::device_span<vertex_t const> edge_dsts,
  bool do_expensive_check) const
{
  CUGRAPH_EXPECTS(
    edge_srcs.size() == edge_dsts.size(),
    "Invalid input arguments: edge_srcs.size() does not coincide with edge_dsts.size().");

  auto edge_first =
    thrust::make_zip_iterator(store_transposed ? edge_dsts.begin() : edge_srcs.begin(),
                              store_transposed ? edge_srcs.begin() : edge_dsts.begin());

  if (do_expensive_check) {
    auto num_invalids =
      detail::count_invalid_vertex_pairs(handle, *this, edge_first, edge_first + edge_srcs.size());
    CUGRAPH_EXPECTS(num_invalids == 0,
                    "Invalid input argument: there are invalid edge (src, dst) pairs.");
  }

  auto edge_mask_view = this->edge_mask_view();

  rmm::device_uvector<bool> ret(edge_srcs.size(), handle.get_stream());

  auto edge_partition =
    edge_partition_device_view_t<vertex_t, edge_t, multi_gpu>(this->local_edge_partition_view());
  auto edge_partition_e_mask =
    edge_mask_view
      ? cuda::std::make_optional<
          detail::edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool>>(
          *edge_mask_view, 0)
      : cuda::std::nullopt;
  thrust::transform(
    handle.get_thrust_policy(),
    edge_first,
    edge_first + edge_srcs.size(),
    ret.begin(),
    [edge_partition, edge_partition_e_mask] __device__(auto e) {
      auto major        = thrust::get<0>(e);
      auto minor        = thrust::get<1>(e);
      auto major_offset = edge_partition.major_offset_from_major_nocheck(major);
      vertex_t const* indices{nullptr};
      edge_t local_edge_offset{};
      edge_t local_degree{};
      thrust::tie(indices, local_edge_offset, local_degree) =
        edge_partition.local_edges(major_offset);
      auto it = thrust::lower_bound(thrust::seq, indices, indices + local_degree, minor);
      if ((it != indices + local_degree) && *it == minor) {
        if (edge_partition_e_mask) {
          return (*edge_partition_e_mask).get(local_edge_offset + cuda::std::distance(indices, it));
        } else {
          return true;
        }
      } else {
        return false;
      }
    });

  return ret;
}

template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
rmm::device_uvector<edge_t>
graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu, std::enable_if_t<multi_gpu>>::
  compute_multiplicity(raft::handle_t const& handle,
                       raft::device_span<vertex_t const> edge_srcs,
                       raft::device_span<vertex_t const> edge_dsts,
                       bool do_expensive_check)
{
  CUGRAPH_EXPECTS(this->is_multigraph(), "Use has_edge() instead for non-multigraphs.");
  CUGRAPH_EXPECTS(
    edge_srcs.size() == edge_dsts.size(),
    "Invalid input arguments: edge_srcs.size() does not coincide with edge_dsts.size().");

  auto edge_first =
    thrust::make_zip_iterator(store_transposed ? edge_dsts.begin() : edge_srcs.begin(),
                              store_transposed ? edge_srcs.begin() : edge_dsts.begin());

  if (do_expensive_check) {
    auto num_invalids =
      detail::count_invalid_vertex_pairs(handle, *this, edge_first, edge_first + edge_srcs.size());
    CUGRAPH_EXPECTS(num_invalids == 0,
                    "Invalid input argument: there are invalid edge (src, dst) pairs.");
  }

  auto [edge_indices, edge_partition_offsets] =
    compute_edge_indices_and_edge_partition_offsets(handle,
                                                    *this,
                                                    store_transposed ? edge_dsts : edge_srcs,
                                                    store_transposed ? edge_srcs : edge_dsts);

  auto edge_mask_view = this->edge_mask_view();

  auto sorted_edge_first = thrust::make_transform_iterator(
    edge_indices.begin(), cugraph::detail::indirection_t<size_t, decltype(edge_first)>{edge_first});
  rmm::device_uvector<edge_t> ret(edge_srcs.size(), handle.get_stream());

  for (size_t i = 0; i < this->number_of_local_edge_partitions(); ++i) {
    auto edge_partition =
      edge_partition_device_view_t<vertex_t, edge_t, multi_gpu>(this->local_edge_partition_view(i));
    auto edge_partition_e_mask =
      edge_mask_view
        ? cuda::std::make_optional<
            detail::edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool>>(
            *edge_mask_view, i)
        : cuda::std::nullopt;
    thrust::transform(
      handle.get_thrust_policy(),
      sorted_edge_first + edge_partition_offsets[i],
      sorted_edge_first + edge_partition_offsets[i + 1],
      thrust::make_permutation_iterator(ret.begin(),
                                        edge_indices.begin() + edge_partition_offsets[i]),
      [edge_partition, edge_partition_e_mask] __device__(auto e) {
        auto major     = thrust::get<0>(e);
        auto minor     = thrust::get<1>(e);
        auto major_idx = edge_partition.major_idx_from_major_nocheck(major);
        if (major_idx) {
          vertex_t const* indices{nullptr};
          edge_t local_edge_offset{};
          edge_t local_degree{};
          thrust::tie(indices, local_edge_offset, local_degree) =
            edge_partition.local_edges(*major_idx);
          auto lower_it = thrust::lower_bound(thrust::seq, indices, indices + local_degree, minor);
          auto upper_it = thrust::upper_bound(thrust::seq, indices, indices + local_degree, minor);
          auto multiplicity = static_cast<edge_t>(cuda::std::distance(lower_it, upper_it));
          if (edge_partition_e_mask && (multiplicity > 0)) {
            multiplicity = static_cast<edge_t>(detail::count_set_bits(
              (*edge_partition_e_mask).value_first(),
              static_cast<size_t>(local_edge_offset + cuda::std::distance(indices, lower_it)),
              static_cast<size_t>(multiplicity)));
          }
          return multiplicity;
        } else {
          return edge_t{0};
        }
      });
  }

  return ret;
}

template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
rmm::device_uvector<edge_t>
graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu, std::enable_if_t<!multi_gpu>>::
  compute_multiplicity(raft::handle_t const& handle,
                       raft::device_span<vertex_t const> edge_srcs,
                       raft::device_span<vertex_t const> edge_dsts,
                       bool do_expensive_check)
{
  CUGRAPH_EXPECTS(this->is_multigraph(), "Use has_edge() instead for non-multigraphs.");
  CUGRAPH_EXPECTS(
    edge_srcs.size() == edge_dsts.size(),
    "Invalid input arguments: edge_srcs.size() does not coincide with edge_dsts.size().");

  auto edge_first =
    thrust::make_zip_iterator(store_transposed ? edge_dsts.begin() : edge_srcs.begin(),
                              store_transposed ? edge_srcs.begin() : edge_dsts.begin());

  if (do_expensive_check) {
    auto num_invalids =
      detail::count_invalid_vertex_pairs(handle, *this, edge_first, edge_first + edge_srcs.size());
    CUGRAPH_EXPECTS(num_invalids == 0,
                    "Invalid input argument: there are invalid edge (src, dst) pairs.");
  }

  auto edge_mask_view = this->edge_mask_view();

  rmm::device_uvector<edge_t> ret(edge_srcs.size(), handle.get_stream());

  auto edge_partition =
    edge_partition_device_view_t<vertex_t, edge_t, multi_gpu>(this->local_edge_partition_view());
  auto edge_partition_e_mask =
    edge_mask_view
      ? cuda::std::make_optional<
          detail::edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool>>(
          *edge_mask_view, 0)
      : cuda::std::nullopt;
  thrust::transform(
    handle.get_thrust_policy(),
    edge_first,
    edge_first + edge_srcs.size(),
    ret.begin(),
    [edge_partition, edge_partition_e_mask] __device__(auto e) {
      auto major        = thrust::get<0>(e);
      auto minor        = thrust::get<1>(e);
      auto major_offset = edge_partition.major_offset_from_major_nocheck(major);
      vertex_t const* indices{nullptr};
      edge_t local_edge_offset{};
      edge_t local_degree{};
      thrust::tie(indices, local_edge_offset, local_degree) =
        edge_partition.local_edges(major_offset);
      auto lower_it     = thrust::lower_bound(thrust::seq, indices, indices + local_degree, minor);
      auto upper_it     = thrust::upper_bound(thrust::seq, indices, indices + local_degree, minor);
      auto multiplicity = static_cast<edge_t>(cuda::std::distance(lower_it, upper_it));
      if (edge_partition_e_mask && (multiplicity > 0)) {
        multiplicity = static_cast<edge_t>(detail::count_set_bits(
          (*edge_partition_e_mask).value_first(),
          static_cast<size_t>(local_edge_offset + cuda::std::distance(indices, lower_it)),
          static_cast<size_t>(multiplicity)));
      }
      return multiplicity;
    });

  return ret;
}

}  // namespace cugraph
