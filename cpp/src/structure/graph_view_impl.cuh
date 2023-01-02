/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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
#include <prims/per_v_transform_reduce_incoming_outgoing_e.cuh>
#include <prims/reduce_op.cuh>
#include <prims/transform_reduce_e.cuh>

#include <cugraph/edge_property.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/optional.h>
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

template <typename vertex_t, typename edge_t>
std::vector<edge_t> update_edge_partition_edge_counts(
  std::vector<edge_t const*> const& edge_partition_offsets,
  std::optional<std::vector<vertex_t>> const& edge_partition_dcs_nzd_vertex_counts,
  partition_t<vertex_t> const& partition,
  std::vector<vertex_t> const& edge_partition_segment_offsets,
  cudaStream_t stream)
{
  std::vector<edge_t> edge_partition_edge_counts(partition.number_of_local_edge_partitions(), 0);
  auto use_dcs = edge_partition_dcs_nzd_vertex_counts.has_value();
  for (size_t i = 0; i < edge_partition_offsets.size(); ++i) {
    auto [major_range_first, major_range_last] = partition.local_edge_partition_major_range(i);
    auto segment_offset_size_per_partition =
      edge_partition_segment_offsets.size() / edge_partition_offsets.size();
    raft::update_host(
      &(edge_partition_edge_counts[i]),
      edge_partition_offsets[i] +
        (use_dcs
           ? (edge_partition_segment_offsets[segment_offset_size_per_partition * i +
                                             detail::num_sparse_segments_per_vertex_partition] +
              (*edge_partition_dcs_nzd_vertex_counts)[i])
           : (major_range_last - major_range_first)),
      1,
      stream);
  }
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  return edge_partition_edge_counts;
}

// compute out-degrees (if we are internally storing edges in the sparse 2D matrix using sources as
// major indices) or in-degrees (otherwise)
template <typename vertex_t, typename edge_t>
rmm::device_uvector<edge_t> compute_major_degrees(
  raft::handle_t const& handle,
  std::vector<edge_t const*> const& edge_partition_offsets,
  std::optional<std::vector<vertex_t const*>> const& edge_partition_dcs_nzd_vertices,
  std::optional<std::vector<vertex_t>> const& edge_partition_dcs_nzd_vertex_counts,
  partition_t<vertex_t> const& partition,
  std::vector<vertex_t> const& edge_partition_segment_offsets)
{
  auto& row_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().row_name());
  auto const row_comm_rank = row_comm.get_rank();
  auto const row_comm_size = row_comm.get_size();
  auto& col_comm           = handle.get_subcomm(cugraph::partition_2d::key_naming_t().col_name());
  auto const col_comm_rank = col_comm.get_rank();
  auto const col_comm_size = col_comm.get_size();

  auto use_dcs = edge_partition_dcs_nzd_vertices.has_value();

  rmm::device_uvector<edge_t> local_degrees(0, handle.get_stream());
  rmm::device_uvector<edge_t> degrees(0, handle.get_stream());

  vertex_t max_num_local_degrees{0};
  for (int i = 0; i < col_comm_size; ++i) {
    auto vertex_partition_idx  = static_cast<size_t>(i * row_comm_size + row_comm_rank);
    auto vertex_partition_size = partition.vertex_partition_range_size(vertex_partition_idx);
    max_num_local_degrees      = std::max(max_num_local_degrees, vertex_partition_size);
    if (i == col_comm_rank) { degrees.resize(vertex_partition_size, handle.get_stream()); }
  }
  local_degrees.resize(max_num_local_degrees, handle.get_stream());
  for (int i = 0; i < col_comm_size; ++i) {
    auto vertex_partition_idx = static_cast<size_t>(i * row_comm_size + row_comm_rank);
    vertex_t major_range_first{};
    vertex_t major_range_last{};
    std::tie(major_range_first, major_range_last) =
      partition.vertex_partition_range(vertex_partition_idx);
    auto p_offsets = edge_partition_offsets[i];
    auto segment_offset_size_per_partition =
      edge_partition_segment_offsets.size() / static_cast<size_t>(col_comm_size);
    auto major_hypersparse_first =
      use_dcs ? major_range_first +
                  edge_partition_segment_offsets[segment_offset_size_per_partition * i +
                                                 detail::num_sparse_segments_per_vertex_partition]
              : major_range_last;
    auto execution_policy = handle.get_thrust_policy();
    thrust::transform(execution_policy,
                      thrust::make_counting_iterator(vertex_t{0}),
                      thrust::make_counting_iterator(major_hypersparse_first - major_range_first),
                      local_degrees.begin(),
                      [p_offsets] __device__(auto i) { return p_offsets[i + 1] - p_offsets[i]; });
    if (use_dcs) {
      auto p_dcs_nzd_vertices   = (*edge_partition_dcs_nzd_vertices)[i];
      auto dcs_nzd_vertex_count = (*edge_partition_dcs_nzd_vertex_counts)[i];
      thrust::fill(execution_policy,
                   local_degrees.begin() + (major_hypersparse_first - major_range_first),
                   local_degrees.begin() + (major_range_last - major_range_first),
                   edge_t{0});
      thrust::for_each(execution_policy,
                       thrust::make_counting_iterator(vertex_t{0}),
                       thrust::make_counting_iterator(dcs_nzd_vertex_count),
                       [p_offsets,
                        p_dcs_nzd_vertices,
                        major_range_first,
                        major_hypersparse_first,
                        local_degrees = local_degrees.data()] __device__(auto i) {
                         auto d = p_offsets[(major_hypersparse_first - major_range_first) + i + 1] -
                                  p_offsets[(major_hypersparse_first - major_range_first) + i];
                         auto v                               = p_dcs_nzd_vertices[i];
                         local_degrees[v - major_range_first] = d;
                       });
    }
    col_comm.reduce(local_degrees.data(),
                    i == col_comm_rank ? degrees.data() : static_cast<edge_t*>(nullptr),
                    static_cast<size_t>(major_range_last - major_range_first),
                    raft::comms::op_t::SUM,
                    i,
                    handle.get_stream());
  }

  return degrees;
}

// compute out-degrees (if we are internally storing edges in the sparse 2D matrix using sources as
// major indices) or in-degrees (otherwise)
template <typename vertex_t, typename edge_t>
rmm::device_uvector<edge_t> compute_major_degrees(raft::handle_t const& handle,
                                                  edge_t const* offsets,
                                                  vertex_t number_of_vertices)
{
  rmm::device_uvector<edge_t> degrees(number_of_vertices, handle.get_stream());
  thrust::tabulate(
    handle.get_thrust_policy(), degrees.begin(), degrees.end(), [offsets] __device__(auto i) {
      return offsets[i + 1] - offsets[i];
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
__global__ void for_all_major_for_all_nbr_mid_degree(
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
  if (threadIdx.x == 0) { atomic_add_edge_op_result(count, count_sum); }
}

template <typename vertex_t, typename edge_t, bool multi_gpu>
__global__ void for_all_major_for_all_nbr_high_degree(
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
  if (threadIdx.x == 0) { atomic_add_edge_op_result(count, count_sum); }
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
        [edge_partition] __device__(auto major) {
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
        },
        edge_t{0},
        thrust::plus<edge_t>{});
    }
    if (edge_partition.dcs_nzd_vertex_count() && (*(edge_partition.dcs_nzd_vertex_count()) > 0)) {
      ret += thrust::transform_reduce(
        execution_policy,
        thrust::make_counting_iterator(vertex_t{0}),
        thrust::make_counting_iterator(*(edge_partition.dcs_nzd_vertex_count())),
        [edge_partition, major_start_offset = (*segment_offsets)[3]] __device__(auto idx) {
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
        },
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
      [edge_partition] __device__(auto major) {
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
      },
      edge_t{0},
      thrust::plus<edge_t>{});
  }
}

}  // namespace

template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu, std::enable_if_t<multi_gpu>>::
  graph_view_t(raft::handle_t const& handle,
               std::vector<edge_t const*> const& edge_partition_offsets,
               std::vector<vertex_t const*> const& edge_partition_indices,
               std::optional<std::vector<vertex_t const*>> const& edge_partition_dcs_nzd_vertices,
               std::optional<std::vector<vertex_t>> const& edge_partition_dcs_nzd_vertex_counts,
               graph_view_meta_t<vertex_t, edge_t, store_transposed, multi_gpu> meta)
  : detail::graph_base_t<vertex_t, edge_t>(
      handle, meta.number_of_vertices, meta.number_of_edges, meta.properties),
    edge_partition_offsets_(edge_partition_offsets),
    edge_partition_indices_(edge_partition_indices),
    edge_partition_dcs_nzd_vertices_(edge_partition_dcs_nzd_vertices),
    edge_partition_dcs_nzd_vertex_counts_(edge_partition_dcs_nzd_vertex_counts),
    edge_partition_number_of_edges_(
      update_edge_partition_edge_counts(edge_partition_offsets,
                                        edge_partition_dcs_nzd_vertex_counts,
                                        meta.partition,
                                        meta.edge_partition_segment_offsets,
                                        handle.get_stream())),
    partition_(meta.partition),
    edge_partition_segment_offsets_(meta.edge_partition_segment_offsets),
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

  auto const col_comm_size =
    this->handle_ptr()->get_subcomm(cugraph::partition_2d::key_naming_t().col_name()).get_size();

  auto use_dcs = edge_partition_dcs_nzd_vertices.has_value();

  CUGRAPH_EXPECTS(edge_partition_offsets.size() == edge_partition_indices.size(),
                  "Internal Error: edge_partition_offsets.size() and "
                  "edge_partition_indices.size() should coincide.");
  CUGRAPH_EXPECTS(edge_partition_dcs_nzd_vertex_counts.has_value() == use_dcs,
                  "edge_partition_dcs_nzd_vertices.has_value() and "
                  "edge_partition_dcs_nzd_vertex_counts.has_value() should coincide");
  CUGRAPH_EXPECTS(!use_dcs || ((*edge_partition_dcs_nzd_vertices).size() ==
                               (*edge_partition_dcs_nzd_vertex_counts).size()),
                  "Internal Error: edge_partition_dcs_nzd_vertices.size() and "
                  "edge_partition_dcs_nzd_vertex_counts.size() should coincide (if used).");
  CUGRAPH_EXPECTS(
    !use_dcs || ((*edge_partition_dcs_nzd_vertices).size() == edge_partition_offsets.size()),
    "Internal Error: edge_partition_dcs_nzd_vertices.size() should coincide "
    "with edge_partition_offsets.size() (if used).");

  CUGRAPH_EXPECTS(edge_partition_offsets.size() == static_cast<size_t>(col_comm_size),
                  "Internal Error: erroneous edge_partition_offsets.size().");

  CUGRAPH_EXPECTS(
    meta.edge_partition_segment_offsets.size() ==
      col_comm_size * (detail::num_sparse_segments_per_vertex_partition + (use_dcs ? 3 : 2)),
    "Internal Error: invalid edge_partition_segment_offsets.size().");

  // skip expensive error checks as this function is only called by graph_t
}

template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu, std::enable_if_t<!multi_gpu>>::
  graph_view_t(raft::handle_t const& handle,
               edge_t const* offsets,
               vertex_t const* indices,
               graph_view_meta_t<vertex_t, edge_t, store_transposed, multi_gpu> meta)
  : detail::graph_base_t<vertex_t, edge_t>(
      handle, meta.number_of_vertices, meta.number_of_edges, meta.properties),
    offsets_(offsets),
    indices_(indices),
    segment_offsets_(meta.segment_offsets)
{
  // cheap error checks

  CUGRAPH_EXPECTS(
    !(meta.segment_offsets).has_value() ||
      ((*(meta.segment_offsets)).size() == (detail::num_sparse_segments_per_vertex_partition + 2)),
    "Internal Error: (*(meta.segment_offsets)).size() returns an invalid value.");

  // skip expensive error checks as this function is only called by graph_t
}

template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
rmm::device_uvector<edge_t>
graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu, std::enable_if_t<multi_gpu>>::
  compute_in_degrees(raft::handle_t const& handle) const
{
  if (store_transposed) {
    return compute_major_degrees(handle,
                                 this->edge_partition_offsets_,
                                 this->edge_partition_dcs_nzd_vertices_,
                                 this->edge_partition_dcs_nzd_vertex_counts_,
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
    return compute_major_degrees(handle, this->offsets_, this->local_vertex_partition_range_size());
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
    return compute_major_degrees(handle,
                                 this->edge_partition_offsets_,
                                 this->edge_partition_dcs_nzd_vertices_,
                                 this->edge_partition_dcs_nzd_vertex_counts_,
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
    return compute_major_degrees(handle, this->offsets_, this->local_vertex_partition_range_size());
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
  return transform_reduce_e(
    handle,
    *this,
    edge_src_dummy_property_t{}.view(),
    edge_dst_dummy_property_t{}.view(),
    edge_dummy_property_t{}.view(),
    [] __device__(vertex_t src, vertex_t dst, auto, auto, auto) {
      return src == dst ? edge_t{1} : edge_t{0};
    },
    edge_t{0});
}

template <typename vertex_t, typename edge_t, bool store_transposed, bool multi_gpu>
edge_t graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu, std::enable_if_t<!multi_gpu>>::
  count_self_loops(raft::handle_t const& handle) const
{
  return transform_reduce_e(
    handle,
    *this,
    edge_src_dummy_property_t{}.view(),
    edge_dst_dummy_property_t{}.view(),
    edge_dummy_property_t{}.view(),
    [] __device__(vertex_t src, vertex_t dst, auto, auto, auto) {
      return src == dst ? edge_t{1} : edge_t{0};
    },
    edge_t{0});
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

}  // namespace cugraph
