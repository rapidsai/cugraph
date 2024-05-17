/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include "prims/detail/partition_v_frontier.cuh"
#include "prims/detail/transform_v_frontier_e.cuh"
#include "prims/property_op_utils.cuh"

#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/edge_partition_edge_property_device_view.cuh>
#include <cugraph/edge_partition_endpoint_property_device_view.cuh>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/mask_utils.cuh>
#include <cugraph/utilities/misc_utils.cuh>
#include <cugraph/utilities/shuffle_comm.cuh>
#include <cugraph/vertex_partition_device_view.cuh>

#include <raft/random/rng.cuh>
#ifndef NO_CUGRAPH_OPS
#include <cugraph-ops/graph/sampling.hpp>
#endif

#include <cub/cub.cuh>
#include <cuda/atomic>
#include <cuda/functional>
#include <thrust/adjacent_difference.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/optional.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include <optional>
#include <tuple>

namespace cugraph {

namespace detail {

int32_t constexpr sample_and_compute_local_nbr_indices_block_size = 256;

size_t constexpr compute_valid_local_nbr_count_inclusive_sum_local_degree_threshold =
  packed_bools_per_word() *
  size_t{4} /* tuning parameter */;  // minimum local degree to compute inclusive sums of valid
                                     // local neighbors per word to accelerate finding n'th local
                                     // neighbor vertex
size_t constexpr compute_valid_local_nbr_count_inclusive_sum_mid_local_degree_threshold =
  packed_bools_per_word() * static_cast<size_t>(raft::warp_size()) *
  size_t{4} /* tuning parameter */;  // minimum local degree to use a CUDA warp
size_t constexpr compute_valid_local_nbr_count_inclusive_sum_high_local_degree_threshold =
  packed_bools_per_word() * static_cast<size_t>(sample_and_compute_local_nbr_indices_block_size) *
  size_t{4} /* tuning parameter */;  // minimum local degree to use a CUDA block

template <typename value_t>
struct compute_local_value_displacements_and_global_value_t {
  raft::device_span<value_t const> gathered_local_values{};
  raft::device_span<value_t>
    partitioned_local_value_displacements{};  // one partition per gpu in the same minor_comm
  raft::device_span<value_t> global_values{};
  int minor_comm_size{};

  __device__ void operator()(size_t i) const
  {
    constexpr int buffer_size = 8;  // tuning parameter
    value_t displacements[buffer_size];
    value_t sum{0};
    for (int round = 0; round < (minor_comm_size + buffer_size - 1) / buffer_size; ++round) {
      auto loop_count = std::min(buffer_size, minor_comm_size - round * buffer_size);
      for (int j = 0; j < loop_count; ++j) {
        displacements[j] = sum;
        sum += gathered_local_values[i + (round * buffer_size + j) * global_values.size()];
      }
      thrust::copy(
        thrust::seq,
        displacements,
        displacements + loop_count,
        partitioned_local_value_displacements.begin() + i * minor_comm_size + round * buffer_size);
    }
    global_values[i] = sum;
  }
};

// convert a (neighbor value, key index) pair  to a (minor_comm_rank, intra-partition offset, local
// neighbor value, key index) quadruplet, minor_comm_rank is set to -1 if an neighbor value is
// invalid
template <typename value_t>
struct convert_pair_to_quadruplet_t {
  raft::device_span<value_t const>
    partitioned_local_value_displacements{};  // one partition per gpu in the same minor_comm
  raft::device_span<size_t> tx_counts{};
  size_t stride{};
  int minor_comm_size{};
  value_t invalid_value{};

  __device__ thrust::tuple<int, size_t, value_t, size_t> operator()(
    thrust::tuple<value_t, size_t> pair) const
  {
    auto nbr_value       = thrust::get<0>(pair);
    auto key_idx         = thrust::get<1>(pair);
    auto local_nbr_value = nbr_value;
    int minor_comm_rank{-1};
    size_t intra_partition_offset{};
    if (nbr_value != invalid_value) {
      auto displacement_first =
        partitioned_local_value_displacements.begin() + key_idx * minor_comm_size;
      minor_comm_rank =
        static_cast<int>(thrust::distance(
          displacement_first,
          thrust::upper_bound(
            thrust::seq, displacement_first, displacement_first + minor_comm_size, nbr_value))) -
        1;
      local_nbr_value -= *(displacement_first + minor_comm_rank);
      cuda::atomic_ref<size_t, cuda::thread_scope_device> counter(tx_counts[minor_comm_rank]);
      intra_partition_offset = counter.fetch_add(size_t{1}, cuda::std::memory_order_relaxed);
    }
    return thrust::make_tuple(minor_comm_rank, intra_partition_offset, local_nbr_value, key_idx);
  }
};

struct shuffle_index_compute_offset_t {
  raft::device_span<int const> minor_comm_ranks{};
  raft::device_span<size_t const> intra_partition_displacements{};
  raft::device_span<size_t const> tx_displacements{};

  __device__ size_t operator()(size_t i) const
  {
    auto minor_comm_rank = minor_comm_ranks[i];
    assert(minor_comm_rank != -1);
    return tx_displacements[minor_comm_rank] + intra_partition_displacements[i];
  }
};

template <typename GraphViewType, typename EdgePartitionEdgeMaskWrapper, typename VertexIterator>
struct find_nth_valid_nbr_idx_t {
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu> edge_partition{};
  EdgePartitionEdgeMaskWrapper edge_partition_e_mask;
  VertexIterator major_first{};
  thrust::tuple<raft::device_span<size_t const>, raft::device_span<edge_t const>>
    major_valid_local_nbr_count_inclusive_sums{};

  __device__ edge_t operator()(thrust::tuple<edge_t, size_t> pair) const
  {
    edge_t local_nbr_idx = thrust::get<0>(pair);
    size_t major_idx     = thrust::get<1>(pair);
    auto major           = *(major_first + major_idx);
    auto major_offset    = edge_partition.major_offset_from_major_nocheck(major);
    vertex_t const* indices{nullptr};
    edge_t edge_offset{0};
    [[maybe_unused]] edge_t local_degree{0};
    if constexpr (GraphViewType::is_multi_gpu) {
      auto major_hypersparse_first = edge_partition.major_hypersparse_first();
      if (major_hypersparse_first && (major >= *major_hypersparse_first)) {
        auto major_hypersparse_idx = edge_partition.major_hypersparse_idx_from_major_nocheck(major);
        if (major_hypersparse_idx) {
          thrust::tie(indices, edge_offset, local_degree) = edge_partition.local_edges(
            edge_partition.major_offset_from_major_nocheck(*major_hypersparse_first) +
            *major_hypersparse_idx);
        }
      } else {
        thrust::tie(indices, edge_offset, local_degree) = edge_partition.local_edges(major_offset);
      }
    } else {
      thrust::tie(indices, edge_offset, local_degree) = edge_partition.local_edges(major_offset);
    }

    if (local_degree < compute_valid_local_nbr_count_inclusive_sum_local_degree_threshold) {
      local_nbr_idx = find_nth_set_bits(
        (*edge_partition_e_mask).value_first(), edge_offset, local_degree, local_nbr_idx + 1);
    } else {
      auto inclusive_sum_first = thrust::get<1>(major_valid_local_nbr_count_inclusive_sums).begin();
      auto start_offset = thrust::get<0>(major_valid_local_nbr_count_inclusive_sums)[major_idx];
      auto end_offset   = thrust::get<0>(major_valid_local_nbr_count_inclusive_sums)[major_idx + 1];
      auto word_idx =
        static_cast<edge_t>(thrust::distance(inclusive_sum_first + start_offset,
                                             thrust::upper_bound(thrust::seq,
                                                                 inclusive_sum_first + start_offset,
                                                                 inclusive_sum_first + end_offset,
                                                                 local_nbr_idx)));
      local_nbr_idx =
        word_idx * packed_bools_per_word() +
        find_nth_set_bits(
          (*edge_partition_e_mask).value_first(),
          edge_offset + word_idx * packed_bools_per_word(),
          local_degree - word_idx * packed_bools_per_word(),
          (local_nbr_idx + 1) -
            ((word_idx > 0) ? *(inclusive_sum_first + start_offset + word_idx - 1) : edge_t{0}));
    }
    return local_nbr_idx;
  }
};

template <typename vertex_t, typename edge_t, bool multi_gpu>
__global__ static void compute_valid_local_nbr_count_inclusive_sums_mid_local_degree(
  edge_partition_device_view_t<vertex_t, edge_t, multi_gpu> edge_partition,
  edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool> edge_partition_e_mask,
  vertex_t const* edge_partition_frontier_major_first,
  raft::device_span<size_t const> inclusive_sum_offsets,
  raft::device_span<size_t const> frontier_indices,
  raft::device_span<edge_t> inclusive_sums)
{
  static_assert(sample_and_compute_local_nbr_indices_block_size % raft::warp_size() == 0);

  auto const tid     = threadIdx.x + blockIdx.x * blockDim.x;
  auto const lane_id = tid % raft::warp_size();

  auto idx = static_cast<size_t>(tid / raft::warp_size());

  using WarpScan = cub::WarpScan<edge_t, raft::warp_size()>;
  __shared__ typename WarpScan::TempStorage temp_storage;

  while (idx < frontier_indices.size()) {
    auto frontier_idx = frontier_indices[idx];
    auto major        = *(edge_partition_frontier_major_first + frontier_idx);
    vertex_t major_idx{};
    if constexpr (multi_gpu) {
      major_idx = *(edge_partition.major_idx_from_major_nocheck(major));
    } else {
      major_idx = edge_partition.major_offset_from_major_nocheck(major);
    }
    auto edge_offset  = edge_partition.local_offset(major_idx);
    auto local_degree = edge_partition.local_degree(major_idx);

    auto start_offset       = inclusive_sum_offsets[frontier_idx];
    auto end_offset         = inclusive_sum_offsets[frontier_idx + 1];
    auto num_inclusive_sums = end_offset - start_offset;
    auto rounded_up_num_inclusive_sums =
      ((num_inclusive_sums + raft::warp_size() - 1) / raft::warp_size()) * raft::warp_size();
    edge_t sum{0};
    for (size_t j = lane_id; j <= rounded_up_num_inclusive_sums; j += raft::warp_size()) {
      auto inc =
        (j < num_inclusive_sums)
          ? static_cast<edge_t>(count_set_bits(
              edge_partition_e_mask.value_first(),
              edge_offset + packed_bools_per_word() * j,
              cuda::std::min(packed_bools_per_word(), local_degree - packed_bools_per_word() * j)))
          : edge_t{0};
      WarpScan(temp_storage).InclusiveSum(inc, inc);
      inclusive_sums[start_offset + j] = sum + inc;
      sum += __shfl_sync(raft::warp_full_mask(), inc, raft::warp_size() - 1);
    }

    idx += gridDim.x * (blockDim.x / raft::warp_size());
  }
}

template <typename vertex_t, typename edge_t, bool multi_gpu>
__global__ static void compute_valid_local_nbr_count_inclusive_sums_high_local_degree(
  edge_partition_device_view_t<vertex_t, edge_t, multi_gpu> edge_partition,
  edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool> edge_partition_e_mask,
  vertex_t const* edge_partition_frontier_major_first,
  raft::device_span<size_t const> inclusive_sum_offsets,
  raft::device_span<size_t const> frontier_indices,
  raft::device_span<edge_t> inclusive_sums)
{
  static_assert(sample_and_compute_local_nbr_indices_block_size % raft::warp_size() == 0);

  auto idx = static_cast<size_t>(blockIdx.x);

  using BlockScan = cub::BlockScan<edge_t, sample_and_compute_local_nbr_indices_block_size>;
  __shared__ typename BlockScan::TempStorage temp_storage;

  __shared__ edge_t sum;

  while (idx < frontier_indices.size()) {
    auto frontier_idx = frontier_indices[idx];
    auto major        = *(edge_partition_frontier_major_first + frontier_idx);
    vertex_t major_idx{};
    if constexpr (multi_gpu) {
      major_idx = *(edge_partition.major_idx_from_major_nocheck(major));
    } else {
      major_idx = edge_partition.major_offset_from_major_nocheck(major);
    }
    auto edge_offset  = edge_partition.local_offset(major_idx);
    auto local_degree = edge_partition.local_degree(major_idx);

    auto start_offset       = inclusive_sum_offsets[frontier_idx];
    auto end_offset         = inclusive_sum_offsets[frontier_idx + 1];
    auto num_inclusive_sums = end_offset - start_offset;
    auto rounded_up_num_inclusive_sums =
      ((num_inclusive_sums + sample_and_compute_local_nbr_indices_block_size - 1) /
       sample_and_compute_local_nbr_indices_block_size) *
      sample_and_compute_local_nbr_indices_block_size;
    if (threadIdx.x == sample_and_compute_local_nbr_indices_block_size - 1) { sum = 0; }
    for (size_t j = threadIdx.x; j <= rounded_up_num_inclusive_sums; j += blockDim.x) {
      auto inc =
        (j < num_inclusive_sums)
          ? static_cast<edge_t>(count_set_bits(
              edge_partition_e_mask.value_first(),
              edge_offset + packed_bools_per_word() * j,
              cuda::std::min(packed_bools_per_word(), local_degree - packed_bools_per_word() * j)))
          : edge_t{0};
      BlockScan(temp_storage).InclusiveSum(inc, inc);
      inclusive_sums[start_offset + j] = sum + inc;
      __syncthreads();
      if (threadIdx.x == sample_and_compute_local_nbr_indices_block_size - 1) { sum += inc; }
    }

    idx += gridDim.x;
  }
}

template <typename value_t>
std::tuple<rmm::device_uvector<value_t>, rmm::device_uvector<value_t>>
compute_frontier_value_sums_and_partitioned_local_value_sum_displacements(
  raft::handle_t const& handle,
  raft::device_span<value_t const> aggregate_local_frontier_local_value_sums,
  std::vector<size_t> const& local_frontier_displacements,
  std::vector<size_t> const& local_frontier_sizes)
{
  auto& minor_comm     = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
  auto minor_comm_rank = minor_comm.get_rank();
  auto minor_comm_size = minor_comm.get_size();

  rmm::device_uvector<value_t> frontier_gathered_local_value_sums(0, handle.get_stream());
  std::tie(frontier_gathered_local_value_sums, std::ignore) =
    shuffle_values(minor_comm,
                   aggregate_local_frontier_local_value_sums.begin(),
                   local_frontier_sizes,
                   handle.get_stream());

  rmm::device_uvector<value_t> frontier_value_sums(local_frontier_sizes[minor_comm_rank],
                                                   handle.get_stream());
  rmm::device_uvector<value_t> frontier_partitioned_local_value_sum_displacements(
    frontier_value_sums.size() * minor_comm_size, handle.get_stream());

  thrust::for_each(
    handle.get_thrust_policy(),
    thrust::make_counting_iterator(size_t{0}),
    thrust::make_counting_iterator(frontier_value_sums.size()),
    compute_local_value_displacements_and_global_value_t<value_t>{
      raft::device_span<value_t const>(frontier_gathered_local_value_sums.data(),
                                       frontier_gathered_local_value_sums.size()),
      raft::device_span<value_t>(frontier_partitioned_local_value_sum_displacements.data(),
                                 frontier_partitioned_local_value_sum_displacements.size()),
      raft::device_span<value_t>(frontier_value_sums.data(), frontier_value_sums.size()),
      minor_comm_size});

  return std::make_tuple(std::move(frontier_value_sums),
                         std::move(frontier_partitioned_local_value_sum_displacements));
}

template <typename GraphViewType, typename VertexIterator>
std::vector<
  std::tuple<rmm::device_uvector<size_t>, rmm::device_uvector<typename GraphViewType::edge_type>>>
compute_valid_local_nbr_count_inclusive_sums(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  VertexIterator aggregate_local_frontier_major_first,
  std::vector<size_t> const& local_frontier_displacements,
  std::vector<size_t> const& local_frontier_sizes)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  static_assert(
    std::is_same_v<typename thrust::iterator_traits<VertexIterator>::value_type, vertex_t>);

  auto edge_mask_view = graph_view.edge_mask_view();

  std::vector<std::tuple<rmm::device_uvector<size_t>, rmm::device_uvector<edge_t>>>
    local_frontier_valid_local_nbr_count_inclusive_sums{};
  local_frontier_valid_local_nbr_count_inclusive_sums.reserve(
    graph_view.number_of_local_edge_partitions());

  for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
    auto edge_partition =
      edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
        graph_view.local_edge_partition_view(i));
    auto edge_partition_e_mask =
      edge_mask_view
        ? thrust::make_optional<
            detail::edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool>>(
            *edge_mask_view, i)
        : thrust::nullopt;

    auto edge_partition_frontier_major_first =
      aggregate_local_frontier_major_first + local_frontier_displacements[i];

    auto edge_partition_local_degrees = edge_partition.compute_local_degrees(
      edge_partition_frontier_major_first,
      edge_partition_frontier_major_first + local_frontier_sizes[i],
      handle.get_stream());
    auto inclusive_sum_offsets =
      rmm::device_uvector<size_t>(local_frontier_sizes[i] + 1, handle.get_stream());
    inclusive_sum_offsets.set_element_to_zero_async(0, handle.get_stream());
    auto size_first = thrust::make_transform_iterator(
      edge_partition_local_degrees.begin(),
      cuda::proclaim_return_type<size_t>([] __device__(edge_t local_degree) {
        return static_cast<size_t>((local_degree + packed_bools_per_word() - 1) /
                                   packed_bools_per_word());
      }));
    thrust::inclusive_scan(handle.get_thrust_policy(),
                           size_first,
                           size_first + edge_partition_local_degrees.size(),
                           inclusive_sum_offsets.begin() + 1);

    auto [edge_partition_frontier_indices, frontier_partition_offsets] = partition_v_frontier(
      handle,
      edge_partition_local_degrees.begin(),
      edge_partition_local_degrees.end(),
      std::vector<edge_t>{
        static_cast<edge_t>(compute_valid_local_nbr_count_inclusive_sum_local_degree_threshold),
        static_cast<edge_t>(compute_valid_local_nbr_count_inclusive_sum_mid_local_degree_threshold),
        static_cast<edge_t>(
          compute_valid_local_nbr_count_inclusive_sum_high_local_degree_threshold)});

    rmm::device_uvector<edge_t> inclusive_sums(
      inclusive_sum_offsets.back_element(handle.get_stream()), handle.get_stream());

    thrust::for_each(
      handle.get_thrust_policy(),
      edge_partition_frontier_indices.begin() + frontier_partition_offsets[1],
      edge_partition_frontier_indices.begin() + frontier_partition_offsets[2],
      [edge_partition,
       edge_partition_e_mask,
       edge_partition_frontier_major_first,
       inclusive_sum_offsets = raft::device_span<size_t const>(inclusive_sum_offsets.data(),
                                                               inclusive_sum_offsets.size()),
       inclusive_sums        = raft::device_span<edge_t>(inclusive_sums.data(),
                                                  inclusive_sums.size())] __device__(size_t i) {
        auto major = *(edge_partition_frontier_major_first + i);
        vertex_t major_idx{};
        if constexpr (GraphViewType::is_multi_gpu) {
          major_idx = *(edge_partition.major_idx_from_major_nocheck(major));
        } else {
          major_idx = edge_partition.major_offset_from_major_nocheck(major);
        }
        auto edge_offset  = edge_partition.local_offset(major_idx);
        auto local_degree = edge_partition.local_degree(major_idx);
        edge_t sum{0};
        auto start_offset = inclusive_sum_offsets[i];
        auto end_offset   = inclusive_sum_offsets[i + 1];
        for (size_t j = 0; j < end_offset - start_offset; ++j) {
          sum += count_set_bits(
            (*edge_partition_e_mask).value_first(),
            edge_offset + packed_bools_per_word() * j,
            cuda::std::min(packed_bools_per_word(), local_degree - packed_bools_per_word() * j));
          inclusive_sums[start_offset + j] = sum;
        }
      });

    auto mid_partition_size = frontier_partition_offsets[3] - frontier_partition_offsets[2];
    if (mid_partition_size > 0) {
      raft::grid_1d_warp_t update_grid(mid_partition_size,
                                       sample_and_compute_local_nbr_indices_block_size,
                                       handle.get_device_properties().maxGridSize[0]);
      compute_valid_local_nbr_count_inclusive_sums_mid_local_degree<<<update_grid.num_blocks,
                                                                      update_grid.block_size,
                                                                      0,
                                                                      handle.get_stream()>>>(
        edge_partition,
        *edge_partition_e_mask,
        edge_partition_frontier_major_first,
        raft::device_span<size_t const>(inclusive_sum_offsets.data(), inclusive_sum_offsets.size()),
        raft::device_span<size_t const>(
          edge_partition_frontier_indices.data() + frontier_partition_offsets[2],
          frontier_partition_offsets[3] - frontier_partition_offsets[2]),
        raft::device_span<edge_t>(inclusive_sums.data(), inclusive_sums.size()));
    }

    auto high_partition_size = frontier_partition_offsets[4] - frontier_partition_offsets[3];
    if (high_partition_size > 0) {
      raft::grid_1d_block_t update_grid(high_partition_size,
                                        sample_and_compute_local_nbr_indices_block_size,
                                        handle.get_device_properties().maxGridSize[0]);
      compute_valid_local_nbr_count_inclusive_sums_high_local_degree<<<update_grid.num_blocks,
                                                                       update_grid.block_size,
                                                                       0,
                                                                       handle.get_stream()>>>(
        edge_partition,
        *edge_partition_e_mask,
        edge_partition_frontier_major_first,
        raft::device_span<size_t const>(inclusive_sum_offsets.data(), inclusive_sum_offsets.size()),
        raft::device_span<size_t const>(
          edge_partition_frontier_indices.data() + frontier_partition_offsets[3],
          frontier_partition_offsets[4] - frontier_partition_offsets[3]),
        raft::device_span<edge_t>(inclusive_sums.data(), inclusive_sums.size()));
    }

    local_frontier_valid_local_nbr_count_inclusive_sums.push_back(
      std::make_tuple(std::move(inclusive_sum_offsets), std::move(inclusive_sums)));
  }

  return local_frontier_valid_local_nbr_count_inclusive_sums;
}

template <typename edge_t>
rmm::device_uvector<edge_t> compute_uniform_sampling_index_without_replacement(
  raft::handle_t const& handle,
  rmm::device_uvector<edge_t>&& frontier_degrees,
  raft::random::RngState& rng_state,
  size_t K)
{
#ifndef NO_CUGRAPH_OPS
  edge_t mid_partition_degree_range_last = static_cast<edge_t>(K * 10);  // tuning parameter
  assert(mid_partition_degree_range_last > K);
  size_t high_partition_oversampling_K = K * 2;  // tuning parameter
  assert(high_partition_oversampling_K > K);

  auto [frontier_indices, frontier_partition_offsets] = partition_v_frontier(
    handle,
    frontier_degrees.begin(),
    frontier_degrees.end(),
    std::vector<edge_t>{static_cast<edge_t>(K + 1), mid_partition_degree_range_last + 1});

  rmm::device_uvector<edge_t> nbr_indices(frontier_degrees.size() * K, handle.get_stream());

  auto low_partition_size = frontier_partition_offsets[1];
  if (low_partition_size > 0) {
    thrust::for_each(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator(size_t{0}),
      thrust::make_counting_iterator(low_partition_size * K),
      [K,
       frontier_indices =
         raft::device_span<size_t const>(frontier_indices.data(), low_partition_size),
       frontier_degrees =
         raft::device_span<edge_t const>(frontier_degrees.data(), frontier_degrees.size()),
       nbr_indices = raft::device_span<edge_t>(nbr_indices.data(), nbr_indices.size()),
       invalid_idx = cugraph::ops::graph::INVALID_ID<edge_t>] __device__(size_t i) {
        auto frontier_idx = frontier_indices[i / K];
        auto degree       = frontier_degrees[frontier_idx];
        auto sample_idx   = static_cast<edge_t>(i % K);
        nbr_indices[frontier_idx * K + sample_idx] =
          (sample_idx < degree) ? sample_idx : invalid_idx;
      });
  }

  auto mid_partition_size = frontier_partition_offsets[2] - frontier_partition_offsets[1];
  if (mid_partition_size > 0) {
    // FIXME: tmp_degrees & tmp_nbr_indices can be avoided if we customize
    // cugraph::ops::get_sampling_index
    rmm::device_uvector<edge_t> tmp_degrees(mid_partition_size, handle.get_stream());
    rmm::device_uvector<edge_t> tmp_nbr_indices(mid_partition_size * K, handle.get_stream());
    thrust::gather(handle.get_thrust_policy(),
                   frontier_indices.begin() + frontier_partition_offsets[1],
                   frontier_indices.begin() + frontier_partition_offsets[2],
                   frontier_degrees.begin(),
                   tmp_degrees.begin());
    cugraph::ops::graph::get_sampling_index(tmp_nbr_indices.data(),
                                            rng_state,
                                            tmp_degrees.data(),
                                            mid_partition_size,
                                            static_cast<int32_t>(K),
                                            false,
                                            handle.get_stream());
    thrust::for_each(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator(size_t{0}),
      thrust::make_counting_iterator(mid_partition_size * K),
      [K,
       frontier_indices = raft::device_span<size_t const>(
         frontier_indices.data() + frontier_partition_offsets[1], mid_partition_size),
       tmp_nbr_indices = tmp_nbr_indices.data(),
       nbr_indices     = nbr_indices.data()] __device__(size_t i) {
        auto frontier_idx                          = frontier_indices[i / K];
        auto sample_idx                            = static_cast<edge_t>(i % K);
        nbr_indices[frontier_idx * K + sample_idx] = tmp_nbr_indices[i];
      });
  }

  auto high_partition_size = frontier_partition_offsets[3] - frontier_partition_offsets[2];
  if (high_partition_size > 0) {
    // to limit memory footprint ((1 << 20) is a tuning parameter), std::max for forward progress
    // guarantee when high_partition_oversampling_K is exorbitantly large
    auto seeds_to_sort_per_iteration =
      std::max(static_cast<size_t>(handle.get_device_properties().multiProcessorCount * (1 << 20)) /
                 high_partition_oversampling_K,
               size_t{1});

    rmm::device_uvector<edge_t> tmp_nbr_indices(
      seeds_to_sort_per_iteration * high_partition_oversampling_K, handle.get_stream());
    assert(high_partition_oversampling_K * 2 <=
           static_cast<size_t>(std::numeric_limits<int32_t>::max()));
    rmm::device_uvector<int32_t> tmp_sample_indices(
      tmp_nbr_indices.size(),
      handle.get_stream());  // sample indices ([0, high_partition_oversampling_K)) within a segment
                             // (one segment per seed)

    rmm::device_uvector<edge_t> segment_sorted_tmp_nbr_indices(tmp_nbr_indices.size(),
                                                               handle.get_stream());
    rmm::device_uvector<int32_t> segment_sorted_tmp_sample_indices(tmp_nbr_indices.size(),
                                                                   handle.get_stream());

    rmm::device_uvector<std::byte> d_tmp_storage(0, handle.get_stream());
    size_t tmp_storage_bytes{0};

    auto num_chunks =
      (high_partition_size + seeds_to_sort_per_iteration - 1) / seeds_to_sort_per_iteration;
    for (size_t i = 0; i < num_chunks; ++i) {
      size_t num_segments = std::min(seeds_to_sort_per_iteration,
                                     high_partition_size - seeds_to_sort_per_iteration * i);

      rmm::device_uvector<edge_t> unique_counts(num_segments, handle.get_stream());

      std::optional<rmm::device_uvector<size_t>> retry_segment_indices{std::nullopt};
      std::optional<rmm::device_uvector<edge_t>> retry_degrees{std::nullopt};
      std::optional<rmm::device_uvector<edge_t>> retry_nbr_indices{std::nullopt};
      std::optional<rmm::device_uvector<int32_t>> retry_sample_indices{std::nullopt};
      std::optional<rmm::device_uvector<edge_t>> retry_segment_sorted_nbr_indices{std::nullopt};
      std::optional<rmm::device_uvector<int32_t>> retry_segment_sorted_sample_indices{std::nullopt};
      while (true) {
        auto segment_frontier_index_first = frontier_indices.begin() +
                                            frontier_partition_offsets[2] +
                                            seeds_to_sort_per_iteration * i;
        auto segment_frontier_degree_first = thrust::make_transform_iterator(
          segment_frontier_index_first,
          indirection_t<size_t, decltype(frontier_degrees.begin())>{frontier_degrees.begin()});

        if (retry_segment_indices) {
          retry_degrees =
            rmm::device_uvector<edge_t>((*retry_segment_indices).size(), handle.get_stream());
          thrust::gather(handle.get_thrust_policy(),
                         (*retry_segment_indices).begin(),
                         (*retry_segment_indices).end(),
                         segment_frontier_degree_first,
                         (*retry_degrees).begin());
          retry_nbr_indices = rmm::device_uvector<edge_t>(
            (*retry_segment_indices).size() * high_partition_oversampling_K, handle.get_stream());
          retry_sample_indices =
            rmm::device_uvector<int32_t>((*retry_nbr_indices).size(), handle.get_stream());
          retry_segment_sorted_nbr_indices =
            rmm::device_uvector<edge_t>((*retry_nbr_indices).size(), handle.get_stream());
          retry_segment_sorted_sample_indices =
            rmm::device_uvector<int32_t>((*retry_nbr_indices).size(), handle.get_stream());
        }

        if (retry_segment_indices) {
          cugraph::ops::graph::get_sampling_index(
            (*retry_nbr_indices).data(),
            rng_state,
            (*retry_degrees).begin(),
            (*retry_degrees).size(),
            static_cast<int32_t>(high_partition_oversampling_K),
            true,
            handle.get_stream());
        } else {
          // FIXME: this temporary is unnecessary if we update get_sampling_index to take a thrust
          // iterator
          rmm::device_uvector<edge_t> tmp_degrees(num_segments, handle.get_stream());
          thrust::copy(handle.get_thrust_policy(),
                       segment_frontier_degree_first,
                       segment_frontier_degree_first + num_segments,
                       tmp_degrees.begin());
          cugraph::ops::graph::get_sampling_index(
            tmp_nbr_indices.data(),
            rng_state,
            tmp_degrees.data(),
            num_segments,
            static_cast<int32_t>(high_partition_oversampling_K),
            true,
            handle.get_stream());
        }

        if (retry_segment_indices) {
          thrust::for_each(
            handle.get_thrust_policy(),
            thrust::make_counting_iterator(size_t{0}),
            thrust::make_counting_iterator((*retry_segment_indices).size() *
                                           high_partition_oversampling_K),
            [high_partition_oversampling_K,
             unique_counts                  = unique_counts.data(),
             segment_sorted_tmp_nbr_indices = segment_sorted_tmp_nbr_indices.data(),
             retry_segment_indices          = (*retry_segment_indices).data(),
             retry_nbr_indices              = (*retry_nbr_indices).data(),
             retry_sample_indices           = (*retry_sample_indices).data()] __device__(size_t i) {
              auto segment_idx  = retry_segment_indices[i / high_partition_oversampling_K];
              auto sample_idx   = static_cast<edge_t>(i % high_partition_oversampling_K);
              auto unique_count = unique_counts[segment_idx];
              auto output_first = thrust::make_zip_iterator(
                thrust::make_tuple(retry_nbr_indices, retry_sample_indices));
              // sample index for the previously selected neighbor indices should be smaller than
              // the new candidates to ensure that the previously selected neighbor indices will be
              // selected again
              if (sample_idx < unique_count) {
                *(output_first + i) = thrust::make_tuple(
                  segment_sorted_tmp_nbr_indices[segment_idx * high_partition_oversampling_K +
                                                 sample_idx],
                  static_cast<int32_t>(sample_idx));
              } else {
                *(output_first + i) =
                  thrust::make_tuple(retry_nbr_indices[i],
                                     high_partition_oversampling_K + (sample_idx - unique_count));
              }
            });
        } else {
          thrust::tabulate(
            handle.get_thrust_policy(),
            tmp_sample_indices.begin(),
            tmp_sample_indices.begin() + num_segments * high_partition_oversampling_K,
            [high_partition_oversampling_K] __device__(size_t i) {
              return static_cast<int32_t>(i % high_partition_oversampling_K);
            });
        }

        // sort the (sample neighbor index, sample index) pairs (key: sample neighbor index)

        cub::DeviceSegmentedSort::SortPairs(
          static_cast<void*>(nullptr),
          tmp_storage_bytes,
          retry_segment_indices ? (*retry_nbr_indices).data() : tmp_nbr_indices.data(),
          retry_segment_indices ? (*retry_segment_sorted_nbr_indices).data()
                                : segment_sorted_tmp_nbr_indices.data(),
          retry_segment_indices ? (*retry_sample_indices).data() : tmp_sample_indices.data(),
          retry_segment_indices ? (*retry_segment_sorted_sample_indices).data()
                                : segment_sorted_tmp_sample_indices.data(),
          (retry_segment_indices ? (*retry_segment_indices).size() : num_segments) *
            high_partition_oversampling_K,
          retry_segment_indices ? (*retry_segment_indices).size() : num_segments,
          thrust::make_transform_iterator(thrust::make_counting_iterator(size_t{0}),
                                          multiplier_t<size_t>{high_partition_oversampling_K}),
          thrust::make_transform_iterator(thrust::make_counting_iterator(size_t{1}),
                                          multiplier_t<size_t>{high_partition_oversampling_K}),
          handle.get_stream());
        if (tmp_storage_bytes > d_tmp_storage.size()) {
          d_tmp_storage = rmm::device_uvector<std::byte>(tmp_storage_bytes, handle.get_stream());
        }
        cub::DeviceSegmentedSort::SortPairs(
          d_tmp_storage.data(),
          tmp_storage_bytes,
          retry_segment_indices ? (*retry_nbr_indices).data() : tmp_nbr_indices.data(),
          retry_segment_indices ? (*retry_segment_sorted_nbr_indices).data()
                                : segment_sorted_tmp_nbr_indices.data(),
          retry_segment_indices ? (*retry_sample_indices).data() : tmp_sample_indices.data(),
          retry_segment_indices ? (*retry_segment_sorted_sample_indices).data()
                                : segment_sorted_tmp_sample_indices.data(),
          (retry_segment_indices ? (*retry_segment_indices).size() : num_segments) *
            high_partition_oversampling_K,
          retry_segment_indices ? (*retry_segment_indices).size() : num_segments,
          thrust::make_transform_iterator(thrust::make_counting_iterator(size_t{0}),
                                          multiplier_t<size_t>{high_partition_oversampling_K}),
          thrust::make_transform_iterator(thrust::make_counting_iterator(size_t{1}),
                                          multiplier_t<size_t>{high_partition_oversampling_K}),
          handle.get_stream());

        // count the number of unique neighbor indices

        if (retry_segment_indices) {
          thrust::for_each(
            handle.get_thrust_policy(),
            thrust::make_counting_iterator(size_t{0}),
            thrust::make_counting_iterator((*retry_segment_indices).size()),
            [high_partition_oversampling_K,
             unique_counts                   = unique_counts.data(),
             retry_segment_indices           = (*retry_segment_indices).data(),
             retry_segment_sorted_pair_first = thrust::make_zip_iterator(
               thrust::make_tuple((*retry_segment_sorted_nbr_indices).begin(),
                                  (*retry_segment_sorted_sample_indices).begin())),
             segment_sorted_pair_first = thrust::make_zip_iterator(thrust::make_tuple(
               segment_sorted_tmp_nbr_indices.begin(),
               segment_sorted_tmp_sample_indices.begin()))] __device__(size_t i) {
              auto unique_count          = static_cast<edge_t>(thrust::distance(
                retry_segment_sorted_pair_first + high_partition_oversampling_K * i,
                thrust::unique(
                  thrust::seq,
                  retry_segment_sorted_pair_first + high_partition_oversampling_K * i,
                  retry_segment_sorted_pair_first + high_partition_oversampling_K * (i + 1),
                  [] __device__(auto lhs, auto rhs) {
                    return thrust::get<0>(lhs) == thrust::get<0>(rhs);
                  })));
              auto segment_idx           = retry_segment_indices[i];
              unique_counts[segment_idx] = unique_count;
              thrust::copy(
                thrust::seq,
                retry_segment_sorted_pair_first + high_partition_oversampling_K * i,
                retry_segment_sorted_pair_first + high_partition_oversampling_K * i + unique_count,
                segment_sorted_pair_first + high_partition_oversampling_K * segment_idx);
            });
        } else {
          thrust::tabulate(
            handle.get_thrust_policy(),
            unique_counts.begin(),
            unique_counts.end(),
            [high_partition_oversampling_K,
             segment_sorted_pair_first = thrust::make_zip_iterator(thrust::make_tuple(
               segment_sorted_tmp_nbr_indices.begin(),
               segment_sorted_tmp_sample_indices.begin()))] __device__(size_t i) {
              return static_cast<edge_t>(thrust::distance(
                segment_sorted_pair_first + high_partition_oversampling_K * i,
                thrust::unique(thrust::seq,
                               segment_sorted_pair_first + high_partition_oversampling_K * i,
                               segment_sorted_pair_first + high_partition_oversampling_K * (i + 1),
                               [] __device__(auto lhs, auto rhs) {
                                 return thrust::get<0>(lhs) == thrust::get<0>(rhs);
                               })));
            });
        }

        auto num_retry_segments =
          thrust::count_if(handle.get_thrust_policy(),
                           unique_counts.begin(),
                           unique_counts.end(),
                           [K] __device__(auto count) { return count < K; });
        if (num_retry_segments > 0) {
          retry_segment_indices =
            rmm::device_uvector<size_t>(num_retry_segments, handle.get_stream());
          thrust::copy_if(handle.get_thrust_policy(),
                          thrust::make_counting_iterator(size_t{0}),
                          thrust::make_counting_iterator(num_segments),
                          (*retry_segment_indices).begin(),
                          [K, unique_counts = unique_counts.data()] __device__(size_t i) {
                            return unique_counts[i] < K;
                          });
        } else {
          break;
        }
      }

      // sort the segment-sorted (sample index, sample neighbor index) pairs (key: sample index)

      cub::DeviceSegmentedSort::SortPairs(
        static_cast<void*>(nullptr),
        tmp_storage_bytes,
        segment_sorted_tmp_sample_indices.data(),
        tmp_sample_indices.data(),
        segment_sorted_tmp_nbr_indices.data(),
        tmp_nbr_indices.data(),
        num_segments * high_partition_oversampling_K,
        num_segments,
        thrust::make_transform_iterator(thrust::make_counting_iterator(size_t{0}),
                                        multiplier_t<size_t>{high_partition_oversampling_K}),
        thrust::make_transform_iterator(
          thrust::make_counting_iterator(size_t{0}),
          cuda::proclaim_return_type<size_t>(
            [high_partition_oversampling_K, unique_counts = unique_counts.data()] __device__(
              size_t i) { return i * high_partition_oversampling_K + unique_counts[i]; })),
        handle.get_stream());
      if (tmp_storage_bytes > d_tmp_storage.size()) {
        d_tmp_storage = rmm::device_uvector<std::byte>(tmp_storage_bytes, handle.get_stream());
      }
      cub::DeviceSegmentedSort::SortPairs(
        d_tmp_storage.data(),
        tmp_storage_bytes,
        segment_sorted_tmp_sample_indices.data(),
        tmp_sample_indices.data(),
        segment_sorted_tmp_nbr_indices.data(),
        tmp_nbr_indices.data(),
        num_segments * high_partition_oversampling_K,
        num_segments,
        thrust::make_transform_iterator(thrust::make_counting_iterator(size_t{0}),
                                        multiplier_t<size_t>{high_partition_oversampling_K}),
        thrust::make_transform_iterator(
          thrust::make_counting_iterator(size_t{0}),
          cuda::proclaim_return_type<size_t>(
            [high_partition_oversampling_K, unique_counts = unique_counts.data()] __device__(
              size_t i) { return i * high_partition_oversampling_K + unique_counts[i]; })),
        handle.get_stream());

      // copy the neighbor indices back to nbr_indices

      thrust::for_each(
        handle.get_thrust_policy(),
        thrust::make_counting_iterator(size_t{0}),
        thrust::make_counting_iterator(num_segments * K),
        [K,
         high_partition_oversampling_K,
         frontier_indices = frontier_indices.begin() + frontier_partition_offsets[2] +
                            seeds_to_sort_per_iteration * i,
         tmp_nbr_indices = tmp_nbr_indices.data(),
         nbr_indices     = nbr_indices.data()] __device__(size_t i) {
          auto seed_idx   = *(frontier_indices + i / K);
          auto sample_idx = static_cast<edge_t>(i % K);
          *(nbr_indices + seed_idx * K + sample_idx) =
            *(tmp_nbr_indices + (i / K) * high_partition_oversampling_K + sample_idx);
        });
    }
  }

  frontier_degrees.resize(0, handle.get_stream());
  frontier_degrees.shrink_to_fit(handle.get_stream());

  return nbr_indices;
#else
  CUGRAPH_FAIL("unimplemented.");
#endif
}

template <typename edge_t, typename bias_t>
void compute_biased_sampling_index_without_replacement(
  raft::handle_t const& handle,
  std::optional<raft::device_span<size_t const>>
    input_frontier_indices,  // input_biases & input_degree_offsets
                             // are already packed if std::nullopt
  raft::device_span<size_t const> input_degree_offsets,
  raft::device_span<bias_t const> input_biases,
  std::optional<raft::device_span<size_t const>>
    output_frontier_indices,  // output_biases is already packed if std::nullopt
  raft::device_span<edge_t> output_nbr_indices,
  std::optional<raft::device_span<bias_t>> output_keys,
  raft::random::RngState& rng_state,
  size_t K,
  bool jump)
{
  if (jump) {  // Algorithm A-ExpJ
    CUGRAPH_FAIL("unimplemented.");
  } else {  // Algorithm A-Res
    // update packed input degree offsets if input_frontier_indices.has_value() is true

    auto packed_input_degree_offsets =
      input_frontier_indices ? std::make_optional<rmm::device_uvector<size_t>>(
                                 (*input_frontier_indices).size() + 1, handle.get_stream())
                             : std::nullopt;
    if (packed_input_degree_offsets) {
      (*packed_input_degree_offsets).set_element_to_zero_async(0, handle.get_stream());
      auto degree_first = thrust::make_transform_iterator(
        (*input_frontier_indices).begin(),
        cuda::proclaim_return_type<size_t>([input_degree_offsets] __device__(size_t i) {
          return input_degree_offsets[i + 1] - input_degree_offsets[i];
        }));
      thrust::inclusive_scan(handle.get_thrust_policy(),
                             degree_first,
                             degree_first + (*input_frontier_indices).size(),
                             (*packed_input_degree_offsets).begin() + 1);
    }

    // generate (key, nbr_index) pairs

    size_t num_pairs{};
    raft::update_host(
      &num_pairs,
      packed_input_degree_offsets
        ? (*packed_input_degree_offsets).data() + ((*packed_input_degree_offsets).size() - 1)
        : input_degree_offsets.data() + (input_degree_offsets.size() - 1),
      1,
      handle.get_stream());
    handle.sync_stream();
    rmm::device_uvector<bias_t> keys(num_pairs, handle.get_stream());

    cugraph::detail::uniform_random_fill(
      handle.get_stream(), keys.data(), keys.size(), bias_t{0.0}, bias_t{1.0}, rng_state);

    if (input_frontier_indices) {
      auto bias_first = thrust::make_transform_iterator(
        thrust::make_counting_iterator(size_t{0}),
        cuda::proclaim_return_type<bias_t>(
          [input_biases,
           input_degree_offsets,
           frontier_indices            = *input_frontier_indices,
           packed_input_degree_offsets = raft::device_span<size_t const>(
             (*packed_input_degree_offsets).data(),
             (*packed_input_degree_offsets).size())] __device__(size_t i) {
            auto it           = thrust::upper_bound(thrust::seq,
                                          packed_input_degree_offsets.begin() + 1,
                                          packed_input_degree_offsets.end(),
                                          i);
            auto idx          = thrust::distance(packed_input_degree_offsets.begin() + 1, it);
            auto frontier_idx = frontier_indices[idx];
            return input_biases[input_degree_offsets[frontier_idx] +
                                (i - packed_input_degree_offsets[idx])];
          }));
      thrust::transform(handle.get_thrust_policy(),
                        keys.begin(),
                        keys.end(),
                        bias_first,
                        keys.begin(),
                        cuda::proclaim_return_type<bias_t>([] __device__(bias_t r, bias_t b) {
                          return b > 0.0 ? -log(r) / b : std::numeric_limits<bias_t>::infinity();
                        }));
    } else {
      thrust::transform(handle.get_thrust_policy(),
                        keys.begin(),
                        keys.end(),
                        input_biases.begin(),
                        keys.begin(),
                        cuda::proclaim_return_type<bias_t>([] __device__(bias_t r, bias_t b) {
                          return b > 0.0 ? -log(r) / b : std::numeric_limits<bias_t>::infinity();
                        }));
    }

    rmm::device_uvector<edge_t> nbr_indices(keys.size(), handle.get_stream());
    thrust::tabulate(
      handle.get_thrust_policy(),
      nbr_indices.begin(),
      nbr_indices.end(),
      [offsets = packed_input_degree_offsets
                   ? raft::device_span<size_t const>((*packed_input_degree_offsets).data(),
                                                     (*packed_input_degree_offsets).size())
                   : input_degree_offsets] __device__(size_t i) {
        auto it  = thrust::upper_bound(thrust::seq, offsets.begin() + 1, offsets.end(), i);
        auto idx = thrust::distance(offsets.begin() + 1, it);
        return static_cast<edge_t>(i - offsets[idx]);
      });

    // pick top K for each frontier index

    rmm::device_uvector<std::byte> d_tmp_storage(0, handle.get_stream());
    size_t tmp_storage_bytes{0};

    rmm::device_uvector<bias_t> segment_sorted_keys(keys.size(), handle.get_stream());
    rmm::device_uvector<edge_t> segment_sorted_nbr_indices(nbr_indices.size(), handle.get_stream());

    cub::DeviceSegmentedSort::SortPairs(
      static_cast<void*>(nullptr),
      tmp_storage_bytes,
      keys.data(),
      segment_sorted_keys.data(),
      nbr_indices.data(),
      segment_sorted_nbr_indices.data(),
      keys.size(),
      input_frontier_indices ? (*input_frontier_indices).size() : (input_degree_offsets.size() - 1),
      packed_input_degree_offsets ? (*packed_input_degree_offsets).begin()
                                  : input_degree_offsets.begin(),
      (packed_input_degree_offsets ? (*packed_input_degree_offsets).begin()
                                   : input_degree_offsets.begin()) +
        1,
      handle.get_stream());
    if (tmp_storage_bytes > d_tmp_storage.size()) {
      d_tmp_storage = rmm::device_uvector<std::byte>(tmp_storage_bytes, handle.get_stream());
    }
    cub::DeviceSegmentedSort::SortPairs(
      d_tmp_storage.data(),
      tmp_storage_bytes,
      keys.data(),
      segment_sorted_keys.data(),
      nbr_indices.data(),
      segment_sorted_nbr_indices.data(),
      keys.size(),
      input_frontier_indices ? (*input_frontier_indices).size() : input_degree_offsets.size() - 1,
      packed_input_degree_offsets ? (*packed_input_degree_offsets).begin()
                                  : input_degree_offsets.begin(),
      (packed_input_degree_offsets ? (*packed_input_degree_offsets).begin()
                                   : input_degree_offsets.begin()) +
        1,
      handle.get_stream());

    // FIXME: how should we handle bias 0? return invalid_idx?
    if (output_frontier_indices) {
      thrust::for_each(
        handle.get_thrust_policy(),
        thrust::make_counting_iterator(size_t{0}),
        thrust::make_counting_iterator((*output_frontier_indices).size() * K),
        [input_degree_offsets =
           packed_input_degree_offsets
             ? raft::device_span<size_t const>((*packed_input_degree_offsets).data(),
                                               (*packed_input_degree_offsets).size())
             : input_degree_offsets,
         output_frontier_indices = *output_frontier_indices,
         output_keys,
         output_nbr_indices,
         segment_sorted_keys =
           raft::device_span<bias_t const>(segment_sorted_keys.data(), segment_sorted_keys.size()),
         segment_sorted_nbr_indices = raft::device_span<edge_t const>(
           segment_sorted_nbr_indices.data(), segment_sorted_nbr_indices.size()),
         K,
         invalid_idx = cugraph::ops::graph::INVALID_ID<edge_t>] __device__(size_t i) {
          auto output_frontier_idx = output_frontier_indices[i / K];
          auto output_idx          = output_frontier_idx * K + (i % K);
          auto degree              = input_degree_offsets[i / K + 1] - input_degree_offsets[i / K];
          if (i % K < degree) {
            auto input_idx = input_degree_offsets[i / K] + (i % K);
            if (output_keys) { (*output_keys)[output_idx] = segment_sorted_keys[input_idx]; }
            output_nbr_indices[output_idx] = segment_sorted_nbr_indices[input_idx];
          } else {
            if (output_keys) {
              (*output_keys)[output_idx] = std::numeric_limits<bias_t>::infinity();
            }
            output_nbr_indices[output_idx] = invalid_idx;
          }
        });
    } else {
      thrust::for_each(
        handle.get_thrust_policy(),
        thrust::make_counting_iterator(size_t{0}),
        thrust::make_counting_iterator(output_nbr_indices.size()),
        [input_degree_offsets =
           packed_input_degree_offsets
             ? raft::device_span<size_t const>((*packed_input_degree_offsets).data(),
                                               (*packed_input_degree_offsets).size())
             : input_degree_offsets,
         output_keys,
         output_nbr_indices,
         segment_sorted_keys =
           raft::device_span<bias_t const>(segment_sorted_keys.data(), segment_sorted_keys.size()),
         segment_sorted_nbr_indices = raft::device_span<edge_t const>(
           segment_sorted_nbr_indices.data(), segment_sorted_nbr_indices.size()),
         K,
         invalid_idx = cugraph::ops::graph::INVALID_ID<edge_t>] __device__(size_t i) {
          auto degree = input_degree_offsets[i / K + 1] - input_degree_offsets[i / K];
          if (i % K < degree) {
            auto input_idx = input_degree_offsets[i / K] + (i % K);
            if (output_keys) { (*output_keys)[i] = segment_sorted_keys[input_idx]; }
            output_nbr_indices[i] = segment_sorted_nbr_indices[input_idx];
          } else {
            if (output_keys) { (*output_keys)[i] = std::numeric_limits<bias_t>::infinity(); }
            output_nbr_indices[i] = invalid_idx;
          }
        });
    }
  }

  return;
}

template <typename GraphViewType, typename VertexIterator>
rmm::device_uvector<typename GraphViewType::edge_type>
compute_aggregate_local_frontier_local_degrees(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  VertexIterator aggregate_local_frontier_major_first,
  std::vector<size_t> const& local_frontier_displacements,
  std::vector<size_t> const& local_frontier_sizes)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  static_assert(
    std::is_same_v<typename thrust::iterator_traits<VertexIterator>::value_type, vertex_t>);

  auto edge_mask_view = graph_view.edge_mask_view();

  auto aggregate_local_frontier_local_degrees = rmm::device_uvector<edge_t>(
    local_frontier_displacements.back() + local_frontier_sizes.back(), handle.get_stream());
  for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
    auto edge_partition =
      edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
        graph_view.local_edge_partition_view(i));
    auto edge_partition_e_mask =
      edge_mask_view
        ? thrust::make_optional<
            detail::edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool>>(
            *edge_mask_view, i)
        : thrust::nullopt;

    auto edge_partition_frontier_major_first =
      aggregate_local_frontier_major_first + local_frontier_displacements[i];
    auto edge_partition_frontier_local_degrees =
      !edge_partition_e_mask ? edge_partition.compute_local_degrees(
                                 edge_partition_frontier_major_first,
                                 edge_partition_frontier_major_first + local_frontier_sizes[i],
                                 handle.get_stream())
                             : edge_partition.compute_local_degrees_with_mask(
                                 (*edge_partition_e_mask).value_first(),
                                 edge_partition_frontier_major_first,
                                 edge_partition_frontier_major_first + local_frontier_sizes[i],
                                 handle.get_stream());

    // FIXME: this copy is unnecessary if edge_partition.compute_local_degrees() takes a pointer
    // to the output array
    thrust::copy(handle.get_thrust_policy(),
                 edge_partition_frontier_local_degrees.begin(),
                 edge_partition_frontier_local_degrees.end(),
                 aggregate_local_frontier_local_degrees.begin() + local_frontier_displacements[i]);
  }

  return aggregate_local_frontier_local_degrees;
}

// return (bias segmented local inclusive sums, segment offsets) pairs for each key in th eaggregate
// local frontier
template <typename GraphViewType,
          typename KeyIterator,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename EdgeBiasOp>
std::tuple<rmm::device_uvector<
             typename edge_op_result_type<typename thrust::iterator_traits<KeyIterator>::value_type,
                                          typename GraphViewType::vertex_type,
                                          typename EdgeSrcValueInputWrapper::value_type,
                                          typename EdgeDstValueInputWrapper::value_type,
                                          typename EdgeValueInputWrapper::value_type,
                                          EdgeBiasOp>::type>,
           rmm::device_uvector<size_t>>
compute_aggregate_local_frontier_biases(raft::handle_t const& handle,
                                        GraphViewType const& graph_view,
                                        KeyIterator aggregate_local_frontier_key_first,
                                        EdgeSrcValueInputWrapper edge_src_value_input,
                                        EdgeDstValueInputWrapper edge_dst_value_input,
                                        EdgeValueInputWrapper edge_value_input,
                                        EdgeBiasOp e_bias_op,
                                        std::vector<size_t> const& local_frontier_displacements,
                                        std::vector<size_t> const& local_frontier_sizes)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using key_t    = typename thrust::iterator_traits<KeyIterator>::value_type;

  using bias_t = typename edge_op_result_type<key_t,
                                              vertex_t,
                                              typename EdgeSrcValueInputWrapper::value_type,
                                              typename EdgeDstValueInputWrapper::value_type,
                                              typename EdgeValueInputWrapper::value_type,
                                              EdgeBiasOp>::type;

  auto [aggregate_local_frontier_biases, aggregate_local_frontier_local_degree_offsets] =
    transform_v_frontier_e(handle,
                           graph_view,
                           aggregate_local_frontier_key_first,
                           edge_src_value_input,
                           edge_dst_value_input,
                           edge_value_input,
                           e_bias_op,
                           local_frontier_displacements,
                           local_frontier_sizes);

  return std::make_tuple(std::move(aggregate_local_frontier_biases),
                         std::move(aggregate_local_frontier_local_degree_offsets));
}

template <typename value_t, bool multi_gpu>
std::tuple<rmm::device_uvector<value_t>,
           std::optional<rmm::device_uvector<size_t>>,
           std::vector<size_t>>
shuffle_and_compute_local_nbr_values(raft::handle_t const& handle,
                                     rmm::device_uvector<value_t>&& sample_nbr_values,
                                     std::optional<raft::device_span<value_t const>>
                                       frontier_partitioned_value_local_sum_displacements,
                                     std::vector<size_t> const& local_frontier_displacements,
                                     std::vector<size_t> const& local_frontier_sizes,
                                     size_t K)
{
  int minor_comm_rank{0};
  int minor_comm_size{1};
  if constexpr (multi_gpu) {
    auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    minor_comm_rank  = minor_comm.get_rank();
    minor_comm_size  = minor_comm.get_size();
  }

  auto sample_local_nbr_values = std::move(
    sample_nbr_values);  // neighbor value within an edge partition (note that each vertex's
                         // neighbors are distributed in minor_comm_size partitions)
  std::optional<rmm::device_uvector<size_t>> key_indices{
    std::nullopt};  // relevant only when (minor_comm_size > 1)
  std::vector<size_t> local_frontier_sample_offsets{};
  if (minor_comm_size > 1) {
    auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());

    key_indices = rmm::device_uvector<size_t>(sample_local_nbr_values.size(), handle.get_stream());
    auto minor_comm_ranks =
      rmm::device_uvector<int>(sample_local_nbr_values.size(), handle.get_stream());
    auto intra_partition_displacements =
      rmm::device_uvector<size_t>(sample_local_nbr_values.size(), handle.get_stream());
    rmm::device_uvector<size_t> d_tx_counts(minor_comm_size, handle.get_stream());
    thrust::fill(handle.get_thrust_policy(), d_tx_counts.begin(), d_tx_counts.end(), size_t{0});
    auto input_pair_first = thrust::make_zip_iterator(
      thrust::make_tuple(sample_local_nbr_values.begin(),
                         thrust::make_transform_iterator(thrust::make_counting_iterator(size_t{0}),
                                                         divider_t<size_t>{K})));
    value_t invalid_value{};
    static_assert(std::is_arithmetic_v<value_t>);
    if constexpr (std::is_floating_point_v<value_t>) {
      invalid_value = std::numeric_limits<value_t>::infinity();
    } else {
      invalid_value = cugraph::ops::graph::INVALID_ID<value_t>;
    }
    thrust::transform(
      handle.get_thrust_policy(),
      input_pair_first,
      input_pair_first + sample_local_nbr_values.size(),
      thrust::make_zip_iterator(thrust::make_tuple(minor_comm_ranks.begin(),
                                                   intra_partition_displacements.begin(),
                                                   sample_local_nbr_values.begin(),
                                                   (*key_indices).begin())),
      convert_pair_to_quadruplet_t<value_t>{
        raft::device_span<value_t const>(
          (*frontier_partitioned_value_local_sum_displacements).data(),
          (*frontier_partitioned_value_local_sum_displacements).size()),
        raft::device_span<size_t>(d_tx_counts.data(), d_tx_counts.size()),
        local_frontier_sizes[minor_comm_rank],
        minor_comm_size,
        invalid_value});
    rmm::device_uvector<size_t> tx_displacements(minor_comm_size, handle.get_stream());
    thrust::exclusive_scan(
      handle.get_thrust_policy(), d_tx_counts.begin(), d_tx_counts.end(), tx_displacements.begin());
    auto tmp_sample_local_nbr_values =
      rmm::device_uvector<value_t>(tx_displacements.back_element(handle.get_stream()) +
                                     d_tx_counts.back_element(handle.get_stream()),
                                   handle.get_stream());
    auto tmp_key_indices =
      rmm::device_uvector<size_t>(tmp_sample_local_nbr_values.size(), handle.get_stream());
    auto pair_first = thrust::make_zip_iterator(
      thrust::make_tuple(sample_local_nbr_values.begin(), (*key_indices).begin()));
    thrust::scatter_if(
      handle.get_thrust_policy(),
      pair_first,
      pair_first + sample_local_nbr_values.size(),
      thrust::make_transform_iterator(
        thrust::make_counting_iterator(size_t{0}),
        shuffle_index_compute_offset_t{
          raft::device_span<int const>(minor_comm_ranks.data(), minor_comm_ranks.size()),
          raft::device_span<size_t const>(intra_partition_displacements.data(),
                                          intra_partition_displacements.size()),
          raft::device_span<size_t const>(tx_displacements.data(), tx_displacements.size())}),
      minor_comm_ranks.begin(),
      thrust::make_zip_iterator(
        thrust::make_tuple(tmp_sample_local_nbr_values.begin(), tmp_key_indices.begin())),
      is_not_equal_t<int>{-1});

    sample_local_nbr_values = std::move(tmp_sample_local_nbr_values);
    key_indices             = std::move(tmp_key_indices);

    std::vector<size_t> h_tx_counts(d_tx_counts.size());
    raft::update_host(
      h_tx_counts.data(), d_tx_counts.data(), d_tx_counts.size(), handle.get_stream());
    handle.sync_stream();

    pair_first = thrust::make_zip_iterator(
      thrust::make_tuple(sample_local_nbr_values.begin(), (*key_indices).begin()));
    auto [rx_value_buffer, rx_counts] =
      shuffle_values(minor_comm, pair_first, h_tx_counts, handle.get_stream());

    sample_local_nbr_values          = std::move(std::get<0>(rx_value_buffer));
    key_indices                      = std::move(std::get<1>(rx_value_buffer));
    local_frontier_sample_offsets    = std::vector<size_t>(rx_counts.size() + 1);
    local_frontier_sample_offsets[0] = size_t{0};
    std::inclusive_scan(
      rx_counts.begin(), rx_counts.end(), local_frontier_sample_offsets.begin() + 1);
  } else {
    local_frontier_sample_offsets =
      std::vector<size_t>{size_t{0}, local_frontier_sizes[minor_comm_rank] * K};
  }

  return std::make_tuple(std::move(sample_local_nbr_values),
                         std::move(key_indices),
                         std::move(local_frontier_sample_offsets));
}

template <typename GraphViewType, typename VertexIterator>
rmm::device_uvector<typename GraphViewType::edge_type> convert_to_unmasked_local_nbr_idx(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  VertexIterator aggregate_local_frontier_major_first,
  rmm::device_uvector<typename GraphViewType::edge_type>&& local_nbr_indices,
  std::optional<raft::device_span<size_t const>> key_indices,
  std::vector<size_t> const& local_frontier_sample_offsets,
  std::vector<size_t> const& local_frontier_displacements,
  std::vector<size_t> const& local_frontier_sizes,
  size_t K)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  static_assert(
    std::is_same_v<vertex_t, typename thrust::iterator_traits<VertexIterator>::value_type>);

  auto edge_mask_view = graph_view.edge_mask_view();

  // to avoid searching the entire neighbor list K times for high degree vertices with edge masking
  auto local_frontier_valid_local_nbr_count_inclusive_sums =
    compute_valid_local_nbr_count_inclusive_sums(handle,
                                                 graph_view,
                                                 aggregate_local_frontier_major_first,
                                                 local_frontier_displacements,
                                                 local_frontier_sizes);

  auto sample_major_idx_first = thrust::make_transform_iterator(
    thrust::make_counting_iterator(size_t{0}),
    cuda::proclaim_return_type<size_t>(
      [K,
       key_indices = key_indices ? thrust::make_optional<raft::device_span<size_t const>>(
                                     (*key_indices).data(), (*key_indices).size())
                                 : thrust::nullopt] __device__(size_t i) {
        return key_indices ? (*key_indices)[i] : i / K;
      }));
  auto pair_first = thrust::make_zip_iterator(local_nbr_indices.begin(), sample_major_idx_first);
  for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
    auto edge_partition =
      edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
        graph_view.local_edge_partition_view(i));
    auto edge_partition_e_mask =
      edge_mask_view
        ? thrust::make_optional<
            detail::edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool>>(
            *edge_mask_view, i)
        : thrust::nullopt;

    auto edge_partition_frontier_major_first =
      aggregate_local_frontier_major_first + local_frontier_displacements[i];
    thrust::transform_if(
      handle.get_thrust_policy(),
      pair_first + local_frontier_sample_offsets[i],
      pair_first + local_frontier_sample_offsets[i + 1],
      local_nbr_indices.begin() + local_frontier_sample_offsets[i],
      local_nbr_indices.begin() + local_frontier_sample_offsets[i],
      find_nth_valid_nbr_idx_t<GraphViewType,
                               decltype(edge_partition_e_mask),
                               decltype(edge_partition_frontier_major_first)>{
        edge_partition,
        edge_partition_e_mask,
        edge_partition_frontier_major_first,
        thrust::make_tuple(
          raft::device_span<size_t const>(
            std::get<0>(local_frontier_valid_local_nbr_count_inclusive_sums[i]).data(),
            std::get<0>(local_frontier_valid_local_nbr_count_inclusive_sums[i]).size()),
          raft::device_span<edge_t const>(
            std::get<1>(local_frontier_valid_local_nbr_count_inclusive_sums[i]).data(),
            std::get<1>(local_frontier_valid_local_nbr_count_inclusive_sums[i]).size()))},
      is_not_equal_t<edge_t>{cugraph::ops::graph::INVALID_ID<edge_t>});
  }

  return std::move(local_nbr_indices);
}

template <typename GraphViewType, typename KeyIterator>
std::tuple<rmm::device_uvector<typename GraphViewType::edge_type>,
           std::optional<rmm::device_uvector<size_t>>,
           std::vector<size_t>>
uniform_sample_and_compute_local_nbr_indices(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  KeyIterator aggregate_local_frontier_key_first,
  std::vector<size_t> const& local_frontier_displacements,
  std::vector<size_t> const& local_frontier_sizes,
  raft::random::RngState& rng_state,
  size_t K,
  bool with_replacement)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using key_t    = typename thrust::iterator_traits<KeyIterator>::value_type;

  int minor_comm_size{1};
  if constexpr (GraphViewType::is_multi_gpu) {
    auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    minor_comm_size  = minor_comm.get_size();
  }

  auto aggregate_local_frontier_major_first =
    thrust_tuple_get_or_identity<KeyIterator, 0>(aggregate_local_frontier_key_first);

  auto edge_mask_view = graph_view.edge_mask_view();

  // 1. compute degrees

  rmm::device_uvector<edge_t> frontier_degrees(0, handle.get_stream());
  std::optional<rmm::device_uvector<edge_t>> frontier_partitioned_local_degree_displacements{
    std::nullopt};
  {
    auto aggregate_local_frontier_local_degrees =
      compute_aggregate_local_frontier_local_degrees(handle,
                                                     graph_view,
                                                     aggregate_local_frontier_major_first,
                                                     local_frontier_displacements,
                                                     local_frontier_sizes);

    if (minor_comm_size > 1) {
      std::tie(frontier_degrees, frontier_partitioned_local_degree_displacements) =
        compute_frontier_value_sums_and_partitioned_local_value_sum_displacements(
          handle,
          raft::device_span<edge_t const>(aggregate_local_frontier_local_degrees.data(),
                                          aggregate_local_frontier_local_degrees.size()),
          local_frontier_displacements,
          local_frontier_sizes);
      aggregate_local_frontier_local_degrees.resize(0, handle.get_stream());
      aggregate_local_frontier_local_degrees.shrink_to_fit(handle.get_stream());
    } else {
      frontier_degrees = std::move(aggregate_local_frontier_local_degrees);
    }
  }

  // 2. sample neighbor indices

  rmm::device_uvector<edge_t> nbr_indices(0, handle.get_stream());

  if (with_replacement) {
    if (frontier_degrees.size() > 0) {
      nbr_indices.resize(frontier_degrees.size() * K, handle.get_stream());
      cugraph::ops::graph::get_sampling_index(nbr_indices.data(),
                                              rng_state,
                                              frontier_degrees.data(),
                                              static_cast<edge_t>(frontier_degrees.size()),
                                              static_cast<int32_t>(K),
                                              with_replacement,
                                              handle.get_stream());
      frontier_degrees.resize(0, handle.get_stream());
      frontier_degrees.shrink_to_fit(handle.get_stream());
    }
  } else {
    nbr_indices = compute_uniform_sampling_index_without_replacement(
      handle, std::move(frontier_degrees), rng_state, K);
  }

  // 3. shuffle neighbor indices

  auto [local_nbr_indices, key_indices, local_frontier_sample_offsets] =
    shuffle_and_compute_local_nbr_values<edge_t, GraphViewType::is_multi_gpu>(
      handle,
      std::move(nbr_indices),
      frontier_partitioned_local_degree_displacements
        ? std::make_optional<raft::device_span<edge_t const>>(
            (*frontier_partitioned_local_degree_displacements).data(),
            (*frontier_partitioned_local_degree_displacements).size())
        : std::nullopt,
      local_frontier_displacements,
      local_frontier_sizes,
      K);

  // 4. convert neighbor indices in the neighbor list considering edge mask to neighbor indices in
  // the neighbor list ignoring edge mask

  if (edge_mask_view) {
    local_nbr_indices = convert_to_unmasked_local_nbr_idx(
      handle,
      graph_view,
      aggregate_local_frontier_major_first,
      std::move(local_nbr_indices),
      key_indices ? std::make_optional<raft::device_span<size_t const>>((*key_indices).data(),
                                                                        (*key_indices).size())
                  : std::nullopt,
      local_frontier_sample_offsets,
      local_frontier_displacements,
      local_frontier_sizes,
      K);
  }

  return std::make_tuple(
    std::move(local_nbr_indices), std::move(key_indices), std::move(local_frontier_sample_offsets));
}

template <typename GraphViewType,
          typename KeyIterator,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename EdgeBiasOp>
std::tuple<rmm::device_uvector<typename GraphViewType::edge_type>,
           std::optional<rmm::device_uvector<size_t>>,
           std::vector<size_t>>
biased_sample_and_compute_local_nbr_indices(raft::handle_t const& handle,
                                            GraphViewType const& graph_view,
                                            KeyIterator aggregate_local_frontier_key_first,
                                            EdgeSrcValueInputWrapper edge_src_value_input,
                                            EdgeDstValueInputWrapper edge_dst_value_input,
                                            EdgeValueInputWrapper edge_value_input,
                                            EdgeBiasOp e_bias_op,
                                            std::vector<size_t> const& local_frontier_displacements,
                                            std::vector<size_t> const& local_frontier_sizes,
                                            raft::random::RngState& rng_state,
                                            size_t K,
                                            bool with_replacement)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using key_t    = typename thrust::iterator_traits<KeyIterator>::value_type;

  using bias_t = typename edge_op_result_type<key_t,
                                              vertex_t,
                                              typename EdgeSrcValueInputWrapper::value_type,
                                              typename EdgeDstValueInputWrapper::value_type,
                                              typename EdgeValueInputWrapper::value_type,
                                              EdgeBiasOp>::type;

  int minor_comm_rank{0};
  int minor_comm_size{1};
  if constexpr (GraphViewType::is_multi_gpu) {
    auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    minor_comm_rank  = minor_comm.get_rank();
    minor_comm_size  = minor_comm.get_size();
  }
  assert(minor_comm_size == graph_view.number_of_local_edge_partitions());

  auto aggregate_local_frontier_major_first =
    thrust_tuple_get_or_identity<KeyIterator, 0>(aggregate_local_frontier_key_first);

  auto edge_mask_view = graph_view.edge_mask_view();

  // 1. compute biases

  auto [aggregate_local_frontier_biases, aggregate_local_frontier_local_degree_offsets] =
    compute_aggregate_local_frontier_biases(handle,
                                            graph_view,
                                            aggregate_local_frontier_key_first,
                                            edge_src_value_input,
                                            edge_dst_value_input,
                                            edge_value_input,
                                            e_bias_op,
                                            local_frontier_displacements,
                                            local_frontier_sizes);

  // 2. sample neighbor indices

  rmm::device_uvector<edge_t> local_nbr_indices(0, handle.get_stream());
  std::optional<rmm::device_uvector<size_t>> key_indices{std::nullopt};
  std::vector<size_t> local_frontier_sample_offsets{};
  if (with_replacement) {
    auto key_first = thrust::make_transform_iterator(
      thrust::make_counting_iterator(size_t{0}),
      cuda::proclaim_return_type<size_t>(
        [offsets = raft::device_span<size_t const>(
           aggregate_local_frontier_local_degree_offsets.data(),
           aggregate_local_frontier_local_degree_offsets.size())] __device__(size_t i) {
          return static_cast<size_t>(thrust::distance(
            offsets.begin() + 1,
            thrust::upper_bound(thrust::seq, offsets.begin() + 1, offsets.end(), i)));
        }));
    thrust::inclusive_scan_by_key(handle.get_thrust_policy(),
                                  key_first,
                                  key_first + aggregate_local_frontier_biases.size(),
                                  get_dataframe_buffer_begin(aggregate_local_frontier_biases),
                                  get_dataframe_buffer_begin(aggregate_local_frontier_biases));

    auto aggregate_local_frontier_bias_segmented_local_inclusive_sums =
      std::move(aggregate_local_frontier_biases);

    auto aggregate_local_frontier_bias_local_sums = rmm::device_uvector<bias_t>(
      local_frontier_displacements.back() + local_frontier_sizes.back(), handle.get_stream());
    thrust::tabulate(
      handle.get_thrust_policy(),
      get_dataframe_buffer_begin(aggregate_local_frontier_bias_local_sums),
      get_dataframe_buffer_end(aggregate_local_frontier_bias_local_sums),
      [offsets =
         raft::device_span<size_t const>(aggregate_local_frontier_local_degree_offsets.data(),
                                         aggregate_local_frontier_local_degree_offsets.size()),
       aggregate_local_frontier_bias_segmented_local_inclusive_sums =
         raft::device_span<bias_t const>(
           aggregate_local_frontier_bias_segmented_local_inclusive_sums.data(),
           aggregate_local_frontier_bias_segmented_local_inclusive_sums
             .size())] __device__(size_t i) {
        auto degree = offsets[i + 1] - offsets[i];
        if (degree > 0) {
          return aggregate_local_frontier_bias_segmented_local_inclusive_sums[offsets[i] + degree -
                                                                              1];
        } else {
          return bias_t{0.0};
        }
      });

    rmm::device_uvector<bias_t> frontier_bias_sums(0, handle.get_stream());
    std::optional<rmm::device_uvector<bias_t>> frontier_partitioned_bias_local_sum_displacements{
      std::nullopt};
    if (minor_comm_size > 1) {
      std::tie(frontier_bias_sums, frontier_partitioned_bias_local_sum_displacements) =
        compute_frontier_value_sums_and_partitioned_local_value_sum_displacements(
          handle,
          raft::device_span<bias_t const>(aggregate_local_frontier_bias_local_sums.data(),
                                          aggregate_local_frontier_bias_local_sums.size()),
          local_frontier_displacements,
          local_frontier_sizes);
      aggregate_local_frontier_bias_local_sums.resize(0, handle.get_stream());
      aggregate_local_frontier_bias_local_sums.shrink_to_fit(handle.get_stream());
    } else {
      frontier_bias_sums = std::move(aggregate_local_frontier_bias_local_sums);
    }

    rmm::device_uvector<bias_t> sample_random_numbers(frontier_bias_sums.size() * K,
                                                      handle.get_stream());
    cugraph::detail::uniform_random_fill(handle.get_stream(),
                                         sample_random_numbers.data(),
                                         sample_random_numbers.size(),
                                         bias_t{0.0},
                                         bias_t{1.0},
                                         rng_state);
    thrust::transform(
      handle.get_thrust_policy(),
      sample_random_numbers.begin(),
      sample_random_numbers.end(),
      thrust::make_counting_iterator(size_t{0}),
      sample_random_numbers.begin(),
      [frontier_bias_sums =
         raft::device_span<bias_t const>(frontier_bias_sums.data(), frontier_bias_sums.size()),
       K,
       invalid_value = std::numeric_limits<bias_t>::infinity()] __device__(bias_t r, size_t i) {
        return frontier_bias_sums[i / K] > 0.0 ? r * frontier_bias_sums[i / K] : invalid_value;
      });

    rmm::device_uvector<bias_t> sample_local_random_numbers(0, handle.get_stream());
    std::tie(sample_local_random_numbers, key_indices, local_frontier_sample_offsets) =
      shuffle_and_compute_local_nbr_values<bias_t, GraphViewType::is_multi_gpu>(
        handle,
        std::move(sample_random_numbers),
        frontier_partitioned_bias_local_sum_displacements
          ? std::make_optional<raft::device_span<bias_t const>>(
              (*frontier_partitioned_bias_local_sum_displacements).data(),
              (*frontier_partitioned_bias_local_sum_displacements).size())
          : std::nullopt,
        local_frontier_displacements,
        local_frontier_sizes,
        K);

    local_nbr_indices.resize(sample_local_random_numbers.size(), handle.get_stream());
    for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
      thrust::tabulate(
        handle.get_thrust_policy(),
        local_nbr_indices.begin() + local_frontier_sample_offsets[i],
        local_nbr_indices.begin() + local_frontier_sample_offsets[i + 1],
        [K,
         sample_local_random_numbers = raft::device_span<bias_t>(
           sample_local_random_numbers.data() + local_frontier_sample_offsets[i],
           local_frontier_sample_offsets[i + 1] - local_frontier_sample_offsets[i]),
         key_indices =
           key_indices ? thrust::make_optional<raft::device_span<size_t const>>(
                           (*key_indices).data() + local_frontier_sample_offsets[i],
                           local_frontier_sample_offsets[i + 1] - local_frontier_sample_offsets[i])
                       : thrust::nullopt,
         aggregate_local_frontier_bias_segmented_local_inclusive_sums = raft::device_span<bias_t>(
           aggregate_local_frontier_bias_segmented_local_inclusive_sums.data(),
           aggregate_local_frontier_bias_segmented_local_inclusive_sums.size()),
         local_degree_offsets = raft::device_span<size_t const>(
           aggregate_local_frontier_local_degree_offsets.data() + local_frontier_displacements[i],
           local_frontier_sizes[i] + 1),
         invalid_idx = cugraph::ops::graph::INVALID_ID<edge_t>] __device__(size_t i) {
          auto key_idx = key_indices ? (*key_indices)[i] : (i / K);
          auto local_degree =
            static_cast<edge_t>(local_degree_offsets[key_idx + 1] - local_degree_offsets[key_idx]);
          if (local_degree > 0) {
            auto local_random_number = sample_local_random_numbers[i];
            auto inclusive_sum_first =
              aggregate_local_frontier_bias_segmented_local_inclusive_sums.begin() +
              local_degree_offsets[key_idx];
            auto inclusive_sum_last = inclusive_sum_first + local_degree;
            auto local_nbr_idx      = static_cast<edge_t>(thrust::distance(
              inclusive_sum_first,
              thrust::upper_bound(
                thrust::seq, inclusive_sum_first, inclusive_sum_last, local_random_number)));
            return cuda::std::min(local_nbr_idx, local_degree - 1);
          } else {
            return invalid_idx;
          }
        });
    }
  } else {
    rmm::device_uvector<edge_t> frontier_degrees(0, handle.get_stream());
    std::optional<rmm::device_uvector<edge_t>> frontier_partitioned_local_degree_displacements{
      std::nullopt};
    {
      rmm::device_uvector<edge_t> aggregate_local_frontier_local_degrees(
        local_frontier_displacements.back() + local_frontier_sizes.back(), handle.get_stream());
      thrust::adjacent_difference(handle.get_thrust_policy(),
                                  aggregate_local_frontier_local_degree_offsets.begin() + 1,
                                  aggregate_local_frontier_local_degree_offsets.end(),
                                  aggregate_local_frontier_local_degrees.begin());
      if (minor_comm_size > 1) {
        std::tie(frontier_degrees, frontier_partitioned_local_degree_displacements) =
          compute_frontier_value_sums_and_partitioned_local_value_sum_displacements(
            handle,
            raft::device_span<edge_t const>(aggregate_local_frontier_local_degrees.data(),
                                            aggregate_local_frontier_local_degrees.size()),
            local_frontier_displacements,
            local_frontier_sizes);
      } else {
        frontier_degrees = std::move(aggregate_local_frontier_local_degrees);
      }
    }

    auto [frontier_indices, frontier_partition_offsets] =
      partition_v_frontier(handle,
                           frontier_degrees.begin(),
                           frontier_degrees.end(),
                           std::vector<edge_t>{static_cast<edge_t>(K + 1),
                                               static_cast<edge_t>(minor_comm_size * K * 2)});

    rmm::device_uvector<edge_t> nbr_indices(frontier_degrees.size() * K, handle.get_stream());
    if (frontier_partition_offsets[1] > 0) {
      thrust::for_each(
        handle.get_thrust_policy(),
        frontier_indices.begin(),
        frontier_indices.begin() + frontier_partition_offsets[1],
        [K,
         frontier_degrees =
           raft::device_span<edge_t const>(frontier_degrees.data(), frontier_degrees.size()),
         nbr_indices = raft::device_span<edge_t>(nbr_indices.data(), nbr_indices.size()),
         invalid_idx = cugraph::ops::graph::INVALID_ID<edge_t>] __device__(size_t i) {
          auto degree = frontier_degrees[i];
          thrust::sequence(thrust::seq,
                           nbr_indices.begin() + i * K,
                           nbr_indices.begin() + i * K + degree,
                           edge_t{0});
          thrust::fill(thrust::seq,
                       nbr_indices.begin() + i * K + degree,
                       nbr_indices.begin() + (i + 1) * K,
                       invalid_idx);
        });
    }

    auto mid_frontier_size = frontier_partition_offsets[2] - frontier_partition_offsets[1];
    std::vector<size_t> mid_local_frontier_sizes{};
    if (minor_comm_size > 1) {
      auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
      mid_local_frontier_sizes =
        host_scalar_allgather(minor_comm, mid_frontier_size, handle.get_stream());
    } else {
      mid_local_frontier_sizes = std::vector<size_t>{mid_frontier_size};
    }
    std::vector<size_t> mid_local_frontier_displacements(mid_local_frontier_sizes.size());
    std::exclusive_scan(mid_local_frontier_sizes.begin(),
                        mid_local_frontier_sizes.end(),
                        mid_local_frontier_displacements.begin(),
                        size_t{0});

    if (mid_local_frontier_displacements.back() + mid_local_frontier_sizes.back() > 0) {
      // aggregate frontier indices with their degrees in the medium range

      if (minor_comm_size > 1) {
        auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
        auto aggregate_mid_local_frontier_indices = rmm::device_uvector<size_t>(
          mid_local_frontier_displacements.back() + mid_local_frontier_sizes.back(),
          handle.get_stream());
        device_allgatherv(minor_comm,
                          frontier_indices.begin() + frontier_partition_offsets[1],
                          aggregate_mid_local_frontier_indices.begin(),
                          mid_local_frontier_sizes,
                          mid_local_frontier_displacements,
                          handle.get_stream());

        // compute local degrees for the aggregated frontier indices

        rmm::device_uvector<edge_t> aggregate_mid_local_frontier_local_degrees(
          aggregate_mid_local_frontier_indices.size(), handle.get_stream());
        for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
          thrust::transform(
            handle.get_thrust_policy(),
            aggregate_mid_local_frontier_indices.begin() + mid_local_frontier_displacements[i],
            aggregate_mid_local_frontier_indices.begin() + mid_local_frontier_displacements[i] +
              mid_local_frontier_sizes[i],
            aggregate_mid_local_frontier_local_degrees.begin() +
              mid_local_frontier_displacements[i],
            cuda::proclaim_return_type<edge_t>(
              [offsets = raft::device_span<size_t const>(
                 aggregate_local_frontier_local_degree_offsets.data() +
                   local_frontier_displacements[i],
                 local_frontier_sizes[i] + 1)] __device__(size_t i) {
                return static_cast<edge_t>(offsets[i + 1] - offsets[i]);
              }));
        }

        // gather biases for the aggregated frontier indices

        rmm::device_uvector<bias_t> aggregate_mid_local_frontier_biases(0, handle.get_stream());
        std::vector<size_t> mid_local_frontier_degree_sums(mid_local_frontier_sizes.size());
        {
          rmm::device_uvector<size_t> aggregate_mid_local_frontier_local_degree_offsets(
            aggregate_mid_local_frontier_local_degrees.size() + 1, handle.get_stream());
          aggregate_mid_local_frontier_local_degree_offsets.set_element_to_zero_async(
            0, handle.get_stream());
          thrust::inclusive_scan(handle.get_thrust_policy(),
                                 aggregate_mid_local_frontier_local_degrees.begin(),
                                 aggregate_mid_local_frontier_local_degrees.end(),
                                 aggregate_mid_local_frontier_local_degree_offsets.begin() + 1);
          aggregate_mid_local_frontier_biases.resize(
            aggregate_mid_local_frontier_local_degree_offsets.back_element(handle.get_stream()),
            handle.get_stream());

          std::vector<size_t> mid_local_frontier_degree_sum_lasts(
            mid_local_frontier_degree_sums.size());
          for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
            thrust::for_each(
              handle.get_thrust_policy(),
              thrust::make_counting_iterator(size_t{0}),
              thrust::make_counting_iterator(mid_local_frontier_sizes[i]),
              [aggregate_local_frontier_biases = raft::device_span<bias_t>(
                 aggregate_local_frontier_biases.data(), aggregate_local_frontier_biases.size()),
               aggregate_local_frontier_local_degree_offsets =
                 raft::device_span<size_t>(aggregate_local_frontier_local_degree_offsets.data(),
                                           aggregate_local_frontier_local_degree_offsets.size()),
               mid_local_frontier_indices = raft::device_span<size_t const>(
                 aggregate_mid_local_frontier_indices.data() + mid_local_frontier_displacements[i],
                 mid_local_frontier_sizes[i]),
               aggregate_mid_local_frontier_biases =
                 raft::device_span<bias_t>(aggregate_mid_local_frontier_biases.data(),
                                           aggregate_mid_local_frontier_biases.size()),
               aggregate_mid_local_frontier_local_degree_offsets = raft::device_span<size_t>(
                 aggregate_mid_local_frontier_local_degree_offsets.data(),
                 aggregate_mid_local_frontier_local_degree_offsets.size()),
               input_offset  = local_frontier_displacements[i],
               output_offset = mid_local_frontier_displacements[i]] __device__(size_t i) {
                thrust::copy(
                  thrust::seq,
                  aggregate_local_frontier_biases.begin() +
                    aggregate_local_frontier_local_degree_offsets[input_offset +
                                                                  mid_local_frontier_indices[i]],
                  aggregate_local_frontier_biases.begin() +
                    aggregate_local_frontier_local_degree_offsets
                      [input_offset + (mid_local_frontier_indices[i] + 1)],
                  aggregate_mid_local_frontier_biases.begin() +
                    aggregate_mid_local_frontier_local_degree_offsets[output_offset + i]);
              });
            mid_local_frontier_degree_sum_lasts[i] =
              aggregate_mid_local_frontier_local_degree_offsets.element(
                mid_local_frontier_displacements[i] + mid_local_frontier_sizes[i],
                handle.get_stream());
          }
          std::adjacent_difference(mid_local_frontier_degree_sum_lasts.begin(),
                                   mid_local_frontier_degree_sum_lasts.end(),
                                   mid_local_frontier_degree_sums.begin());
        }
        aggregate_mid_local_frontier_indices.resize(0, handle.get_stream());
        aggregate_mid_local_frontier_indices.shrink_to_fit(handle.get_stream());

        // shuffle local degrees & biases

        rmm::device_uvector<size_t> mid_frontier_gathered_local_degree_offsets(0,
                                                                               handle.get_stream());
        {
          rmm::device_uvector<edge_t> mid_frontier_gathered_local_degrees(0, handle.get_stream());
          std::tie(mid_frontier_gathered_local_degrees, std::ignore) =
            shuffle_values(minor_comm,
                           aggregate_mid_local_frontier_local_degrees.data(),
                           mid_local_frontier_sizes,
                           handle.get_stream());
          aggregate_mid_local_frontier_local_degrees.resize(0, handle.get_stream());
          aggregate_mid_local_frontier_local_degrees.shrink_to_fit(handle.get_stream());
          mid_frontier_gathered_local_degree_offsets.resize(
            mid_frontier_gathered_local_degrees.size() + 1, handle.get_stream());
          mid_frontier_gathered_local_degree_offsets.set_element_to_zero_async(0,
                                                                               handle.get_stream());
          thrust::inclusive_scan(handle.get_thrust_policy(),
                                 mid_frontier_gathered_local_degrees.begin(),
                                 mid_frontier_gathered_local_degrees.end(),
                                 mid_frontier_gathered_local_degree_offsets.begin() + 1);
        }

        rmm::device_uvector<bias_t> mid_frontier_gathered_biases(0, handle.get_stream());
        std::tie(mid_frontier_gathered_biases, std::ignore) =
          shuffle_values(minor_comm,
                         aggregate_mid_local_frontier_biases.data(),
                         mid_local_frontier_degree_sums,
                         handle.get_stream());
        aggregate_mid_local_frontier_biases.resize(0, handle.get_stream());
        aggregate_mid_local_frontier_biases.shrink_to_fit(handle.get_stream());

        auto mid_frontier_degree_first = thrust::make_transform_iterator(
          frontier_indices.begin() + frontier_partition_offsets[1],
          cuda::proclaim_return_type<edge_t>(
            [frontier_degrees = raft::device_span<edge_t>(
               frontier_degrees.data(), frontier_degrees.size())] __device__(size_t i) {
              return frontier_degrees[i];
            }));
        rmm::device_uvector<size_t> mid_frontier_degree_offsets(mid_frontier_size + 1,
                                                                handle.get_stream());
        mid_frontier_degree_offsets.set_element_to_zero_async(0, handle.get_stream());
        thrust::inclusive_scan(handle.get_thrust_policy(),
                               mid_frontier_degree_first,
                               mid_frontier_degree_first + mid_frontier_size,
                               mid_frontier_degree_offsets.begin() + 1);
        rmm::device_uvector<bias_t> mid_frontier_biases(mid_frontier_gathered_biases.size(),
                                                        handle.get_stream());
        thrust::for_each(
          handle.get_thrust_policy(),
          thrust::make_counting_iterator(size_t{0}),
          thrust::make_counting_iterator(mid_frontier_size),
          [mid_frontier_gathered_local_degree_offsets =
             raft::device_span<size_t>(mid_frontier_gathered_local_degree_offsets.data(),
                                       mid_frontier_gathered_local_degree_offsets.size()),
           mid_frontier_gathered_biases = raft::device_span<bias_t const>(
             mid_frontier_gathered_biases.data(), mid_frontier_gathered_biases.size()),
           mid_frontier_degree_offsets = raft::device_span<size_t>(
             mid_frontier_degree_offsets.data(), mid_frontier_degree_offsets.size()),
           mid_frontier_biases =
             raft::device_span<bias_t>(mid_frontier_biases.data(), mid_frontier_biases.size()),
           minor_comm_size,
           mid_frontier_size] __device__(size_t i) {
            auto output_offset = mid_frontier_degree_offsets[i];
            for (int j = 0; j < minor_comm_size; ++j) {
              auto input_offset =
                mid_frontier_gathered_local_degree_offsets[mid_frontier_size * j + i];
              auto input_size =
                mid_frontier_gathered_local_degree_offsets[mid_frontier_size * j + i + 1] -
                input_offset;
              thrust::copy(thrust::seq,
                           mid_frontier_gathered_biases.begin() + input_offset,
                           mid_frontier_gathered_biases.begin() + input_offset + input_size,
                           mid_frontier_biases.begin() + output_offset);
              output_offset += input_size;
            }
          });

        // now sample and update indices

        // FIXME: test with jump = true
        compute_biased_sampling_index_without_replacement<edge_t, bias_t>(
          handle,
          std::nullopt,
          raft::device_span<size_t const>(mid_frontier_degree_offsets.data(),
                                          mid_frontier_degree_offsets.size()),
          raft::device_span<bias_t const>(mid_frontier_biases.data(), mid_frontier_biases.size()),
          std::make_optional<raft::device_span<size_t const>>(
            frontier_indices.begin() + frontier_partition_offsets[1], mid_frontier_size),
          raft::device_span<edge_t>(nbr_indices.data(), nbr_indices.size()),
          std::nullopt,
          rng_state,
          K,
          false);
      } else {
        // FIXME: test with jump = true
        compute_biased_sampling_index_without_replacement<edge_t, bias_t>(
          handle,
          std::make_optional<raft::device_span<size_t const>>(
            frontier_indices.data() + frontier_partition_offsets[1], mid_frontier_size),
          raft::device_span<size_t const>(aggregate_local_frontier_local_degree_offsets.data(),
                                          aggregate_local_frontier_local_degree_offsets.size()),
          raft::device_span<bias_t const>(aggregate_local_frontier_biases.data(),
                                          aggregate_local_frontier_biases.size()),
          std::make_optional<raft::device_span<size_t const>>(
            frontier_indices.data() + frontier_partition_offsets[1], mid_frontier_size),
          raft::device_span<edge_t>(nbr_indices.data(), nbr_indices.size()),
          std::nullopt,
          rng_state,
          K,
          false);
      }
    }

    auto high_frontier_size = frontier_partition_offsets[3] - frontier_partition_offsets[2];
    std::vector<size_t> high_local_frontier_sizes{};
    if (minor_comm_size > 1) {
      auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
      high_local_frontier_sizes =
        host_scalar_allgather(minor_comm, high_frontier_size, handle.get_stream());
    } else {
      high_local_frontier_sizes = std::vector<size_t>{high_frontier_size};
    }
    std::vector<size_t> high_local_frontier_displacements(high_local_frontier_sizes.size());
    std::exclusive_scan(high_local_frontier_sizes.begin(),
                        high_local_frontier_sizes.end(),
                        high_local_frontier_displacements.begin(),
                        size_t{0});
    if (high_local_frontier_displacements.back() + high_local_frontier_sizes.back() > 0) {
      if (minor_comm_size > 1) {
        // aggregate frontier indices wit their degrees in the high range

        auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
        auto aggregate_high_local_frontier_indices = rmm::device_uvector<size_t>(
          high_local_frontier_displacements.back() + high_local_frontier_sizes.back(),
          handle.get_stream());
        device_allgatherv(minor_comm,
                          frontier_indices.begin() + frontier_partition_offsets[2],
                          aggregate_high_local_frontier_indices.begin(),
                          high_local_frontier_sizes,
                          high_local_frontier_displacements,
                          handle.get_stream());

        // local sample and update indices

        rmm::device_uvector<edge_t> aggregate_high_local_frontier_local_nbr_indices(
          (high_local_frontier_displacements.back() + high_local_frontier_sizes.back()) * K,
          handle.get_stream());
        rmm::device_uvector<bias_t> aggregate_high_local_frontier_keys(
          aggregate_high_local_frontier_local_nbr_indices.size(), handle.get_stream());
        for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
          // FIXME: test with jump = true
          compute_biased_sampling_index_without_replacement<edge_t, bias_t>(
            handle,
            std::make_optional<raft::device_span<size_t const>>(
              aggregate_high_local_frontier_indices.data() + high_local_frontier_displacements[i],
              high_local_frontier_sizes[i]),
            raft::device_span<size_t const>(aggregate_local_frontier_local_degree_offsets.data() +
                                              local_frontier_displacements[i],
                                            local_frontier_sizes[i] + 1),
            raft::device_span<bias_t>(aggregate_local_frontier_biases.data(),
                                      aggregate_local_frontier_biases.size()),
            std::nullopt,
            raft::device_span<edge_t>(aggregate_high_local_frontier_local_nbr_indices.data() +
                                        high_local_frontier_displacements[i] * K,
                                      high_local_frontier_sizes[i] * K),
            std::make_optional<raft::device_span<bias_t>>(
              aggregate_high_local_frontier_keys.data() + high_local_frontier_displacements[i] * K,
              high_local_frontier_sizes[i] * K),
            rng_state,
            K,
            false);
        }

        // shuffle local sampling outputs

        std::vector<size_t> tx_counts(high_local_frontier_sizes);
        std::transform(high_local_frontier_sizes.begin(),
                       high_local_frontier_sizes.end(),
                       tx_counts.begin(),
                       [K](size_t size) { return size * K; });
        rmm::device_uvector<edge_t> high_frontier_gathered_local_nbr_indices(0,
                                                                             handle.get_stream());
        std::tie(high_frontier_gathered_local_nbr_indices, std::ignore) =
          shuffle_values(minor_comm,
                         aggregate_high_local_frontier_local_nbr_indices.data(),
                         tx_counts,
                         handle.get_stream());
        rmm::device_uvector<bias_t> high_frontier_gathered_keys(0, handle.get_stream());
        std::tie(high_frontier_gathered_keys, std::ignore) = shuffle_values(
          minor_comm, aggregate_high_local_frontier_keys.data(), tx_counts, handle.get_stream());
        // FIXME: now free aggregate_high_local_frontier_local_nbr_indices & keys?

        // merge local sampling outputs

        rmm::device_uvector<edge_t> high_frontier_local_nbr_indices(
          high_frontier_size * minor_comm_size * K, handle.get_stream());
        rmm::device_uvector<bias_t> high_frontier_keys(high_frontier_local_nbr_indices.size(),
                                                       handle.get_stream());
        auto index_first = thrust::make_transform_iterator(
          thrust::make_counting_iterator(size_t{0}),
          cuda::proclaim_return_type<size_t>(
            [K, minor_comm_rank, minor_comm_size, high_frontier_size] __device__(size_t i) {
              auto idx             = i / (K * minor_comm_size);
              auto minor_comm_rank = (i % (K * minor_comm_size)) / K;
              return minor_comm_rank * (high_frontier_size * K) + idx * K + (i % K);
            }));
        auto high_frontier_gathered_nbr_idx_first = thrust::make_transform_iterator(
          thrust::counting_iterator(size_t{0}),
          cuda::proclaim_return_type<edge_t>(
            [frontier_partitioned_local_degree_displacements = raft::device_span<edge_t const>(
               (*frontier_partitioned_local_degree_displacements).data(),
               (*frontier_partitioned_local_degree_displacements).size()),
             high_frontier_indices = raft::device_span<size_t const>(
               frontier_indices.data() + frontier_partition_offsets[2], high_frontier_size),
             high_frontier_gathered_local_nbr_indices =
               raft::device_span<edge_t const>(high_frontier_gathered_local_nbr_indices.data(),
                                               high_frontier_gathered_local_nbr_indices.size()),
             K,
             minor_comm_size,
             high_frontier_size] __device__(size_t i) {
              auto minor_comm_rank = static_cast<int>(i / (high_frontier_size * K));
              auto frontier_idx    = high_frontier_indices[(i % (high_frontier_size * K)) / K];
              return frontier_partitioned_local_degree_displacements[frontier_idx *
                                                                       minor_comm_size +
                                                                     minor_comm_rank] +
                     high_frontier_gathered_local_nbr_indices[i];
            }));
        thrust::gather(handle.get_thrust_policy(),
                       index_first,
                       index_first + high_frontier_local_nbr_indices.size(),
                       thrust::make_zip_iterator(high_frontier_gathered_nbr_idx_first,
                                                 high_frontier_gathered_keys.begin()),
                       thrust::make_zip_iterator(high_frontier_local_nbr_indices.begin(),
                                                 high_frontier_keys.begin()));
        high_frontier_gathered_local_nbr_indices.resize(0, handle.get_stream());
        high_frontier_gathered_local_nbr_indices.shrink_to_fit(handle.get_stream());
        high_frontier_gathered_keys.resize(0, handle.get_stream());
        high_frontier_gathered_keys.shrink_to_fit(handle.get_stream());

        rmm::device_uvector<std::byte> d_tmp_storage(0, handle.get_stream());
        size_t tmp_storage_bytes{0};

        rmm::device_uvector<edge_t> high_frontier_segment_sorted_local_nbr_indices(
          high_frontier_local_nbr_indices.size(), handle.get_stream());
        rmm::device_uvector<bias_t> high_frontier_segment_sorted_keys(high_frontier_keys.size(),
                                                                      handle.get_stream());
        cub::DeviceSegmentedSort::SortPairs(
          static_cast<void*>(nullptr),
          tmp_storage_bytes,
          high_frontier_keys.data(),
          high_frontier_segment_sorted_keys.data(),
          high_frontier_local_nbr_indices.data(),
          high_frontier_segment_sorted_local_nbr_indices.data(),
          high_frontier_size * K * minor_comm_size,
          high_frontier_size,
          thrust::make_transform_iterator(thrust::make_counting_iterator(size_t{0}),
                                          multiplier_t<size_t>{minor_comm_size * K}),
          thrust::make_transform_iterator(thrust::make_counting_iterator(size_t{1}),
                                          multiplier_t<size_t>{minor_comm_size * K}),
          handle.get_stream());
        if (tmp_storage_bytes > d_tmp_storage.size()) {
          d_tmp_storage = rmm::device_uvector<std::byte>(tmp_storage_bytes, handle.get_stream());
        }
        cub::DeviceSegmentedSort::SortPairs(
          d_tmp_storage.data(),
          tmp_storage_bytes,
          high_frontier_keys.data(),
          high_frontier_segment_sorted_keys.data(),
          high_frontier_local_nbr_indices.data(),
          high_frontier_segment_sorted_local_nbr_indices.data(),
          high_frontier_size * K * minor_comm_size,
          high_frontier_size,
          thrust::make_transform_iterator(thrust::make_counting_iterator(size_t{0}),
                                          multiplier_t<size_t>{minor_comm_size * K}),
          thrust::make_transform_iterator(thrust::make_counting_iterator(size_t{1}),
                                          multiplier_t<size_t>{minor_comm_size * K}),
          handle.get_stream());

        thrust::for_each(
          handle.get_thrust_policy(),
          thrust::make_counting_iterator(size_t{0}),
          thrust::make_counting_iterator(high_frontier_size),
          [high_frontier_indices = raft::device_span<size_t const>(
             frontier_indices.data() + frontier_partition_offsets[2], high_frontier_size),
           high_frontier_segment_sorted_local_nbr_indices =
             raft::device_span<edge_t const>(high_frontier_segment_sorted_local_nbr_indices.data(),
                                             high_frontier_segment_sorted_local_nbr_indices.size()),
           nbr_indices = raft::device_span<edge_t>(nbr_indices.data(), nbr_indices.size()),
           K,
           minor_comm_size] __device__(size_t i) {
            thrust::copy(
              thrust::seq,
              high_frontier_segment_sorted_local_nbr_indices.begin() + (i * K * minor_comm_size),
              high_frontier_segment_sorted_local_nbr_indices.begin() +
                (i * K * minor_comm_size + K),
              nbr_indices.begin() + high_frontier_indices[i] * K);
          });
      } else {
        // FIXME: test with jump = true
        compute_biased_sampling_index_without_replacement<edge_t, bias_t>(
          handle,
          std::make_optional<raft::device_span<size_t const>>(
            frontier_indices.data() + frontier_partition_offsets[2], high_frontier_size),
          raft::device_span<size_t const>(aggregate_local_frontier_local_degree_offsets.data(),
                                          aggregate_local_frontier_local_degree_offsets.size()),
          raft::device_span<bias_t>(aggregate_local_frontier_biases.data(),
                                    aggregate_local_frontier_biases.size()),
          std::make_optional<raft::device_span<size_t const>>(
            frontier_indices.data() + frontier_partition_offsets[2], high_frontier_size),
          raft::device_span<edge_t>(nbr_indices.data(), nbr_indices.size()),
          std::nullopt,
          rng_state,
          K,
          false);
      }
    }

    std::tie(local_nbr_indices, key_indices, local_frontier_sample_offsets) =
      shuffle_and_compute_local_nbr_values<edge_t, GraphViewType::is_multi_gpu>(
        handle,
        std::move(nbr_indices),
        frontier_partitioned_local_degree_displacements
          ? std::make_optional<raft::device_span<edge_t const>>(
              (*frontier_partitioned_local_degree_displacements).data(),
              (*frontier_partitioned_local_degree_displacements).size())
          : std::nullopt,
        local_frontier_displacements,
        local_frontier_sizes,
        K);
  }

  // 3. convert neighbor indices in the neighbor list considering edge mask to neighbor indices in
  // the neighbor list ignoring edge mask

  if (edge_mask_view) {
    local_nbr_indices = convert_to_unmasked_local_nbr_idx(
      handle,
      graph_view,
      aggregate_local_frontier_major_first,
      std::move(local_nbr_indices),
      key_indices ? std::make_optional<raft::device_span<size_t const>>((*key_indices).data(),
                                                                        (*key_indices).size())
                  : std::nullopt,
      local_frontier_sample_offsets,
      local_frontier_displacements,
      local_frontier_sizes,
      K);
  }

  return std::make_tuple(
    std::move(local_nbr_indices), std::move(key_indices), std::move(local_frontier_sample_offsets));
}

}  // namespace detail

}  // namespace cugraph
