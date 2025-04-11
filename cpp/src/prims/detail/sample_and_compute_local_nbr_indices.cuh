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

#include "prims/detail/partition_v_frontier.cuh"
#include "prims/detail/transform_v_frontier_e.cuh"
#include "prims/property_op_utils.cuh"

#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/edge_partition_edge_property_device_view.cuh>
#include <cugraph/edge_partition_endpoint_property_device_view.cuh>
#include <cugraph/graph.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/mask_utils.cuh>
#include <cugraph/utilities/misc_utils.cuh>
#include <cugraph/utilities/shuffle_comm.cuh>
#include <cugraph/vertex_partition_device_view.cuh>

#include <raft/random/rng.cuh>

#include <cub/cub.cuh>
#include <cuda/atomic>
#include <cuda/functional>
#include <cuda/std/iterator>
#include <cuda/std/optional>
#include <thrust/adjacent_difference.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
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

// this functor output will later be used to convert global value (neighbor index, random number) to
// (local value, minor_comm_rank) pairs.
template <typename value_t>
struct compute_local_value_displacements_and_global_value_t {
  raft::device_span<value_t const> gathered_local_values{};
  raft::device_span<value_t>
    partitioned_local_value_displacements{};  // one partition per gpu in the same minor_comm
  raft::device_span<value_t> global_values{};
  int minor_comm_size{};
  size_t num_values_per_key{};

  __device__ void operator()(size_t i) const
  {
    auto key_idx              = i / num_values_per_key;
    auto value_idx            = i % num_values_per_key;
    constexpr int buffer_size = 8;  // tuning parameter
    value_t displacements[buffer_size];
    value_t sum{0};
    for (int round = 0; round < (minor_comm_size + buffer_size - 1) / buffer_size; ++round) {
      auto loop_count = std::min(buffer_size, minor_comm_size - round * buffer_size);
      for (int j = 0; j < loop_count; ++j) {
        displacements[j] = sum;
        sum += gathered_local_values[i + (round * buffer_size + j) * global_values.size()];
      }
      thrust::copy(thrust::seq,
                   displacements,
                   displacements + loop_count,
                   partitioned_local_value_displacements.begin() +
                     key_idx * num_values_per_key * minor_comm_size + value_idx * minor_comm_size +
                     round * buffer_size);
    }
    global_values[i] = sum;
  }
};

// convert a (neighbor value, key index) pair  to a (minor_comm_rank, intra-partition offset, local
// neighbor value, key index) quadruplet, minor_comm_rank is set to -1 if a neighbor value is
// invalid
template <typename InputIterator, typename OutputIterator, typename value_t>
struct convert_value_key_pair_to_shuffle_t {
  InputIterator input_pair_first;
  OutputIterator output_tuple_first;
  raft::device_span<value_t const>
    partitioned_local_value_displacements{};  // one partition per gpu in the same minor_comm
  raft::device_span<size_t> tx_counts{};
  int minor_comm_size{};
  value_t invalid_value{};

  __device__ void operator()(size_t i) const
  {
    auto pair            = *(input_pair_first + i);
    auto nbr_value       = thrust::get<0>(pair);
    auto key_idx         = thrust::get<1>(pair);
    auto local_nbr_value = nbr_value;
    int minor_comm_rank{-1};
    size_t intra_partition_offset{0};
    if (nbr_value != invalid_value) {
      auto displacement_first =
        partitioned_local_value_displacements.begin() + key_idx * minor_comm_size;
      minor_comm_rank =
        static_cast<int>(cuda::std::distance(
          displacement_first,
          thrust::upper_bound(
            thrust::seq, displacement_first, displacement_first + minor_comm_size, nbr_value))) -
        1;
      local_nbr_value -= *(displacement_first + minor_comm_rank);
      cuda::atomic_ref<size_t, cuda::thread_scope_device> counter(tx_counts[minor_comm_rank]);
      intra_partition_offset = counter.fetch_add(size_t{1}, cuda::std::memory_order_relaxed);
    }
    *(output_tuple_first + i) =
      thrust::make_tuple(minor_comm_rank, intra_partition_offset, local_nbr_value, key_idx);
  }
};

// convert a (per-type neighbor value, index) pair  to a (minor_comm_rank, intra-partition offset,
// per-type local neighbor value, type, key index) 5-tuple, minor_comm_rank is set to -1 if a
// neighbor value is invalid
template <typename InputIterator, typename OutputIterator, typename edge_type_t, typename value_t>
struct convert_per_type_value_key_pair_to_shuffle_t {
  InputIterator input_pair_first;
  OutputIterator output_tuple_first;
  raft::device_span<value_t const>
    partitioned_per_type_local_value_displacements{};  // one partition per gpu in the same
                                                       // minor_comm
  raft::device_span<size_t> tx_counts{};
  raft::device_span<size_t const> K_offsets{};
  size_t K_sum;
  int minor_comm_size{};
  value_t invalid_value{};

  __device__ void operator()(size_t i) const
  {
    auto pair                     = *(input_pair_first + i);
    auto num_edge_types           = K_offsets.size() - 1;
    auto per_type_nbr_value       = thrust::get<0>(pair);
    auto idx                      = thrust::get<1>(pair);
    auto key_idx                  = idx / K_sum;
    auto type                     = static_cast<edge_type_t>(cuda::std::distance(
      K_offsets.begin() + 1,
      thrust::upper_bound(thrust::seq, K_offsets.begin() + 1, K_offsets.end(), idx % K_sum)));
    auto per_type_local_nbr_value = per_type_nbr_value;
    int minor_comm_rank{-1};
    size_t intra_partition_offset{0};
    if (per_type_nbr_value != invalid_value) {
      auto displacement_first = partitioned_per_type_local_value_displacements.begin() +
                                (key_idx * num_edge_types + type) * minor_comm_size;
      minor_comm_rank = static_cast<int>(cuda::std::distance(
                          displacement_first,
                          thrust::upper_bound(thrust::seq,
                                              displacement_first,
                                              displacement_first + minor_comm_size,
                                              per_type_nbr_value))) -
                        1;
      per_type_local_nbr_value -= *(displacement_first + minor_comm_rank);
      cuda::atomic_ref<size_t, cuda::thread_scope_device> counter(tx_counts[minor_comm_rank]);
      intra_partition_offset = counter.fetch_add(size_t{1}, cuda::std::memory_order_relaxed);
    }
    *(output_tuple_first + i) = thrust::make_tuple(
      minor_comm_rank, intra_partition_offset, per_type_local_nbr_value, type, key_idx);
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

// to convert neighbor index excluding masked out edges to neighbor index ignoring edge mask
template <typename GraphViewType, typename EdgePartitionEdgeMaskWrapper, typename VertexIterator>
struct find_nth_valid_nbr_idx_t {
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu> edge_partition{};
  EdgePartitionEdgeMaskWrapper edge_partition_e_mask;
  VertexIterator major_first{};
  raft::device_span<size_t const> major_idx_to_unique_major_idx{};
  thrust::tuple<raft::device_span<size_t const>, raft::device_span<edge_t const>>
    unique_major_valid_local_nbr_count_inclusive_sums{};

  __device__ edge_t operator()(thrust::tuple<edge_t, size_t> pair) const
  {
    edge_t local_nbr_idx    = thrust::get<0>(pair);
    size_t major_idx        = thrust::get<1>(pair);
    size_t unique_major_idx = major_idx_to_unique_major_idx[major_idx];
    auto major              = *(major_first + major_idx);
    auto major_offset       = edge_partition.major_offset_from_major_nocheck(major);
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
      auto inclusive_sum_first =
        thrust::get<1>(unique_major_valid_local_nbr_count_inclusive_sums).begin();
      auto start_offset =
        thrust::get<0>(unique_major_valid_local_nbr_count_inclusive_sums)[unique_major_idx];
      auto end_offset =
        thrust::get<0>(unique_major_valid_local_nbr_count_inclusive_sums)[unique_major_idx + 1];
      auto word_idx = static_cast<edge_t>(
        cuda::std::distance(inclusive_sum_first + start_offset,
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

// compute unique keys & keys to unique keys mapping (in each edge partition)
template <typename KeyIterator>
std::tuple<dataframe_buffer_type_t<typename thrust::iterator_traits<KeyIterator>::value_type>,
           rmm::device_uvector<size_t>,
           std::vector<size_t>>
compute_unique_keys(raft::handle_t const& handle,
                    KeyIterator aggregate_local_frontier_key_first,
                    raft::host_span<size_t const> local_frontier_offsets)
{
  using key_t = typename thrust::iterator_traits<KeyIterator>::value_type;

  auto aggregate_local_frontier_unique_keys =
    allocate_dataframe_buffer<key_t>(0, handle.get_stream());
  auto aggregate_local_frontier_key_idx_to_unique_key_idx =
    rmm::device_uvector<size_t>(local_frontier_offsets.back(), handle.get_stream());
  auto local_frontier_unique_key_offsets = std::vector<size_t>(local_frontier_offsets.size(), 0);

  auto tmp_keys =
    allocate_dataframe_buffer<key_t>(local_frontier_offsets.back(), handle.get_stream());
  std::vector<size_t> local_frontier_unique_key_sizes(local_frontier_offsets.size() - 1);
  for (size_t i = 0; i < local_frontier_unique_key_sizes.size(); ++i) {
    thrust::copy(handle.get_thrust_policy(),
                 aggregate_local_frontier_key_first + local_frontier_offsets[i],
                 aggregate_local_frontier_key_first + local_frontier_offsets[i + 1],
                 get_dataframe_buffer_begin(tmp_keys) + local_frontier_offsets[i]);
    thrust::sort(handle.get_thrust_policy(),
                 get_dataframe_buffer_begin(tmp_keys) + local_frontier_offsets[i],
                 get_dataframe_buffer_begin(tmp_keys) + local_frontier_offsets[i + 1]);
    local_frontier_unique_key_sizes[i] = cuda::std::distance(
      get_dataframe_buffer_begin(tmp_keys) + local_frontier_offsets[i],
      thrust::unique(handle.get_thrust_policy(),
                     get_dataframe_buffer_begin(tmp_keys) + local_frontier_offsets[i],
                     get_dataframe_buffer_begin(tmp_keys) + local_frontier_offsets[i + 1]));
  }
  std::inclusive_scan(local_frontier_unique_key_sizes.begin(),
                      local_frontier_unique_key_sizes.end(),
                      local_frontier_unique_key_offsets.begin() + 1);
  resize_dataframe_buffer(aggregate_local_frontier_unique_keys,
                          local_frontier_unique_key_offsets.back(),
                          handle.get_stream());
  for (size_t i = 0; i < local_frontier_unique_key_sizes.size(); ++i) {
    thrust::copy(handle.get_thrust_policy(),
                 get_dataframe_buffer_begin(tmp_keys) + local_frontier_offsets[i],
                 get_dataframe_buffer_begin(tmp_keys) + local_frontier_offsets[i] +
                   local_frontier_unique_key_sizes[i],
                 get_dataframe_buffer_begin(aggregate_local_frontier_unique_keys) +
                   local_frontier_unique_key_offsets[i]);
    thrust::transform(
      handle.get_thrust_policy(),
      aggregate_local_frontier_key_first + local_frontier_offsets[i],
      aggregate_local_frontier_key_first + local_frontier_offsets[i + 1],
      aggregate_local_frontier_key_idx_to_unique_key_idx.begin() + local_frontier_offsets[i],
      cuda::proclaim_return_type<size_t>(
        [unique_key_first = get_dataframe_buffer_begin(aggregate_local_frontier_unique_keys) +
                            local_frontier_unique_key_offsets[i],
         unique_key_last = get_dataframe_buffer_begin(aggregate_local_frontier_unique_keys) +
                           local_frontier_unique_key_offsets[i + 1]] __device__(key_t key) {
          return static_cast<size_t>(cuda::std::distance(
            unique_key_first, thrust::find(thrust::seq, unique_key_first, unique_key_last, key)));
        }));
  }

  return std::make_tuple(std::move(aggregate_local_frontier_unique_keys),
                         std::move(aggregate_local_frontier_key_idx_to_unique_key_idx),
                         std::move(local_frontier_unique_key_offsets));
}

template <typename value_t>
std::tuple<rmm::device_uvector<value_t>, rmm::device_uvector<value_t>>
compute_frontier_value_sums_and_partitioned_local_value_sum_displacements(
  raft::handle_t const& handle,
  raft::device_span<value_t const> aggregate_local_frontier_local_value_sums,
  raft::host_span<size_t const> local_frontier_offsets,
  size_t num_values_per_key)
{
  auto& minor_comm     = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
  auto minor_comm_rank = minor_comm.get_rank();
  auto minor_comm_size = minor_comm.get_size();

  std::vector<size_t> tx_sizes(minor_comm_size * num_values_per_key);
  for (int i = 0; i < minor_comm_size; ++i) {
    tx_sizes[i] = (local_frontier_offsets[i + 1] - local_frontier_offsets[i]) * num_values_per_key;
  }

  rmm::device_uvector<value_t> frontier_gathered_local_value_sums(0, handle.get_stream());
  std::tie(frontier_gathered_local_value_sums, std::ignore) =
    shuffle_values(minor_comm,
                   aggregate_local_frontier_local_value_sums.begin(),
                   raft::host_span<size_t>(tx_sizes.data(), tx_sizes.size()),
                   handle.get_stream());

  rmm::device_uvector<value_t> frontier_value_sums(
    (local_frontier_offsets[minor_comm_rank + 1] - local_frontier_offsets[minor_comm_rank]) *
      num_values_per_key,
    handle.get_stream());
  rmm::device_uvector<value_t> frontier_partitioned_local_value_sum_displacements(
    frontier_value_sums.size() * minor_comm_size * num_values_per_key, handle.get_stream());

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
      minor_comm_size,
      num_values_per_key});

  return std::make_tuple(std::move(frontier_value_sums),
                         std::move(frontier_partitioned_local_value_sum_displacements));
}

template <typename GraphViewType, typename VertexIterator>
std::vector<
  std::tuple<rmm::device_uvector<size_t>, rmm::device_uvector<typename GraphViewType::edge_type>>>
compute_valid_local_nbr_count_inclusive_sums(raft::handle_t const& handle,
                                             GraphViewType const& graph_view,
                                             VertexIterator aggregate_local_frontier_major_first,
                                             raft::host_span<size_t const> local_frontier_offsets)
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
        ? cuda::std::make_optional<
            detail::edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool>>(
            *edge_mask_view, i)
        : cuda::std::nullopt;

    auto edge_partition_local_degrees = edge_partition.compute_local_degrees(
      aggregate_local_frontier_major_first + local_frontier_offsets[i],
      aggregate_local_frontier_major_first + local_frontier_offsets[i + 1],
      handle.get_stream());
    auto inclusive_sum_offsets = rmm::device_uvector<size_t>(
      (local_frontier_offsets[i + 1] - local_frontier_offsets[i]) + 1, handle.get_stream());
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
       edge_partition_frontier_major_first =
         aggregate_local_frontier_major_first + local_frontier_offsets[i],
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
        aggregate_local_frontier_major_first + local_frontier_offsets[i],
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
        aggregate_local_frontier_major_first + local_frontier_offsets[i],
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

template <typename edge_t, typename bias_t>
void sample_nbr_index_with_replacement(
  raft::handle_t const& handle,
  raft::device_span<edge_t const> frontier_degrees,
  std::optional<raft::device_span<size_t const>> frontier_indices,
  raft::device_span<edge_t> nbr_indices /* [OUT] */,
  raft::random::RngState& rng_state,
  size_t K)
{
  auto num_keys = frontier_indices ? (*frontier_indices).size() : frontier_degrees.size();

  rmm::device_uvector<bias_t> sample_random_numbers(num_keys * K, handle.get_stream());
  cugraph::detail::uniform_random_fill(handle.get_stream(),
                                       sample_random_numbers.data(),
                                       sample_random_numbers.size(),
                                       bias_t{0.0},
                                       bias_t{1.0},
                                       rng_state);

  auto pair_first = thrust::make_zip_iterator(thrust::make_counting_iterator(size_t{0}),
                                              sample_random_numbers.begin());
  thrust::for_each(
    handle.get_thrust_policy(),
    pair_first,
    pair_first + num_keys * K,
    [frontier_degrees,
     frontier_indices = frontier_indices
                          ? cuda::std::optional<raft::device_span<size_t const>>(*frontier_indices)
                          : cuda::std::nullopt,
     nbr_indices,
     K,
     invalid_idx = cugraph::invalid_edge_id_v<edge_t>] __device__(auto pair) {
      auto i            = thrust::get<0>(pair);
      auto r            = thrust::get<1>(pair);
      auto frontier_idx = frontier_indices ? (*frontier_indices)[i / K] : i / K;
      auto degree       = frontier_degrees[frontier_idx];
      auto sample_idx   = invalid_idx;
      if (degree > 0) { sample_idx = cuda::std::min(static_cast<edge_t>(r * degree), degree - 1); }
      nbr_indices[frontier_idx * K + (i % K)] = sample_idx;
    });
}

template <typename edge_t, typename edge_type_t, typename bias_t>
void sample_nbr_index_with_replacement(
  raft::handle_t const& handle,
  raft::device_span<edge_t const> frontier_per_type_degrees,
  std::optional<std::tuple<raft::device_span<size_t const>, raft::device_span<edge_type_t const>>>
    frontier_index_type_pairs,
  raft::device_span<edge_t> per_type_nbr_indices /* [OUT] */,
  raft::random::RngState& rng_state,
  raft::device_span<size_t const> K_offsets,
  size_t K_sum)
{
  auto num_edge_types = static_cast<edge_type_t>(K_offsets.size() - 1);

  auto num_keys = frontier_index_type_pairs ? std::get<0>(*frontier_index_type_pairs).size()
                                            : frontier_per_type_degrees.size();
  assert(frontier_index_type_pairs.has_value() || (num_keys % num_edge_types) == 0);
  std::optional<rmm::device_uvector<size_t>> input_r_offsets{std::nullopt};
  if (frontier_index_type_pairs) {
    input_r_offsets = rmm::device_uvector<size_t>(num_keys + 1, handle.get_stream());
    (*input_r_offsets).set_element_to_zero_async(0, handle.get_stream());
    auto k_first = thrust::make_transform_iterator(
      std::get<1>(*frontier_index_type_pairs).begin(),
      cuda::proclaim_return_type<size_t>(
        [K_offsets] __device__(auto type) { return K_offsets[type + 1] - K_offsets[type]; }));
    thrust::inclusive_scan(
      handle.get_thrust_policy(), k_first, k_first + num_keys, (*input_r_offsets).begin() + 1);
  }

  rmm::device_uvector<bias_t> sample_random_numbers(
    input_r_offsets ? (*input_r_offsets).back_element(handle.get_stream())
                    : (num_keys / num_edge_types) * K_sum,
    handle.get_stream());
  cugraph::detail::uniform_random_fill(handle.get_stream(),
                                       sample_random_numbers.data(),
                                       sample_random_numbers.size(),
                                       bias_t{0.0},
                                       bias_t{1.0},
                                       rng_state);

  auto pair_first = thrust::make_zip_iterator(thrust::make_counting_iterator(size_t{0}),
                                              sample_random_numbers.begin());
  if (frontier_index_type_pairs) {
    thrust::for_each(
      handle.get_thrust_policy(),
      pair_first,
      pair_first + sample_random_numbers.size(),
      [frontier_per_type_degrees,
       frontier_indices = std::get<0>(*frontier_index_type_pairs),
       frontier_types   = std::get<1>(*frontier_index_type_pairs),
       input_r_offsets =
         raft::device_span<size_t const>((*input_r_offsets).data(), (*input_r_offsets).size()),
       per_type_nbr_indices,
       K_offsets,
       K_sum,
       invalid_idx = cugraph::invalid_edge_id_v<edge_t>] __device__(auto pair) {
        auto num_edge_types = static_cast<size_t>(K_offsets.size() - 1);
        auto i              = thrust::get<0>(pair);
        auto r              = thrust::get<1>(pair);
        auto idx            = cuda::std::distance(
          input_r_offsets.begin() + 1,
          thrust::upper_bound(thrust::seq, input_r_offsets.begin() + 1, input_r_offsets.end(), i));
        auto frontier_idx = frontier_indices[idx];
        auto type         = frontier_types[idx];
        auto degree       = frontier_per_type_degrees[frontier_idx * num_edge_types + type];
        auto sample_idx   = invalid_idx;
        if (degree > 0) {
          sample_idx = cuda::std::min(static_cast<edge_t>(r * degree), degree - 1);
        }
        per_type_nbr_indices[frontier_idx * K_sum + K_offsets[type] + (i - input_r_offsets[idx])] =
          sample_idx;
      });
  } else {
    thrust::transform(
      handle.get_thrust_policy(),
      pair_first,
      pair_first + sample_random_numbers.size(),
      per_type_nbr_indices.begin(),
      cuda::proclaim_return_type<edge_t>(
        [frontier_per_type_degrees,
         per_type_nbr_indices,
         K_offsets,
         K_sum,
         invalid_idx = cugraph::invalid_edge_id_v<edge_t>] __device__(auto pair) {
          auto num_edge_types = static_cast<edge_type_t>(K_offsets.size() - 1);
          auto i              = thrust::get<0>(pair);
          auto r              = thrust::get<1>(pair);
          auto frontier_idx   = i / K_sum;
          auto type           = static_cast<edge_type_t>(cuda::std::distance(
            K_offsets.begin() + 1,
            thrust::upper_bound(thrust::seq, K_offsets.begin() + 1, K_offsets.end(), i % K_sum)));
          auto degree         = frontier_per_type_degrees[frontier_idx * num_edge_types + type];
          auto sample_idx     = invalid_idx;
          if (degree > 0) {
            sample_idx = cuda::std::min(static_cast<edge_t>(r * degree), degree - 1);
          }
          return sample_idx;
        }));
  }
}

template <typename edge_t, typename bias_t>
void sample_nbr_index_without_replacement(
  raft::handle_t const& handle,
  raft::device_span<edge_t const> frontier_degrees,
  std::optional<raft::device_span<size_t const>> frontier_indices,
  raft::device_span<edge_t> nbr_indices /* [OUT] */,
  raft::random::RngState& rng_state,
  size_t K,
  bool algo_r = true)
{
  auto num_keys = frontier_indices ? (*frontier_indices).size() : frontier_degrees.size();

  // initialize reservoirs

  if (frontier_indices) {
    thrust::for_each(handle.get_thrust_policy(),
                     thrust::make_counting_iterator(size_t{0}),
                     thrust::make_counting_iterator(num_keys * K),
                     [frontier_degrees,
                      frontier_indices = *frontier_indices,
                      nbr_indices,
                      K,
                      invalid_idx = cugraph::invalid_edge_id_v<edge_t>] __device__(auto i) {
                       auto frontier_idx = frontier_indices[i / K];
                       auto d            = static_cast<size_t>(frontier_degrees[frontier_idx]);
                       nbr_indices[frontier_idx * K + (i % K)] =
                         ((i % K) < d) ? static_cast<edge_t>(i % K) : invalid_idx;
                     });
  } else {
    thrust::tabulate(
      handle.get_thrust_policy(),
      nbr_indices.begin(),
      nbr_indices.begin() + num_keys * K,
      [frontier_degrees, K, invalid_idx = cugraph::invalid_edge_id_v<edge_t>] __device__(auto i) {
        auto d = static_cast<size_t>(frontier_degrees[i / K]);
        return ((i % K) < d) ? static_cast<edge_t>(i % K) : invalid_idx;
      });
  }

  if (algo_r) {  // reservoir sampling, algorithm R
    rmm::device_uvector<size_t> input_r_offsets(num_keys + 1, handle.get_stream());
    input_r_offsets.set_element_to_zero_async(0, handle.get_stream());
    if (frontier_indices) {
      auto count_first = thrust::make_transform_iterator(
        (*frontier_indices).begin(),
        cuda::proclaim_return_type<size_t>([frontier_degrees, K] __device__(size_t i) {
          auto d = static_cast<size_t>(frontier_degrees[i]);
          return d > K ? (d - K) : size_t{0};
        }));
      thrust::inclusive_scan(handle.get_thrust_policy(),
                             count_first,
                             count_first + num_keys,
                             input_r_offsets.begin() + 1);

    } else {
      auto count_first = thrust::make_transform_iterator(
        frontier_degrees.begin(), cuda::proclaim_return_type<size_t>([K] __device__(auto degree) {
          auto d = static_cast<size_t>(degree);
          return d > K ? (d - K) : size_t{0};
        }));
      thrust::inclusive_scan(handle.get_thrust_policy(),
                             count_first,
                             count_first + num_keys,
                             input_r_offsets.begin() + 1);
    }

    rmm::device_uvector<bias_t> sample_random_numbers(
      input_r_offsets.back_element(handle.get_stream()), handle.get_stream());
    cugraph::detail::uniform_random_fill(handle.get_stream(),
                                         sample_random_numbers.data(),
                                         sample_random_numbers.size(),
                                         bias_t{0.0},
                                         bias_t{1.0},
                                         rng_state);

    thrust::for_each(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator(size_t{0}),
      thrust::make_counting_iterator(sample_random_numbers.size()),
      [input_r_offsets =
         raft::device_span<size_t const>(input_r_offsets.data(), input_r_offsets.size()),
       sample_random_numbers = raft::device_span<double const>(sample_random_numbers.data(),
                                                               sample_random_numbers.size()),
       frontier_indices =
         frontier_indices ? cuda::std::optional<raft::device_span<size_t const>>(*frontier_indices)
                          : cuda::std::nullopt,
       nbr_indices,
       K] __device__(size_t i) {
        auto idx = cuda::std::distance(
          input_r_offsets.begin() + 1,
          thrust::upper_bound(thrust::seq, input_r_offsets.begin() + 1, input_r_offsets.end(), i));
        auto nbr_idx = K + (i - input_r_offsets[idx]);
        auto r       = static_cast<size_t>(sample_random_numbers[i] * (nbr_idx + 1));
        if (r < K) {
          auto frontier_idx = frontier_indices ? (*frontier_indices)[idx] : idx;
          cuda::atomic_ref<edge_t, cuda::thread_scope_device> sample_nbr_idx(
            nbr_indices[frontier_idx * K + r]);
          sample_nbr_idx.fetch_max(nbr_idx, cuda::std::memory_order_relaxed);
        }
      });
  } else {  // reservoir sampling, algorithm L
    // algorithm L will be effective for high-degree vertices, but when vertex degree >> K, it is
    // generally more efficient to over-sample (with replacement) and take first K unique samples
    CUGRAPH_FAIL("unimplemented.");
  }
}

template <typename edge_t, typename edge_type_t, typename bias_t>
void sample_nbr_index_without_replacement(
  raft::handle_t const& handle,
  raft::device_span<edge_t const> frontier_per_type_degrees,
  std::optional<std::tuple<raft::device_span<size_t const>, raft::device_span<edge_type_t const>>>
    frontier_index_type_pairs,
  raft::device_span<edge_t> per_type_nbr_indices /* [OUT] */,
  raft::random::RngState& rng_state,
  raft::device_span<size_t const> K_offsets,
  size_t K_sum,
  bool algo_r = true)
{
  auto num_edge_types = static_cast<edge_type_t>(K_offsets.size() - 1);

  auto num_keys = frontier_index_type_pairs ? std::get<0>(*frontier_index_type_pairs).size()
                                            : frontier_per_type_degrees.size();
  assert(frontier_index_type_pairs.has_value() || (num_keys % num_edge_types) == 0);

  // initialize reservoirs

  if (frontier_index_type_pairs) {
    rmm::device_uvector<size_t> sample_size_offsets(num_keys + 1, handle.get_stream());
    sample_size_offsets.set_element_to_zero_async(0, handle.get_stream());
    auto k_first = thrust::make_transform_iterator(
      std::get<1>(*frontier_index_type_pairs).begin(),
      cuda::proclaim_return_type<size_t>(
        [K_offsets] __device__(auto type) { return K_offsets[type + 1] - K_offsets[type]; }));
    thrust::inclusive_scan(
      handle.get_thrust_policy(), k_first, k_first + num_keys, sample_size_offsets.begin() + 1);

    thrust::for_each(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator(size_t{0}),
      thrust::make_counting_iterator(sample_size_offsets.back_element(handle.get_stream())),
      [sample_size_offsets =
         raft::device_span<size_t const>(sample_size_offsets.data(), sample_size_offsets.size()),
       frontier_per_type_degrees,
       frontier_indices = std::get<0>(*frontier_index_type_pairs),
       types            = std::get<1>(*frontier_index_type_pairs),
       per_type_nbr_indices,
       K_offsets,
       K_sum,
       invalid_idx = cugraph::invalid_edge_id_v<edge_t>] __device__(auto i) {
        auto num_edge_types = static_cast<edge_type_t>(K_offsets.size() - 1);
        auto idx            = cuda::std::distance(
          sample_size_offsets.begin() + 1,
          thrust::upper_bound(
            thrust::seq, sample_size_offsets.begin() + 1, sample_size_offsets.end(), i));
        auto frontier_idx = frontier_indices[idx];
        auto type         = types[idx];
        auto d            = frontier_per_type_degrees[frontier_idx * num_edge_types + type];
        auto K            = K_offsets[type + 1] - K_offsets[type];
        auto sample_idx   = i - sample_size_offsets[idx];
        per_type_nbr_indices[frontier_idx * K_sum + K_offsets[type] + sample_idx] =
          (sample_idx < d) ? sample_idx : invalid_idx;
      });
  } else {
    thrust::tabulate(
      handle.get_thrust_policy(),
      per_type_nbr_indices.begin(),
      per_type_nbr_indices.begin() + (num_keys / num_edge_types) * K_sum,
      [frontier_per_type_degrees,
       K_offsets,
       K_sum,
       invalid_idx = cugraph::invalid_edge_id_v<edge_t>] __device__(auto i) {
        auto num_edge_types = static_cast<edge_type_t>(K_offsets.size() - 1);
        auto frontier_idx   = i / K_sum;
        auto type           = static_cast<edge_t>(cuda::std::distance(
          K_offsets.begin() + 1,
          thrust::upper_bound(thrust::seq, K_offsets.begin() + 1, K_offsets.end(), i % K_sum)));
        auto d              = frontier_per_type_degrees[frontier_idx * num_edge_types + type];
        auto K              = K_offsets[type + 1] - K_offsets[type];
        auto sample_idx     = static_cast<edge_t>((i % K_sum) - K_offsets[type]);
        return sample_idx < d ? static_cast<edge_t>(sample_idx) : invalid_idx;
      });
  }

  if (algo_r) {  // reservoir sampling, algorithm R
    rmm::device_uvector<size_t> input_r_offsets(num_keys + 1, handle.get_stream());
    input_r_offsets.set_element_to_zero_async(0, handle.get_stream());
    if (frontier_index_type_pairs) {
      auto count_first = thrust::make_transform_iterator(
        thrust::make_zip_iterator(std::get<0>(*frontier_index_type_pairs).begin(),
                                  std::get<1>(*frontier_index_type_pairs).begin()),
        cuda::proclaim_return_type<size_t>(
          [frontier_per_type_degrees, K_offsets] __device__(auto pair) {
            auto num_edge_types = static_cast<edge_type_t>(K_offsets.size() - 1);
            auto frontier_idx   = thrust::get<0>(pair);
            auto type           = thrust::get<1>(pair);
            auto d =
              static_cast<size_t>(frontier_per_type_degrees[frontier_idx * num_edge_types + type]);
            auto K = K_offsets[type + 1] - K_offsets[type];
            return d > K ? (d - K) : size_t{0};
          }));
      thrust::inclusive_scan(handle.get_thrust_policy(),
                             count_first,
                             count_first + num_keys,
                             input_r_offsets.begin() + 1);
    } else {
      auto count_first = thrust::make_transform_iterator(
        thrust::make_counting_iterator(size_t{0}),
        cuda::proclaim_return_type<size_t>(
          [frontier_per_type_degrees, K_offsets] __device__(auto i) {
            auto num_edge_types = static_cast<edge_type_t>(K_offsets.size() - 1);
            auto d              = static_cast<size_t>(frontier_per_type_degrees[i]);
            auto type           = static_cast<edge_type_t>(i % num_edge_types);
            auto K              = K_offsets[type + 1] - K_offsets[type];
            return d > K ? (d - K) : size_t{0};
          }));
      thrust::inclusive_scan(handle.get_thrust_policy(),
                             count_first,
                             count_first + num_keys,
                             input_r_offsets.begin() + 1);
    }

    rmm::device_uvector<bias_t> sample_random_numbers(
      input_r_offsets.back_element(handle.get_stream()), handle.get_stream());
    cugraph::detail::uniform_random_fill(handle.get_stream(),
                                         sample_random_numbers.data(),
                                         sample_random_numbers.size(),
                                         bias_t{0.0},
                                         bias_t{1.0},
                                         rng_state);

    // based on reservoir sampling, algorithm R

    thrust::for_each(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator(size_t{0}),
      thrust::make_counting_iterator(sample_random_numbers.size()),
      [frontier_per_type_degrees,
       frontier_indices = frontier_index_type_pairs
                            ? cuda::std::optional<raft::device_span<size_t const>>(
                                std::get<0>(*frontier_index_type_pairs))
                            : cuda::std::nullopt,
       types            = frontier_index_type_pairs
                            ? cuda::std::optional<raft::device_span<edge_type_t const>>(
                     std::get<1>(*frontier_index_type_pairs))
                            : cuda::std::nullopt,
       per_type_nbr_indices,
       input_r_offsets =
         raft::device_span<size_t const>(input_r_offsets.data(), input_r_offsets.size()),
       sample_random_numbers = raft::device_span<double const>(sample_random_numbers.data(),
                                                               sample_random_numbers.size()),
       K_offsets,
       K_sum,
       invalid_idx = cugraph::invalid_edge_id_v<edge_t>] __device__(size_t i) {
        auto num_edge_types = static_cast<edge_type_t>(K_offsets.size() - 1);
        auto idx            = cuda::std::distance(
          input_r_offsets.begin() + 1,
          thrust::upper_bound(thrust::seq, input_r_offsets.begin() + 1, input_r_offsets.end(), i));
        auto type = types ? (*types)[idx] : static_cast<edge_type_t>(idx % num_edge_types);
        auto K    = K_offsets[type + 1] - K_offsets[type];
        auto per_type_nbr_idx = K + (i - input_r_offsets[idx]);
        auto r = static_cast<size_t>(sample_random_numbers[i] * (per_type_nbr_idx + 1));
        if (r < K) {
          auto frontier_idx = frontier_indices ? (*frontier_indices)[idx] : idx / num_edge_types;
          cuda::atomic_ref<edge_t, cuda::thread_scope_device> sample_nbr_idx(
            per_type_nbr_indices[frontier_idx * K_sum + K_offsets[type] + r]);
          sample_nbr_idx.fetch_max(per_type_nbr_idx, cuda::std::memory_order_relaxed);
        }
      });
  } else {  // reservoir sampling, algorithm L
    // algorithm L will be effective for high-degree vertices, but when vertex degree >> K, it is
    // generally more efficient to over-sample (with replacement) and take first K unique samples
    CUGRAPH_FAIL("unimplemented.");
  }
}

template <typename edge_t>
rmm::device_uvector<edge_t> compute_homogeneous_uniform_sampling_index_without_replacement(
  raft::handle_t const& handle,
  raft::device_span<edge_t const> frontier_degrees,
  raft::random::RngState& rng_state,
  size_t K)
{
  using bias_t = double;

  edge_t low_partition_degree_range_last =
    static_cast<edge_t>(K * 10);  // exclusive, tuning parameter
  assert(low_partition_degree_range_last >= K);
  size_t high_partition_oversampling_K = std::max(K * 2, K + 16);  // tuning parameter
  assert(high_partition_oversampling_K > K);

  auto [frontier_indices, frontier_partition_offsets] =
    partition_v_frontier(handle,
                         frontier_degrees.begin(),
                         frontier_degrees.end(),
                         std::vector<edge_t>{low_partition_degree_range_last});

  rmm::device_uvector<edge_t> nbr_indices(frontier_degrees.size() * K, handle.get_stream());

  auto low_partition_size = frontier_partition_offsets[1];
  if (low_partition_size > 0) {
    sample_nbr_index_without_replacement<edge_t, bias_t>(
      handle,
      frontier_degrees,
      std::make_optional<raft::device_span<size_t const>>(frontier_indices.data(),
                                                          low_partition_size),
      raft::device_span<edge_t>(nbr_indices.data(), nbr_indices.size()),
      rng_state,
      K);
  }

  auto high_partition_size = frontier_partition_offsets[2] - frontier_partition_offsets[1];
  if (high_partition_size > 0) {
    // to limit memory footprint ((1 << 20) is a tuning parameter), std::max for forward progress
    // guarantee when high_partition_oversampling_K is exorbitantly large
    auto keys_to_sort_per_iteration =
      std::max(static_cast<size_t>(handle.get_device_properties().multiProcessorCount * (1 << 20)) /
                 high_partition_oversampling_K,
               size_t{1});

    rmm::device_uvector<edge_t> tmp_nbr_indices(
      std::min(keys_to_sort_per_iteration, high_partition_size) * high_partition_oversampling_K,
      handle.get_stream());
    assert(high_partition_oversampling_K <=
           static_cast<size_t>(std::numeric_limits<int32_t>::max()));
    rmm::device_uvector<int32_t> tmp_sample_indices(
      tmp_nbr_indices.size(),
      handle.get_stream());  // sample indices ([0, high_partition_oversampling_K)) within a segment
                             // (one segment per key)

    rmm::device_uvector<edge_t> segment_sorted_tmp_nbr_indices(tmp_nbr_indices.size(),
                                                               handle.get_stream());
    rmm::device_uvector<int32_t> segment_sorted_tmp_sample_indices(tmp_nbr_indices.size(),
                                                                   handle.get_stream());

    rmm::device_uvector<std::byte> d_tmp_storage(0, handle.get_stream());
    size_t tmp_storage_bytes{0};

    auto num_chunks =
      (high_partition_size + keys_to_sort_per_iteration - 1) / keys_to_sort_per_iteration;
    for (size_t i = 0; i < num_chunks; ++i) {
      size_t num_segments =
        std::min(keys_to_sort_per_iteration, high_partition_size - keys_to_sort_per_iteration * i);

      rmm::device_uvector<edge_t> unique_counts(num_segments, handle.get_stream());
      std::optional<rmm::device_uvector<size_t>> retry_segment_indices{std::nullopt};

      auto segment_frontier_index_first =
        frontier_indices.begin() + frontier_partition_offsets[1] + keys_to_sort_per_iteration * i;
      auto segment_frontier_degree_first = thrust::make_transform_iterator(
        segment_frontier_index_first,
        indirection_t<size_t, decltype(frontier_degrees.begin())>{frontier_degrees.begin()});

      while (true) {
        std::optional<rmm::device_uvector<edge_t>> retry_nbr_indices{std::nullopt};
        std::optional<rmm::device_uvector<int32_t>> retry_sample_indices{std::nullopt};
        std::optional<rmm::device_uvector<edge_t>> retry_segment_sorted_nbr_indices{std::nullopt};
        std::optional<rmm::device_uvector<int32_t>> retry_segment_sorted_sample_indices{
          std::nullopt};

        if (retry_segment_indices) {
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
          rmm::device_uvector<edge_t> tmp_degrees((*retry_segment_indices).size(),
                                                  handle.get_stream());
          thrust::gather(handle.get_thrust_policy(),
                         (*retry_segment_indices).begin(),
                         (*retry_segment_indices).end(),
                         segment_frontier_degree_first,
                         tmp_degrees.begin());
          sample_nbr_index_with_replacement<edge_t, bias_t>(
            handle,
            raft::device_span<edge_t const>(tmp_degrees.data(), tmp_degrees.size()),
            std::nullopt,
            raft::device_span<edge_t>((*retry_nbr_indices).data(), (*retry_nbr_indices).size()),
            rng_state,
            high_partition_oversampling_K);
        } else {
          rmm::device_uvector<edge_t> tmp_degrees(num_segments, handle.get_stream());
          thrust::copy(handle.get_thrust_policy(),
                       segment_frontier_degree_first,
                       segment_frontier_degree_first + num_segments,
                       tmp_degrees.begin());
          sample_nbr_index_with_replacement<edge_t, bias_t>(
            handle,
            raft::device_span<edge_t const>(tmp_degrees.data(), tmp_degrees.size()),
            std::nullopt,
            raft::device_span<edge_t>(tmp_nbr_indices.data(), tmp_nbr_indices.size()),
            rng_state,
            high_partition_oversampling_K);
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
              auto output_first =
                thrust::make_zip_iterator(retry_nbr_indices, retry_sample_indices);
              // sample index for the previously selected neighbor indices should be smaller than
              // the new candidates to ensure that the previously selected neighbor indices will be
              // selected again
              if (sample_idx < unique_count) {  // re-select the previous ones
                *(output_first + i) = thrust::make_tuple(
                  segment_sorted_tmp_nbr_indices[segment_idx * high_partition_oversampling_K +
                                                 sample_idx],
                  static_cast<int32_t>(sample_idx));
              } else {
                *(output_first + i) =
                  thrust::make_tuple(retry_nbr_indices[i], static_cast<int32_t>(sample_idx));
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
             unique_counts = raft::device_span<edge_t>(unique_counts.data(), unique_counts.size()),
             retry_segment_indices = (*retry_segment_indices).data(),
             retry_segment_sorted_pair_first =
               thrust::make_zip_iterator((*retry_segment_sorted_nbr_indices).begin(),
                                         (*retry_segment_sorted_sample_indices).begin()),
             segment_sorted_pair_first = thrust::make_zip_iterator(
               segment_sorted_tmp_nbr_indices.begin(),
               segment_sorted_tmp_sample_indices.begin())] __device__(size_t i) {
              auto input_pair_first =
                retry_segment_sorted_pair_first + high_partition_oversampling_K * i;
              auto segment_idx = retry_segment_indices[i];
              auto output_pair_first =
                segment_sorted_pair_first + high_partition_oversampling_K * segment_idx;
              assert(high_partition_oversampling_K > 0);
              auto prev           = *input_pair_first;
              size_t unique_count = 1;
              for (size_t j = 1; j < high_partition_oversampling_K; ++j) {
                auto cur = *(input_pair_first + j);
                if (thrust::get<0>(cur) ==
                    thrust::get<0>(prev)) {  // update the sample index to the minimum
                  thrust::get<1>(prev) = cuda::std::min(thrust::get<1>(prev), thrust::get<1>(cur));
                } else {  // new unique neighbor index
                  *(output_pair_first + unique_count - 1) = prev;
                  ++unique_count;
                  prev = cur;
                }
              }
              *(output_pair_first + unique_count - 1) = prev;
              unique_counts[segment_idx]              = unique_count;
            });
        } else {
          thrust::for_each(
            handle.get_thrust_policy(),
            thrust::make_counting_iterator(size_t{0}),
            thrust::make_counting_iterator(num_segments),
            [high_partition_oversampling_K,
             unique_counts = raft::device_span<edge_t>(unique_counts.data(), unique_counts.size()),
             segment_sorted_pair_first = thrust::make_zip_iterator(
               segment_sorted_tmp_nbr_indices.begin(),
               segment_sorted_tmp_sample_indices.begin())] __device__(size_t i) {
              auto pair_first = segment_sorted_pair_first + high_partition_oversampling_K * i;
              assert(high_partition_oversampling_K > 0);
              thrust::tuple<edge_t, int32_t> prev = *pair_first;
              size_t unique_count                 = 1;
              for (size_t j = 1; j < high_partition_oversampling_K; ++j) {
                auto cur = *(pair_first + j);
                if (thrust::get<0>(cur) ==
                    thrust::get<0>(prev)) {  // update the sample index to the minimum
                  thrust::get<1>(prev) = cuda::std::min(thrust::get<1>(prev), thrust::get<1>(cur));
                } else {  // new unique neighbor index
                  *(pair_first + unique_count - 1) = prev;
                  ++unique_count;
                  prev = cur;
                }
              }
              *(pair_first + unique_count - 1) = prev;
              unique_counts[i]                 = unique_count;
            });
        }

        if (retry_segment_indices) {
          auto last =
            thrust::remove_if(handle.get_thrust_policy(),
                              (*retry_segment_indices).begin(),
                              (*retry_segment_indices).end(),
                              [unique_counts = raft::device_span<edge_t const>(
                                 unique_counts.data(), unique_counts.size()),
                               K] __device__(auto segment_idx) {
                                return unique_counts[segment_idx] >= static_cast<edge_t>(K);
                              });
          auto num_retry_segments = cuda::std::distance((*retry_segment_indices).begin(), last);
          if (num_retry_segments > 0) {
            (*retry_segment_indices).resize(num_retry_segments, handle.get_stream());
          } else {
            retry_segment_indices = std::nullopt;
          }
        } else {
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
                            [K,
                             unique_counts = raft::device_span<edge_t const>(
                               unique_counts.data(), unique_counts.size())] __device__(size_t i) {
                              return unique_counts[i] < K;
                            });
          }
        }

        if (!retry_segment_indices) { break; }
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
            [high_partition_oversampling_K,
             unique_counts = raft::device_span<edge_t const>(
               unique_counts.data(), unique_counts.size())] __device__(size_t i) {
              return i * high_partition_oversampling_K + unique_counts[i];
            })),
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
            [high_partition_oversampling_K,
             unique_counts = raft::device_span<edge_t const>(
               unique_counts.data(), unique_counts.size())] __device__(size_t i) {
              return i * high_partition_oversampling_K + unique_counts[i];
            })),
        handle.get_stream());

      // copy the neighbor indices back to nbr_indices

      thrust::for_each(
        handle.get_thrust_policy(),
        thrust::make_counting_iterator(size_t{0}),
        thrust::make_counting_iterator(num_segments * K),
        [K,
         high_partition_oversampling_K,
         frontier_indices = frontier_indices.begin() + frontier_partition_offsets[1] +
                            keys_to_sort_per_iteration * i,
         tmp_nbr_indices =
           raft::device_span<edge_t const>(tmp_nbr_indices.data(), tmp_nbr_indices.size()),
         nbr_indices =
           raft::device_span<edge_t>(nbr_indices.data(), nbr_indices.size())] __device__(size_t i) {
          auto key_idx    = *(frontier_indices + i / K);
          auto sample_idx = static_cast<edge_t>(i % K);
          nbr_indices[key_idx * K + sample_idx] =
            tmp_nbr_indices[(i / K) * high_partition_oversampling_K + sample_idx];
        });
    }
  }

  return nbr_indices;
}

template <typename edge_t, typename edge_type_t>
rmm::device_uvector<edge_t> compute_heterogeneous_uniform_sampling_index_without_replacement(
  raft::handle_t const& handle,
  raft::device_span<edge_t const> frontier_per_type_degrees,
  raft::random::RngState& rng_state,
  raft::device_span<size_t const> K_offsets,
  size_t K_sum)
{
  using bias_t = double;

  size_t max_K{0};
  {
    std::vector<size_t> h_K_offsets(K_offsets.size());
    raft::update_host(h_K_offsets.data(), K_offsets.data(), K_offsets.size(), handle.get_stream());
    handle.sync_stream();
    for (size_t i = 0; i < h_K_offsets.size() - 1; ++i) {
      max_K = std::max(max_K, h_K_offsets[i + 1] - h_K_offsets[i]);
    }
  }

  edge_t low_partition_degree_range_last =
    static_cast<edge_t>(max_K * 10);  // exclusive, tuning parameter
  assert(low_partition_degree_range_last >= max_K);
  size_t high_partition_oversampling_K = std::max(max_K * 2, max_K + 16);  // tuning parameter
  assert(high_partition_oversampling_K > max_K);

  auto num_edge_types = static_cast<edge_type_t>(K_offsets.size() - 1);

  std::vector<edge_t> thresholds(num_edge_types);
  std::fill(
    thresholds.begin(), thresholds.end(), static_cast<edge_t>(low_partition_degree_range_last));

  auto [frontier_indices, frontier_edge_types, frontier_partition_offsets] =
    partition_v_frontier_per_value_idx<edge_type_t>(
      handle,
      frontier_per_type_degrees.begin(),
      frontier_per_type_degrees.end(),
      raft::host_span<edge_t const>(thresholds.data(), thresholds.size()),
      num_edge_types);

  rmm::device_uvector<edge_t> per_type_nbr_indices(
    (frontier_per_type_degrees.size() / num_edge_types) * K_sum, handle.get_stream());

  auto low_partition_size = frontier_partition_offsets[1];
  if (low_partition_size > 0) {
    sample_nbr_index_without_replacement<edge_t, edge_type_t, bias_t>(
      handle,
      frontier_per_type_degrees,
      std::make_optional(std::make_tuple(
        raft::device_span<size_t const>(frontier_indices.data(), low_partition_size),
        raft::device_span<edge_type_t const>(frontier_edge_types.data(), low_partition_size))),
      raft::device_span<edge_t>(per_type_nbr_indices.data(), per_type_nbr_indices.size()),
      rng_state,
      K_offsets,
      K_sum);
  }

  auto high_partition_size = frontier_partition_offsets[2] - frontier_partition_offsets[1];
  if (high_partition_size > 0) {
    // to limit memory footprint ((1 << 20) is a tuning parameter), std::max for forward progress
    // guarantee when high_partition_oversampling_K is exorbitantly large
    auto keys_to_sort_per_iteration =
      std::max(static_cast<size_t>(handle.get_device_properties().multiProcessorCount * (1 << 20)) /
                 high_partition_oversampling_K,
               size_t{1});

    rmm::device_uvector<edge_t> tmp_per_type_nbr_indices(
      std::min(keys_to_sort_per_iteration, high_partition_size) * high_partition_oversampling_K,
      handle.get_stream());
    assert(high_partition_oversampling_K <=
           static_cast<size_t>(std::numeric_limits<int32_t>::max()));
    rmm::device_uvector<int32_t> tmp_sample_indices(
      tmp_per_type_nbr_indices.size(),
      handle.get_stream());  // sample indices ([0, high_partition_oversampling_K)) within a
                             // segment (one segment per key)

    rmm::device_uvector<edge_t> segment_sorted_tmp_per_type_nbr_indices(
      tmp_per_type_nbr_indices.size(), handle.get_stream());
    rmm::device_uvector<int32_t> segment_sorted_tmp_sample_indices(tmp_per_type_nbr_indices.size(),
                                                                   handle.get_stream());

    rmm::device_uvector<std::byte> d_tmp_storage(0, handle.get_stream());
    size_t tmp_storage_bytes{0};

    auto num_chunks =
      (high_partition_size + keys_to_sort_per_iteration - 1) / keys_to_sort_per_iteration;
    for (size_t i = 0; i < num_chunks; ++i) {
      size_t num_segments =
        std::min(keys_to_sort_per_iteration, high_partition_size - keys_to_sort_per_iteration * i);

      rmm::device_uvector<edge_t> unique_counts(num_segments, handle.get_stream());
      std::optional<rmm::device_uvector<size_t>> retry_segment_indices{std::nullopt};

      auto segment_frontier_index_first =
        frontier_indices.begin() + frontier_partition_offsets[1] + keys_to_sort_per_iteration * i;
      auto segment_frontier_per_type_degree_first = thrust::make_transform_iterator(
        thrust::make_zip_iterator(frontier_indices.begin(), frontier_edge_types.begin()) +
          frontier_partition_offsets[1] + keys_to_sort_per_iteration * i,
        cuda::proclaim_return_type<edge_t>(
          [frontier_per_type_degrees, num_edge_types] __device__(auto pair) {
            return frontier_per_type_degrees[thrust::get<0>(pair) * num_edge_types +
                                             thrust::get<1>(pair)];
          }));
      auto segment_frontier_type_first = frontier_edge_types.begin() +
                                         frontier_partition_offsets[1] +
                                         keys_to_sort_per_iteration * i;

      while (true) {
        std::optional<rmm::device_uvector<edge_t>> retry_per_type_nbr_indices{std::nullopt};
        std::optional<rmm::device_uvector<int32_t>> retry_sample_indices{std::nullopt};
        std::optional<rmm::device_uvector<edge_t>> retry_segment_sorted_per_type_nbr_indices{
          std::nullopt};
        std::optional<rmm::device_uvector<int32_t>> retry_segment_sorted_sample_indices{
          std::nullopt};

        if (retry_segment_indices) {
          retry_per_type_nbr_indices = rmm::device_uvector<edge_t>(
            (*retry_segment_indices).size() * high_partition_oversampling_K, handle.get_stream());
          retry_sample_indices =
            rmm::device_uvector<int32_t>((*retry_per_type_nbr_indices).size(), handle.get_stream());
          retry_segment_sorted_per_type_nbr_indices =
            rmm::device_uvector<edge_t>((*retry_per_type_nbr_indices).size(), handle.get_stream());
          retry_segment_sorted_sample_indices =
            rmm::device_uvector<int32_t>((*retry_per_type_nbr_indices).size(), handle.get_stream());
        }

        if (retry_segment_indices) {
          rmm::device_uvector<edge_t> tmp_per_type_degrees((*retry_segment_indices).size(),
                                                           handle.get_stream());
          thrust::gather(handle.get_thrust_policy(),
                         (*retry_segment_indices).begin(),
                         (*retry_segment_indices).end(),
                         segment_frontier_per_type_degree_first,
                         tmp_per_type_degrees.begin());
          sample_nbr_index_with_replacement<edge_t, bias_t>(
            handle,
            raft::device_span<edge_t const>(tmp_per_type_degrees.data(),
                                            tmp_per_type_degrees.size()),
            std::nullopt,
            raft::device_span<edge_t>((*retry_per_type_nbr_indices).data(),
                                      (*retry_per_type_nbr_indices).size()),
            rng_state,
            high_partition_oversampling_K);
        } else {
          rmm::device_uvector<edge_t> tmp_per_type_degrees(num_segments, handle.get_stream());
          thrust::copy(handle.get_thrust_policy(),
                       segment_frontier_per_type_degree_first,
                       segment_frontier_per_type_degree_first + num_segments,
                       tmp_per_type_degrees.begin());
          sample_nbr_index_with_replacement<edge_t, bias_t>(
            handle,
            raft::device_span<edge_t const>(tmp_per_type_degrees.data(),
                                            tmp_per_type_degrees.size()),
            std::nullopt,
            raft::device_span<edge_t>(tmp_per_type_nbr_indices.data(),
                                      tmp_per_type_nbr_indices.size()),
            rng_state,
            high_partition_oversampling_K);
        }

        if (retry_segment_indices) {
          thrust::for_each(
            handle.get_thrust_policy(),
            thrust::make_counting_iterator(size_t{0}),
            thrust::make_counting_iterator((*retry_segment_indices).size() *
                                           high_partition_oversampling_K),
            [high_partition_oversampling_K,
             unique_counts =
               raft::device_span<edge_t const>(unique_counts.data(), unique_counts.size()),
             segment_sorted_tmp_per_type_nbr_indices =
               segment_sorted_tmp_per_type_nbr_indices.data(),
             retry_segment_indices      = (*retry_segment_indices).data(),
             retry_per_type_nbr_indices = (*retry_per_type_nbr_indices).data(),
             retry_sample_indices       = (*retry_sample_indices).data()] __device__(size_t i) {
              auto segment_idx  = retry_segment_indices[i / high_partition_oversampling_K];
              auto sample_idx   = static_cast<edge_t>(i % high_partition_oversampling_K);
              auto unique_count = unique_counts[segment_idx];
              auto output_first = thrust::make_zip_iterator(
                thrust::make_tuple(retry_per_type_nbr_indices, retry_sample_indices));
              // sample index for the previously selected neighbor indices should be smaller than
              // the new candidates to ensure that the previously selected neighbor indices will
              // be selected again
              if (sample_idx < unique_count) {  // re-select the previous ones
                *(output_first + i) =
                  thrust::make_tuple(segment_sorted_tmp_per_type_nbr_indices
                                       [segment_idx * high_partition_oversampling_K + sample_idx],
                                     static_cast<int32_t>(sample_idx));
              } else {
                *(output_first + i) = thrust::make_tuple(retry_per_type_nbr_indices[i],
                                                         static_cast<int32_t>(sample_idx));
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
          retry_segment_indices ? (*retry_per_type_nbr_indices).data()
                                : tmp_per_type_nbr_indices.data(),
          retry_segment_indices ? (*retry_segment_sorted_per_type_nbr_indices).data()
                                : segment_sorted_tmp_per_type_nbr_indices.data(),
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
          retry_segment_indices ? (*retry_per_type_nbr_indices).data()
                                : tmp_per_type_nbr_indices.data(),
          retry_segment_indices ? (*retry_segment_sorted_per_type_nbr_indices).data()
                                : segment_sorted_tmp_per_type_nbr_indices.data(),
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
             unique_counts = raft::device_span<edge_t>(unique_counts.data(), unique_counts.size()),
             retry_segment_indices           = (*retry_segment_indices).data(),
             retry_segment_sorted_pair_first = thrust::make_zip_iterator(
               thrust::make_tuple((*retry_segment_sorted_per_type_nbr_indices).begin(),
                                  (*retry_segment_sorted_sample_indices).begin())),
             segment_sorted_pair_first = thrust::make_zip_iterator(
               segment_sorted_tmp_per_type_nbr_indices.begin(),
               segment_sorted_tmp_sample_indices.begin())] __device__(size_t i) {
              auto input_pair_first =
                retry_segment_sorted_pair_first + high_partition_oversampling_K * i;
              auto segment_idx = retry_segment_indices[i];
              auto output_pair_first =
                segment_sorted_pair_first + high_partition_oversampling_K * segment_idx;
              assert(high_partition_oversampling_K > 0);
              auto prev           = *input_pair_first;
              size_t unique_count = 1;
              for (size_t j = 1; j < high_partition_oversampling_K; ++j) {
                auto cur = *(input_pair_first + j);
                if (thrust::get<0>(cur) ==
                    thrust::get<0>(prev)) {  // update the sample index to the minimum
                  thrust::get<1>(prev) = cuda::std::min(thrust::get<1>(prev), thrust::get<1>(cur));
                } else {  // new unique neighbor index
                  *(output_pair_first + unique_count - 1) = prev;
                  ++unique_count;
                  prev = cur;
                }
              }
              *(output_pair_first + unique_count - 1) = prev;
              unique_counts[segment_idx]              = unique_count;
            });
        } else {
          thrust::for_each(
            handle.get_thrust_policy(),
            thrust::make_counting_iterator(size_t{0}),
            thrust::make_counting_iterator(num_segments),
            [high_partition_oversampling_K,
             unique_counts = raft::device_span<edge_t>(unique_counts.data(), unique_counts.size()),
             segment_sorted_pair_first = thrust::make_zip_iterator(
               segment_sorted_tmp_per_type_nbr_indices.begin(),
               segment_sorted_tmp_sample_indices.begin())] __device__(size_t i) {
              auto pair_first = segment_sorted_pair_first + high_partition_oversampling_K * i;
              assert(high_partition_oversampling_K > 0);
              thrust::tuple<edge_t, int32_t> prev = *pair_first;
              size_t unique_count                 = 1;
              for (size_t j = 1; j < high_partition_oversampling_K; ++j) {
                auto cur = *(pair_first + j);
                if (thrust::get<0>(cur) ==
                    thrust::get<0>(prev)) {  // update the sample index to the minimum
                  thrust::get<1>(prev) = cuda::std::min(thrust::get<1>(prev), thrust::get<1>(cur));
                } else {  // new unique neighbor index
                  *(pair_first + unique_count - 1) = prev;
                  ++unique_count;
                  prev = cur;
                }
              }
              *(pair_first + unique_count - 1) = prev;
              unique_counts[i]                 = unique_count;
            });
        }

        auto pair_first =
          thrust::make_zip_iterator(unique_counts.begin(), segment_frontier_type_first);
        if (retry_segment_indices) {
          auto last =
            thrust::remove_if(handle.get_thrust_policy(),
                              (*retry_segment_indices).begin(),
                              (*retry_segment_indices).end(),
                              [pair_first, K_offsets] __device__(auto segment_idx) {
                                auto pair = *(pair_first + segment_idx);
                                auto type = thrust::get<1>(pair);
                                return thrust::get<0>(pair) >=
                                       static_cast<edge_t>(K_offsets[type + 1] - K_offsets[type]);
                              });
          auto num_retry_segments = cuda::std::distance((*retry_segment_indices).begin(), last);
          if (num_retry_segments > 0) {
            (*retry_segment_indices).resize(num_retry_segments, handle.get_stream());
          } else {
            retry_segment_indices = std::nullopt;
          }
        } else {
          auto num_retry_segments = thrust::count_if(
            handle.get_thrust_policy(),
            pair_first,
            pair_first + unique_counts.size(),
            [K_offsets] __device__(auto pair) {
              auto count = thrust::get<0>(pair);
              auto type  = thrust::get<1>(pair);
              return count < static_cast<edge_t>(K_offsets[type + 1] - K_offsets[type]);
            });
          if (num_retry_segments > 0) {
            retry_segment_indices =
              rmm::device_uvector<size_t>(num_retry_segments, handle.get_stream());
            thrust::copy_if(handle.get_thrust_policy(),
                            thrust::make_counting_iterator(size_t{0}),
                            thrust::make_counting_iterator(num_segments),
                            (*retry_segment_indices).begin(),
                            [unique_counts = raft::device_span<edge_t const>(unique_counts.data(),
                                                                             unique_counts.size()),
                             segment_frontier_type_first,
                             K_offsets] __device__(size_t i) {
                              auto type = *(segment_frontier_type_first + i);
                              return unique_counts[i] <
                                     static_cast<edge_t>(K_offsets[type + 1] - K_offsets[type]);
                            });
          }
        }

        if (!retry_segment_indices) { break; }
      }

      // sort the segment-sorted (sample index, sample per-type neighbor index) pairs (key: sample
      // index)

      cub::DeviceSegmentedSort::SortPairs(
        static_cast<void*>(nullptr),
        tmp_storage_bytes,
        segment_sorted_tmp_sample_indices.data(),
        tmp_sample_indices.data(),
        segment_sorted_tmp_per_type_nbr_indices.data(),
        tmp_per_type_nbr_indices.data(),
        num_segments * high_partition_oversampling_K,
        num_segments,
        thrust::make_transform_iterator(thrust::make_counting_iterator(size_t{0}),
                                        multiplier_t<size_t>{high_partition_oversampling_K}),
        thrust::make_transform_iterator(
          thrust::make_counting_iterator(size_t{0}),
          cuda::proclaim_return_type<size_t>(
            [high_partition_oversampling_K,
             unique_counts = raft::device_span<edge_t const>(
               unique_counts.data(), unique_counts.size())] __device__(size_t i) {
              return i * high_partition_oversampling_K + unique_counts[i];
            })),
        handle.get_stream());
      if (tmp_storage_bytes > d_tmp_storage.size()) {
        d_tmp_storage = rmm::device_uvector<std::byte>(tmp_storage_bytes, handle.get_stream());
      }
      cub::DeviceSegmentedSort::SortPairs(
        d_tmp_storage.data(),
        tmp_storage_bytes,
        segment_sorted_tmp_sample_indices.data(),
        tmp_sample_indices.data(),
        segment_sorted_tmp_per_type_nbr_indices.data(),
        tmp_per_type_nbr_indices.data(),
        num_segments * high_partition_oversampling_K,
        num_segments,
        thrust::make_transform_iterator(thrust::make_counting_iterator(size_t{0}),
                                        multiplier_t<size_t>{high_partition_oversampling_K}),
        thrust::make_transform_iterator(
          thrust::make_counting_iterator(size_t{0}),
          cuda::proclaim_return_type<size_t>(
            [high_partition_oversampling_K,
             unique_counts = raft::device_span<edge_t const>(
               unique_counts.data(), unique_counts.size())] __device__(size_t i) {
              return i * high_partition_oversampling_K + unique_counts[i];
            })),
        handle.get_stream());

      // copy the neighbor indices back to nbr_indices

      rmm::device_uvector<size_t> output_count_offsets(num_segments + 1, handle.get_stream());
      output_count_offsets.set_element_to_zero_async(0, handle.get_stream());
      auto k_first = thrust::make_transform_iterator(
        segment_frontier_type_first,
        cuda::proclaim_return_type<size_t>(
          [K_offsets] __device__(auto type) { return K_offsets[type + 1] - K_offsets[type]; }));
      thrust::inclusive_scan(handle.get_thrust_policy(),
                             k_first,
                             k_first + num_segments,
                             output_count_offsets.begin() + 1);
      thrust::for_each(
        handle.get_thrust_policy(),
        thrust::make_counting_iterator(size_t{0}),
        thrust::make_counting_iterator(output_count_offsets.back_element(handle.get_stream())),
        [high_partition_oversampling_K,
         segment_frontier_index_first,
         segment_frontier_type_first,
         tmp_per_type_nbr_indices = raft::device_span<edge_t const>(
           tmp_per_type_nbr_indices.data(), tmp_per_type_nbr_indices.size()),
         output_count_offsets = raft::device_span<size_t const>(output_count_offsets.data(),
                                                                output_count_offsets.size()),
         per_type_nbr_indices =
           raft::device_span<edge_t>(per_type_nbr_indices.data(), per_type_nbr_indices.size()),
         K_offsets,
         K_sum] __device__(size_t i) {
          auto idx = cuda::std::distance(
            output_count_offsets.begin() + 1,
            thrust::upper_bound(
              thrust::seq, output_count_offsets.begin() + 1, output_count_offsets.end(), i));
          auto frontier_idx = *(segment_frontier_index_first + idx);
          auto type         = *(segment_frontier_type_first + idx);
          auto sample_idx   = static_cast<edge_t>(i - output_count_offsets[idx]);
          *(per_type_nbr_indices.begin() + frontier_idx * K_sum + K_offsets[type] + sample_idx) =
            *(tmp_per_type_nbr_indices.begin() + idx * high_partition_oversampling_K + sample_idx);
        });
    }
  }

  return per_type_nbr_indices;
}

template <typename edge_t, typename bias_t>
void compute_homogeneous_biased_sampling_index_without_replacement(
  raft::handle_t const& handle,
  std::optional<raft::device_span<size_t const>>
    input_frontier_indices,  // input_degree_offsets & input_biases
                             // are already packed if std::nullopt
  raft::device_span<size_t const> input_degree_offsets,
  raft::device_span<bias_t const> input_biases,  // bias 0 edges can't be selected
  std::optional<raft::device_span<size_t const>>
    output_frontier_indices,  // output_nbr_indices is already packed if std::nullopt
  raft::device_span<edge_t> output_nbr_indices,
  std::optional<raft::device_span<bias_t>> output_keys,
  raft::random::RngState& rng_state,
  size_t K,
  bool jump)
{
  if (jump) {  // Algorithm A-ExpJ
    CUGRAPH_FAIL(
      "unimplemented.");  // FIXME: this could be faster especially for high-degree vertices
  } else {                // Algorithm A-Res
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
        ? (*packed_input_degree_offsets).data() + (*packed_input_degree_offsets).size() - 1
        : input_degree_offsets.data() + input_degree_offsets.size() - 1,
      1,
      handle.get_stream());
    handle.sync_stream();

    auto approx_edges_to_process_per_iteration =
      static_cast<size_t>(handle.get_device_properties().multiProcessorCount) *
      (1 << 18) /* tuning parameter */;
    auto [chunk_offsets, element_offsets] = cugraph::detail::compute_offset_aligned_element_chunks(
      handle,
      raft::device_span<size_t const>(
        packed_input_degree_offsets ? (*packed_input_degree_offsets).data()
                                    : input_degree_offsets.data(),
        packed_input_degree_offsets ? (*packed_input_degree_offsets).size()
                                    : input_degree_offsets.size()),
      num_pairs,
      approx_edges_to_process_per_iteration);
    auto num_chunks = chunk_offsets.size() - 1;
    for (size_t i = 0; i < num_chunks; ++i) {
      auto num_chunk_pairs = element_offsets[i + 1] - element_offsets[i];
      rmm::device_uvector<bias_t> keys(num_chunk_pairs, handle.get_stream());

      cugraph::detail::uniform_random_fill(
        handle.get_stream(), keys.data(), keys.size(), bias_t{0.0}, bias_t{1.0}, rng_state);

      if (packed_input_degree_offsets) {
        auto bias_first = thrust::make_transform_iterator(
          thrust::make_counting_iterator(element_offsets[i]),
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
              auto idx          = cuda::std::distance(packed_input_degree_offsets.begin() + 1, it);
              auto frontier_idx = frontier_indices[idx];
              return input_biases[input_degree_offsets[frontier_idx] +
                                  (i - packed_input_degree_offsets[idx])];
            }));
        thrust::transform(
          handle.get_thrust_policy(),
          keys.begin(),
          keys.end(),
          bias_first,
          keys.begin(),
          cuda::proclaim_return_type<bias_t>([] __device__(bias_t r, bias_t b) {
            assert(b >
                   0.0);  // 0 bias neighbors shold be pre-filtered before invoking this function
            return cuda::std::min(-cuda::std::log(r) / b, std::numeric_limits<bias_t>::max());
          }));
      } else {
        thrust::transform(
          handle.get_thrust_policy(),
          keys.begin(),
          keys.end(),
          input_biases.begin() + element_offsets[i],
          keys.begin(),
          cuda::proclaim_return_type<bias_t>([] __device__(bias_t r, bias_t b) {
            assert(b >
                   0.0);  // 0 bias neighbors shold be pre-filtered before invoking this function
            return cuda::std::min(-cuda::std::log(r) / b, std::numeric_limits<bias_t>::max());
          }));
      }

      rmm::device_uvector<edge_t> nbr_indices(keys.size(), handle.get_stream());
      thrust::tabulate(handle.get_thrust_policy(),
                       nbr_indices.begin(),
                       nbr_indices.end(),
                       [offsets        = packed_input_degree_offsets
                                           ? raft::device_span<size_t const>(
                                        (*packed_input_degree_offsets).data() + chunk_offsets[i],
                                        chunk_offsets[i + 1] - chunk_offsets[i])
                                           : raft::device_span<size_t const>(
                                        input_degree_offsets.data() + chunk_offsets[i],
                                        chunk_offsets[i + 1] - chunk_offsets[i]),
                        element_offset = element_offsets[i]] __device__(size_t i) {
                         auto idx = cuda::std::distance(
                           offsets.begin() + 1,
                           thrust::upper_bound(
                             thrust::seq, offsets.begin() + 1, offsets.end(), element_offset + i));
                         return static_cast<edge_t>((element_offset + i) - offsets[idx]);
                       });

      // pick top K for each frontier index

      rmm::device_uvector<std::byte> d_tmp_storage(0, handle.get_stream());
      size_t tmp_storage_bytes{0};

      rmm::device_uvector<bias_t> segment_sorted_keys(keys.size(), handle.get_stream());
      rmm::device_uvector<edge_t> segment_sorted_nbr_indices(nbr_indices.size(),
                                                             handle.get_stream());

      auto offset_first = thrust::make_transform_iterator(
        (packed_input_degree_offsets ? (*packed_input_degree_offsets).begin()
                                     : input_degree_offsets.begin()) +
          chunk_offsets[i],
        detail::shift_left_t<size_t>{element_offsets[i]});
      cub::DeviceSegmentedSort::SortPairs(static_cast<void*>(nullptr),
                                          tmp_storage_bytes,
                                          keys.data(),
                                          segment_sorted_keys.data(),
                                          nbr_indices.data(),
                                          segment_sorted_nbr_indices.data(),
                                          keys.size(),
                                          chunk_offsets[i + 1] - chunk_offsets[i],
                                          offset_first,
                                          offset_first + 1,
                                          handle.get_stream());
      if (tmp_storage_bytes > d_tmp_storage.size()) {
        d_tmp_storage = rmm::device_uvector<std::byte>(tmp_storage_bytes, handle.get_stream());
      }
      cub::DeviceSegmentedSort::SortPairs(d_tmp_storage.data(),
                                          tmp_storage_bytes,
                                          keys.data(),
                                          segment_sorted_keys.data(),
                                          nbr_indices.data(),
                                          segment_sorted_nbr_indices.data(),
                                          keys.size(),
                                          chunk_offsets[i + 1] - chunk_offsets[i],
                                          offset_first,
                                          offset_first + 1,
                                          handle.get_stream());

      if (output_frontier_indices) {
        thrust::for_each(
          handle.get_thrust_policy(),
          thrust::make_counting_iterator(size_t{0}),
          thrust::make_counting_iterator((chunk_offsets[i + 1] - chunk_offsets[i]) * K),
          [input_degree_offsets =
             packed_input_degree_offsets
               ? raft::device_span<size_t const>((*packed_input_degree_offsets).data(),
                                                 (*packed_input_degree_offsets).size())
               : input_degree_offsets,
           idx_offset              = chunk_offsets[i] * K,
           output_frontier_indices = *output_frontier_indices,
           output_keys,
           output_nbr_indices,
           segment_sorted_keys        = raft::device_span<bias_t const>(segment_sorted_keys.data(),
                                                                 segment_sorted_keys.size()),
           segment_sorted_nbr_indices = raft::device_span<edge_t const>(
             segment_sorted_nbr_indices.data(), segment_sorted_nbr_indices.size()),
           K,
           invalid_idx = cugraph::invalid_edge_id_v<edge_t>] __device__(size_t i) {
            auto idx                 = idx_offset + i;
            auto key_idx             = idx / K;
            auto output_frontier_idx = output_frontier_indices[key_idx];
            auto output_idx          = output_frontier_idx * K + (idx % K);
            auto degree = input_degree_offsets[key_idx + 1] - input_degree_offsets[key_idx];
            auto segment_sorted_input_idx =
              (input_degree_offsets[key_idx] - input_degree_offsets[idx_offset / K]) + (idx % K);
            if ((idx % K) < degree) {
              if (output_keys) {
                (*output_keys)[output_idx] = segment_sorted_keys[segment_sorted_input_idx];
              }
              output_nbr_indices[output_idx] = segment_sorted_nbr_indices[segment_sorted_input_idx];
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
          thrust::make_counting_iterator((chunk_offsets[i + 1] - chunk_offsets[i]) * K),
          [input_degree_offsets =
             packed_input_degree_offsets
               ? raft::device_span<size_t const>((*packed_input_degree_offsets).data(),
                                                 (*packed_input_degree_offsets).size())
               : input_degree_offsets,
           idx_offset = chunk_offsets[i] * K,
           output_keys,
           output_nbr_indices,
           segment_sorted_keys        = raft::device_span<bias_t const>(segment_sorted_keys.data(),
                                                                 segment_sorted_keys.size()),
           segment_sorted_nbr_indices = raft::device_span<edge_t const>(
             segment_sorted_nbr_indices.data(), segment_sorted_nbr_indices.size()),
           K,
           invalid_idx = cugraph::invalid_edge_id_v<edge_t>] __device__(size_t i) {
            auto idx     = idx_offset + i;
            auto key_idx = idx / K;
            auto degree  = input_degree_offsets[key_idx + 1] - input_degree_offsets[key_idx];
            auto segment_sorted_input_idx =
              (input_degree_offsets[key_idx] - input_degree_offsets[idx_offset / K]) + (idx % K);
            if ((idx % K) < degree) {
              if (output_keys) {
                (*output_keys)[idx] = segment_sorted_keys[segment_sorted_input_idx];
              }
              output_nbr_indices[idx] = segment_sorted_nbr_indices[segment_sorted_input_idx];
            } else {
              if (output_keys) { (*output_keys)[idx] = std::numeric_limits<bias_t>::infinity(); }
              output_nbr_indices[idx] = invalid_idx;
            }
          });
      }
    }
  }

  return;
}

template <typename edge_t, typename edge_type_t, typename bias_t>
void compute_heterogeneous_biased_sampling_index_without_replacement(
  raft::handle_t const& handle,
  std::optional<raft::device_span<size_t const>>
    input_frontier_indices,  // input_per_tyep_degree_offsets & input_biases are already packed if
                             // std::nullopt
  raft::device_span<edge_type_t const> input_frontier_edge_types,
  raft::device_span<size_t const> input_per_type_degree_offsets,
  raft::device_span<bias_t const> input_biases,  // bias 0 edges can't be selected
  raft::device_span<size_t const> output_start_displacements,
  raft::device_span<edge_t> output_per_type_nbr_indices,
  std::optional<raft::device_span<bias_t>> output_keys,
  raft::random::RngState& rng_state,
  raft::device_span<size_t const> K_offsets,
  bool jump)
{
  auto num_edge_types = static_cast<edge_type_t>(K_offsets.size() - 1);

  if (jump) {  // Algorithm A-ExpJ
    CUGRAPH_FAIL(
      "unimplemented.");  // FIXME: this could be faster especially for high-degree vertices
  } else {                // Algorithm A-Res
    // update packed input degree offsets if input_frontier_indices.has_value() is true

    auto packed_input_per_type_degree_offsets =
      input_frontier_indices ? std::make_optional<rmm::device_uvector<size_t>>(
                                 (*input_frontier_indices).size() + 1, handle.get_stream())
                             : std::nullopt;
    if (packed_input_per_type_degree_offsets) {
      (*packed_input_per_type_degree_offsets).set_element_to_zero_async(0, handle.get_stream());
      auto per_type_degree_first = thrust::make_transform_iterator(
        thrust::make_zip_iterator((*input_frontier_indices).begin(),
                                  input_frontier_edge_types.begin()),
        cuda::proclaim_return_type<size_t>(
          [input_per_type_degree_offsets, num_edge_types] __device__(auto pair) {
            auto idx  = thrust::get<0>(pair);
            auto type = thrust::get<1>(pair);
            return input_per_type_degree_offsets[idx * num_edge_types + type + 1] -
                   input_per_type_degree_offsets[idx * num_edge_types + type];
          }));
      thrust::inclusive_scan(handle.get_thrust_policy(),
                             per_type_degree_first,
                             per_type_degree_first + (*input_frontier_indices).size(),
                             (*packed_input_per_type_degree_offsets).begin() + 1);
    }

    // generate (key, nbr_index) pairs

    size_t num_pairs{};
    raft::update_host(
      &num_pairs,
      packed_input_per_type_degree_offsets
        ? (*packed_input_per_type_degree_offsets).data() +
            (*packed_input_per_type_degree_offsets).size() - 1
        : input_per_type_degree_offsets.data() + input_per_type_degree_offsets.size() - 1,
      1,
      handle.get_stream());
    handle.sync_stream();

    auto approx_edges_to_process_per_iteration =
      static_cast<size_t>(handle.get_device_properties().multiProcessorCount) *
      (1 << 18) /* tuning parameter */;
    auto [chunk_offsets, element_offsets] = cugraph::detail::compute_offset_aligned_element_chunks(
      handle,
      raft::device_span<size_t const>(
        packed_input_per_type_degree_offsets ? (*packed_input_per_type_degree_offsets).data()
                                             : input_per_type_degree_offsets.data(),
        packed_input_per_type_degree_offsets ? (*packed_input_per_type_degree_offsets).size()
                                             : input_per_type_degree_offsets.size()),
      num_pairs,
      approx_edges_to_process_per_iteration);
    auto num_chunks = chunk_offsets.size() - 1;
    for (size_t i = 0; i < num_chunks; ++i) {
      auto num_chunk_pairs = element_offsets[i + 1] - element_offsets[i];
      rmm::device_uvector<bias_t> keys(num_chunk_pairs, handle.get_stream());

      cugraph::detail::uniform_random_fill(
        handle.get_stream(), keys.data(), keys.size(), bias_t{0.0}, bias_t{1.0}, rng_state);

      if (packed_input_per_type_degree_offsets) {
        auto bias_first = thrust::make_transform_iterator(
          thrust::make_counting_iterator(element_offsets[i]),
          cuda::proclaim_return_type<bias_t>(
            [input_biases,
             input_per_type_degree_offsets,
             frontier_indices = *input_frontier_indices,
             frontier_types   = input_frontier_edge_types,
             packed_input_per_type_degree_offsets =
               raft::device_span<size_t const>((*packed_input_per_type_degree_offsets).data(),
                                               (*packed_input_per_type_degree_offsets).size()),
             num_edge_types] __device__(size_t i) {
              auto it  = thrust::upper_bound(thrust::seq,
                                            packed_input_per_type_degree_offsets.begin() + 1,
                                            packed_input_per_type_degree_offsets.end(),
                                            i);
              auto idx = cuda::std::distance(packed_input_per_type_degree_offsets.begin() + 1, it);
              auto frontier_idx = frontier_indices[idx];
              auto type         = frontier_types[idx];
              return input_biases[input_per_type_degree_offsets[frontier_idx * num_edge_types +
                                                                type] +
                                  (i - packed_input_per_type_degree_offsets[idx])];
            }));
        thrust::transform(
          handle.get_thrust_policy(),
          keys.begin(),
          keys.end(),
          bias_first,
          keys.begin(),
          cuda::proclaim_return_type<bias_t>([] __device__(bias_t r, bias_t b) {
            assert(b >
                   0.0);  // 0 bias neighbors shold be pre-filtered before invoking this function
            return cuda::std::min(-cuda::std::log(r) / b, std::numeric_limits<bias_t>::max());
          }));
      } else {
        thrust::transform(
          handle.get_thrust_policy(),
          keys.begin(),
          keys.end(),
          input_biases.begin() + element_offsets[i],
          keys.begin(),
          cuda::proclaim_return_type<bias_t>([] __device__(bias_t r, bias_t b) {
            assert(b >
                   0.0);  // 0 bias neighbors shold be pre-filtered before invoking this function
            return cuda::std::min(-cuda::std::log(r) / b, std::numeric_limits<bias_t>::max());
          }));
      }

      rmm::device_uvector<edge_t> per_type_nbr_indices(keys.size(), handle.get_stream());
      thrust::tabulate(
        handle.get_thrust_policy(),
        per_type_nbr_indices.begin(),
        per_type_nbr_indices.end(),
        [offsets        = packed_input_per_type_degree_offsets
                            ? raft::device_span<size_t const>(
                         (*packed_input_per_type_degree_offsets).data() + chunk_offsets[i],
                         chunk_offsets[i + 1] - chunk_offsets[i])
                            : raft::device_span<size_t const>(
                         input_per_type_degree_offsets.data() + chunk_offsets[i],
                         chunk_offsets[i + 1] - chunk_offsets[i]),
         element_offset = element_offsets[i]] __device__(size_t i) {
          auto idx = cuda::std::distance(
            offsets.begin() + 1,
            thrust::upper_bound(
              thrust::seq, offsets.begin() + 1, offsets.end(), element_offset + i));
          return static_cast<edge_t>((element_offset + i) - offsets[idx]);
        });

      // pick top K for each frontier index

      rmm::device_uvector<std::byte> d_tmp_storage(0, handle.get_stream());
      size_t tmp_storage_bytes{0};

      rmm::device_uvector<bias_t> segment_sorted_keys(keys.size(), handle.get_stream());
      rmm::device_uvector<edge_t> segment_sorted_per_type_nbr_indices(per_type_nbr_indices.size(),
                                                                      handle.get_stream());

      auto offset_first = thrust::make_transform_iterator(
        (packed_input_per_type_degree_offsets ? (*packed_input_per_type_degree_offsets).begin()
                                              : input_per_type_degree_offsets.begin()) +
          chunk_offsets[i],
        detail::shift_left_t<size_t>{element_offsets[i]});
      cub::DeviceSegmentedSort::SortPairs(static_cast<void*>(nullptr),
                                          tmp_storage_bytes,
                                          keys.data(),
                                          segment_sorted_keys.data(),
                                          per_type_nbr_indices.data(),
                                          segment_sorted_per_type_nbr_indices.data(),
                                          keys.size(),
                                          chunk_offsets[i + 1] - chunk_offsets[i],
                                          offset_first,
                                          offset_first + 1,
                                          handle.get_stream());
      if (tmp_storage_bytes > d_tmp_storage.size()) {
        d_tmp_storage = rmm::device_uvector<std::byte>(tmp_storage_bytes, handle.get_stream());
      }
      cub::DeviceSegmentedSort::SortPairs(d_tmp_storage.data(),
                                          tmp_storage_bytes,
                                          keys.data(),
                                          segment_sorted_keys.data(),
                                          per_type_nbr_indices.data(),
                                          segment_sorted_per_type_nbr_indices.data(),
                                          keys.size(),
                                          chunk_offsets[i + 1] - chunk_offsets[i],
                                          offset_first,
                                          offset_first + 1,
                                          handle.get_stream());

      thrust::for_each(
        handle.get_thrust_policy(),
        thrust::make_counting_iterator(size_t{0}),
        thrust::make_counting_iterator(chunk_offsets[i + 1] - chunk_offsets[i]),
        [input_frontier_edge_types,
         input_per_type_degree_offsets =
           packed_input_per_type_degree_offsets
             ? raft::device_span<size_t const>((*packed_input_per_type_degree_offsets).data(),
                                               (*packed_input_per_type_degree_offsets).size())
             : input_per_type_degree_offsets,
         chunk_offset = chunk_offsets[i],
         output_start_displacements,
         output_per_type_nbr_indices,
         output_keys,
         segment_sorted_keys =
           raft::device_span<bias_t const>(segment_sorted_keys.data(), segment_sorted_keys.size()),
         segment_sorted_per_type_nbr_indices = raft::device_span<edge_t const>(
           segment_sorted_per_type_nbr_indices.data(), segment_sorted_per_type_nbr_indices.size()),
         K_offsets,
         invalid_idx = cugraph::invalid_edge_id_v<edge_t>] __device__(size_t i) {
          auto key_idx         = chunk_offset + i;
          auto type            = input_frontier_edge_types[key_idx];
          auto K               = static_cast<edge_t>(K_offsets[type + 1] - K_offsets[type]);
          auto per_type_degree = static_cast<edge_t>(input_per_type_degree_offsets[key_idx + 1] -
                                                     input_per_type_degree_offsets[key_idx]);
          auto segment_sorted_input_start_offset =
            input_per_type_degree_offsets[key_idx] - input_per_type_degree_offsets[chunk_offset];
          auto output_start_offset = output_start_displacements[key_idx];
          edge_t j                 = 0;
          for (; j < cuda::std::min(per_type_degree, K); ++j) {
            auto segment_sorted_input_idx = segment_sorted_input_start_offset + j;
            auto output_idx               = output_start_offset + j;
            if (output_keys) {
              (*output_keys)[output_idx] = segment_sorted_keys[segment_sorted_input_idx];
            }
            output_per_type_nbr_indices[output_idx] =
              segment_sorted_per_type_nbr_indices[segment_sorted_input_idx];
          }
          for (; j < K; ++j) {
            auto output_idx = output_start_offset + j;
            if (output_keys) {
              (*output_keys)[output_idx] = std::numeric_limits<bias_t>::infinity();
            }
            output_per_type_nbr_indices[output_idx] = invalid_idx;
          }
        });
    }
  }

  return;
}

template <typename GraphViewType, typename VertexIterator>
rmm::device_uvector<typename GraphViewType::edge_type>
compute_aggregate_local_frontier_local_degrees(raft::handle_t const& handle,
                                               GraphViewType const& graph_view,
                                               VertexIterator aggregate_local_frontier_major_first,
                                               raft::host_span<size_t const> local_frontier_offsets)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  static_assert(
    std::is_same_v<typename thrust::iterator_traits<VertexIterator>::value_type, vertex_t>);

  auto edge_mask_view = graph_view.edge_mask_view();

  auto aggregate_local_frontier_local_degrees =
    rmm::device_uvector<edge_t>(local_frontier_offsets.back(), handle.get_stream());
  for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
    auto edge_partition =
      edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
        graph_view.local_edge_partition_view(i));
    auto edge_partition_e_mask =
      edge_mask_view
        ? cuda::std::make_optional<
            detail::edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool>>(
            *edge_mask_view, i)
        : cuda::std::nullopt;

    auto edge_partition_frontier_local_degrees =
      !edge_partition_e_mask
        ? edge_partition.compute_local_degrees(
            aggregate_local_frontier_major_first + local_frontier_offsets[i],
            aggregate_local_frontier_major_first + local_frontier_offsets[i + 1],
            handle.get_stream())
        : edge_partition.compute_local_degrees_with_mask(
            (*edge_partition_e_mask).value_first(),
            aggregate_local_frontier_major_first + local_frontier_offsets[i],
            aggregate_local_frontier_major_first + local_frontier_offsets[i + 1],
            handle.get_stream());

    // FIXME: this copy is unnecessary if edge_partition.compute_local_degrees() takes a pointer
    // to the output array
    thrust::copy(handle.get_thrust_policy(),
                 edge_partition_frontier_local_degrees.begin(),
                 edge_partition_frontier_local_degrees.end(),
                 aggregate_local_frontier_local_degrees.begin() + local_frontier_offsets[i]);
  }

  return aggregate_local_frontier_local_degrees;
}

template <typename edge_t, typename edge_type_t>
rmm::device_uvector<edge_t> compute_aggregate_local_frontier_per_type_local_degrees(
  raft::handle_t const& handle,
  raft::device_span<size_t const> aggregate_local_frontier_key_idx_to_unique_key_idx,
  raft::host_span<size_t const> local_frontier_offsets,
  raft::device_span<edge_type_t const>
    aggregate_local_frontier_unique_key_edge_types,  // sorted by (key, edge_type)
  raft::device_span<size_t const> aggregate_local_frontier_unique_key_local_degree_offsets,
  raft::host_span<size_t const> local_frontier_unique_key_offsets,
  size_t num_edge_types)
{
  auto aggregate_local_frontier_per_type_local_degrees = rmm::device_uvector<edge_t>(
    local_frontier_offsets.back() * num_edge_types, handle.get_stream());
  for (size_t i = 0; i < local_frontier_offsets.size() - 1; ++i) {
    thrust::tabulate(
      handle.get_thrust_policy(),
      aggregate_local_frontier_per_type_local_degrees.begin() +
        local_frontier_offsets[i] * num_edge_types,
      aggregate_local_frontier_per_type_local_degrees.begin() +
        local_frontier_offsets[i + 1] * num_edge_types,
      [key_idx_to_unique_key_idx = raft::device_span<size_t const>(
         aggregate_local_frontier_key_idx_to_unique_key_idx.data() + local_frontier_offsets[i],
         local_frontier_offsets[i + 1] - local_frontier_offsets[i]),
       aggregate_local_frontier_unique_key_edge_types = raft::device_span<edge_type_t const>(
         aggregate_local_frontier_unique_key_edge_types.data(),
         aggregate_local_frontier_unique_key_edge_types.size()),
       unique_key_local_degree_offsets = raft::device_span<size_t const>(
         aggregate_local_frontier_unique_key_local_degree_offsets.data() +
           local_frontier_unique_key_offsets[i],
         (local_frontier_unique_key_offsets[i + 1] - local_frontier_unique_key_offsets[i]) + 1),
       num_edge_types] __device__(size_t i) {
        auto key_idx        = i / num_edge_types;
        auto edge_type      = static_cast<edge_type_t>(i % num_edge_types);
        auto unique_key_idx = key_idx_to_unique_key_idx[key_idx];
        auto start_offset   = unique_key_local_degree_offsets[unique_key_idx];
        auto end_offset     = unique_key_local_degree_offsets[unique_key_idx + 1];
        auto edge_type_first =
          aggregate_local_frontier_unique_key_edge_types.begin() + start_offset;
        auto edge_type_last = aggregate_local_frontier_unique_key_edge_types.begin() + end_offset;
        return static_cast<edge_t>(cuda::std::distance(
          thrust::lower_bound(thrust::seq, edge_type_first, edge_type_last, edge_type),
          thrust::upper_bound(thrust::seq, edge_type_first, edge_type_last, edge_type)));
      });
  }

  return aggregate_local_frontier_per_type_local_degrees;
}

// return (bias values, local neighbor indices with non-zero bias values, segment offsets) pairs for
// each key in th eaggregate local frontier
template <typename GraphViewType,
          typename KeyIterator,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename BiasEdgeOp>
std::tuple<rmm::device_uvector<
             typename edge_op_result_type<typename thrust::iterator_traits<KeyIterator>::value_type,
                                          typename GraphViewType::vertex_type,
                                          typename EdgeSrcValueInputWrapper::value_type,
                                          typename EdgeDstValueInputWrapper::value_type,
                                          typename EdgeValueInputWrapper::value_type,
                                          BiasEdgeOp>::type>,
           rmm::device_uvector<typename GraphViewType::edge_type>,
           rmm::device_uvector<size_t>>
compute_aggregate_local_frontier_biases(raft::handle_t const& handle,
                                        GraphViewType const& graph_view,
                                        KeyIterator aggregate_local_frontier_key_first,
                                        EdgeSrcValueInputWrapper edge_src_value_input,
                                        EdgeDstValueInputWrapper edge_dst_value_input,
                                        EdgeValueInputWrapper edge_value_input,
                                        BiasEdgeOp bias_e_op,
                                        raft::host_span<size_t const> local_frontier_offsets,
                                        bool do_expensive_check)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using key_t    = typename thrust::iterator_traits<KeyIterator>::value_type;

  using bias_t = typename edge_op_result_type<key_t,
                                              vertex_t,
                                              typename EdgeSrcValueInputWrapper::value_type,
                                              typename EdgeDstValueInputWrapper::value_type,
                                              typename EdgeValueInputWrapper::value_type,
                                              BiasEdgeOp>::type;

  // 1. collect bias values from local neighbors

  std::vector<size_t> local_frontier_sizes(local_frontier_offsets.size() - 1);
  std::adjacent_difference(
    local_frontier_offsets.begin() + 1, local_frontier_offsets.end(), local_frontier_sizes.begin());
  auto [aggregate_local_frontier_biases, aggregate_local_frontier_local_degree_offsets] =
    transform_v_frontier_e(
      handle,
      graph_view,
      aggregate_local_frontier_key_first,
      edge_src_value_input,
      edge_dst_value_input,
      edge_value_input,
      bias_e_op,
      raft::host_span<size_t const>(local_frontier_offsets.data(), local_frontier_offsets.size()));

  // 2. expensive check

  if (do_expensive_check) {
    auto num_invalid_biases = thrust::count_if(
      handle.get_thrust_policy(),
      aggregate_local_frontier_biases.begin(),
      aggregate_local_frontier_biases.end(),
      check_out_of_range_t<bias_t>{bias_t{0.0}, std::numeric_limits<bias_t>::max()});
    if constexpr (GraphViewType::is_multi_gpu) {
      num_invalid_biases = host_scalar_allreduce(
        handle.get_comms(), num_invalid_biases, raft::comms::op_t::SUM, handle.get_stream());
    }
    CUGRAPH_EXPECTS(num_invalid_biases == 0,
                    "invalid_input_argument: bias_e_op return values should be non-negative and "
                    "should not exceed std::numeirc_limits<bias_t>::max().");
  }

  // 3. exclude 0 bias neighbors & update offsets

  rmm::device_uvector<edge_t> aggregate_local_frontier_nz_bias_indices(
    aggregate_local_frontier_biases.size(), handle.get_stream());
  thrust::tabulate(handle.get_thrust_policy(),
                   aggregate_local_frontier_nz_bias_indices.begin(),
                   aggregate_local_frontier_nz_bias_indices.end(),
                   [offsets = raft::device_span<size_t const>(
                      aggregate_local_frontier_local_degree_offsets.data(),
                      aggregate_local_frontier_local_degree_offsets.size())] __device__(size_t i) {
                     auto it =
                       thrust::upper_bound(thrust::seq, offsets.begin() + 1, offsets.end(), i);
                     auto idx = cuda::std::distance(offsets.begin() + 1, it);
                     return static_cast<edge_t>(i - offsets[idx]);
                   });

  rmm::device_uvector<size_t> aggregate_local_frontier_local_degrees(local_frontier_offsets.back(),
                                                                     handle.get_stream());
  thrust::adjacent_difference(handle.get_thrust_policy(),
                              aggregate_local_frontier_local_degree_offsets.begin() + 1,
                              aggregate_local_frontier_local_degree_offsets.end(),
                              aggregate_local_frontier_local_degrees.begin());

  {
    auto pair_first = thrust::make_zip_iterator(aggregate_local_frontier_biases.begin(),
                                                thrust::make_counting_iterator(size_t{0}));
    thrust::for_each(handle.get_thrust_policy(),
                     pair_first,
                     pair_first + aggregate_local_frontier_biases.size(),
                     [offsets = raft::device_span<size_t const>(
                        aggregate_local_frontier_local_degree_offsets.data(),
                        aggregate_local_frontier_local_degree_offsets.size()),
                      degrees = raft::device_span<size_t>(
                        aggregate_local_frontier_local_degrees.data(),
                        aggregate_local_frontier_local_degrees.size())] __device__(auto pair) {
                       auto bias = thrust::get<0>(pair);
                       if (bias == 0.0) {
                         auto i   = thrust::get<1>(pair);
                         auto idx = cuda::std::distance(
                           offsets.begin() + 1,
                           thrust::upper_bound(thrust::seq, offsets.begin() + 1, offsets.end(), i));
                         cuda::atomic_ref<size_t, cuda::thread_scope_device> degree(degrees[idx]);
                         degree.fetch_sub(size_t{1}, cuda::std::memory_order_relaxed);
                       }
                     });
  }

  thrust::inclusive_scan(handle.get_thrust_policy(),
                         aggregate_local_frontier_local_degrees.begin(),
                         aggregate_local_frontier_local_degrees.end(),
                         aggregate_local_frontier_local_degree_offsets.begin() + 1);

  {
    auto pair_first = thrust::make_zip_iterator(aggregate_local_frontier_biases.begin(),
                                                aggregate_local_frontier_nz_bias_indices.begin());
    auto pair_last =
      thrust::remove_if(handle.get_thrust_policy(),
                        pair_first,
                        pair_first + aggregate_local_frontier_biases.size(),
                        [] __device__(auto pair) { return thrust::get<0>(pair) == 0.0; });
    aggregate_local_frontier_biases.resize(cuda::std::distance(pair_first, pair_last),
                                           handle.get_stream());
    aggregate_local_frontier_nz_bias_indices.resize(cuda::std::distance(pair_first, pair_last),
                                                    handle.get_stream());
    aggregate_local_frontier_biases.shrink_to_fit(handle.get_stream());
    aggregate_local_frontier_nz_bias_indices.shrink_to_fit(handle.get_stream());
  }

  return std::make_tuple(std::move(aggregate_local_frontier_biases),
                         std::move(aggregate_local_frontier_nz_bias_indices),
                         std::move(aggregate_local_frontier_local_degree_offsets));
}

// return (bias values, edge types, local neighbor indices with non-zero bias values, segment
// offsets) triplets for each key in th eaggregate local frontier
template <typename GraphViewType,
          typename KeyIterator,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename BiasEdgeOp,
          typename EdgeTypeInputWrapper>
std::tuple<rmm::device_uvector<
             typename edge_op_result_type<typename thrust::iterator_traits<KeyIterator>::value_type,
                                          typename GraphViewType::vertex_type,
                                          typename EdgeSrcValueInputWrapper::value_type,
                                          typename EdgeDstValueInputWrapper::value_type,
                                          typename EdgeValueInputWrapper::value_type,
                                          BiasEdgeOp>::type>,
           rmm::device_uvector<typename EdgeTypeInputWrapper::value_type>,
           rmm::device_uvector<typename GraphViewType::edge_type>,
           rmm::device_uvector<size_t>>
compute_aggregate_local_frontier_bias_type_pairs(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  KeyIterator aggregate_local_frontier_key_first,
  EdgeSrcValueInputWrapper edge_src_value_input,
  EdgeDstValueInputWrapper edge_dst_value_input,
  EdgeValueInputWrapper edge_value_input,
  BiasEdgeOp bias_e_op,
  EdgeTypeInputWrapper edge_type_input,
  raft::host_span<size_t const> local_frontier_offsets,
  bool do_expensive_check)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using key_t    = typename thrust::iterator_traits<KeyIterator>::value_type;

  using edge_value_t = typename EdgeValueInputWrapper::value_type;
  using edge_type_t  = typename EdgeTypeInputWrapper::value_type;
  using bias_t       = typename edge_op_result_type<key_t,
                                              vertex_t,
                                              typename EdgeSrcValueInputWrapper::value_type,
                                              typename EdgeDstValueInputWrapper::value_type,
                                              typename EdgeValueInputWrapper::value_type,
                                              BiasEdgeOp>::type;
  static_assert(std::is_arithmetic_v<bias_t>);
  static_assert(std::is_integral_v<edge_type_t>);

  std::vector<size_t> local_frontier_sizes(local_frontier_offsets.size() - 1);
  std::adjacent_difference(
    local_frontier_offsets.begin() + 1, local_frontier_offsets.end(), local_frontier_sizes.begin());

  rmm::device_uvector<bias_t> aggregate_local_frontier_biases(0, handle.get_stream());
  rmm::device_uvector<edge_type_t> aggregate_local_frontier_edge_types(0, handle.get_stream());
  rmm::device_uvector<size_t> aggregate_local_frontier_local_degree_offsets(0, handle.get_stream());
  if constexpr (std::is_same_v<edge_value_t, cuda::std::nullopt_t>) {
    std::forward_as_tuple(
      std::tie(aggregate_local_frontier_biases, aggregate_local_frontier_edge_types),
      aggregate_local_frontier_local_degree_offsets) =
      transform_v_frontier_e(
        handle,
        graph_view,
        aggregate_local_frontier_key_first,
        edge_src_value_input,
        edge_dst_value_input,
        edge_type_input,
        cuda::proclaim_return_type<thrust::tuple<bias_t, edge_type_t>>(
          [bias_e_op] __device__(auto src, auto dst, auto src_val, auto dst_val, auto e_val) {
            return thrust::make_tuple(bias_e_op(src, dst, src_val, dst_val, cuda::std::nullopt),
                                      e_val);
          }),
        raft::host_span<size_t const>(local_frontier_offsets.data(),
                                      local_frontier_offsets.size()));
  } else {
    std::forward_as_tuple(
      std::tie(aggregate_local_frontier_biases, aggregate_local_frontier_edge_types),
      aggregate_local_frontier_local_degree_offsets) =
      transform_v_frontier_e(
        handle,
        graph_view,
        aggregate_local_frontier_key_first,
        edge_src_value_input,
        edge_dst_value_input,
        view_concat(edge_value_input, edge_type_input),
        cuda::proclaim_return_type<thrust::tuple<bias_t, edge_type_t>>(
          [bias_e_op] __device__(auto src, auto dst, auto src_val, auto dst_val, auto e_val) {
            using tuple_type          = decltype(e_val);
            auto constexpr tuple_size = thrust::tuple_size<tuple_type>::value;
            edge_value_t bias_e_op_e_val{};
            if constexpr (std::is_arithmetic_v<edge_value_t>) {
              bias_e_op_e_val = thrust::get<0>(e_val);
            } else {
              bias_e_op_e_val = thrust_tuple_slice<tuple_type, size_t{0}, tuple_size - 1>(e_val);
            }
            return thrust::make_tuple(bias_e_op(src, dst, src_val, dst_val, bias_e_op_e_val),
                                      thrust::get<tuple_size - 1>(e_val));
          }),
        raft::host_span<size_t const>(local_frontier_offsets.data(),
                                      local_frontier_offsets.size()));
  }

  if (do_expensive_check) {
    auto num_invalid_biases = thrust::count_if(
      handle.get_thrust_policy(),
      aggregate_local_frontier_biases.begin(),
      aggregate_local_frontier_biases.end(),
      check_out_of_range_t<bias_t>{bias_t{0.0}, std::numeric_limits<bias_t>::max()});
    if constexpr (GraphViewType::is_multi_gpu) {
      num_invalid_biases = host_scalar_allreduce(
        handle.get_comms(), num_invalid_biases, raft::comms::op_t::SUM, handle.get_stream());
    }
    CUGRAPH_EXPECTS(num_invalid_biases == 0,
                    "invalid_input_argument: bias_e_op return values should be non-negative and "
                    "should not exceed std::numeirc_limits<bias_t>::max().");
  }

  // 3. exclude 0 bias neighbors & update offsets

  rmm::device_uvector<edge_t> aggregate_local_frontier_nz_bias_indices(
    aggregate_local_frontier_biases.size(), handle.get_stream());
  thrust::tabulate(handle.get_thrust_policy(),
                   aggregate_local_frontier_nz_bias_indices.begin(),
                   aggregate_local_frontier_nz_bias_indices.end(),
                   [offsets = raft::device_span<size_t const>(
                      aggregate_local_frontier_local_degree_offsets.data(),
                      aggregate_local_frontier_local_degree_offsets.size())] __device__(size_t i) {
                     auto it =
                       thrust::upper_bound(thrust::seq, offsets.begin() + 1, offsets.end(), i);
                     auto idx = cuda::std::distance(offsets.begin() + 1, it);
                     return static_cast<edge_t>(i - offsets[idx]);
                   });

  rmm::device_uvector<size_t> aggregate_local_frontier_local_degrees(local_frontier_offsets.back(),
                                                                     handle.get_stream());
  thrust::adjacent_difference(handle.get_thrust_policy(),
                              aggregate_local_frontier_local_degree_offsets.begin() + 1,
                              aggregate_local_frontier_local_degree_offsets.end(),
                              aggregate_local_frontier_local_degrees.begin());

  {
    auto pair_first = thrust::make_zip_iterator(aggregate_local_frontier_biases.begin(),
                                                thrust::make_counting_iterator(size_t{0}));
    thrust::for_each(handle.get_thrust_policy(),
                     pair_first,
                     pair_first + aggregate_local_frontier_biases.size(),
                     [offsets = raft::device_span<size_t const>(
                        aggregate_local_frontier_local_degree_offsets.data(),
                        aggregate_local_frontier_local_degree_offsets.size()),
                      degrees = raft::device_span<size_t>(
                        aggregate_local_frontier_local_degrees.data(),
                        aggregate_local_frontier_local_degrees.size())] __device__(auto pair) {
                       auto bias = thrust::get<0>(pair);
                       if (bias == 0.0) {
                         auto i = thrust::get<1>(pair);
                         auto it =
                           thrust::upper_bound(thrust::seq, offsets.begin() + 1, offsets.end(), i);
                         auto idx = cuda::std::distance(offsets.begin() + 1, it);
                         cuda::atomic_ref<size_t, cuda::thread_scope_device> degree(degrees[idx]);
                         degree.fetch_sub(size_t{1}, cuda::std::memory_order_relaxed);
                       }
                     });
  }

  thrust::inclusive_scan(handle.get_thrust_policy(),
                         aggregate_local_frontier_local_degrees.begin(),
                         aggregate_local_frontier_local_degrees.end(),
                         aggregate_local_frontier_local_degree_offsets.begin() + 1);

  {
    auto triplet_first =
      thrust::make_zip_iterator(aggregate_local_frontier_biases.begin(),
                                aggregate_local_frontier_edge_types.begin(),
                                aggregate_local_frontier_nz_bias_indices.begin());
    auto triplet_last =
      thrust::remove_if(handle.get_thrust_policy(),
                        triplet_first,
                        triplet_first + aggregate_local_frontier_biases.size(),
                        [] __device__(auto triplet) { return thrust::get<0>(triplet) == 0.0; });
    aggregate_local_frontier_biases.resize(cuda::std::distance(triplet_first, triplet_last),
                                           handle.get_stream());
    aggregate_local_frontier_edge_types.resize(cuda::std::distance(triplet_first, triplet_last),
                                               handle.get_stream());
    aggregate_local_frontier_nz_bias_indices.resize(
      cuda::std::distance(triplet_first, triplet_last), handle.get_stream());
    aggregate_local_frontier_biases.shrink_to_fit(handle.get_stream());
    aggregate_local_frontier_edge_types.shrink_to_fit(handle.get_stream());
    aggregate_local_frontier_nz_bias_indices.shrink_to_fit(handle.get_stream());
  }

  return std::make_tuple(std::move(aggregate_local_frontier_biases),
                         std::move(aggregate_local_frontier_edge_types),
                         std::move(aggregate_local_frontier_nz_bias_indices),
                         std::move(aggregate_local_frontier_local_degree_offsets));
}

// return (edge types, segment offsets) pairs for each key in the aggregate local frontier
template <typename GraphViewType, typename KeyIterator, typename EdgeTypeInputWrapper>
std::tuple<rmm::device_uvector<typename EdgeTypeInputWrapper::value_type>,
           rmm::device_uvector<size_t>>
compute_aggregate_local_frontier_edge_types(raft::handle_t const& handle,
                                            GraphViewType const& graph_view,
                                            KeyIterator aggregate_local_frontier_key_first,
                                            EdgeTypeInputWrapper edge_type_input,
                                            raft::host_span<size_t const> local_frontier_offsets)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using key_t    = typename thrust::iterator_traits<KeyIterator>::value_type;

  std::vector<size_t> local_frontier_sizes(local_frontier_offsets.size() - 1);
  std::adjacent_difference(
    local_frontier_offsets.begin() + 1, local_frontier_offsets.end(), local_frontier_sizes.begin());
  auto [aggregate_local_frontier_types, aggregate_local_frontier_local_degree_offsets] =
    transform_v_frontier_e(
      handle,
      graph_view,
      aggregate_local_frontier_key_first,
      edge_src_dummy_property_t{}.view(),
      edge_dst_dummy_property_t{}.view(),
      edge_type_input,
      [] __device__(auto, auto, auto, auto, auto e_val) { return e_val; },
      raft::host_span<size_t const>(local_frontier_offsets.data(), local_frontier_offsets.size()));

  return std::make_tuple(std::move(aggregate_local_frontier_types),
                         std::move(aggregate_local_frontier_local_degree_offsets));
}

// elements with invalid values are dropped before shuffling
template <typename value_t>
std::tuple<rmm::device_uvector<value_t>, rmm::device_uvector<size_t>, std::vector<size_t>>
shuffle_and_compute_local_nbr_values(
  raft::handle_t const& handle,
  rmm::device_uvector<value_t>&& sample_nbr_values,
  raft::device_span<value_t const> frontier_partitioned_value_local_sum_displacements,
  size_t K,
  value_t invalid_value)
{
  auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
  auto const minor_comm_size = minor_comm.get_size();

  auto sample_local_nbr_values = std::move(
    sample_nbr_values);  // neighbor value within an edge partition (note that each vertex's
                         // neighbors are distributed in minor_comm_size partitions)
  rmm::device_uvector<size_t> key_indices(sample_local_nbr_values.size(), handle.get_stream());
  auto minor_comm_ranks =
    rmm::device_uvector<int>(sample_local_nbr_values.size(), handle.get_stream());
  auto intra_partition_displacements =
    rmm::device_uvector<size_t>(sample_local_nbr_values.size(), handle.get_stream());

  rmm::device_uvector<size_t> d_tx_counts(minor_comm_size, handle.get_stream());
  thrust::fill(handle.get_thrust_policy(), d_tx_counts.begin(), d_tx_counts.end(), size_t{0});
  auto input_pair_first =
    thrust::make_zip_iterator(sample_local_nbr_values.begin(),
                              thrust::make_transform_iterator(
                                thrust::make_counting_iterator(size_t{0}), divider_t<size_t>{K}));
  auto output_tuple_first = thrust::make_zip_iterator(minor_comm_ranks.begin(),
                                                      intra_partition_displacements.begin(),
                                                      sample_local_nbr_values.begin(),
                                                      key_indices.begin());
  thrust::for_each(
    handle.get_thrust_policy(),
    thrust::make_counting_iterator(size_t{0}),
    thrust::make_counting_iterator(sample_local_nbr_values.size()),
    convert_value_key_pair_to_shuffle_t<decltype(input_pair_first),
                                        decltype(output_tuple_first),
                                        value_t>{
      input_pair_first,
      output_tuple_first,
      raft::device_span<value_t const>(frontier_partitioned_value_local_sum_displacements.data(),
                                       frontier_partitioned_value_local_sum_displacements.size()),
      raft::device_span<size_t>(d_tx_counts.data(), d_tx_counts.size()),
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
  auto pair_first = thrust::make_zip_iterator(sample_local_nbr_values.begin(), key_indices.begin());
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
    thrust::make_tuple(sample_local_nbr_values.begin(), key_indices.begin()));
  auto [rx_value_buffer, rx_counts] =
    shuffle_values(minor_comm,
                   pair_first,
                   raft::host_span<size_t const>(h_tx_counts.data(), h_tx_counts.size()),
                   handle.get_stream());

  sample_local_nbr_values            = std::move(std::get<0>(rx_value_buffer));
  key_indices                        = std::move(std::get<1>(rx_value_buffer));
  auto local_frontier_sample_offsets = std::vector<size_t>(rx_counts.size() + 1);
  local_frontier_sample_offsets[0]   = size_t{0};
  std::inclusive_scan(
    rx_counts.begin(), rx_counts.end(), local_frontier_sample_offsets.begin() + 1);

  return std::make_tuple(std::move(sample_local_nbr_values),
                         std::move(key_indices),
                         std::move(local_frontier_sample_offsets));
}

// elements with invalid values are dropped before shuffling
template <typename edge_type_t, typename value_t>
std::tuple<rmm::device_uvector<value_t>,
           rmm::device_uvector<edge_type_t>,
           rmm::device_uvector<size_t>,
           std::vector<size_t>>
shuffle_and_compute_per_type_local_nbr_values(
  raft::handle_t const& handle,
  rmm::device_uvector<value_t>&& sample_per_type_nbr_values,
  raft::device_span<value_t const> frontier_partitioned_per_type_value_local_sum_displacements,
  raft::device_span<size_t const> K_offsets,
  size_t K_sum,
  value_t invalid_value)
{
  auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
  auto const minor_comm_size = minor_comm.get_size();

  auto num_edge_types = static_cast<edge_type_t>(K_offsets.size() - 1);

  auto sample_per_type_local_nbr_values =
    std::move(sample_per_type_nbr_values);  // neighbor value within an edge partition (note that
                                            // each vertex's neighbors are distributed in
                                            // minor_comm_size partitions)
  rmm::device_uvector<edge_type_t> edge_types(sample_per_type_local_nbr_values.size(),
                                              handle.get_stream());
  rmm::device_uvector<size_t> key_indices(sample_per_type_local_nbr_values.size(),
                                          handle.get_stream());
  auto minor_comm_ranks =
    rmm::device_uvector<int>(sample_per_type_local_nbr_values.size(), handle.get_stream());
  auto intra_partition_displacements =
    rmm::device_uvector<size_t>(sample_per_type_local_nbr_values.size(), handle.get_stream());

  rmm::device_uvector<size_t> d_tx_counts(minor_comm_size, handle.get_stream());
  thrust::fill(handle.get_thrust_policy(), d_tx_counts.begin(), d_tx_counts.end(), size_t{0});
  auto input_pair_first   = thrust::make_zip_iterator(sample_per_type_local_nbr_values.begin(),
                                                    thrust::make_counting_iterator(size_t{0}));
  auto output_tuple_first = thrust::make_zip_iterator(minor_comm_ranks.begin(),
                                                      intra_partition_displacements.begin(),
                                                      sample_per_type_local_nbr_values.begin(),
                                                      edge_types.begin(),
                                                      key_indices.begin());
  thrust::for_each(handle.get_thrust_policy(),
                   thrust::make_counting_iterator(size_t{0}),
                   thrust::make_counting_iterator(sample_per_type_local_nbr_values.size()),
                   convert_per_type_value_key_pair_to_shuffle_t<decltype(input_pair_first),
                                                                decltype(output_tuple_first),
                                                                edge_type_t,
                                                                value_t>{
                     input_pair_first,
                     output_tuple_first,
                     raft::device_span<value_t const>(
                       frontier_partitioned_per_type_value_local_sum_displacements.data(),
                       frontier_partitioned_per_type_value_local_sum_displacements.size()),
                     raft::device_span<size_t>(d_tx_counts.data(), d_tx_counts.size()),
                     K_offsets,
                     K_sum,
                     minor_comm_size,
                     invalid_value});
  rmm::device_uvector<size_t> tx_displacements(minor_comm_size, handle.get_stream());
  thrust::exclusive_scan(
    handle.get_thrust_policy(), d_tx_counts.begin(), d_tx_counts.end(), tx_displacements.begin());
  auto tmp_sample_per_type_local_nbr_values =
    rmm::device_uvector<value_t>(tx_displacements.back_element(handle.get_stream()) +
                                   d_tx_counts.back_element(handle.get_stream()),
                                 handle.get_stream());
  auto tmp_edge_types = rmm::device_uvector<edge_type_t>(
    tmp_sample_per_type_local_nbr_values.size(), handle.get_stream());
  auto tmp_key_indices =
    rmm::device_uvector<size_t>(tmp_sample_per_type_local_nbr_values.size(), handle.get_stream());
  auto triplet_first = thrust::make_zip_iterator(
    sample_per_type_local_nbr_values.begin(), edge_types.begin(), key_indices.begin());
  thrust::scatter_if(
    handle.get_thrust_policy(),
    triplet_first,
    triplet_first + sample_per_type_local_nbr_values.size(),
    thrust::make_transform_iterator(
      thrust::make_counting_iterator(size_t{0}),
      shuffle_index_compute_offset_t{
        raft::device_span<int const>(minor_comm_ranks.data(), minor_comm_ranks.size()),
        raft::device_span<size_t const>(intra_partition_displacements.data(),
                                        intra_partition_displacements.size()),
        raft::device_span<size_t const>(tx_displacements.data(), tx_displacements.size())}),
    minor_comm_ranks.begin(),
    thrust::make_zip_iterator(tmp_sample_per_type_local_nbr_values.begin(),
                              tmp_edge_types.begin(),
                              tmp_key_indices.begin()),
    is_not_equal_t<int>{-1});

  sample_per_type_local_nbr_values = std::move(tmp_sample_per_type_local_nbr_values);
  edge_types                       = std::move(tmp_edge_types);
  key_indices                      = std::move(tmp_key_indices);

  std::vector<size_t> h_tx_counts(d_tx_counts.size());
  raft::update_host(
    h_tx_counts.data(), d_tx_counts.data(), d_tx_counts.size(), handle.get_stream());
  handle.sync_stream();

  triplet_first = thrust::make_zip_iterator(
    sample_per_type_local_nbr_values.begin(), edge_types.begin(), key_indices.begin());
  auto [rx_value_buffer, rx_counts] =
    shuffle_values(minor_comm,
                   triplet_first,
                   raft::host_span<size_t const>(h_tx_counts.data(), h_tx_counts.size()),
                   handle.get_stream());

  sample_per_type_local_nbr_values   = std::move(std::get<0>(rx_value_buffer));
  edge_types                         = std::move(std::get<1>(rx_value_buffer));
  key_indices                        = std::move(std::get<2>(rx_value_buffer));
  auto local_frontier_sample_offsets = std::vector<size_t>(rx_counts.size() + 1);
  local_frontier_sample_offsets[0]   = size_t{0};
  std::inclusive_scan(
    rx_counts.begin(), rx_counts.end(), local_frontier_sample_offsets.begin() + 1);

  return std::make_tuple(std::move(sample_per_type_local_nbr_values),
                         std::move(edge_types),
                         std::move(key_indices),
                         std::move(local_frontier_sample_offsets));
}

template <typename edge_t, typename edge_type_t>
rmm::device_uvector<edge_t> compute_local_nbr_indices_from_per_type_local_nbr_indices(
  raft::handle_t const& handle,
  raft::device_span<size_t const> aggregate_local_frontier_key_idx_to_unique_key_idx,
  raft::host_span<size_t const> local_frontier_offsets,
  raft::device_span<size_t const> aggregate_local_frontier_unique_key_per_type_local_degree_offsets,
  raft::host_span<size_t const> local_frontier_unique_key_offsets,
  std::optional<std::tuple<raft::device_span<edge_type_t const>, raft::device_span<size_t const>>>
    edge_type_key_idx_pairs,
  rmm::device_uvector<edge_t>&& per_type_local_nbr_indices,
  raft::host_span<size_t const> local_frontier_sample_offsets,
  raft::device_span<size_t const> K_offsets,
  size_t K_sum)
{
  auto num_edge_types = static_cast<edge_type_t>(K_offsets.size() - 1);

  auto local_nbr_indices = std::move(per_type_local_nbr_indices);
  if (edge_type_key_idx_pairs) {
    auto triplet_first = thrust::make_zip_iterator(local_nbr_indices.begin(),
                                                   std::get<0>(*edge_type_key_idx_pairs).begin(),
                                                   std::get<1>(*edge_type_key_idx_pairs).begin());
    for (size_t i = 0; i < local_frontier_sample_offsets.size() - 1; ++i) {
      thrust::transform(
        handle.get_thrust_policy(),
        triplet_first + local_frontier_sample_offsets[i],
        triplet_first + local_frontier_sample_offsets[i + 1],
        local_nbr_indices.begin() + local_frontier_sample_offsets[i],
        cuda::proclaim_return_type<edge_t>(
          [key_idx_to_unique_key_idx = raft::device_span<size_t const>(
             aggregate_local_frontier_key_idx_to_unique_key_idx.data() + local_frontier_offsets[i],
             local_frontier_offsets[i + 1] - local_frontier_offsets[i]),
           unique_key_per_type_local_degree_offsets = raft::device_span<size_t const>(
             aggregate_local_frontier_unique_key_per_type_local_degree_offsets.data() +
               local_frontier_unique_key_offsets[i] * num_edge_types,
             (local_frontier_unique_key_offsets[i + 1] - local_frontier_unique_key_offsets[i]) *
                 num_edge_types +
               1),
           num_edge_types,
           invalid_idx = cugraph::invalid_edge_id_v<edge_t>] __device__(auto triplet) {
            auto per_type_local_nbr_idx = thrust::get<0>(triplet);
            if (per_type_local_nbr_idx != invalid_idx) {
              auto type              = thrust::get<1>(triplet);
              auto key_idx           = thrust::get<2>(triplet);
              auto unique_key_idx    = key_idx_to_unique_key_idx[key_idx];
              auto type_start_offset = static_cast<edge_t>(
                unique_key_per_type_local_degree_offsets[unique_key_idx * num_edge_types + type] -
                unique_key_per_type_local_degree_offsets[unique_key_idx * num_edge_types]);
              return type_start_offset + per_type_local_nbr_idx;
            } else {
              return invalid_idx;
            }
          }));
    }
  } else {
    auto pair_first = thrust::make_zip_iterator(local_nbr_indices.begin(),
                                                thrust::make_counting_iterator(size_t{0}));
    thrust::transform(
      handle.get_thrust_policy(),
      pair_first,
      pair_first + local_nbr_indices.size(),
      local_nbr_indices.begin(),
      cuda::proclaim_return_type<edge_t>(
        [key_idx_to_unique_key_idx = raft::device_span<size_t const>(
           aggregate_local_frontier_key_idx_to_unique_key_idx.data(),
           aggregate_local_frontier_key_idx_to_unique_key_idx.size()),
         unique_key_per_type_local_degree_offsets = raft::device_span<size_t const>(
           aggregate_local_frontier_unique_key_per_type_local_degree_offsets.data(),
           aggregate_local_frontier_unique_key_per_type_local_degree_offsets.size()),
         K_offsets,
         K_sum,
         invalid_idx = cugraph::invalid_edge_id_v<edge_t>] __device__(auto pair) {
          auto num_edge_types         = static_cast<edge_type_t>(K_offsets.size() - 1);
          auto per_type_local_nbr_idx = thrust::get<0>(pair);
          if (per_type_local_nbr_idx != invalid_idx) {
            auto i                 = thrust::get<1>(pair);
            auto key_idx           = i / K_sum;
            auto unique_key_idx    = key_idx_to_unique_key_idx[key_idx];
            auto type              = static_cast<edge_type_t>(cuda::std::distance(
              K_offsets.begin() + 1,
              thrust::upper_bound(thrust::seq, K_offsets.begin() + 1, K_offsets.end(), i % K_sum)));
            auto type_start_offset = static_cast<edge_t>(
              unique_key_per_type_local_degree_offsets[unique_key_idx * num_edge_types + type] -
              unique_key_per_type_local_degree_offsets[unique_key_idx * num_edge_types]);
            return type_start_offset + per_type_local_nbr_idx;
          } else {
            return invalid_idx;
          }
        }));

    // In each K_sum sampled indices, place invalid indices at the end.

    thrust::for_each(handle.get_thrust_policy(),
                     thrust::make_counting_iterator(size_t{0}),
                     thrust::make_counting_iterator(local_frontier_offsets.back()),
                     [local_nbr_indices = raft::device_span<edge_t>(local_nbr_indices.data(),
                                                                    local_nbr_indices.size()),
                      K_sum,
                      invalid_idx = cugraph::invalid_edge_id_v<edge_t>] __device__(size_t i) {
                       thrust::partition(thrust::seq,
                                         local_nbr_indices.begin() + i * K_sum,
                                         local_nbr_indices.begin() + (i + 1) * K_sum,
                                         [invalid_idx] __device__(auto local_nbr_idx) {
                                           return local_nbr_idx != invalid_idx;
                                         });
                     });
  }

  return local_nbr_indices;
}

template <typename edge_t, typename edge_type_t, typename bias_t, bool multi_gpu>
std::tuple<rmm::device_uvector<edge_t> /* local_nbr_indices */,
           std::optional<rmm::device_uvector<size_t>> /* key_indices */,
           std::vector<size_t> /* local_frontier_sample_offsets */>
biased_sample_with_replacement(
  raft::handle_t const& handle,
  raft::device_span<size_t const> aggregate_local_frontier_key_idx_to_unique_key_idx,
  raft::host_span<size_t const> local_frontier_offsets,
  raft::device_span<bias_t const> aggregate_local_frontier_unique_key_biases,
  raft::device_span<size_t const> aggregate_local_frontier_unique_key_per_type_local_degree_offsets,
  raft::host_span<size_t const> local_frontier_unique_key_offsets,
  raft::random::RngState& rng_state,
  raft::host_span<size_t const> Ks)
{
  int minor_comm_rank{0};
  int minor_comm_size{1};
  if constexpr (multi_gpu) {
    auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    minor_comm_rank  = minor_comm.get_rank();
    minor_comm_size  = minor_comm.get_size();
  }

  auto num_local_edge_partitions = local_frontier_offsets.size() - 1;
  auto num_edge_types            = static_cast<edge_type_t>(Ks.size());

  rmm::device_uvector<edge_t> local_nbr_indices(0, handle.get_stream());
  std::optional<rmm::device_uvector<size_t>> key_indices{std::nullopt};
  std::vector<size_t> local_frontier_sample_offsets{};

  auto K_sum = std::accumulate(Ks.begin(), Ks.end(), size_t{0});

  rmm::device_uvector<size_t> d_K_offsets(Ks.size() + 1, handle.get_stream());
  {
    std::vector<size_t> h_K_offsets(d_K_offsets.size());
    h_K_offsets[0] = 0;
    std::inclusive_scan(Ks.begin(), Ks.end(), h_K_offsets.begin() + 1);
    raft::update_device(
      d_K_offsets.data(), h_K_offsets.data(), h_K_offsets.size(), handle.get_stream());
  }

  // compute segmented inclusive sums (one segment per key & type pair)

  rmm::device_uvector<bias_t>
    aggregate_local_frontier_unique_key_bias_segmented_local_inclusive_sums(
      aggregate_local_frontier_unique_key_biases.size(), handle.get_stream());
  {
    auto unique_key_first = thrust::make_transform_iterator(
      thrust::make_counting_iterator(size_t{0}),
      cuda::proclaim_return_type<size_t>(
        [offsets = raft::device_span<size_t const>(
           aggregate_local_frontier_unique_key_per_type_local_degree_offsets.data(),
           aggregate_local_frontier_unique_key_per_type_local_degree_offsets
             .size())] __device__(size_t i) {
          return static_cast<size_t>(cuda::std::distance(
            offsets.begin() + 1,
            thrust::upper_bound(thrust::seq, offsets.begin() + 1, offsets.end(), i)));
        }));
    thrust::inclusive_scan_by_key(
      handle.get_thrust_policy(),
      unique_key_first,
      unique_key_first + aggregate_local_frontier_unique_key_biases.size(),
      aggregate_local_frontier_unique_key_biases.begin(),
      aggregate_local_frontier_unique_key_bias_segmented_local_inclusive_sums.begin());
  }

  rmm::device_uvector<bias_t> sample_local_random_numbers(0, handle.get_stream());
  std::optional<rmm::device_uvector<edge_type_t>> edge_types{std::nullopt};
  {
    // sum local bias values (one value per key & type pair) and collect local bias sums

    auto aggregate_local_frontier_per_type_bias_local_sums = rmm::device_uvector<bias_t>(
      local_frontier_offsets.back() * num_edge_types, handle.get_stream());
    for (size_t i = 0; i < num_local_edge_partitions; ++i) {
      thrust::tabulate(
        handle.get_thrust_policy(),
        get_dataframe_buffer_begin(aggregate_local_frontier_per_type_bias_local_sums) +
          local_frontier_offsets[i] * num_edge_types,
        get_dataframe_buffer_begin(aggregate_local_frontier_per_type_bias_local_sums) +
          local_frontier_offsets[i + 1] * num_edge_types,
        [key_idx_to_unique_key_idx = raft::device_span<size_t const>(
           aggregate_local_frontier_key_idx_to_unique_key_idx.data() + local_frontier_offsets[i],
           local_frontier_offsets[i + 1] - local_frontier_offsets[i]),
         unique_key_per_type_local_degree_offsets = raft::device_span<size_t const>(
           aggregate_local_frontier_unique_key_per_type_local_degree_offsets.data() +
             local_frontier_unique_key_offsets[i] * num_edge_types,
           (local_frontier_unique_key_offsets[i + 1] - local_frontier_unique_key_offsets[i]) *
               num_edge_types +
             1),
         aggregate_local_frontier_unique_key_bias_segmented_local_inclusive_sums =
           raft::device_span<bias_t const>(
             aggregate_local_frontier_unique_key_bias_segmented_local_inclusive_sums.data(),
             aggregate_local_frontier_unique_key_bias_segmented_local_inclusive_sums.size()),
         num_edge_types] __device__(size_t i) {
          auto key_idx        = i / num_edge_types;
          auto type           = static_cast<edge_t>(i % num_edge_types);
          auto unique_key_idx = key_idx_to_unique_key_idx[key_idx];
          auto degree =
            unique_key_per_type_local_degree_offsets[unique_key_idx * num_edge_types + type + 1] -
            unique_key_per_type_local_degree_offsets[unique_key_idx * num_edge_types + type];
          if (degree > 0) {
            return aggregate_local_frontier_unique_key_bias_segmented_local_inclusive_sums
              [unique_key_per_type_local_degree_offsets[unique_key_idx * num_edge_types + type] +
               degree - 1];
          } else {
            return bias_t{0.0};
          }
        });
    }

    rmm::device_uvector<bias_t> frontier_per_type_bias_sums(0, handle.get_stream());
    std::optional<rmm::device_uvector<bias_t>>
      frontier_partitioned_per_type_bias_local_sum_displacements{std::nullopt};
    if (minor_comm_size > 1) {
      std::tie(frontier_per_type_bias_sums,
               frontier_partitioned_per_type_bias_local_sum_displacements) =
        compute_frontier_value_sums_and_partitioned_local_value_sum_displacements(
          handle,
          raft::device_span<bias_t const>(aggregate_local_frontier_per_type_bias_local_sums.data(),
                                          aggregate_local_frontier_per_type_bias_local_sums.size()),
          local_frontier_offsets,
          num_edge_types);
      aggregate_local_frontier_per_type_bias_local_sums.resize(0, handle.get_stream());
      aggregate_local_frontier_per_type_bias_local_sums.shrink_to_fit(handle.get_stream());
    } else {
      frontier_per_type_bias_sums = std::move(aggregate_local_frontier_per_type_bias_local_sums);
    }

    // generate & shuffle random numbers

    rmm::device_uvector<bias_t> sample_random_numbers(
      (local_frontier_offsets[minor_comm_rank + 1] - local_frontier_offsets[minor_comm_rank]) *
        K_sum,
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
      cuda::proclaim_return_type<bias_t>(
        [frontier_per_type_bias_sums = raft::device_span<bias_t const>(
           frontier_per_type_bias_sums.data(), frontier_per_type_bias_sums.size()),
         K_offsets = raft::device_span<size_t const>(d_K_offsets.data(), d_K_offsets.size()),
         K_sum,
         invalid_value = std::numeric_limits<bias_t>::infinity()] __device__(bias_t r, size_t i) {
          auto num_edge_types = static_cast<edge_type_t>(K_offsets.size() - 1);
          auto type           = static_cast<edge_type_t>(cuda::std::distance(
            K_offsets.begin() + 1,
            thrust::upper_bound(thrust::seq, K_offsets.begin() + 1, K_offsets.end(), i % K_sum)));
          // bias_sum will be 0 if degree is 0 or all the edges have 0 bias
          auto bias_sum = frontier_per_type_bias_sums[(i / K_sum) * num_edge_types + type];
          return bias_sum > 0.0 ? r * bias_sum : invalid_value;
        }));

    if (minor_comm_size > 1) {
      if (num_edge_types > 1) {
        std::tie(
          sample_local_random_numbers, edge_types, key_indices, local_frontier_sample_offsets) =
          shuffle_and_compute_per_type_local_nbr_values<edge_type_t, bias_t>(
            handle,
            std::move(sample_random_numbers),
            raft::device_span<bias_t const>(
              (*frontier_partitioned_per_type_bias_local_sum_displacements).data(),
              (*frontier_partitioned_per_type_bias_local_sum_displacements).size()),
            raft::device_span<size_t const>(d_K_offsets.data(), d_K_offsets.size()),
            K_sum,
            std::numeric_limits<bias_t>::infinity());
      } else {
        std::tie(sample_local_random_numbers, key_indices, local_frontier_sample_offsets) =
          shuffle_and_compute_local_nbr_values<bias_t>(
            handle,
            std::move(sample_random_numbers),
            raft::device_span<bias_t const>(
              (*frontier_partitioned_per_type_bias_local_sum_displacements).data(),
              (*frontier_partitioned_per_type_bias_local_sum_displacements).size()),
            K_sum,
            std::numeric_limits<bias_t>::infinity());
      }
    } else {
      sample_local_random_numbers   = std::move(sample_random_numbers);
      local_frontier_sample_offsets = {size_t{0}, sample_local_random_numbers.size()};
    }
  }

  rmm::device_uvector<edge_t> per_type_local_nbr_indices(sample_local_random_numbers.size(),
                                                         handle.get_stream());
  for (size_t i = 0; i < num_local_edge_partitions; ++i) {
    thrust::tabulate(
      handle.get_thrust_policy(),
      per_type_local_nbr_indices.begin() + local_frontier_sample_offsets[i],
      per_type_local_nbr_indices.begin() + local_frontier_sample_offsets[i + 1],
      [sample_local_random_numbers = raft::device_span<bias_t>(
         sample_local_random_numbers.data() + local_frontier_sample_offsets[i],
         local_frontier_sample_offsets[i + 1] - local_frontier_sample_offsets[i]),
       key_indices               = key_indices
                                     ? cuda::std::make_optional<raft::device_span<size_t const>>(
                           (*key_indices).data() + local_frontier_sample_offsets[i],
                           local_frontier_sample_offsets[i + 1] - local_frontier_sample_offsets[i])
                                     : cuda::std::nullopt,
       edge_types                = edge_types
                                     ? cuda::std::make_optional<raft::device_span<edge_type_t const>>(
                          (*edge_types).data() + local_frontier_sample_offsets[i],
                          local_frontier_sample_offsets[i + 1] - local_frontier_sample_offsets[i])
                                     : cuda::std::nullopt,
       key_idx_to_unique_key_idx = raft::device_span<size_t const>(
         aggregate_local_frontier_key_idx_to_unique_key_idx.data() + local_frontier_offsets[i],
         local_frontier_offsets[i + 1] - local_frontier_offsets[i]),
       aggregate_local_frontier_unique_key_bias_segmented_local_inclusive_sums =
         raft::device_span<bias_t const>(
           aggregate_local_frontier_unique_key_bias_segmented_local_inclusive_sums.data(),
           aggregate_local_frontier_unique_key_bias_segmented_local_inclusive_sums.size()),
       unique_key_per_type_local_degree_offsets = raft::device_span<size_t const>(
         aggregate_local_frontier_unique_key_per_type_local_degree_offsets.data() +
           local_frontier_unique_key_offsets[i] * num_edge_types,
         (local_frontier_unique_key_offsets[i + 1] - local_frontier_unique_key_offsets[i]) *
             num_edge_types +
           1),
       K_offsets = raft::device_span<size_t const>(d_K_offsets.data(), d_K_offsets.size()),
       K_sum,
       num_edge_types,
       invalid_random_number = std::numeric_limits<bias_t>::infinity(),
       invalid_idx           = cugraph::invalid_edge_id_v<edge_t>] __device__(size_t i) {
        auto local_random_number = sample_local_random_numbers[i];
        if (local_random_number != invalid_random_number) {
          auto key_idx = key_indices ? (*key_indices)[i] : (i / K_sum);
          auto type =
            num_edge_types > 1
              ? (edge_types ? (*edge_types)[i]
                            : static_cast<edge_type_t>(cuda::std::distance(
                                K_offsets.begin() + 1,
                                thrust::upper_bound(
                                  thrust::seq, K_offsets.begin() + 1, K_offsets.end(), i % K_sum))))
              : edge_type_t{0};
          auto unique_key_idx = key_idx_to_unique_key_idx[key_idx];
          auto local_degree   = static_cast<edge_t>(
            unique_key_per_type_local_degree_offsets[unique_key_idx * num_edge_types + type + 1] -
            unique_key_per_type_local_degree_offsets[unique_key_idx * num_edge_types + type]);
          auto inclusive_sum_first =
            aggregate_local_frontier_unique_key_bias_segmented_local_inclusive_sums.begin() +
            unique_key_per_type_local_degree_offsets[unique_key_idx * num_edge_types + type];
          auto inclusive_sum_last     = inclusive_sum_first + local_degree;
          auto per_type_local_nbr_idx = static_cast<edge_t>(cuda::std::distance(
            inclusive_sum_first,
            thrust::upper_bound(
              thrust::seq, inclusive_sum_first, inclusive_sum_last, local_random_number)));
          return cuda::std::min(per_type_local_nbr_idx, local_degree - 1);
        } else {
          return invalid_idx;
        }
      });
  }

  if (num_edge_types > 1) {
    // per-type local neighbor indices => local neighbor indices

    assert(edge_types.has_value() == key_indices.has_value());
    local_nbr_indices =
      compute_local_nbr_indices_from_per_type_local_nbr_indices<edge_t, edge_type_t>(
        handle,
        aggregate_local_frontier_key_idx_to_unique_key_idx,
        local_frontier_offsets,
        aggregate_local_frontier_unique_key_per_type_local_degree_offsets,
        local_frontier_unique_key_offsets,
        edge_types
          ? std::make_optional(std::make_tuple(
              raft::device_span<edge_type_t const>((*edge_types).data(), (*edge_types).size()),
              raft::device_span<size_t const>((*key_indices).data(), (*key_indices).size())))
          : std::nullopt,
        std::move(per_type_local_nbr_indices),
        raft::host_span<size_t const>(local_frontier_sample_offsets.data(),
                                      local_frontier_sample_offsets.size()),
        raft::device_span<size_t const>(d_K_offsets.data(), d_K_offsets.size()),
        K_sum);
  } else {
    local_nbr_indices = std::move(per_type_local_nbr_indices);
  }

  return std::make_tuple(
    std::move(local_nbr_indices), std::move(key_indices), std::move(local_frontier_sample_offsets));
}

template <typename edge_t, typename bias_t, bool multi_gpu>
std::tuple<rmm::device_uvector<edge_t> /* local_nbr_indices */,
           std::optional<rmm::device_uvector<size_t>> /* key_indices */,
           std::vector<size_t> /* local_frontier_sample_offsets */>
homogeneous_biased_sample_without_replacement(
  raft::handle_t const& handle,
  raft::device_span<size_t const> aggregate_local_frontier_key_idx_to_unique_key_idx,
  raft::host_span<size_t const> local_frontier_offsets,
  raft::device_span<bias_t const> aggregate_local_frontier_unique_key_biases,
  raft::device_span<size_t const> aggregate_local_frontier_unique_key_local_degree_offsets,
  raft::host_span<size_t const> local_frontier_unique_key_offsets,
  raft::random::RngState& rng_state,
  size_t K)
{
  int minor_comm_rank{0};
  int minor_comm_size{1};
  if constexpr (multi_gpu) {
    auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    minor_comm_rank  = minor_comm.get_rank();
    minor_comm_size  = minor_comm.get_size();
  }

  auto num_local_edge_partitions = local_frontier_offsets.size() - 1;

  rmm::device_uvector<edge_t> local_nbr_indices(0, handle.get_stream());
  std::optional<rmm::device_uvector<size_t>> key_indices{std::nullopt};
  std::vector<size_t> local_frontier_sample_offsets{};

  rmm::device_uvector<edge_t> frontier_degrees(0, handle.get_stream());
  std::optional<rmm::device_uvector<edge_t>> frontier_partitioned_local_degree_displacements{
    std::nullopt};
  {
    rmm::device_uvector<edge_t> aggregate_local_frontier_local_degrees(
      local_frontier_offsets.back(), handle.get_stream());
    for (size_t i = 0; i < num_local_edge_partitions; ++i) {
      thrust::tabulate(
        handle.get_thrust_policy(),
        aggregate_local_frontier_local_degrees.begin() + local_frontier_offsets[i],
        aggregate_local_frontier_local_degrees.begin() + local_frontier_offsets[i + 1],
        [key_idx_to_unique_key_idx = raft::device_span<size_t const>(
           aggregate_local_frontier_key_idx_to_unique_key_idx.data() + local_frontier_offsets[i],
           local_frontier_offsets[i + 1] - local_frontier_offsets[i]),
         unique_key_local_degree_offsets = raft::device_span<size_t const>(
           aggregate_local_frontier_unique_key_local_degree_offsets.data() +
             local_frontier_unique_key_offsets[i],
           (local_frontier_unique_key_offsets[i + 1] - local_frontier_unique_key_offsets[i]) +
             1)] __device__(size_t i) {
          auto unique_key_idx = key_idx_to_unique_key_idx[i];
          return unique_key_local_degree_offsets[unique_key_idx + 1] -
                 unique_key_local_degree_offsets[unique_key_idx];
        });
    }
    if (minor_comm_size > 1) {
      std::tie(frontier_degrees, frontier_partitioned_local_degree_displacements) =
        compute_frontier_value_sums_and_partitioned_local_value_sum_displacements(
          handle,
          raft::device_span<edge_t const>(aggregate_local_frontier_local_degrees.data(),
                                          aggregate_local_frontier_local_degrees.size()),
          local_frontier_offsets,
          1);
    } else {
      frontier_degrees = std::move(aggregate_local_frontier_local_degrees);
    }
  }

  auto [frontier_indices, frontier_partition_offsets] = partition_v_frontier(
    handle,
    frontier_degrees.begin(),
    frontier_degrees.end(),
    std::vector<edge_t>{static_cast<edge_t>(K + 1), static_cast<edge_t>(minor_comm_size * K * 2)});

  if (minor_comm_size > 1) {
    rmm::device_uvector<edge_t> nbr_indices(frontier_degrees.size() * K, handle.get_stream());

    auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());

    if (frontier_partition_offsets[1] > 0) {
      thrust::for_each(
        handle.get_thrust_policy(),
        frontier_indices.begin(),
        frontier_indices.begin() + frontier_partition_offsets[1],
        [frontier_degrees =
           raft::device_span<edge_t const>(frontier_degrees.data(), frontier_degrees.size()),
         nbr_indices = raft::device_span<edge_t>(nbr_indices.data(), nbr_indices.size()),
         K,
         invalid_idx = cugraph::invalid_edge_id_v<edge_t>] __device__(size_t i) {
          auto degree = frontier_degrees[i];
          thrust::sequence(thrust::seq,
                           nbr_indices.begin() + i * K,
                           nbr_indices.begin() + i * K + degree,
                           edge_t{0});
          thrust::fill(thrust::seq,
                       nbr_indices.begin() + i * K + +degree,
                       nbr_indices.begin() + (i + 1) * K,
                       invalid_idx);
        });
    }

    auto mid_frontier_size = frontier_partition_offsets[2] - frontier_partition_offsets[1];
    std::vector<size_t> mid_local_frontier_sizes{};
    mid_local_frontier_sizes =
      host_scalar_allgather(minor_comm, mid_frontier_size, handle.get_stream());
    std::vector<size_t> mid_local_frontier_offsets(mid_local_frontier_sizes.size() + 1);
    mid_local_frontier_offsets[0] = 0;
    std::inclusive_scan(mid_local_frontier_sizes.begin(),
                        mid_local_frontier_sizes.end(),
                        mid_local_frontier_offsets.begin() + 1);

    if (mid_local_frontier_offsets.back() > 0) {
      // aggregate frontier indices with their degrees in the medium range

      auto aggregate_mid_local_frontier_indices =
        rmm::device_uvector<size_t>(mid_local_frontier_offsets.back(), handle.get_stream());
      device_allgatherv(minor_comm,
                        frontier_indices.begin() + frontier_partition_offsets[1],
                        aggregate_mid_local_frontier_indices.begin(),
                        raft::host_span<size_t const>(mid_local_frontier_sizes.data(),
                                                      mid_local_frontier_sizes.size()),
                        raft::host_span<size_t const>(mid_local_frontier_offsets.data(),
                                                      mid_local_frontier_offsets.size() - 1),
                        handle.get_stream());

      // compute local degrees for the aggregated frontier indices

      rmm::device_uvector<edge_t> aggregate_mid_local_frontier_local_degrees(
        aggregate_mid_local_frontier_indices.size(), handle.get_stream());
      for (size_t i = 0; i < num_local_edge_partitions; ++i) {
        thrust::transform(
          handle.get_thrust_policy(),
          aggregate_mid_local_frontier_indices.begin() + mid_local_frontier_offsets[i],
          aggregate_mid_local_frontier_indices.begin() + mid_local_frontier_offsets[i + 1],
          aggregate_mid_local_frontier_local_degrees.begin() + mid_local_frontier_offsets[i],
          cuda::proclaim_return_type<edge_t>(
            [key_idx_to_unique_key_idx = raft::device_span<size_t const>(
               aggregate_local_frontier_key_idx_to_unique_key_idx.data() +
                 local_frontier_offsets[i],
               local_frontier_offsets[i + 1] - local_frontier_offsets[i]),
             unique_key_local_degree_offsets = raft::device_span<size_t const>(
               aggregate_local_frontier_unique_key_local_degree_offsets.data() +
                 local_frontier_unique_key_offsets[i],
               (local_frontier_unique_key_offsets[i + 1] - local_frontier_unique_key_offsets[i]) +
                 1)] __device__(size_t i) {
              auto unique_key_idx = key_idx_to_unique_key_idx[i];
              return static_cast<edge_t>(unique_key_local_degree_offsets[unique_key_idx + 1] -
                                         unique_key_local_degree_offsets[unique_key_idx]);
            }));
      }

      // gather biases for the aggregated frontier indices

      rmm::device_uvector<bias_t> aggregate_mid_local_frontier_biases(0, handle.get_stream());
      std::vector<size_t> tx_counts(mid_local_frontier_sizes.size());
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

        for (size_t i = 0; i < num_local_edge_partitions; ++i) {
          thrust::for_each(
            handle.get_thrust_policy(),
            thrust::make_counting_iterator(size_t{0}),
            thrust::make_counting_iterator(mid_local_frontier_sizes[i]),
            [key_idx_to_unique_key_idx = raft::device_span<size_t const>(
               aggregate_local_frontier_key_idx_to_unique_key_idx.data() +
                 local_frontier_offsets[i],
               local_frontier_offsets[i + 1] - local_frontier_offsets[i]),
             aggregate_local_frontier_unique_key_biases =
               raft::device_span<bias_t const>(aggregate_local_frontier_unique_key_biases.data(),
                                               aggregate_local_frontier_unique_key_biases.size()),
             unique_key_local_degree_offsets = raft::device_span<size_t const>(
               aggregate_local_frontier_unique_key_local_degree_offsets.data() +
                 local_frontier_unique_key_offsets[i],
               (local_frontier_unique_key_offsets[i + 1] - local_frontier_unique_key_offsets[i]) +
                 1),
             mid_local_frontier_indices = raft::device_span<size_t const>(
               aggregate_mid_local_frontier_indices.data() + mid_local_frontier_offsets[i],
               mid_local_frontier_sizes[i]),
             aggregate_mid_local_frontier_biases =
               raft::device_span<bias_t>(aggregate_mid_local_frontier_biases.data(),
                                         aggregate_mid_local_frontier_biases.size()),
             aggregate_mid_local_frontier_local_degree_offsets = raft::device_span<size_t const>(
               aggregate_mid_local_frontier_local_degree_offsets.data(),
               aggregate_mid_local_frontier_local_degree_offsets.size()),
             output_offset = mid_local_frontier_offsets[i]] __device__(size_t i) {
              auto unique_key_idx = key_idx_to_unique_key_idx[mid_local_frontier_indices[i]];
              thrust::copy(thrust::seq,
                           aggregate_local_frontier_unique_key_biases.begin() +
                             unique_key_local_degree_offsets[unique_key_idx],
                           aggregate_local_frontier_unique_key_biases.begin() +
                             unique_key_local_degree_offsets[unique_key_idx + 1],
                           aggregate_mid_local_frontier_biases.begin() +
                             aggregate_mid_local_frontier_local_degree_offsets[output_offset + i]);
            });
        }

        rmm::device_uvector<size_t> d_mid_local_frontier_offsets(mid_local_frontier_offsets.size(),
                                                                 handle.get_stream());
        raft::update_device(d_mid_local_frontier_offsets.data(),
                            mid_local_frontier_offsets.data(),
                            mid_local_frontier_offsets.size(),
                            handle.get_stream());
        rmm::device_uvector<size_t> d_lasts(num_local_edge_partitions, handle.get_stream());
        auto map_first = thrust::make_transform_iterator(
          thrust::make_counting_iterator(size_t{0}),
          cuda::proclaim_return_type<size_t>(
            [mid_local_frontier_offsets = raft::device_span<size_t const>(
               d_mid_local_frontier_offsets.data(),
               d_mid_local_frontier_offsets.size())] __device__(size_t i) {
              return mid_local_frontier_offsets[i + 1];
            }));
        thrust::gather(handle.get_thrust_policy(),
                       map_first,
                       map_first + num_local_edge_partitions,
                       aggregate_mid_local_frontier_local_degree_offsets.begin(),
                       d_lasts.begin());
        std::vector<size_t> h_lasts(d_lasts.size());
        raft::update_host(h_lasts.data(), d_lasts.data(), d_lasts.size(), handle.get_stream());
        handle.sync_stream();
        std::adjacent_difference(h_lasts.begin(), h_lasts.end(), tx_counts.begin());
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
                         raft::host_span<size_t const>(mid_local_frontier_sizes.data(),
                                                       mid_local_frontier_sizes.size()),
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
                       raft::host_span<size_t const>(tx_counts.data(), tx_counts.size()),
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

      compute_homogeneous_biased_sampling_index_without_replacement<edge_t, bias_t>(
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
    }

    auto high_frontier_size = frontier_partition_offsets[3] - frontier_partition_offsets[2];
    std::vector<size_t> high_local_frontier_sizes{};
    high_local_frontier_sizes =
      host_scalar_allgather(minor_comm, high_frontier_size, handle.get_stream());

    std::vector<size_t> high_local_frontier_offsets(high_local_frontier_sizes.size() + 1);
    high_local_frontier_offsets[0] = 0;
    std::inclusive_scan(high_local_frontier_sizes.begin(),
                        high_local_frontier_sizes.end(),
                        high_local_frontier_offsets.begin() + 1);
    if (high_local_frontier_offsets.back() > 0) {
      // aggregate frontier indices with their degrees in the high range

      auto aggregate_high_local_frontier_indices =
        rmm::device_uvector<size_t>(high_local_frontier_offsets.back(), handle.get_stream());
      device_allgatherv(minor_comm,
                        frontier_indices.begin() + frontier_partition_offsets[2],
                        aggregate_high_local_frontier_indices.begin(),
                        raft::host_span<size_t const>(high_local_frontier_sizes.data(),
                                                      high_local_frontier_sizes.size()),
                        raft::host_span<size_t const>(high_local_frontier_offsets.data(),
                                                      high_local_frontier_offsets.size() - 1),
                        handle.get_stream());

      // local sample and update indices

      rmm::device_uvector<edge_t> aggregate_high_local_frontier_local_nbr_indices(
        high_local_frontier_offsets.back() * K, handle.get_stream());
      rmm::device_uvector<bias_t> aggregate_high_local_frontier_keys(
        aggregate_high_local_frontier_local_nbr_indices.size(), handle.get_stream());
      for (size_t i = 0; i < num_local_edge_partitions; ++i) {
        rmm::device_uvector<size_t> unique_key_indices_for_key_indices(high_local_frontier_sizes[i],
                                                                       handle.get_stream());
        thrust::gather(
          handle.get_thrust_policy(),
          aggregate_high_local_frontier_indices.begin() + high_local_frontier_offsets[i],
          aggregate_high_local_frontier_indices.begin() + high_local_frontier_offsets[i + 1],
          aggregate_local_frontier_key_idx_to_unique_key_idx.data() + local_frontier_offsets[i],
          unique_key_indices_for_key_indices.begin());
        compute_homogeneous_biased_sampling_index_without_replacement<edge_t, bias_t>(
          handle,
          std::make_optional<raft::device_span<size_t const>>(
            unique_key_indices_for_key_indices.data(), unique_key_indices_for_key_indices.size()),
          raft::device_span<size_t const>(
            aggregate_local_frontier_unique_key_local_degree_offsets.data() +
              local_frontier_unique_key_offsets[i],
            (local_frontier_unique_key_offsets[i + 1] - local_frontier_unique_key_offsets[i]) + 1),
          raft::device_span<bias_t const>(aggregate_local_frontier_unique_key_biases.data(),
                                          aggregate_local_frontier_unique_key_biases.size()),
          std::nullopt,
          raft::device_span<edge_t>(aggregate_high_local_frontier_local_nbr_indices.data() +
                                      high_local_frontier_offsets[i] * K,
                                    high_local_frontier_sizes[i] * K),
          std::make_optional<raft::device_span<bias_t>>(
            aggregate_high_local_frontier_keys.data() + high_local_frontier_offsets[i] * K,
            high_local_frontier_sizes[i] * K),
          rng_state,
          K,
          false);
      }

      // shuffle local sampling outputs

      std::vector<size_t> tx_counts(high_local_frontier_sizes.size());
      std::transform(high_local_frontier_sizes.begin(),
                     high_local_frontier_sizes.end(),
                     tx_counts.begin(),
                     [K](size_t size) { return size * K; });
      rmm::device_uvector<edge_t> high_frontier_gathered_local_nbr_indices(0, handle.get_stream());
      std::tie(high_frontier_gathered_local_nbr_indices, std::ignore) =
        shuffle_values(minor_comm,
                       aggregate_high_local_frontier_local_nbr_indices.data(),
                       raft::host_span<size_t const>(tx_counts.data(), tx_counts.size()),
                       handle.get_stream());
      rmm::device_uvector<bias_t> high_frontier_gathered_keys(0, handle.get_stream());
      std::tie(high_frontier_gathered_keys, std::ignore) =
        shuffle_values(minor_comm,
                       aggregate_high_local_frontier_keys.data(),
                       raft::host_span<size_t const>(tx_counts.data(), tx_counts.size()),
                       handle.get_stream());
      aggregate_high_local_frontier_local_nbr_indices.resize(0, handle.get_stream());
      aggregate_high_local_frontier_local_nbr_indices.shrink_to_fit(handle.get_stream());
      aggregate_high_local_frontier_keys.resize(0, handle.get_stream());
      aggregate_high_local_frontier_keys.shrink_to_fit(handle.get_stream());

      // merge local sampling outputs

      rmm::device_uvector<edge_t> high_frontier_nbr_indices(
        high_frontier_size * minor_comm_size * K, handle.get_stream());
      rmm::device_uvector<bias_t> high_frontier_keys(high_frontier_nbr_indices.size(),
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
            return frontier_partitioned_local_degree_displacements[frontier_idx * minor_comm_size +
                                                                   minor_comm_rank] +
                   high_frontier_gathered_local_nbr_indices[i];
          }));
      thrust::gather(
        handle.get_thrust_policy(),
        index_first,
        index_first + high_frontier_nbr_indices.size(),
        thrust::make_zip_iterator(high_frontier_gathered_nbr_idx_first,
                                  high_frontier_gathered_keys.begin()),
        thrust::make_zip_iterator(high_frontier_nbr_indices.begin(), high_frontier_keys.begin()));
      high_frontier_gathered_local_nbr_indices.resize(0, handle.get_stream());
      high_frontier_gathered_local_nbr_indices.shrink_to_fit(handle.get_stream());
      high_frontier_gathered_keys.resize(0, handle.get_stream());
      high_frontier_gathered_keys.shrink_to_fit(handle.get_stream());

      rmm::device_uvector<std::byte> d_tmp_storage(0, handle.get_stream());
      size_t tmp_storage_bytes{0};

      rmm::device_uvector<edge_t> high_frontier_segment_sorted_nbr_indices(
        high_frontier_nbr_indices.size(), handle.get_stream());
      rmm::device_uvector<bias_t> high_frontier_segment_sorted_keys(high_frontier_keys.size(),
                                                                    handle.get_stream());
      cub::DeviceSegmentedSort::SortPairs(
        static_cast<void*>(nullptr),
        tmp_storage_bytes,
        high_frontier_keys.data(),
        high_frontier_segment_sorted_keys.data(),
        high_frontier_nbr_indices.data(),
        high_frontier_segment_sorted_nbr_indices.data(),
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
        high_frontier_nbr_indices.data(),
        high_frontier_segment_sorted_nbr_indices.data(),
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
         high_frontier_segment_sorted_nbr_indices =
           raft::device_span<edge_t const>(high_frontier_segment_sorted_nbr_indices.data(),
                                           high_frontier_segment_sorted_nbr_indices.size()),
         nbr_indices = raft::device_span<edge_t>(nbr_indices.data(), nbr_indices.size()),
         K,
         minor_comm_size] __device__(size_t i) {
          thrust::copy(
            thrust::seq,
            high_frontier_segment_sorted_nbr_indices.begin() + (i * K * minor_comm_size),
            high_frontier_segment_sorted_nbr_indices.begin() + (i * K * minor_comm_size + K),
            nbr_indices.begin() + high_frontier_indices[i] * K);
        });
    }

    std::tie(local_nbr_indices, key_indices, local_frontier_sample_offsets) =
      shuffle_and_compute_local_nbr_values<edge_t>(
        handle,
        std::move(nbr_indices),
        raft::device_span<edge_t const>((*frontier_partitioned_local_degree_displacements).data(),
                                        (*frontier_partitioned_local_degree_displacements).size()),
        K,
        cugraph::invalid_edge_id_v<edge_t>);

  } else {  // minor_comm_size == 1
    local_nbr_indices.resize(frontier_degrees.size() * K, handle.get_stream());
    // sample from low-degree vertices

    if (frontier_partition_offsets[1] > 0) {
      thrust::for_each(handle.get_thrust_policy(),
                       frontier_indices.begin(),
                       frontier_indices.begin() + frontier_partition_offsets[1],
                       [frontier_degrees  = raft::device_span<edge_t const>(frontier_degrees.data(),
                                                                           frontier_degrees.size()),
                        local_nbr_indices = raft::device_span<edge_t>(local_nbr_indices.data(),
                                                                      local_nbr_indices.size()),
                        K,
                        invalid_idx = cugraph::invalid_edge_id_v<edge_t>] __device__(size_t i) {
                         auto degree = frontier_degrees[i];
                         thrust::sequence(thrust::seq,
                                          local_nbr_indices.begin() + i * K,
                                          local_nbr_indices.begin() + i * K + degree,
                                          edge_t{0});
                         thrust::fill(thrust::seq,
                                      local_nbr_indices.begin() + i * K + +degree,
                                      local_nbr_indices.begin() + (i + 1) * K,
                                      invalid_idx);
                       });
    }

    // sample from mid & high-degree vertices

    auto mid_and_high_frontier_size = frontier_partition_offsets[3] - frontier_partition_offsets[1];

    if (mid_and_high_frontier_size > 0) {
      rmm::device_uvector<size_t> unique_key_indices_for_key_indices(mid_and_high_frontier_size,
                                                                     handle.get_stream());
      thrust::gather(
        handle.get_thrust_policy(),
        frontier_indices.begin() + frontier_partition_offsets[1],
        frontier_indices.begin() + frontier_partition_offsets[1] + mid_and_high_frontier_size,
        aggregate_local_frontier_key_idx_to_unique_key_idx.begin(),
        unique_key_indices_for_key_indices.begin());
      compute_homogeneous_biased_sampling_index_without_replacement<edge_t, bias_t>(
        handle,
        std::make_optional<raft::device_span<size_t const>>(
          unique_key_indices_for_key_indices.data(), unique_key_indices_for_key_indices.size()),
        raft::device_span<size_t const>(
          aggregate_local_frontier_unique_key_local_degree_offsets.data(),
          aggregate_local_frontier_unique_key_local_degree_offsets.size()),
        raft::device_span<bias_t const>(aggregate_local_frontier_unique_key_biases.data(),
                                        aggregate_local_frontier_unique_key_biases.size()),
        std::make_optional<raft::device_span<size_t const>>(
          frontier_indices.data() + frontier_partition_offsets[1], mid_and_high_frontier_size),
        raft::device_span<edge_t>(local_nbr_indices.data(), local_nbr_indices.size()),
        std::nullopt,
        rng_state,
        K,
        false);
    }

    local_frontier_sample_offsets = std::vector<size_t>{0, local_nbr_indices.size()};
  }

  return std::make_tuple(
    std::move(local_nbr_indices), std::move(key_indices), std::move(local_frontier_sample_offsets));
}

template <typename edge_t, typename edge_type_t, typename bias_t, bool multi_gpu>
std::tuple<rmm::device_uvector<edge_t> /* local_nbr_indices */,
           std::optional<rmm::device_uvector<size_t>> /* key_indices */,
           std::vector<size_t> /* local_frontier_sample_offsets */>
heterogeneous_biased_sample_without_replacement(
  raft::handle_t const& handle,
  raft::device_span<size_t const> aggregate_local_frontier_key_idx_to_unique_key_idx,
  raft::host_span<size_t const> local_frontier_offsets,
  raft::device_span<bias_t const> aggregate_local_frontier_unique_key_biases,
  raft::device_span<size_t const> aggregate_local_frontier_unique_key_per_type_local_degree_offsets,
  raft::host_span<size_t const> local_frontier_unique_key_offsets,
  raft::random::RngState& rng_state,
  raft::host_span<size_t const> Ks)
{
  int minor_comm_rank{0};
  int minor_comm_size{1};
  if constexpr (multi_gpu) {
    auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    minor_comm_rank  = minor_comm.get_rank();
    minor_comm_size  = minor_comm.get_size();
  }

  auto num_local_edge_partitions = local_frontier_offsets.size() - 1;
  auto num_edge_types            = static_cast<edge_type_t>(Ks.size());

  rmm::device_uvector<edge_t> local_nbr_indices(0, handle.get_stream());
  std::optional<rmm::device_uvector<size_t>> key_indices{std::nullopt};
  std::vector<size_t> local_frontier_sample_offsets{};

  rmm::device_uvector<edge_t> frontier_per_type_degrees(0, handle.get_stream());
  std::optional<rmm::device_uvector<edge_t>>
    frontier_partitioned_per_type_local_degree_displacements{std::nullopt};
  {
    rmm::device_uvector<edge_t> aggregate_local_frontier_per_type_local_degrees(
      local_frontier_offsets.back() * num_edge_types, handle.get_stream());
    for (size_t i = 0; i < num_local_edge_partitions; ++i) {
      thrust::tabulate(
        handle.get_thrust_policy(),
        aggregate_local_frontier_per_type_local_degrees.begin() +
          local_frontier_offsets[i] * num_edge_types,
        aggregate_local_frontier_per_type_local_degrees.begin() +
          local_frontier_offsets[i + 1] * num_edge_types,
        [key_idx_to_unique_key_idx = raft::device_span<size_t const>(
           aggregate_local_frontier_key_idx_to_unique_key_idx.data() + local_frontier_offsets[i],
           local_frontier_offsets[i + 1] - local_frontier_offsets[i]),
         unique_key_per_type_local_degree_offsets = raft::device_span<size_t const>(
           aggregate_local_frontier_unique_key_per_type_local_degree_offsets.data() +
             local_frontier_unique_key_offsets[i] * num_edge_types,
           (local_frontier_unique_key_offsets[i + 1] - local_frontier_unique_key_offsets[i]) *
               num_edge_types +
             1),
         num_edge_types] __device__(size_t i) {
          auto key_idx        = i / num_edge_types;
          auto edge_type      = static_cast<edge_type_t>(i % num_edge_types);
          auto unique_key_idx = key_idx_to_unique_key_idx[key_idx];
          return static_cast<edge_t>(
            unique_key_per_type_local_degree_offsets[unique_key_idx * num_edge_types + edge_type +
                                                     1] -
            unique_key_per_type_local_degree_offsets[unique_key_idx * num_edge_types + edge_type]);
        });
    }
    if (minor_comm_size > 1) {
      std::tie(frontier_per_type_degrees,
               frontier_partitioned_per_type_local_degree_displacements) =
        compute_frontier_value_sums_and_partitioned_local_value_sum_displacements(
          handle,
          raft::device_span<edge_t const>(aggregate_local_frontier_per_type_local_degrees.data(),
                                          aggregate_local_frontier_per_type_local_degrees.size()),
          local_frontier_offsets,
          num_edge_types);
    } else {
      frontier_per_type_degrees = std::move(aggregate_local_frontier_per_type_local_degrees);
    }
  }

  std::vector<edge_t> thresholds(num_edge_types * 2);
  for (edge_type_t i = 0; i < num_edge_types; ++i) {
    thresholds[i * 2]     = static_cast<edge_t>(Ks[i] + 1);
    thresholds[i * 2 + 1] = static_cast<edge_t>(minor_comm_size * Ks[i] * 2);
  }
  auto [frontier_indices, frontier_edge_types, frontier_partition_offsets] =
    partition_v_frontier_per_value_idx<edge_type_t>(
      handle,
      frontier_per_type_degrees.begin(),
      frontier_per_type_degrees.end(),
      raft::host_span<edge_t const>(thresholds.data(), thresholds.size()),
      num_edge_types);

  auto K_sum = std::accumulate(Ks.begin(), Ks.end(), size_t{0});

  rmm::device_uvector<size_t> d_K_offsets(Ks.size() + 1, handle.get_stream());
  {
    std::vector<size_t> h_K_offsets(d_K_offsets.size());
    h_K_offsets[0] = 0;
    std::inclusive_scan(Ks.begin(), Ks.end(), h_K_offsets.begin() + 1);
    raft::update_device(
      d_K_offsets.data(), h_K_offsets.data(), h_K_offsets.size(), handle.get_stream());
  }

  rmm::device_uvector<edge_t> per_type_local_nbr_indices(0, handle.get_stream());
  std::optional<rmm::device_uvector<edge_type_t>> edge_types{std::nullopt};
  if (minor_comm_size > 1) {
    auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());

    rmm::device_uvector<edge_t> per_type_nbr_indices(
      (frontier_per_type_degrees.size() / num_edge_types) * K_sum, handle.get_stream());

    if (frontier_partition_offsets[1] > 0) {
      auto pair_first =
        thrust::make_zip_iterator(frontier_indices.begin(), frontier_edge_types.begin());
      thrust::for_each(
        handle.get_thrust_policy(),
        pair_first,
        pair_first + frontier_partition_offsets[1],
        [frontier_per_type_degrees = raft::device_span<edge_t const>(
           frontier_per_type_degrees.data(), frontier_per_type_degrees.size()),
         per_type_nbr_indices =
           raft::device_span<edge_t>(per_type_nbr_indices.data(), per_type_nbr_indices.size()),
         K_offsets = raft::device_span<size_t const>(d_K_offsets.data(), d_K_offsets.size()),
         K_sum,
         invalid_idx = cugraph::invalid_edge_id_v<edge_t>] __device__(auto pair) {
          auto num_edge_types  = static_cast<edge_type_t>(K_offsets.size() - 1);
          auto idx             = thrust::get<0>(pair);
          auto type            = thrust::get<1>(pair);
          auto per_type_degree = frontier_per_type_degrees[idx * num_edge_types + type];
          thrust::sequence(
            thrust::seq,
            per_type_nbr_indices.begin() + idx * K_sum + K_offsets[type],
            per_type_nbr_indices.begin() + idx * K_sum + K_offsets[type] + per_type_degree,
            edge_t{0});
          thrust::fill(
            thrust::seq,
            per_type_nbr_indices.begin() + idx * K_sum + K_offsets[type] + per_type_degree,
            per_type_nbr_indices.begin() + idx * K_sum + K_offsets[type + 1],
            invalid_idx);
        });
    }

    auto mid_frontier_size = frontier_partition_offsets[2] - frontier_partition_offsets[1];
    auto mid_local_frontier_sizes =
      host_scalar_allgather(minor_comm, mid_frontier_size, handle.get_stream());
    std::vector<size_t> mid_local_frontier_offsets(mid_local_frontier_sizes.size() + 1);
    mid_local_frontier_offsets[0] = 0;
    std::inclusive_scan(mid_local_frontier_sizes.begin(),
                        mid_local_frontier_sizes.end(),
                        mid_local_frontier_offsets.begin() + 1);

    if (mid_local_frontier_offsets.back() > 0) {
      // aggregate frontier index type pairs with their degrees in the medium range

      auto aggregate_mid_local_frontier_index_type_pairs =
        allocate_dataframe_buffer<thrust::tuple<size_t, edge_type_t>>(
          mid_local_frontier_offsets.back(), handle.get_stream());
      device_allgatherv(
        minor_comm,
        thrust::make_zip_iterator(frontier_indices.begin(), frontier_edge_types.begin()) +
          frontier_partition_offsets[1],
        get_dataframe_buffer_begin(aggregate_mid_local_frontier_index_type_pairs),
        raft::host_span<size_t const>(mid_local_frontier_sizes.data(),
                                      mid_local_frontier_sizes.size()),
        raft::host_span<size_t const>(mid_local_frontier_offsets.data(),
                                      mid_local_frontier_offsets.size() - 1),
        handle.get_stream());

      // compute per-type local degrees for the aggregated frontier index type pairs

      rmm::device_uvector<edge_t> aggregate_mid_local_frontier_per_type_local_degrees(
        size_dataframe_buffer(aggregate_mid_local_frontier_index_type_pairs), handle.get_stream());
      for (size_t i = 0; i < num_local_edge_partitions; ++i) {
        thrust::transform(
          handle.get_thrust_policy(),
          get_dataframe_buffer_begin(aggregate_mid_local_frontier_index_type_pairs) +
            mid_local_frontier_offsets[i],
          get_dataframe_buffer_begin(aggregate_mid_local_frontier_index_type_pairs) +
            mid_local_frontier_offsets[i + 1],
          aggregate_mid_local_frontier_per_type_local_degrees.begin() +
            mid_local_frontier_offsets[i],
          cuda::proclaim_return_type<edge_t>(
            [key_idx_to_unique_key_idx = raft::device_span<size_t const>(
               aggregate_local_frontier_key_idx_to_unique_key_idx.data() +
                 local_frontier_offsets[i],
               local_frontier_offsets[i + 1] - local_frontier_offsets[i]),
             unique_key_per_type_local_degree_offsets = raft::device_span<size_t const>(
               aggregate_local_frontier_unique_key_per_type_local_degree_offsets.data() +
                 local_frontier_unique_key_offsets[i] * num_edge_types,
               (local_frontier_unique_key_offsets[i + 1] - local_frontier_unique_key_offsets[i]) *
                   num_edge_types +
                 1),
             num_edge_types] __device__(auto pair) {
              auto key_idx        = thrust::get<0>(pair);
              auto type           = thrust::get<1>(pair);
              auto unique_key_idx = key_idx_to_unique_key_idx[key_idx];
              return static_cast<edge_t>(
                unique_key_per_type_local_degree_offsets[unique_key_idx * num_edge_types + type +
                                                         1] -
                unique_key_per_type_local_degree_offsets[unique_key_idx * num_edge_types + type]);
            }));
      }

      // gather biases for the aggregated frontier index type pairs

      rmm::device_uvector<bias_t> aggregate_mid_local_frontier_biases(0, handle.get_stream());
      std::vector<size_t> tx_counts(mid_local_frontier_sizes.size(), 0);
      {
        rmm::device_uvector<size_t> aggregate_mid_local_frontier_per_type_local_degree_offsets(
          aggregate_mid_local_frontier_per_type_local_degrees.size() + 1, handle.get_stream());
        aggregate_mid_local_frontier_per_type_local_degree_offsets.set_element_to_zero_async(
          0, handle.get_stream());
        thrust::inclusive_scan(
          handle.get_thrust_policy(),
          aggregate_mid_local_frontier_per_type_local_degrees.begin(),
          aggregate_mid_local_frontier_per_type_local_degrees.end(),
          aggregate_mid_local_frontier_per_type_local_degree_offsets.begin() + 1);
        aggregate_mid_local_frontier_biases.resize(
          aggregate_mid_local_frontier_per_type_local_degree_offsets.back_element(
            handle.get_stream()),
          handle.get_stream());
        for (size_t i = 0; i < num_local_edge_partitions; ++i) {
          thrust::for_each(
            handle.get_thrust_policy(),
            thrust::make_counting_iterator(size_t{0}),
            thrust::make_counting_iterator(mid_local_frontier_offsets[i + 1] -
                                           mid_local_frontier_offsets[i]),
            [key_idx_to_unique_key_idx = raft::device_span<size_t const>(
               aggregate_local_frontier_key_idx_to_unique_key_idx.data() +
                 local_frontier_offsets[i],
               local_frontier_offsets[i + 1] - local_frontier_offsets[i]),
             aggregate_local_frontier_unique_key_biases =
               raft::device_span<bias_t const>(aggregate_local_frontier_unique_key_biases.data(),
                                               aggregate_local_frontier_unique_key_biases.size()),
             unique_key_per_type_local_degree_offsets = raft::device_span<size_t const>(
               aggregate_local_frontier_unique_key_per_type_local_degree_offsets.data() +
                 local_frontier_unique_key_offsets[i] * num_edge_types,
               (local_frontier_unique_key_offsets[i + 1] - local_frontier_unique_key_offsets[i]) *
                   num_edge_types +
                 1),
             mid_local_frontier_indices = raft::device_span<size_t const>(
               std::get<0>(aggregate_mid_local_frontier_index_type_pairs).data() +
                 mid_local_frontier_offsets[i],
               mid_local_frontier_sizes[i]),
             mid_local_frontier_types = raft::device_span<edge_type_t const>(
               std::get<1>(aggregate_mid_local_frontier_index_type_pairs).data() +
                 mid_local_frontier_offsets[i],
               mid_local_frontier_sizes[i]),
             mid_local_frontier_per_type_local_degree_offsets = raft::device_span<size_t const>(
               aggregate_mid_local_frontier_per_type_local_degree_offsets.data() +
                 mid_local_frontier_offsets[i],
               mid_local_frontier_sizes[i]),
             aggregate_mid_local_frontier_biases =
               raft::device_span<bias_t>(aggregate_mid_local_frontier_biases.data(),
                                         aggregate_mid_local_frontier_biases.size()),
             num_edge_types] __device__(size_t i) {
              auto unique_key_idx = key_idx_to_unique_key_idx[mid_local_frontier_indices[i]];
              auto type           = mid_local_frontier_types[i];
              thrust::copy(
                thrust::seq,
                aggregate_local_frontier_unique_key_biases.begin() +
                  unique_key_per_type_local_degree_offsets[unique_key_idx * num_edge_types + type],
                aggregate_local_frontier_unique_key_biases.begin() +
                  unique_key_per_type_local_degree_offsets[unique_key_idx * num_edge_types + type +
                                                           1],
                aggregate_mid_local_frontier_biases.begin() +
                  mid_local_frontier_per_type_local_degree_offsets[i]);
            });
        }
        std::get<0>(aggregate_mid_local_frontier_index_type_pairs).resize(0, handle.get_stream());
        std::get<1>(aggregate_mid_local_frontier_index_type_pairs).resize(0, handle.get_stream());
        std::get<0>(aggregate_mid_local_frontier_index_type_pairs)
          .shrink_to_fit(handle.get_stream());
        std::get<1>(aggregate_mid_local_frontier_index_type_pairs)
          .shrink_to_fit(handle.get_stream());

        rmm::device_uvector<size_t> d_mid_local_frontier_offsets(mid_local_frontier_offsets.size(),
                                                                 handle.get_stream());
        raft::update_device(d_mid_local_frontier_offsets.data(),
                            mid_local_frontier_offsets.data(),
                            mid_local_frontier_offsets.size(),
                            handle.get_stream());
        rmm::device_uvector<size_t> d_lasts(num_local_edge_partitions, handle.get_stream());
        thrust::gather(handle.get_thrust_policy(),
                       d_mid_local_frontier_offsets.begin() + 1,
                       d_mid_local_frontier_offsets.end(),
                       aggregate_mid_local_frontier_per_type_local_degree_offsets.begin(),
                       d_lasts.begin());
        std::vector<size_t> h_lasts(d_lasts.size());
        raft::update_host(h_lasts.data(), d_lasts.data(), d_lasts.size(), handle.get_stream());
        handle.sync_stream();
        std::adjacent_difference(h_lasts.begin(), h_lasts.end(), tx_counts.begin());
      }

      // shuffle local degrees & biases

      rmm::device_uvector<size_t> mid_frontier_gathered_per_type_local_degree_offsets(
        0, handle.get_stream());
      {
        rmm::device_uvector<edge_t> mid_frontier_gathered_per_type_local_degrees(
          0, handle.get_stream());
        std::tie(mid_frontier_gathered_per_type_local_degrees, std::ignore) =
          shuffle_values(minor_comm,
                         aggregate_mid_local_frontier_per_type_local_degrees.data(),
                         raft::host_span<size_t const>(mid_local_frontier_sizes.data(),
                                                       mid_local_frontier_sizes.size()),
                         handle.get_stream());
        aggregate_mid_local_frontier_per_type_local_degrees.resize(0, handle.get_stream());
        aggregate_mid_local_frontier_per_type_local_degrees.shrink_to_fit(handle.get_stream());
        mid_frontier_gathered_per_type_local_degree_offsets.resize(
          mid_frontier_gathered_per_type_local_degrees.size() + 1, handle.get_stream());
        mid_frontier_gathered_per_type_local_degree_offsets.set_element_to_zero_async(
          0, handle.get_stream());
        thrust::inclusive_scan(handle.get_thrust_policy(),
                               mid_frontier_gathered_per_type_local_degrees.begin(),
                               mid_frontier_gathered_per_type_local_degrees.end(),
                               mid_frontier_gathered_per_type_local_degree_offsets.begin() + 1);
      }

      rmm::device_uvector<bias_t> mid_frontier_gathered_biases(0, handle.get_stream());
      std::tie(mid_frontier_gathered_biases, std::ignore) =
        shuffle_values(minor_comm,
                       aggregate_mid_local_frontier_biases.data(),
                       raft::host_span<size_t const>(tx_counts.data(), tx_counts.size()),
                       handle.get_stream());
      aggregate_mid_local_frontier_biases.resize(0, handle.get_stream());
      aggregate_mid_local_frontier_biases.shrink_to_fit(handle.get_stream());

      auto mid_frontier_per_type_degree_first = thrust::make_transform_iterator(
        thrust::make_zip_iterator(frontier_indices.begin(), frontier_edge_types.begin()) +
          frontier_partition_offsets[1],
        cuda::proclaim_return_type<edge_t>(
          [frontier_per_type_degrees = raft::device_span<edge_t>(frontier_per_type_degrees.data(),
                                                                 frontier_per_type_degrees.size()),
           num_edge_types] __device__(auto pair) {
            return frontier_per_type_degrees[thrust::get<0>(pair) * num_edge_types +
                                             thrust::get<1>(pair)];
          }));
      rmm::device_uvector<size_t> mid_frontier_per_type_degree_offsets(mid_frontier_size + 1,
                                                                       handle.get_stream());
      mid_frontier_per_type_degree_offsets.set_element_to_zero_async(0, handle.get_stream());
      thrust::inclusive_scan(handle.get_thrust_policy(),
                             mid_frontier_per_type_degree_first,
                             mid_frontier_per_type_degree_first + mid_frontier_size,
                             mid_frontier_per_type_degree_offsets.begin() + 1);
      rmm::device_uvector<bias_t> mid_frontier_biases(mid_frontier_gathered_biases.size(),
                                                      handle.get_stream());
      thrust::for_each(
        handle.get_thrust_policy(),
        thrust::make_counting_iterator(size_t{0}),
        thrust::make_counting_iterator(mid_frontier_size),
        [mid_frontier_gathered_per_type_local_degree_offsets =
           raft::device_span<size_t>(mid_frontier_gathered_per_type_local_degree_offsets.data(),
                                     mid_frontier_gathered_per_type_local_degree_offsets.size()),
         mid_frontier_gathered_biases = raft::device_span<bias_t const>(
           mid_frontier_gathered_biases.data(), mid_frontier_gathered_biases.size()),
         mid_frontier_per_type_degree_offsets =
           raft::device_span<size_t>(mid_frontier_per_type_degree_offsets.data(),
                                     mid_frontier_per_type_degree_offsets.size()),
         mid_frontier_biases =
           raft::device_span<bias_t>(mid_frontier_biases.data(), mid_frontier_biases.size()),
         minor_comm_size,
         mid_frontier_size] __device__(size_t i) {
          auto output_offset = mid_frontier_per_type_degree_offsets[i];
          for (int j = 0; j < minor_comm_size; ++j) {
            auto input_offset =
              mid_frontier_gathered_per_type_local_degree_offsets[mid_frontier_size * j + i];
            auto input_size =
              mid_frontier_gathered_per_type_local_degree_offsets[mid_frontier_size * j + i + 1] -
              input_offset;
            thrust::copy(thrust::seq,
                         mid_frontier_gathered_biases.begin() + input_offset,
                         mid_frontier_gathered_biases.begin() + input_offset + input_size,
                         mid_frontier_biases.begin() + output_offset);
            output_offset += input_size;
          }
        });

      // now sample and update indices

      rmm::device_uvector<size_t> mid_frontier_output_start_displacements(mid_frontier_size,
                                                                          handle.get_stream());
      auto pair_first =
        thrust::make_zip_iterator(frontier_indices.begin(), frontier_edge_types.begin()) +
        frontier_partition_offsets[1];
      thrust::transform(
        handle.get_thrust_policy(),
        pair_first,
        pair_first + mid_frontier_size,
        mid_frontier_output_start_displacements.begin(),
        cuda::proclaim_return_type<size_t>(
          [K_offsets = raft::device_span<size_t const>(d_K_offsets.data(), d_K_offsets.size()),
           K_sum] __device__(auto pair) {
            auto idx  = thrust::get<0>(pair);
            auto type = thrust::get<1>(pair);
            return idx * K_sum + K_offsets[type];
          }));

      compute_heterogeneous_biased_sampling_index_without_replacement<edge_t, edge_type_t, bias_t>(
        handle,
        std::nullopt,
        raft::device_span<edge_type_t const>(
          frontier_edge_types.data() + frontier_partition_offsets[1], mid_frontier_size),
        raft::device_span<size_t const>(mid_frontier_per_type_degree_offsets.data(),
                                        mid_frontier_per_type_degree_offsets.size()),
        raft::device_span<bias_t const>(mid_frontier_biases.data(), mid_frontier_biases.size()),
        raft::device_span<size_t const>(mid_frontier_output_start_displacements.data(),
                                        mid_frontier_output_start_displacements.size()),
        raft::device_span<edge_t>(per_type_nbr_indices.data(), per_type_nbr_indices.size()),
        std::nullopt,
        rng_state,
        raft::device_span<size_t const>(d_K_offsets.data(), d_K_offsets.size()),
        false);
    }

    auto high_frontier_size = frontier_partition_offsets[3] - frontier_partition_offsets[2];

    auto high_local_frontier_sizes =
      host_scalar_allgather(minor_comm, high_frontier_size, handle.get_stream());
    std::vector<size_t> high_local_frontier_offsets(high_local_frontier_sizes.size() + 1);
    high_local_frontier_offsets[0] = 0;
    std::inclusive_scan(high_local_frontier_sizes.begin(),
                        high_local_frontier_sizes.end(),
                        high_local_frontier_offsets.begin() + 1);

    if (high_local_frontier_offsets.back() > 0) {
      // aggregate frontier index & type pairs with their degrees in the high range

      auto aggregate_high_local_frontier_index_type_pairs =
        allocate_dataframe_buffer<thrust::tuple<size_t, edge_type_t>>(
          high_local_frontier_offsets.back(), handle.get_stream());
      device_allgatherv(
        minor_comm,
        thrust::make_zip_iterator(frontier_indices.begin(), frontier_edge_types.begin()) +
          frontier_partition_offsets[2],
        get_dataframe_buffer_begin(aggregate_high_local_frontier_index_type_pairs),
        raft::host_span<size_t const>(high_local_frontier_sizes.data(),
                                      high_local_frontier_sizes.size()),
        raft::host_span<size_t const>(high_local_frontier_offsets.data(),
                                      high_local_frontier_offsets.size() - 1),
        handle.get_stream());

      // local sample and update indices

      rmm::device_uvector<size_t> aggregate_high_local_frontier_output_offsets(
        high_local_frontier_offsets.back() + 1, handle.get_stream());
      {
        auto K_first = thrust::make_transform_iterator(
          std::get<1>(aggregate_high_local_frontier_index_type_pairs).begin(),
          cuda::proclaim_return_type<size_t>(
            [d_K_offsets = raft::device_span<size_t const>(
               d_K_offsets.data(), d_K_offsets.size())] __device__(auto type) {
              return d_K_offsets[type + 1] - d_K_offsets[type];
            }));
        aggregate_high_local_frontier_output_offsets.set_element_to_zero_async(0,
                                                                               handle.get_stream());
        thrust::inclusive_scan(
          handle.get_thrust_policy(),
          K_first,
          K_first + std::get<1>(aggregate_high_local_frontier_index_type_pairs).size(),
          aggregate_high_local_frontier_output_offsets.begin() + 1);
      }

      rmm::device_uvector<edge_t> aggregate_high_local_frontier_per_type_local_nbr_indices(
        aggregate_high_local_frontier_output_offsets.back_element(handle.get_stream()),
        handle.get_stream());
      rmm::device_uvector<bias_t> aggregate_high_local_frontier_keys(
        aggregate_high_local_frontier_per_type_local_nbr_indices.size(), handle.get_stream());
      for (size_t i = 0; i < num_local_edge_partitions; ++i) {
        rmm::device_uvector<size_t> unique_key_indices_for_key_indices(high_local_frontier_sizes[i],
                                                                       handle.get_stream());
        thrust::gather(
          handle.get_thrust_policy(),
          std::get<0>(aggregate_high_local_frontier_index_type_pairs).begin() +
            high_local_frontier_offsets[i],
          std::get<0>(aggregate_high_local_frontier_index_type_pairs).begin() +
            high_local_frontier_offsets[i + 1],
          aggregate_local_frontier_key_idx_to_unique_key_idx.begin() + local_frontier_offsets[i],
          unique_key_indices_for_key_indices.begin());
        compute_heterogeneous_biased_sampling_index_without_replacement<edge_t,
                                                                        edge_type_t,
                                                                        bias_t>(
          handle,
          std::make_optional<raft::device_span<size_t const>>(
            unique_key_indices_for_key_indices.data(), unique_key_indices_for_key_indices.size()),
          raft::device_span<edge_type_t const>(
            std::get<1>(aggregate_high_local_frontier_index_type_pairs).data() +
              high_local_frontier_offsets[i],
            high_local_frontier_offsets[i + 1] - high_local_frontier_offsets[i]),
          raft::device_span<size_t const>(
            aggregate_local_frontier_unique_key_per_type_local_degree_offsets.data() +
              local_frontier_unique_key_offsets[i] * num_edge_types,
            (local_frontier_unique_key_offsets[i + 1] - local_frontier_unique_key_offsets[i]) *
                num_edge_types +
              1),
          raft::device_span<bias_t const>(aggregate_local_frontier_unique_key_biases.data(),
                                          aggregate_local_frontier_unique_key_biases.size()),
          raft::device_span<size_t const>(
            aggregate_high_local_frontier_output_offsets.data() + high_local_frontier_offsets[i],
            (high_local_frontier_offsets[i + 1] - high_local_frontier_offsets[i]) + 1),
          raft::device_span<edge_t>(
            aggregate_high_local_frontier_per_type_local_nbr_indices.data(),
            aggregate_high_local_frontier_per_type_local_nbr_indices.size()),
          std::make_optional<raft::device_span<bias_t>>(aggregate_high_local_frontier_keys.data(),
                                                        aggregate_high_local_frontier_keys.size()),
          rng_state,
          raft::device_span<size_t const>(d_K_offsets.data(), d_K_offsets.size()),
          false);
      }

      // shuffle local sampling outputs

      std::vector<size_t> tx_counts(high_local_frontier_sizes.size());
      {
        rmm::device_uvector<size_t> d_high_local_frontier_offsets(
          high_local_frontier_offsets.size(), handle.get_stream());
        raft::update_device(d_high_local_frontier_offsets.data(),
                            high_local_frontier_offsets.data(),
                            high_local_frontier_offsets.size(),
                            handle.get_stream());
        rmm::device_uvector<size_t> d_lasts(num_local_edge_partitions, handle.get_stream());
        thrust::gather(handle.get_thrust_policy(),
                       d_high_local_frontier_offsets.begin() + 1,
                       d_high_local_frontier_offsets.end(),
                       aggregate_high_local_frontier_output_offsets.begin(),
                       d_lasts.begin());
        std::vector<size_t> h_lasts(d_lasts.size());
        raft::update_host(h_lasts.data(), d_lasts.data(), d_lasts.size(), handle.get_stream());
        handle.sync_stream();
        std::adjacent_difference(h_lasts.begin(), h_lasts.end(), tx_counts.begin());
      }
      rmm::device_uvector<edge_t> high_frontier_gathered_per_type_local_nbr_indices(
        0, handle.get_stream());
      std::tie(high_frontier_gathered_per_type_local_nbr_indices, std::ignore) =
        shuffle_values(minor_comm,
                       aggregate_high_local_frontier_per_type_local_nbr_indices.data(),
                       raft::host_span<size_t const>(tx_counts.data(), tx_counts.size()),
                       handle.get_stream());
      rmm::device_uvector<bias_t> high_frontier_gathered_keys(0, handle.get_stream());
      std::tie(high_frontier_gathered_keys, std::ignore) =
        shuffle_values(minor_comm,
                       aggregate_high_local_frontier_keys.data(),
                       raft::host_span<size_t const>(tx_counts.data(), tx_counts.size()),
                       handle.get_stream());
      aggregate_high_local_frontier_per_type_local_nbr_indices.resize(0, handle.get_stream());
      aggregate_high_local_frontier_per_type_local_nbr_indices.shrink_to_fit(handle.get_stream());
      aggregate_high_local_frontier_keys.resize(0, handle.get_stream());
      aggregate_high_local_frontier_keys.shrink_to_fit(handle.get_stream());

      // merge local sampling outputs

      rmm::device_uvector<size_t> high_frontier_output_offsets(high_frontier_size + 1,
                                                               handle.get_stream());
      {
        auto K_first = thrust::make_transform_iterator(
          frontier_edge_types.begin() + frontier_partition_offsets[2],
          cuda::proclaim_return_type<size_t>(
            [K_offsets = raft::device_span<size_t const>(
               d_K_offsets.data(), d_K_offsets.size())] __device__(auto type) {
              return K_offsets[type + 1] - K_offsets[type];
            }));
        high_frontier_output_offsets.set_element_to_zero_async(0, handle.get_stream());
        thrust::inclusive_scan(handle.get_thrust_policy(),
                               K_first,
                               K_first + high_frontier_size,
                               high_frontier_output_offsets.begin() + 1);
      }

      rmm::device_uvector<edge_t> high_frontier_per_type_nbr_indices(
        high_frontier_output_offsets.back_element(handle.get_stream()) * minor_comm_size,
        handle.get_stream());
      rmm::device_uvector<bias_t> high_frontier_keys(high_frontier_per_type_nbr_indices.size(),
                                                     handle.get_stream());
      auto index_first = thrust::make_transform_iterator(
        thrust::make_counting_iterator(size_t{0}),
        cuda::proclaim_return_type<size_t>(
          [offsets = raft::device_span<size_t const>(high_frontier_output_offsets.data(),
                                                     high_frontier_output_offsets.size()),
           minor_comm_size] __device__(size_t i) {
            auto idx = cuda::std::distance(
              offsets.begin() + 1,
              thrust::upper_bound(
                thrust::seq, offsets.begin() + 1, offsets.end(), i / minor_comm_size));
            auto K               = offsets[idx + 1] - offsets[idx];
            auto minor_comm_rank = (i - offsets[idx] * minor_comm_size) / K;
            return minor_comm_rank * offsets[offsets.size() - 1] + offsets[idx] +
                   ((i - offsets[idx] * minor_comm_size) % K);
          }));
      auto high_frontier_gathered_per_type_nbr_idx_first = thrust::make_transform_iterator(
        thrust::counting_iterator(size_t{0}),
        cuda::proclaim_return_type<edge_t>(
          [frontier_partitioned_per_type_local_degree_displacements =
             raft::device_span<edge_t const>(
               (*frontier_partitioned_per_type_local_degree_displacements).data(),
               (*frontier_partitioned_per_type_local_degree_displacements).size()),
           high_frontier_indices = raft::device_span<size_t const>(
             frontier_indices.data() + frontier_partition_offsets[2], high_frontier_size),
           high_frontier_edge_types = raft::device_span<edge_type_t const>(
             frontier_edge_types.data() + frontier_partition_offsets[2], high_frontier_size),
           high_frontier_gathered_per_type_local_nbr_indices = raft::device_span<edge_t const>(
             high_frontier_gathered_per_type_local_nbr_indices.data(),
             high_frontier_gathered_per_type_local_nbr_indices.size()),
           offsets = raft::device_span<size_t const>(high_frontier_output_offsets.data(),
                                                     high_frontier_output_offsets.size()),
           num_edge_types,
           minor_comm_size] __device__(size_t i) {
            auto minor_comm_rank = static_cast<int>(i / offsets[offsets.size() - 1]);
            auto idx             = cuda::std::distance(
              offsets.begin() + 1,
              thrust::upper_bound(
                thrust::seq, offsets.begin() + 1, offsets.end(), i % offsets[offsets.size() - 1]));
            auto frontier_idx = high_frontier_indices[idx];
            auto type         = high_frontier_edge_types[idx];
            return frontier_partitioned_per_type_local_degree_displacements
                     [(frontier_idx * num_edge_types + type) * minor_comm_size + minor_comm_rank] +
                   high_frontier_gathered_per_type_local_nbr_indices[i];
          }));
      thrust::gather(handle.get_thrust_policy(),
                     index_first,
                     index_first + high_frontier_per_type_nbr_indices.size(),
                     thrust::make_zip_iterator(high_frontier_gathered_per_type_nbr_idx_first,
                                               high_frontier_gathered_keys.begin()),
                     thrust::make_zip_iterator(high_frontier_per_type_nbr_indices.begin(),
                                               high_frontier_keys.begin()));
      high_frontier_gathered_per_type_local_nbr_indices.resize(0, handle.get_stream());
      high_frontier_gathered_per_type_local_nbr_indices.shrink_to_fit(handle.get_stream());
      high_frontier_gathered_keys.resize(0, handle.get_stream());
      high_frontier_gathered_keys.shrink_to_fit(handle.get_stream());

      rmm::device_uvector<std::byte> d_tmp_storage(0, handle.get_stream());
      size_t tmp_storage_bytes{0};

      rmm::device_uvector<edge_t> high_frontier_segment_sorted_per_type_nbr_indices(
        high_frontier_per_type_nbr_indices.size(), handle.get_stream());
      rmm::device_uvector<bias_t> high_frontier_segment_sorted_keys(high_frontier_keys.size(),
                                                                    handle.get_stream());
      auto offset_first = thrust::make_transform_iterator(
        thrust::make_counting_iterator(size_t{0}),
        cuda::proclaim_return_type<size_t>(
          [offsets = raft::device_span<size_t const>(high_frontier_output_offsets.data(),
                                                     high_frontier_output_offsets.size()),
           minor_comm_size] __device__(auto i) { return offsets[i] * minor_comm_size; }));
      cub::DeviceSegmentedSort::SortPairs(
        static_cast<void*>(nullptr),
        tmp_storage_bytes,
        high_frontier_keys.data(),
        high_frontier_segment_sorted_keys.data(),
        high_frontier_per_type_nbr_indices.data(),
        high_frontier_segment_sorted_per_type_nbr_indices.data(),
        high_frontier_output_offsets.back_element(handle.get_stream()) * minor_comm_size,
        high_frontier_size,
        offset_first,
        offset_first + 1,
        handle.get_stream());
      if (tmp_storage_bytes > d_tmp_storage.size()) {
        d_tmp_storage = rmm::device_uvector<std::byte>(tmp_storage_bytes, handle.get_stream());
      }
      cub::DeviceSegmentedSort::SortPairs(
        d_tmp_storage.data(),
        tmp_storage_bytes,
        high_frontier_keys.data(),
        high_frontier_segment_sorted_keys.data(),
        high_frontier_per_type_nbr_indices.data(),
        high_frontier_segment_sorted_per_type_nbr_indices.data(),
        high_frontier_output_offsets.back_element(handle.get_stream()) * minor_comm_size,
        high_frontier_size,
        offset_first,
        offset_first + 1,
        handle.get_stream());

      thrust::for_each(
        handle.get_thrust_policy(),
        thrust::make_counting_iterator(size_t{0}),
        thrust::make_counting_iterator(high_frontier_size),
        [high_frontier_indices = raft::device_span<size_t const>(
           frontier_indices.data() + frontier_partition_offsets[2], high_frontier_size),
         high_frontier_edge_types = raft::device_span<edge_type_t const>(
           frontier_edge_types.data() + frontier_partition_offsets[2], high_frontier_size),
         high_frontier_segment_sorted_nbr_indices = raft::device_span<edge_t const>(
           high_frontier_segment_sorted_per_type_nbr_indices.data(),
           high_frontier_segment_sorted_per_type_nbr_indices.size()),
         offsets = raft::device_span<size_t const>(high_frontier_output_offsets.data(),
                                                   high_frontier_output_offsets.size()),
         per_type_nbr_indices =
           raft::device_span<edge_t>(per_type_nbr_indices.data(), per_type_nbr_indices.size()),
         K_offsets = raft::device_span<size_t const>(d_K_offsets.data(), d_K_offsets.size()),
         K_sum,
         minor_comm_size] __device__(size_t i) {
          auto type = high_frontier_edge_types[i];
          auto K    = K_offsets[type + 1] - K_offsets[type];
          thrust::copy(
            thrust::seq,
            high_frontier_segment_sorted_nbr_indices.begin() + offsets[i] * minor_comm_size,
            high_frontier_segment_sorted_nbr_indices.begin() + offsets[i] * minor_comm_size + K,
            per_type_nbr_indices.begin() + high_frontier_indices[i] * K_sum + K_offsets[type]);
        });
    }

    std::tie(per_type_local_nbr_indices, edge_types, key_indices, local_frontier_sample_offsets) =
      shuffle_and_compute_per_type_local_nbr_values<edge_type_t, edge_t>(
        handle,
        std::move(per_type_nbr_indices),
        raft::device_span<edge_t const>(
          (*frontier_partitioned_per_type_local_degree_displacements).data(),
          (*frontier_partitioned_per_type_local_degree_displacements).size()),
        raft::device_span<size_t const>(d_K_offsets.data(), d_K_offsets.size()),
        K_sum,
        cugraph::invalid_edge_id_v<edge_t>);
  } else {  // minor_comm_size == 1
    per_type_local_nbr_indices.resize(local_frontier_offsets.back() * K_sum, handle.get_stream());

    // sample from low-degree vertices

    if (frontier_partition_offsets[1] > 0) {
      auto pair_first =
        thrust::make_zip_iterator(frontier_indices.begin(), frontier_edge_types.begin());
      thrust::for_each(
        handle.get_thrust_policy(),
        pair_first,
        pair_first + frontier_partition_offsets[1],
        [frontier_per_type_degrees = raft::device_span<edge_t const>(
           frontier_per_type_degrees.data(), frontier_per_type_degrees.size()),
         per_type_local_nbr_indices = raft::device_span<edge_t>(per_type_local_nbr_indices.data(),
                                                                per_type_local_nbr_indices.size()),
         K_offsets = raft::device_span<size_t const>(d_K_offsets.data(), d_K_offsets.size()),
         K_sum,
         invalid_idx = cugraph::invalid_edge_id_v<edge_t>] __device__(auto pair) {
          auto num_edge_types  = static_cast<edge_type_t>(K_offsets.size() - 1);
          auto idx             = thrust::get<0>(pair);
          auto type            = thrust::get<1>(pair);
          auto per_type_degree = frontier_per_type_degrees[idx * num_edge_types + type];
          thrust::sequence(
            thrust::seq,
            per_type_local_nbr_indices.begin() + idx * K_sum + K_offsets[type],
            per_type_local_nbr_indices.begin() + idx * K_sum + K_offsets[type] + per_type_degree,
            edge_t{0});
          thrust::fill(
            thrust::seq,
            per_type_local_nbr_indices.begin() + idx * K_sum + K_offsets[type] + per_type_degree,
            per_type_local_nbr_indices.begin() + idx * K_sum + K_offsets[type + 1],
            invalid_idx);
        });
    }

    // sample from mid & high-degree vertices

    auto mid_and_high_frontier_size = frontier_partition_offsets[3] - frontier_partition_offsets[1];

    if (mid_and_high_frontier_size > 0) {
      rmm::device_uvector<size_t> unique_key_indices_for_key_indices(mid_and_high_frontier_size,
                                                                     handle.get_stream());
      thrust::gather(
        handle.get_thrust_policy(),
        frontier_indices.begin() + frontier_partition_offsets[1],
        frontier_indices.begin() + frontier_partition_offsets[1] + mid_and_high_frontier_size,
        aggregate_local_frontier_key_idx_to_unique_key_idx.begin(),
        unique_key_indices_for_key_indices.begin());

      rmm::device_uvector<size_t> mid_and_high_frontier_output_start_displacements(
        mid_and_high_frontier_size, handle.get_stream());
      auto pair_first =
        thrust::make_zip_iterator(frontier_indices.begin(), frontier_edge_types.begin());
      thrust::transform(
        handle.get_thrust_policy(),
        pair_first + frontier_partition_offsets[1],
        pair_first + frontier_partition_offsets[1] + mid_and_high_frontier_size,
        mid_and_high_frontier_output_start_displacements.begin(),
        cuda::proclaim_return_type<size_t>(
          [K_offsets = raft::device_span<size_t const>(d_K_offsets.data(), d_K_offsets.size()),
           K_sum] __device__(auto pair) {
            auto idx  = thrust::get<0>(pair);
            auto type = thrust::get<1>(pair);
            return idx * K_sum + K_offsets[type];
          }));

      compute_heterogeneous_biased_sampling_index_without_replacement<edge_t, edge_type_t, bias_t>(
        handle,
        std::make_optional<raft::device_span<size_t const>>(
          unique_key_indices_for_key_indices.data(), unique_key_indices_for_key_indices.size()),
        raft::device_span<edge_type_t const>(
          frontier_edge_types.data() + frontier_partition_offsets[1], mid_and_high_frontier_size),
        raft::device_span<size_t const>(
          aggregate_local_frontier_unique_key_per_type_local_degree_offsets.data(),
          aggregate_local_frontier_unique_key_per_type_local_degree_offsets.size()),
        raft::device_span<bias_t const>(aggregate_local_frontier_unique_key_biases.data(),
                                        aggregate_local_frontier_unique_key_biases.size()),
        raft::device_span<size_t const>(mid_and_high_frontier_output_start_displacements.data(),
                                        mid_and_high_frontier_output_start_displacements.size()),
        raft::device_span<edge_t>(per_type_local_nbr_indices.data(),
                                  per_type_local_nbr_indices.size()),
        std::nullopt,
        rng_state,
        raft::device_span<size_t const>(d_K_offsets.data(), d_K_offsets.size()),
        false);
    }

    local_frontier_sample_offsets = std::vector<size_t>{0, per_type_local_nbr_indices.size()};
  }

  // per-type local neighbor indices => local neighbor indices

  assert(edge_types.has_value() == key_indices.has_value());
  local_nbr_indices =
    compute_local_nbr_indices_from_per_type_local_nbr_indices<edge_t, edge_type_t>(
      handle,
      aggregate_local_frontier_key_idx_to_unique_key_idx,
      local_frontier_offsets,
      aggregate_local_frontier_unique_key_per_type_local_degree_offsets,
      local_frontier_unique_key_offsets,
      edge_types
        ? std::make_optional(std::make_tuple(
            raft::device_span<edge_type_t const>((*edge_types).data(), (*edge_types).size()),
            raft::device_span<size_t const>((*key_indices).data(), (*key_indices).size())))
        : std::nullopt,
      std::move(per_type_local_nbr_indices),
      raft::host_span<size_t const>(local_frontier_sample_offsets.data(),
                                    local_frontier_sample_offsets.size()),
      raft::device_span<size_t const>(d_K_offsets.data(), d_K_offsets.size()),
      K_sum);

  return std::make_tuple(
    std::move(local_nbr_indices), std::move(key_indices), std::move(local_frontier_sample_offsets));
}

// skip conversion if local neighbor index is cugraph::invalid_edge_id_v<edge_t>
template <typename edge_t>
rmm::device_uvector<edge_t> remap_local_nbr_indices(
  raft::handle_t const& handle,
  raft::device_span<size_t const> aggregate_local_frontier_key_idx_to_unique_key_idx,
  raft::host_span<size_t const> local_frontier_offsets,
  raft::device_span<edge_t const> aggregate_local_frontier_unique_key_org_indices,
  raft::device_span<size_t const> aggregate_local_frontier_unique_key_local_degree_offsets,
  raft::host_span<size_t const> local_frontier_unique_key_offsets,
  rmm::device_uvector<edge_t>&& local_nbr_indices,
  std::optional<raft::device_span<size_t const>> key_indices,
  raft::host_span<size_t const> local_frontier_sample_offsets,
  size_t K)
{
  if (key_indices) {
    auto pair_first = thrust::make_zip_iterator(local_nbr_indices.begin(), (*key_indices).begin());
    for (size_t i = 0; i < local_frontier_offsets.size() - 1; ++i) {
      thrust::transform(
        handle.get_thrust_policy(),
        pair_first + local_frontier_sample_offsets[i],
        pair_first + local_frontier_sample_offsets[i + 1],
        local_nbr_indices.begin() + local_frontier_sample_offsets[i],
        cuda::proclaim_return_type<edge_t>(
          [key_idx_to_unique_key_idx = raft::device_span<size_t const>(
             aggregate_local_frontier_key_idx_to_unique_key_idx.data() + local_frontier_offsets[i],
             local_frontier_offsets[i + 1] - local_frontier_offsets[i]),
           unique_key_local_degree_offsets = raft::device_span<size_t const>(
             aggregate_local_frontier_unique_key_local_degree_offsets.data() +
               local_frontier_unique_key_offsets[i],
             (local_frontier_unique_key_offsets[i + 1], local_frontier_unique_key_offsets[i]) + 1),
           aggregate_local_frontier_unique_key_org_indices = raft::device_span<edge_t const>(
             aggregate_local_frontier_unique_key_org_indices.data(),
             aggregate_local_frontier_unique_key_org_indices.size()),
           invalid_idx = cugraph::invalid_edge_id_v<edge_t>] __device__(auto pair) {
            auto local_nbr_idx = thrust::get<0>(pair);
            if (local_nbr_idx != invalid_idx) {
              auto key_idx        = thrust::get<1>(pair);
              auto unique_key_idx = key_idx_to_unique_key_idx[key_idx];
              return aggregate_local_frontier_unique_key_org_indices
                [unique_key_local_degree_offsets[unique_key_idx] + local_nbr_idx];
            } else {
              return invalid_idx;
            }
          }));
    }
  } else {
    auto pair_first = thrust::make_zip_iterator(local_nbr_indices.begin(),
                                                thrust::make_counting_iterator(size_t{0}));
    for (size_t i = 0; i < local_frontier_offsets.size() - 1; ++i) {
      thrust::transform(
        handle.get_thrust_policy(),
        pair_first + local_frontier_sample_offsets[i],
        pair_first + local_frontier_sample_offsets[i + 1],
        local_nbr_indices.begin() + local_frontier_sample_offsets[i],
        cuda::proclaim_return_type<edge_t>(
          [key_idx_to_unique_key_idx = raft::device_span<size_t const>(
             aggregate_local_frontier_key_idx_to_unique_key_idx.data() + local_frontier_offsets[i],
             local_frontier_offsets[i + 1] - local_frontier_offsets[i]),
           unique_key_local_degree_offsets = raft::device_span<size_t const>(
             aggregate_local_frontier_unique_key_local_degree_offsets.data() +
               local_frontier_unique_key_offsets[i],
             (local_frontier_unique_key_offsets[i + 1] - local_frontier_unique_key_offsets[i]) + 1),
           aggregate_local_frontier_unique_key_org_indices = raft::device_span<edge_t const>(
             aggregate_local_frontier_unique_key_org_indices.data(),
             aggregate_local_frontier_unique_key_org_indices.size()),
           K,
           invalid_idx = cugraph::invalid_edge_id_v<edge_t>] __device__(auto pair) {
            auto local_nbr_idx = thrust::get<0>(pair);
            if (local_nbr_idx != invalid_idx) {
              auto key_idx        = thrust::get<1>(pair) / K;
              auto unique_key_idx = key_idx_to_unique_key_idx[key_idx];
              return aggregate_local_frontier_unique_key_org_indices
                [unique_key_local_degree_offsets[unique_key_idx] + local_nbr_idx];
            } else {
              return invalid_idx;
            }
          }));
    }
  }

  return std::move(local_nbr_indices);
}

// skip conversion if local neighbor index is cugraph::invalid_edge_id_v<edge_t>
template <typename GraphViewType, typename VertexIterator>
rmm::device_uvector<typename GraphViewType::edge_type> convert_to_unmasked_local_nbr_idx(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  VertexIterator aggregate_local_frontier_major_first,
  rmm::device_uvector<typename GraphViewType::edge_type>&& local_nbr_indices,
  std::optional<raft::device_span<size_t const>> key_indices,
  raft::host_span<size_t const> local_frontier_sample_offsets,
  raft::host_span<size_t const> local_frontier_offsets,
  size_t K)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  static_assert(
    std::is_same_v<vertex_t, typename thrust::iterator_traits<VertexIterator>::value_type>);

  auto edge_mask_view = graph_view.edge_mask_view();

  auto [aggregate_local_frontier_unique_majors,
        aggregate_local_frontier_major_idx_to_unique_major_idx,
        local_frontier_unique_major_offsets] =
    compute_unique_keys(handle, aggregate_local_frontier_major_first, local_frontier_offsets);

  // to avoid searching the entire neighbor list K times for high degree vertices with edge
  // masking
  auto local_frontier_unique_major_valid_local_nbr_count_inclusive_sums =
    compute_valid_local_nbr_count_inclusive_sums(
      handle,
      graph_view,
      aggregate_local_frontier_unique_majors.begin(),
      raft::host_span<size_t const>(local_frontier_unique_major_offsets.data(),
                                    local_frontier_unique_major_offsets.size()));

  auto sample_major_idx_first = thrust::make_transform_iterator(
    thrust::make_counting_iterator(size_t{0}),
    cuda::proclaim_return_type<size_t>(
      [K,
       key_indices = key_indices ? cuda::std::make_optional<raft::device_span<size_t const>>(
                                     (*key_indices).data(), (*key_indices).size())
                                 : cuda::std::nullopt] __device__(size_t i) {
        return key_indices ? (*key_indices)[i] : i / K;
      }));
  auto pair_first = thrust::make_zip_iterator(local_nbr_indices.begin(), sample_major_idx_first);
  for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
    auto edge_partition =
      edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
        graph_view.local_edge_partition_view(i));
    auto edge_partition_e_mask =
      edge_mask_view
        ? cuda::std::make_optional<
            detail::edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool>>(
            *edge_mask_view, i)
        : cuda::std::nullopt;

    auto edge_partition_frontier_major_first =
      aggregate_local_frontier_major_first + local_frontier_offsets[i];
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
        raft::device_span<size_t const>(
          aggregate_local_frontier_major_idx_to_unique_major_idx.data() + local_frontier_offsets[i],
          local_frontier_offsets[i + 1] - local_frontier_offsets[i]),
        thrust::make_tuple(
          raft::device_span<size_t const>(
            std::get<0>(local_frontier_unique_major_valid_local_nbr_count_inclusive_sums[i]).data(),
            std::get<0>(local_frontier_unique_major_valid_local_nbr_count_inclusive_sums[i])
              .size()),
          raft::device_span<edge_t const>(
            std::get<1>(local_frontier_unique_major_valid_local_nbr_count_inclusive_sums[i]).data(),
            std::get<1>(local_frontier_unique_major_valid_local_nbr_count_inclusive_sums[i])
              .size()))},
      is_not_equal_t<edge_t>{cugraph::invalid_edge_id_v<edge_t>});
  }

  return std::move(local_nbr_indices);
}

template <typename GraphViewType, typename KeyIterator>
std::tuple<rmm::device_uvector<typename GraphViewType::edge_type>,
           std::optional<rmm::device_uvector<size_t>>,
           std::vector<size_t>>
homogeneous_uniform_sample_and_compute_local_nbr_indices(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  KeyIterator aggregate_local_frontier_key_first,
  raft::host_span<size_t const> local_frontier_offsets,
  raft::random::RngState& rng_state,
  size_t K,
  bool with_replacement)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using key_t    = typename thrust::iterator_traits<KeyIterator>::value_type;
  using bias_t   = double;

  int minor_comm_size{1};
  if constexpr (GraphViewType::is_multi_gpu) {
    auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    minor_comm_size  = minor_comm.get_size();
  }
  assert(minor_comm_size == graph_view.number_of_local_edge_partitions());

  auto aggregate_local_frontier_major_first =
    thrust_tuple_get_or_identity<KeyIterator, 0>(aggregate_local_frontier_key_first);

  auto edge_mask_view = graph_view.edge_mask_view();

  // 1. compute degrees

  rmm::device_uvector<edge_t> frontier_degrees(0, handle.get_stream());
  std::optional<rmm::device_uvector<edge_t>> frontier_partitioned_local_degree_displacements{
    std::nullopt};
  {
    auto aggregate_local_frontier_local_degrees = compute_aggregate_local_frontier_local_degrees(
      handle, graph_view, aggregate_local_frontier_major_first, local_frontier_offsets);

    if (minor_comm_size > 1) {
      std::tie(frontier_degrees, frontier_partitioned_local_degree_displacements) =
        compute_frontier_value_sums_and_partitioned_local_value_sum_displacements(
          handle,
          raft::device_span<edge_t const>(aggregate_local_frontier_local_degrees.data(),
                                          aggregate_local_frontier_local_degrees.size()),
          local_frontier_offsets,
          1);
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
      sample_nbr_index_with_replacement<edge_t, bias_t>(
        handle,
        raft::device_span<edge_t const>(frontier_degrees.data(), frontier_degrees.size()),
        std::nullopt,
        raft::device_span<edge_t>(nbr_indices.data(), nbr_indices.size()),
        rng_state,
        K);
      frontier_degrees.resize(0, handle.get_stream());
      frontier_degrees.shrink_to_fit(handle.get_stream());
    }
  } else {
    nbr_indices = compute_homogeneous_uniform_sampling_index_without_replacement(
      handle,
      raft::device_span<edge_t const>(frontier_degrees.data(), frontier_degrees.size()),
      rng_state,
      K);
  }
  frontier_degrees.resize(0, handle.get_stream());
  frontier_degrees.shrink_to_fit(handle.get_stream());

  // 3. shuffle neighbor indices

  rmm::device_uvector<edge_t> local_nbr_indices(0, handle.get_stream());
  std::optional<rmm::device_uvector<size_t>> key_indices{std::nullopt};
  std::vector<size_t> local_frontier_sample_offsets{};
  if (minor_comm_size > 1) {
    std::tie(local_nbr_indices, key_indices, local_frontier_sample_offsets) =
      shuffle_and_compute_local_nbr_values<edge_t>(
        handle,
        std::move(nbr_indices),
        raft::device_span<edge_t const>((*frontier_partitioned_local_degree_displacements).data(),
                                        (*frontier_partitioned_local_degree_displacements).size()),
        K,
        cugraph::invalid_edge_id_v<edge_t>);
  } else {
    local_nbr_indices             = std::move(nbr_indices);
    local_frontier_sample_offsets = {size_t{0}, local_nbr_indices.size()};
  }

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
      raft::host_span<size_t const>(local_frontier_sample_offsets.data(),
                                    local_frontier_sample_offsets.size()),
      local_frontier_offsets,
      K);
  }

  return std::make_tuple(
    std::move(local_nbr_indices), std::move(key_indices), std::move(local_frontier_sample_offsets));
}

template <typename GraphViewType, typename KeyIterator, typename EdgeTypeInputWrapper>
std::tuple<rmm::device_uvector<typename GraphViewType::edge_type>,
           std::optional<rmm::device_uvector<size_t>>,
           std::vector<size_t>>
heterogeneous_uniform_sample_and_compute_local_nbr_indices(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  KeyIterator aggregate_local_frontier_key_first,
  EdgeTypeInputWrapper edge_type_input,
  raft::host_span<size_t const> local_frontier_offsets,
  raft::random::RngState& rng_state,
  raft::host_span<size_t const> Ks,
  bool with_replacement)
{
  using vertex_t    = typename GraphViewType::vertex_type;
  using edge_t      = typename GraphViewType::edge_type;
  using key_t       = typename thrust::iterator_traits<KeyIterator>::value_type;
  using edge_type_t = typename EdgeTypeInputWrapper::value_type;
  using bias_t      = double;

  int minor_comm_rank{0};
  int minor_comm_size{1};
  if constexpr (GraphViewType::is_multi_gpu) {
    auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    minor_comm_rank  = minor_comm.get_rank();
    minor_comm_size  = minor_comm.get_size();
  }
  assert(minor_comm_size == graph_view.number_of_local_edge_partitions());

  auto num_edge_types = static_cast<edge_type_t>(Ks.size());

  auto edge_mask_view = graph_view.edge_mask_view();

  auto K_sum = std::accumulate(Ks.begin(), Ks.end(), size_t{0});

  rmm::device_uvector<size_t> d_K_offsets(Ks.size() + 1, handle.get_stream());
  {
    std::vector<size_t> h_K_offsets(d_K_offsets.size());
    h_K_offsets[0] = 0;
    std::inclusive_scan(Ks.begin(), Ks.end(), h_K_offsets.begin() + 1);
    raft::update_device(
      d_K_offsets.data(), h_K_offsets.data(), h_K_offsets.size(), handle.get_stream());
  }

  // 1. compute types for unique keys (to reduce memory footprint)

  auto [aggregate_local_frontier_unique_keys,
        aggregate_local_frontier_key_idx_to_unique_key_idx,
        local_frontier_unique_key_offsets] =
    compute_unique_keys(handle, aggregate_local_frontier_key_first, local_frontier_offsets);

  auto [aggregate_local_frontier_unique_key_edge_types,
        aggregate_local_frontier_unique_key_local_degree_offsets] =
    compute_aggregate_local_frontier_edge_types(
      handle,
      graph_view,
      get_dataframe_buffer_begin(aggregate_local_frontier_unique_keys),
      edge_type_input,
      raft::host_span<size_t const>(local_frontier_unique_key_offsets.data(),
                                    local_frontier_unique_key_offsets.size()));

  // 2. Segment-sort (index, type) pairs on types (1 segment per key)

  rmm::device_uvector<edge_t> aggregate_local_frontier_unique_key_org_indices(
    aggregate_local_frontier_unique_key_edge_types.size(), handle.get_stream());

  {
    // to limit memory footprint ((1 << 20) is a tuning parameter)
    auto approx_nbrs_to_sort_per_iteration =
      static_cast<size_t>(handle.get_device_properties().multiProcessorCount * (1 << 20));

    auto [h_key_offsets, h_nbr_offsets] = detail::compute_offset_aligned_element_chunks(
      handle,
      raft::device_span<size_t const>(
        aggregate_local_frontier_unique_key_local_degree_offsets.data(),
        aggregate_local_frontier_unique_key_local_degree_offsets.size()),
      aggregate_local_frontier_unique_key_edge_types.size(),
      approx_nbrs_to_sort_per_iteration);

    rmm::device_uvector<std::byte> d_tmp_storage(0, handle.get_stream());

    auto num_chunks = h_key_offsets.size() - 1;
    for (size_t i = 0; i < num_chunks; ++i) {
      size_t tmp_storage_bytes{0};

      rmm::device_uvector<edge_type_t> segment_sorted_types(h_nbr_offsets[i + 1] - h_nbr_offsets[i],
                                                            handle.get_stream());
      rmm::device_uvector<edge_t> nbr_indices(h_nbr_offsets[i + 1] - h_nbr_offsets[i],
                                              handle.get_stream());
      thrust::tabulate(
        handle.get_thrust_policy(),
        nbr_indices.begin(),
        nbr_indices.end(),
        [offsets = raft::device_span<size_t const>(
           aggregate_local_frontier_unique_key_local_degree_offsets.data() + h_key_offsets[i],
           (h_key_offsets[i + 1] - h_key_offsets[i]) + 1),
         start_offset = h_nbr_offsets[i]] __device__(size_t i) {
          auto idx = cuda::std::distance(
            offsets.begin() + 1,
            thrust::upper_bound(thrust::seq, offsets.begin() + 1, offsets.end(), start_offset + i));
          return static_cast<edge_t>((start_offset + i) - offsets[idx]);
        });
      raft::device_span<edge_t> segment_sorted_nbr_indices(
        aggregate_local_frontier_unique_key_org_indices.data() + h_nbr_offsets[i],
        h_nbr_offsets[i + 1] - h_nbr_offsets[i]);

      auto offset_first = thrust::make_transform_iterator(
        aggregate_local_frontier_unique_key_local_degree_offsets.data() + h_key_offsets[i],
        detail::shift_left_t<size_t>{h_nbr_offsets[i]});
      cub::DeviceSegmentedSort::SortPairs(
        static_cast<void*>(nullptr),
        tmp_storage_bytes,
        aggregate_local_frontier_unique_key_edge_types.begin() + h_nbr_offsets[i],
        segment_sorted_types.begin(),
        nbr_indices.begin(),
        segment_sorted_nbr_indices.begin(),
        h_nbr_offsets[i + 1] - h_nbr_offsets[i],
        h_key_offsets[i + 1] - h_key_offsets[i],
        offset_first,
        offset_first + 1,
        handle.get_stream());
      if (tmp_storage_bytes > d_tmp_storage.size()) {
        d_tmp_storage = rmm::device_uvector<std::byte>(tmp_storage_bytes, handle.get_stream());
      }
      cub::DeviceSegmentedSort::SortPairs(
        d_tmp_storage.data(),
        tmp_storage_bytes,
        aggregate_local_frontier_unique_key_edge_types.begin() + h_nbr_offsets[i],
        segment_sorted_types.begin(),
        nbr_indices.begin(),
        segment_sorted_nbr_indices.begin(),
        h_nbr_offsets[i + 1] - h_nbr_offsets[i],
        h_key_offsets[i + 1] - h_key_offsets[i],
        offset_first,
        offset_first + 1,
        handle.get_stream());
      thrust::copy(handle.get_thrust_policy(),
                   segment_sorted_types.begin(),
                   segment_sorted_types.end(),
                   aggregate_local_frontier_unique_key_edge_types.begin() + h_nbr_offsets[i]);
    }
  }

  // 3. sample neighbor indices and shuffle neighbor indices

  rmm::device_uvector<edge_t> local_nbr_indices(0, handle.get_stream());
  std::optional<rmm::device_uvector<size_t>> key_indices{std::nullopt};
  std::vector<size_t> local_frontier_sample_offsets{};

  {
    rmm::device_uvector<edge_t> frontier_per_type_degrees(0, handle.get_stream());
    std::optional<rmm::device_uvector<edge_t>>
      frontier_partitioned_per_type_local_degree_displacements{std::nullopt};
    {
      auto aggregate_local_frontier_per_type_local_degrees =
        compute_aggregate_local_frontier_per_type_local_degrees<edge_t, edge_type_t>(
          handle,
          raft::device_span<size_t const>(
            aggregate_local_frontier_key_idx_to_unique_key_idx.data(),
            aggregate_local_frontier_key_idx_to_unique_key_idx.size()),
          local_frontier_offsets,
          raft::device_span<edge_type_t const>(
            aggregate_local_frontier_unique_key_edge_types.data(),
            aggregate_local_frontier_unique_key_edge_types.size()),
          raft::device_span<size_t const>(
            aggregate_local_frontier_unique_key_local_degree_offsets.data(),
            aggregate_local_frontier_unique_key_local_degree_offsets.size()),
          raft::host_span<size_t const>(local_frontier_unique_key_offsets.data(),
                                        local_frontier_unique_key_offsets.size()),
          num_edge_types);

      if (minor_comm_size > 1) {
        std::tie(frontier_per_type_degrees,
                 frontier_partitioned_per_type_local_degree_displacements) =
          compute_frontier_value_sums_and_partitioned_local_value_sum_displacements(
            handle,
            raft::device_span<edge_t const>(aggregate_local_frontier_per_type_local_degrees.data(),
                                            aggregate_local_frontier_per_type_local_degrees.size()),
            local_frontier_offsets,
            num_edge_types);
      } else {
        frontier_per_type_degrees = std::move(aggregate_local_frontier_per_type_local_degrees);
      }
    }

    rmm::device_uvector<edge_t> per_type_nbr_indices(0, handle.get_stream());

    if (with_replacement) {
      per_type_nbr_indices.resize((frontier_per_type_degrees.size() / num_edge_types) * K_sum,
                                  handle.get_stream());
      sample_nbr_index_with_replacement<edge_t, edge_type_t, bias_t>(
        handle,
        raft::device_span<edge_t const>(frontier_per_type_degrees.data(),
                                        frontier_per_type_degrees.size()),
        std::nullopt,
        raft::device_span<edge_t>(per_type_nbr_indices.data(), per_type_nbr_indices.size()),
        rng_state,
        raft::device_span<size_t const>(d_K_offsets.data(), d_K_offsets.size()),
        K_sum);
    } else {
      per_type_nbr_indices =
        compute_heterogeneous_uniform_sampling_index_without_replacement<edge_t, edge_type_t>(
          handle,
          raft::device_span<edge_t const>(frontier_per_type_degrees.data(),
                                          frontier_per_type_degrees.size()),
          rng_state,
          raft::device_span<size_t const>(d_K_offsets.data(), d_K_offsets.size()),
          K_sum);
    }

    rmm::device_uvector<edge_t> per_type_local_nbr_indices(0, handle.get_stream());
    std::optional<rmm::device_uvector<edge_type_t>> edge_types{std::nullopt};
    if (minor_comm_size > 1) {
      std::tie(per_type_local_nbr_indices, edge_types, key_indices, local_frontier_sample_offsets) =
        shuffle_and_compute_per_type_local_nbr_values<edge_type_t, edge_t>(
          handle,
          std::move(per_type_nbr_indices),
          raft::device_span<edge_t const>(
            (*frontier_partitioned_per_type_local_degree_displacements).data(),
            (*frontier_partitioned_per_type_local_degree_displacements).size()),
          raft::device_span<size_t const>(d_K_offsets.data(), d_K_offsets.size()),
          K_sum,
          cugraph::invalid_edge_id_v<edge_t>);
    } else {
      per_type_local_nbr_indices    = std::move(per_type_nbr_indices);
      local_frontier_sample_offsets = {size_t{0}, per_type_local_nbr_indices.size()};
    }

    rmm::device_uvector<size_t> aggregate_local_frontier_unique_key_per_type_local_degree_offsets(
      local_frontier_unique_key_offsets.back() * num_edge_types + 1, handle.get_stream());
    {
      rmm::device_uvector<size_t> aggregate_local_frontier_unique_key_indices(
        local_frontier_unique_key_offsets.back(), handle.get_stream());
      for (size_t i = 0; i < local_frontier_unique_key_offsets.size() - 1; ++i) {
        thrust::sequence(handle.get_thrust_policy(),
                         aggregate_local_frontier_unique_key_indices.begin() +
                           local_frontier_unique_key_offsets[i],
                         aggregate_local_frontier_unique_key_indices.begin() +
                           local_frontier_unique_key_offsets[i + 1],
                         size_t{0});
      }

      auto aggregate_local_frontier_unique_key_per_type_local_degrees =
        compute_aggregate_local_frontier_per_type_local_degrees<edge_t, edge_type_t>(
          handle,
          raft::device_span<size_t const>(aggregate_local_frontier_unique_key_indices.data(),
                                          aggregate_local_frontier_unique_key_indices.size()),
          raft::host_span<size_t const>(local_frontier_unique_key_offsets.data(),
                                        local_frontier_unique_key_offsets.size()),
          raft::device_span<edge_type_t const>(
            aggregate_local_frontier_unique_key_edge_types.data(),
            aggregate_local_frontier_unique_key_edge_types.size()),
          raft::device_span<size_t const>(
            aggregate_local_frontier_unique_key_local_degree_offsets.data(),
            aggregate_local_frontier_unique_key_local_degree_offsets.size()),
          raft::host_span<size_t const>(local_frontier_unique_key_offsets.data(),
                                        local_frontier_unique_key_offsets.size()),
          num_edge_types);

      aggregate_local_frontier_unique_key_per_type_local_degree_offsets.set_element_to_zero_async(
        0, handle.get_stream());
      thrust::inclusive_scan(
        handle.get_thrust_policy(),
        aggregate_local_frontier_unique_key_per_type_local_degrees.begin(),
        aggregate_local_frontier_unique_key_per_type_local_degrees.end(),
        aggregate_local_frontier_unique_key_per_type_local_degree_offsets.begin() + 1);
    }

    assert(edge_types.has_value() == key_indices.has_value());
    local_nbr_indices =
      compute_local_nbr_indices_from_per_type_local_nbr_indices<edge_t, edge_type_t>(
        handle,
        raft::device_span<size_t const>(aggregate_local_frontier_key_idx_to_unique_key_idx.data(),
                                        aggregate_local_frontier_key_idx_to_unique_key_idx.size()),
        local_frontier_offsets,
        raft::device_span<size_t const>(
          aggregate_local_frontier_unique_key_per_type_local_degree_offsets.data(),
          aggregate_local_frontier_unique_key_per_type_local_degree_offsets.size()),
        raft::host_span<size_t const>(local_frontier_unique_key_offsets.data(),
                                      local_frontier_unique_key_offsets.size()),
        edge_types
          ? std::make_optional(std::make_tuple(
              raft::device_span<edge_type_t const>((*edge_types).data(), (*edge_types).size()),
              raft::device_span<size_t const>((*key_indices).data(), (*key_indices).size())))
          : std::nullopt,
        std::move(per_type_local_nbr_indices),
        raft::host_span<size_t const>(local_frontier_sample_offsets.data(),
                                      local_frontier_sample_offsets.size()),
        raft::device_span<size_t const>(d_K_offsets.data(), d_K_offsets.size()),
        K_sum);
  }

  // 4. Re-map local neighbor indices

  local_nbr_indices = remap_local_nbr_indices(
    handle,
    raft::device_span<size_t const>(aggregate_local_frontier_key_idx_to_unique_key_idx.data(),
                                    aggregate_local_frontier_key_idx_to_unique_key_idx.size()),
    local_frontier_offsets,
    raft::device_span<edge_t const>(aggregate_local_frontier_unique_key_org_indices.data(),
                                    aggregate_local_frontier_unique_key_org_indices.size()),
    raft::device_span<size_t const>(
      aggregate_local_frontier_unique_key_local_degree_offsets.data(),
      aggregate_local_frontier_unique_key_local_degree_offsets.size()),
    raft::host_span<size_t const>(local_frontier_unique_key_offsets.data(),
                                  local_frontier_unique_key_offsets.size()),
    std::move(local_nbr_indices),
    key_indices ? std::make_optional<raft::device_span<size_t const>>((*key_indices).data(),
                                                                      (*key_indices).size())
                : std::nullopt,
    raft::host_span<size_t const>(local_frontier_sample_offsets.data(),
                                  local_frontier_sample_offsets.size()),
    K_sum);

  // 5. convert neighbor indices in the neighbor list considering edge mask to neighbor indices in
  // the neighbor list ignoring edge mask

  if (edge_mask_view) {
    local_nbr_indices = convert_to_unmasked_local_nbr_idx(
      handle,
      graph_view,
      thrust_tuple_get_or_identity<KeyIterator, 0>(aggregate_local_frontier_key_first),
      std::move(local_nbr_indices),
      key_indices ? std::make_optional<raft::device_span<size_t const>>((*key_indices).data(),
                                                                        (*key_indices).size())
                  : std::nullopt,
      raft::host_span<size_t const>(local_frontier_sample_offsets.data(),
                                    local_frontier_sample_offsets.size()),
      local_frontier_offsets,
      K_sum);
  }

  return std::make_tuple(
    std::move(local_nbr_indices), std::move(key_indices), std::move(local_frontier_sample_offsets));
}

template <typename GraphViewType,
          typename KeyIterator,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename BiasEdgeOp>
std::tuple<rmm::device_uvector<typename GraphViewType::edge_type>,
           std::optional<rmm::device_uvector<size_t>>,
           std::vector<size_t>>
homogeneous_biased_sample_and_compute_local_nbr_indices(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  KeyIterator aggregate_local_frontier_key_first,
  EdgeSrcValueInputWrapper edge_src_value_input,
  EdgeDstValueInputWrapper edge_dst_value_input,
  EdgeValueInputWrapper edge_value_input,
  BiasEdgeOp bias_e_op,
  raft::host_span<size_t const> local_frontier_offsets,
  raft::random::RngState& rng_state,
  size_t K,
  bool with_replacement,
  bool do_expensive_check /* check bias_e_op return values */)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using key_t    = typename thrust::iterator_traits<KeyIterator>::value_type;

  using bias_t      = typename edge_op_result_type<key_t,
                                              vertex_t,
                                              typename EdgeSrcValueInputWrapper::value_type,
                                              typename EdgeDstValueInputWrapper::value_type,
                                              typename EdgeValueInputWrapper::value_type,
                                              BiasEdgeOp>::type;
  using edge_type_t = int32_t;  // dummy

  int minor_comm_rank{0};
  int minor_comm_size{1};
  if constexpr (GraphViewType::is_multi_gpu) {
    auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    minor_comm_rank  = minor_comm.get_rank();
    minor_comm_size  = minor_comm.get_size();
  }
  assert(minor_comm_size == graph_view.number_of_local_edge_partitions());

  auto edge_mask_view = graph_view.edge_mask_view();

  // 1. compute biases for unique keys (to reduce memory footprint)

  auto [aggregate_local_frontier_unique_keys,
        aggregate_local_frontier_key_idx_to_unique_key_idx,
        local_frontier_unique_key_offsets] =
    compute_unique_keys(handle, aggregate_local_frontier_key_first, local_frontier_offsets);

  auto [aggregate_local_frontier_unique_key_biases,
        aggregate_local_frontier_unique_key_nz_bias_indices,
        aggregate_local_frontier_unique_key_local_degree_offsets] =
    compute_aggregate_local_frontier_biases(
      handle,
      graph_view,
      get_dataframe_buffer_begin(aggregate_local_frontier_unique_keys),
      edge_src_value_input,
      edge_dst_value_input,
      edge_value_input,
      bias_e_op,
      raft::host_span<size_t const>(local_frontier_unique_key_offsets.data(),
                                    local_frontier_unique_key_offsets.size()),
      do_expensive_check);

  // 2. sample neighbor indices and shuffle neighbor indices

  rmm::device_uvector<edge_t> local_nbr_indices(0, handle.get_stream());
  std::optional<rmm::device_uvector<size_t>> key_indices{std::nullopt};
  std::vector<size_t> local_frontier_sample_offsets{};
  if (with_replacement) {
    std::tie(local_nbr_indices, key_indices, local_frontier_sample_offsets) =
      biased_sample_with_replacement<edge_t, edge_type_t, bias_t, GraphViewType::is_multi_gpu>(
        handle,
        raft::device_span<size_t const>(aggregate_local_frontier_key_idx_to_unique_key_idx.data(),
                                        aggregate_local_frontier_key_idx_to_unique_key_idx.size()),
        local_frontier_offsets,
        raft::device_span<bias_t const>(aggregate_local_frontier_unique_key_biases.data(),
                                        aggregate_local_frontier_unique_key_biases.size()),
        raft::device_span<size_t const>(
          aggregate_local_frontier_unique_key_local_degree_offsets.data(),
          aggregate_local_frontier_unique_key_local_degree_offsets.size()),
        raft::host_span<size_t const>(local_frontier_unique_key_offsets.data(),
                                      local_frontier_unique_key_offsets.size()),
        rng_state,
        raft::host_span<size_t const>(&K, size_t{1}));
  } else {
    std::tie(local_nbr_indices, key_indices, local_frontier_sample_offsets) =
      homogeneous_biased_sample_without_replacement<edge_t, bias_t, GraphViewType::is_multi_gpu>(
        handle,
        raft::device_span<size_t const>(aggregate_local_frontier_key_idx_to_unique_key_idx.data(),
                                        aggregate_local_frontier_key_idx_to_unique_key_idx.size()),
        local_frontier_offsets,
        raft::device_span<bias_t const>(aggregate_local_frontier_unique_key_biases.data(),
                                        aggregate_local_frontier_unique_key_biases.size()),
        raft::device_span<size_t const>(
          aggregate_local_frontier_unique_key_local_degree_offsets.data(),
          aggregate_local_frontier_unique_key_local_degree_offsets.size()),
        raft::host_span<size_t const>(local_frontier_unique_key_offsets.data(),
                                      local_frontier_unique_key_offsets.size()),
        rng_state,
        K);
  }

  // 3. remap non-zero bias local neighbor indices to local neighbor indices

  if (key_indices) {
    auto pair_first = thrust::make_zip_iterator(local_nbr_indices.begin(), (*key_indices).begin());
    for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
      thrust::transform(
        handle.get_thrust_policy(),
        pair_first + local_frontier_sample_offsets[i],
        pair_first + local_frontier_sample_offsets[i + 1],
        local_nbr_indices.begin() + local_frontier_sample_offsets[i],
        cuda::proclaim_return_type<edge_t>(
          [key_idx_to_unique_key_idx = raft::device_span<size_t const>(
             aggregate_local_frontier_key_idx_to_unique_key_idx.data() + local_frontier_offsets[i],
             local_frontier_offsets[i + 1] - local_frontier_offsets[i]),
           unique_key_local_degree_offsets = raft::device_span<size_t const>(
             aggregate_local_frontier_unique_key_local_degree_offsets.data() +
               local_frontier_unique_key_offsets[i],
             (local_frontier_unique_key_offsets[i + 1] - local_frontier_unique_key_offsets[i]) + 1),
           unique_key_nz_bias_indices = raft::device_span<edge_t const>(
             aggregate_local_frontier_unique_key_nz_bias_indices.data(),
             aggregate_local_frontier_unique_key_nz_bias_indices.size())] __device__(auto pair) {
            auto nz_bias_idx    = thrust::get<0>(pair);
            auto key_idx        = thrust::get<1>(pair);
            auto unique_key_idx = key_idx_to_unique_key_idx[key_idx];
            return unique_key_nz_bias_indices[unique_key_local_degree_offsets[unique_key_idx] +
                                              nz_bias_idx];
          }));
    }
  } else {
    auto pair_first = thrust::make_zip_iterator(local_nbr_indices.begin(),
                                                thrust::make_counting_iterator(size_t{0}));
    for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
      thrust::transform_if(
        handle.get_thrust_policy(),
        pair_first + local_frontier_sample_offsets[i],
        pair_first + local_frontier_sample_offsets[i + 1],
        local_nbr_indices.begin() + local_frontier_sample_offsets[i],
        local_nbr_indices.begin() + local_frontier_sample_offsets[i],
        cuda::proclaim_return_type<edge_t>(
          [key_idx_to_unique_key_idx = raft::device_span<size_t const>(
             aggregate_local_frontier_key_idx_to_unique_key_idx.data() + local_frontier_offsets[i],
             local_frontier_offsets[i + 1] - local_frontier_offsets[i]),
           unique_key_local_degree_offsets = raft::device_span<size_t const>(
             aggregate_local_frontier_unique_key_local_degree_offsets.data() +
               local_frontier_unique_key_offsets[i],
             (local_frontier_unique_key_offsets[i + 1] - local_frontier_unique_key_offsets[i]) + 1),
           unique_key_nz_bias_indices = raft::device_span<edge_t const>(
             aggregate_local_frontier_unique_key_nz_bias_indices.data(),
             aggregate_local_frontier_unique_key_nz_bias_indices.size()),
           K] __device__(auto pair) {
            auto nz_bias_idx    = thrust::get<0>(pair);
            auto key_idx        = thrust::get<1>(pair) / K;
            auto unique_key_idx = key_idx_to_unique_key_idx[key_idx];
            return unique_key_nz_bias_indices[unique_key_local_degree_offsets[unique_key_idx] +
                                              nz_bias_idx];
          }),
        is_not_equal_t<edge_t>{cugraph::invalid_edge_id_v<edge_t>});
    }
  }

  // 4. convert neighbor indices in the neighbor list considering edge mask to neighbor indices in
  // the neighbor list ignoring edge mask

  if (edge_mask_view) {
    local_nbr_indices = convert_to_unmasked_local_nbr_idx(
      handle,
      graph_view,
      thrust_tuple_get_or_identity<KeyIterator, 0>(aggregate_local_frontier_key_first),
      std::move(local_nbr_indices),
      key_indices ? std::make_optional<raft::device_span<size_t const>>((*key_indices).data(),
                                                                        (*key_indices).size())
                  : std::nullopt,
      raft::host_span<size_t const>(local_frontier_sample_offsets.data(),
                                    local_frontier_sample_offsets.size()),
      local_frontier_offsets,
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
          typename BiasEdgeOp,
          typename EdgeTypeInputWrapper>
std::tuple<rmm::device_uvector<typename GraphViewType::edge_type>,
           std::optional<rmm::device_uvector<size_t>>,
           std::vector<size_t>>
heterogeneous_biased_sample_and_compute_local_nbr_indices(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  KeyIterator aggregate_local_frontier_key_first,
  EdgeSrcValueInputWrapper edge_src_value_input,
  EdgeDstValueInputWrapper edge_dst_value_input,
  EdgeValueInputWrapper edge_value_input,
  BiasEdgeOp bias_e_op,
  EdgeTypeInputWrapper edge_type_input,
  raft::host_span<size_t const> local_frontier_offsets,
  raft::random::RngState& rng_state,
  raft::host_span<size_t const> Ks,
  bool with_replacement,
  bool do_expensive_check /* check bias_e_op return values */)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using key_t    = typename thrust::iterator_traits<KeyIterator>::value_type;

  using bias_t      = typename edge_op_result_type<key_t,
                                              vertex_t,
                                              typename EdgeSrcValueInputWrapper::value_type,
                                              typename EdgeDstValueInputWrapper::value_type,
                                              typename EdgeValueInputWrapper::value_type,
                                              BiasEdgeOp>::type;
  using edge_type_t = typename EdgeTypeInputWrapper::value_type;

  int minor_comm_rank{0};
  int minor_comm_size{1};
  if constexpr (GraphViewType::is_multi_gpu) {
    auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    minor_comm_rank  = minor_comm.get_rank();
    minor_comm_size  = minor_comm.get_size();
  }
  assert(minor_comm_size == graph_view.number_of_local_edge_partitions());

  auto num_edge_types = static_cast<edge_type_t>(Ks.size());

  auto edge_mask_view = graph_view.edge_mask_view();

  // 1. compute (bias, type) pairs for unique keys (to reduce memory footprint)

  auto [aggregate_local_frontier_unique_keys,
        aggregate_local_frontier_key_idx_to_unique_key_idx,
        local_frontier_unique_key_offsets] =
    compute_unique_keys(handle, aggregate_local_frontier_key_first, local_frontier_offsets);

  auto [aggregate_local_frontier_unique_key_biases,
        aggregate_local_frontier_unique_key_edge_types,
        aggregate_local_frontier_unique_key_nz_bias_indices,
        aggregate_local_frontier_unique_key_local_degree_offsets] =
    compute_aggregate_local_frontier_bias_type_pairs(
      handle,
      graph_view,
      get_dataframe_buffer_begin(aggregate_local_frontier_unique_keys),
      edge_src_value_input,
      edge_dst_value_input,
      edge_value_input,
      bias_e_op,
      edge_type_input,
      raft::host_span<size_t const>(local_frontier_unique_key_offsets.data(),
                                    local_frontier_unique_key_offsets.size()),
      do_expensive_check);

  // 2. Segmented-sort (index, bias, type) triplets based on types (1 segment per key)

  {
    // to limit memory footprint ((1 << 20) is a tuning parameter)
    auto approx_nbrs_to_sort_per_iteration =
      static_cast<size_t>(handle.get_device_properties().multiProcessorCount * (1 << 20));

    auto [h_key_offsets, h_nbr_offsets] = detail::compute_offset_aligned_element_chunks(
      handle,
      raft::device_span<size_t const>(
        aggregate_local_frontier_unique_key_local_degree_offsets.data(),
        aggregate_local_frontier_unique_key_local_degree_offsets.size()),
      aggregate_local_frontier_unique_key_biases.size(),
      approx_nbrs_to_sort_per_iteration);

    rmm::device_uvector<std::byte> d_tmp_storage(0, handle.get_stream());

    auto num_chunks = h_key_offsets.size() - 1;
    for (size_t i = 0; i < num_chunks; ++i) {
      size_t tmp_storage_bytes{0};

      rmm::device_uvector<edge_type_t> segment_sorted_types(h_nbr_offsets[i + 1] - h_nbr_offsets[i],
                                                            handle.get_stream());
      rmm::device_uvector<size_t> sequences(h_nbr_offsets[i + 1] - h_nbr_offsets[i],
                                            handle.get_stream());
      thrust::sequence(handle.get_thrust_policy(), sequences.begin(), sequences.end(), size_t{0});
      rmm::device_uvector<size_t> segment_sorted_sequences(h_nbr_offsets[i + 1] - h_nbr_offsets[i],
                                                           handle.get_stream());

      auto offset_first = thrust::make_transform_iterator(
        aggregate_local_frontier_unique_key_local_degree_offsets.data() + h_key_offsets[i],
        detail::shift_left_t<size_t>{h_nbr_offsets[i]});
      cub::DeviceSegmentedSort::SortPairs(
        static_cast<void*>(nullptr),
        tmp_storage_bytes,
        aggregate_local_frontier_unique_key_edge_types.begin() + h_nbr_offsets[i],
        segment_sorted_types.begin(),
        sequences.begin(),
        segment_sorted_sequences.begin(),
        h_nbr_offsets[i + 1] - h_nbr_offsets[i],
        h_key_offsets[i + 1] - h_key_offsets[i],
        offset_first,
        offset_first + 1,
        handle.get_stream());
      if (tmp_storage_bytes > d_tmp_storage.size()) {
        d_tmp_storage = rmm::device_uvector<std::byte>(tmp_storage_bytes, handle.get_stream());
      }
      cub::DeviceSegmentedSort::SortPairs(
        d_tmp_storage.data(),
        tmp_storage_bytes,
        aggregate_local_frontier_unique_key_edge_types.begin() + h_nbr_offsets[i],
        segment_sorted_types.begin(),
        sequences.begin(),
        segment_sorted_sequences.begin(),
        h_nbr_offsets[i + 1] - h_nbr_offsets[i],
        h_key_offsets[i + 1] - h_key_offsets[i],
        offset_first,
        offset_first + 1,
        handle.get_stream());

      thrust::copy(handle.get_thrust_policy(),
                   segment_sorted_types.begin(),
                   segment_sorted_types.end(),
                   aggregate_local_frontier_unique_key_edge_types.begin() + h_nbr_offsets[i]);

      rmm::device_uvector<bias_t> segment_sorted_biases(h_nbr_offsets[i + 1] - h_nbr_offsets[i],
                                                        handle.get_stream());
      rmm::device_uvector<edge_t> segment_sorted_nz_bias_indices(
        h_nbr_offsets[i + 1] - h_nbr_offsets[i], handle.get_stream());
      thrust::gather(
        handle.get_thrust_policy(),
        segment_sorted_sequences.begin(),
        segment_sorted_sequences.end(),
        thrust::make_zip_iterator(aggregate_local_frontier_unique_key_biases.begin(),
                                  aggregate_local_frontier_unique_key_nz_bias_indices.begin()) +
          h_nbr_offsets[i],
        thrust::make_zip_iterator(segment_sorted_biases.begin(),
                                  segment_sorted_nz_bias_indices.begin()));
      auto segment_sorted_pair_first = thrust::make_zip_iterator(
        segment_sorted_biases.begin(), segment_sorted_nz_bias_indices.begin());
      thrust::copy(
        handle.get_thrust_policy(),
        segment_sorted_pair_first,
        segment_sorted_pair_first + segment_sorted_biases.size(),
        thrust::make_zip_iterator(aggregate_local_frontier_unique_key_biases.begin(),
                                  aggregate_local_frontier_unique_key_nz_bias_indices.begin()) +
          h_nbr_offsets[i]);
    }
  }

  // 3. sample neighbor indices and shuffle neighbor indices

  rmm::device_uvector<edge_t> local_nbr_indices(0, handle.get_stream());
  std::optional<rmm::device_uvector<size_t>> key_indices{std::nullopt};
  std::vector<size_t> local_frontier_sample_offsets{};
  {
    rmm::device_uvector<size_t> aggregate_local_frontier_unique_key_per_type_local_degree_offsets(
      0, handle.get_stream());
    {
      rmm::device_uvector<size_t> aggregate_local_frontier_unique_key_indices(
        local_frontier_unique_key_offsets.back(), handle.get_stream());
      for (size_t i = 0; i < local_frontier_unique_key_offsets.size() - 1; ++i) {
        thrust::sequence(handle.get_thrust_policy(),
                         aggregate_local_frontier_unique_key_indices.begin() +
                           local_frontier_unique_key_offsets[i],
                         aggregate_local_frontier_unique_key_indices.begin() +
                           local_frontier_unique_key_offsets[i + 1],
                         size_t{0});
      }

      auto aggregate_local_frontier_unique_key_per_type_local_degrees =
        compute_aggregate_local_frontier_per_type_local_degrees<edge_t, edge_type_t>(
          handle,
          raft::device_span<size_t const>(aggregate_local_frontier_unique_key_indices.data(),
                                          aggregate_local_frontier_unique_key_indices.size()),
          raft::host_span<size_t const>(local_frontier_unique_key_offsets.data(),
                                        local_frontier_unique_key_offsets.size()),
          raft::device_span<edge_type_t const>(
            aggregate_local_frontier_unique_key_edge_types.data(),
            aggregate_local_frontier_unique_key_edge_types.size()),
          raft::device_span<size_t const>(
            aggregate_local_frontier_unique_key_local_degree_offsets.data(),
            aggregate_local_frontier_unique_key_local_degree_offsets.size()),
          raft::host_span<size_t const>(local_frontier_unique_key_offsets.data(),
                                        local_frontier_unique_key_offsets.size()),
          num_edge_types);

      aggregate_local_frontier_unique_key_per_type_local_degree_offsets.resize(
        aggregate_local_frontier_unique_key_per_type_local_degrees.size() + 1, handle.get_stream());
      aggregate_local_frontier_unique_key_per_type_local_degree_offsets.set_element_to_zero_async(
        0, handle.get_stream());
      thrust::inclusive_scan(
        handle.get_thrust_policy(),
        aggregate_local_frontier_unique_key_per_type_local_degrees.begin(),
        aggregate_local_frontier_unique_key_per_type_local_degrees.end(),
        aggregate_local_frontier_unique_key_per_type_local_degree_offsets.begin() + 1);
      aggregate_local_frontier_unique_key_edge_types.resize(0, handle.get_stream());
      aggregate_local_frontier_unique_key_edge_types.shrink_to_fit(handle.get_stream());
    }

    if (with_replacement) {
      std::tie(local_nbr_indices, key_indices, local_frontier_sample_offsets) =
        biased_sample_with_replacement<edge_t, edge_type_t, bias_t, GraphViewType::is_multi_gpu>(
          handle,
          raft::device_span<size_t const>(
            aggregate_local_frontier_key_idx_to_unique_key_idx.data(),
            aggregate_local_frontier_key_idx_to_unique_key_idx.size()),
          local_frontier_offsets,
          raft::device_span<bias_t const>(aggregate_local_frontier_unique_key_biases.data(),
                                          aggregate_local_frontier_unique_key_biases.size()),
          raft::device_span<size_t const>(
            aggregate_local_frontier_unique_key_per_type_local_degree_offsets.data(),
            aggregate_local_frontier_unique_key_per_type_local_degree_offsets.size()),
          raft::host_span<size_t const>(local_frontier_unique_key_offsets.data(),
                                        local_frontier_unique_key_offsets.size()),
          rng_state,
          Ks);
    } else {
      std::tie(local_nbr_indices, key_indices, local_frontier_sample_offsets) =
        heterogeneous_biased_sample_without_replacement<edge_t,
                                                        edge_type_t,
                                                        bias_t,
                                                        GraphViewType::is_multi_gpu>(
          handle,
          raft::device_span<size_t const>(
            aggregate_local_frontier_key_idx_to_unique_key_idx.data(),
            aggregate_local_frontier_key_idx_to_unique_key_idx.size()),
          local_frontier_offsets,
          raft::device_span<bias_t const>(aggregate_local_frontier_unique_key_biases.data(),
                                          aggregate_local_frontier_unique_key_biases.size()),
          raft::device_span<size_t const>(
            aggregate_local_frontier_unique_key_per_type_local_degree_offsets.data(),
            aggregate_local_frontier_unique_key_per_type_local_degree_offsets.size()),
          raft::host_span<size_t const>(local_frontier_unique_key_offsets.data(),
                                        local_frontier_unique_key_offsets.size()),
          rng_state,
          Ks);
    }
  }

  // 4. Re-map local neighbor indices

  auto K_sum = std::accumulate(Ks.begin(), Ks.end(), size_t{0});

  local_nbr_indices = remap_local_nbr_indices(
    handle,
    raft::device_span<size_t const>(aggregate_local_frontier_key_idx_to_unique_key_idx.data(),
                                    aggregate_local_frontier_key_idx_to_unique_key_idx.size()),
    local_frontier_offsets,
    raft::device_span<edge_t const>(aggregate_local_frontier_unique_key_nz_bias_indices.data(),
                                    aggregate_local_frontier_unique_key_nz_bias_indices.size()),
    raft::device_span<size_t const>(
      aggregate_local_frontier_unique_key_local_degree_offsets.data(),
      aggregate_local_frontier_unique_key_local_degree_offsets.size()),
    raft::host_span<size_t const>(local_frontier_unique_key_offsets.data(),
                                  local_frontier_unique_key_offsets.size()),
    std::move(local_nbr_indices),
    key_indices ? std::make_optional<raft::device_span<size_t const>>((*key_indices).data(),
                                                                      (*key_indices).size())
                : std::nullopt,
    raft::host_span<size_t const>(local_frontier_sample_offsets.data(),
                                  local_frontier_sample_offsets.size()),
    K_sum);

  // 5. convert neighbor indices in the neighbor list considering edge mask to neighbor indices in
  // the neighbor list ignoring edge mask

  if (edge_mask_view) {
    local_nbr_indices = convert_to_unmasked_local_nbr_idx(
      handle,
      graph_view,
      thrust_tuple_get_or_identity<KeyIterator, 0>(aggregate_local_frontier_key_first),
      std::move(local_nbr_indices),
      key_indices ? std::make_optional<raft::device_span<size_t const>>((*key_indices).data(),
                                                                        (*key_indices).size())
                  : std::nullopt,
      raft::host_span<size_t const>(local_frontier_sample_offsets.data(),
                                    local_frontier_sample_offsets.size()),
      local_frontier_offsets,
      K_sum);
  }

  return std::make_tuple(
    std::move(local_nbr_indices), std::move(key_indices), std::move(local_frontier_sample_offsets));
}

}  // namespace detail

}  // namespace cugraph
