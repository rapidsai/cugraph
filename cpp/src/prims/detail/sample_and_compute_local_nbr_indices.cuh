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

#include "prims/property_op_utils.cuh"

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

int32_t constexpr per_v_random_select_transform_outgoing_e_block_size = 256;

size_t constexpr compute_valid_local_nbr_count_inclusive_sum_local_degree_threshold =
  packed_bools_per_word() *
  size_t{4} /* tuning parameter */;  // minimum local degree to compute inclusive sums of valid
                                     // local neighbors per word to accelerate finding n'th local
                                     // neighbor vertex
size_t constexpr compute_valid_local_nbr_count_inclusive_sum_mid_local_degree_threshold =
  packed_bools_per_word() * static_cast<size_t>(raft::warp_size()) *
  size_t{4} /* tuning parameter */;  // minimum local degree to use a CUDA warp
size_t constexpr compute_valid_local_nbr_count_inclusive_sum_high_local_degree_threshold =
  packed_bools_per_word() *
  static_cast<size_t>(per_v_random_select_transform_outgoing_e_block_size) *
  size_t{4} /* tuning parameter */;  // minimum local degree to use a CUDA block

template <typename GraphViewType,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename key_t>
struct constant_e_bias_op_t {
  __device__ float operator()(key_t,
                              typename GraphViewType::vertex_type,
                              typename EdgeSrcValueInputWrapper::value_type,
                              typename EdgeDstValueInputWrapper::value_type,
                              typename EdgeValueInputWrapper::value_type) const
  {
    return 1.0;
  }
};

template <typename edge_t>
struct compute_local_degree_displacements_and_global_degree_t {
  raft::device_span<edge_t const> gathered_local_degrees{};
  raft::device_span<edge_t>
    partitioned_local_degree_displacements{};  // one partition per gpu in the same minor_comm
  raft::device_span<edge_t> global_degrees{};
  int minor_comm_size{};

  __device__ void operator()(size_t i) const
  {
    constexpr int buffer_size = 8;  // tuning parameter
    edge_t displacements[buffer_size];
    edge_t sum{0};
    for (int round = 0; round < (minor_comm_size + buffer_size - 1) / buffer_size; ++round) {
      auto loop_count = std::min(buffer_size, minor_comm_size - round * buffer_size);
      for (int j = 0; j < loop_count; ++j) {
        displacements[j] = sum;
        sum += gathered_local_degrees[i + (round * buffer_size + j) * global_degrees.size()];
      }
      thrust::copy(
        thrust::seq,
        displacements,
        displacements + loop_count,
        partitioned_local_degree_displacements.begin() + i * minor_comm_size + round * buffer_size);
    }
    global_degrees[i] = sum;
  }
};

// convert a (neighbor index, key index) pair  to a (minor_comm_rank, intra-partition offset,
// neighbor index, key index) quadruplet, minor_comm_rank is set to -1 if an neighbor index is
// invalid
template <typename edge_t>
struct convert_pair_to_quadruplet_t {
  raft::device_span<edge_t const>
    partitioned_local_degree_displacements{};  // one partition per gpu in the same minor_comm
  raft::device_span<size_t> tx_counts{};
  size_t stride{};
  int minor_comm_size{};
  edge_t invalid_idx{};

  __device__ thrust::tuple<int, size_t, edge_t, size_t> operator()(
    thrust::tuple<edge_t, size_t> index_pair) const
  {
    auto nbr_idx       = thrust::get<0>(index_pair);
    auto key_idx       = thrust::get<1>(index_pair);
    auto local_nbr_idx = nbr_idx;
    int minor_comm_rank{-1};
    size_t intra_partition_offset{};
    if (nbr_idx != invalid_idx) {
      auto displacement_first =
        partitioned_local_degree_displacements.begin() + key_idx * minor_comm_size;
      minor_comm_rank =
        static_cast<int>(thrust::distance(
          displacement_first,
          thrust::upper_bound(
            thrust::seq, displacement_first, displacement_first + minor_comm_size, nbr_idx))) -
        1;
      local_nbr_idx -= *(displacement_first + minor_comm_rank);
      cuda::atomic_ref<size_t, cuda::thread_scope_device> counter(tx_counts[minor_comm_rank]);
      intra_partition_offset = counter.fetch_add(size_t{1}, cuda::std::memory_order_relaxed);
    }
    return thrust::make_tuple(minor_comm_rank, intra_partition_offset, local_nbr_idx, key_idx);
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

template <typename GraphViewType, typename EdgePartitionEdgeMaskWrapper, typename KeyIterator>
struct find_nth_valid_nbr_idx_t {
  using key_t    = typename thrust::iterator_traits<KeyIterator>::value_type;
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu> edge_partition{};
  EdgePartitionEdgeMaskWrapper edge_partition_e_mask;
  KeyIterator key_first{};
  thrust::tuple<raft::device_span<size_t const>, raft::device_span<edge_t const>>
    key_valid_local_nbr_count_inclusive_sums{};

  __device__ edge_t operator()(thrust::tuple<edge_t, size_t> pair) const
  {
    edge_t local_nbr_idx = thrust::get<0>(pair);
    size_t key_idx       = thrust::get<1>(pair);
    auto key             = *(key_first + key_idx);
    vertex_t major{};
    if constexpr (std::is_same_v<key_t, vertex_t>) {
      major = key;
    } else {
      major = thrust::get<0>(key);
    }
    auto major_offset = edge_partition.major_offset_from_major_nocheck(major);
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
      auto inclusive_sum_first = thrust::get<1>(key_valid_local_nbr_count_inclusive_sums).begin();
      auto start_offset        = thrust::get<0>(key_valid_local_nbr_count_inclusive_sums)[key_idx];
      auto end_offset = thrust::get<0>(key_valid_local_nbr_count_inclusive_sums)[key_idx + 1];
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
  raft::device_span<vertex_t const> edge_partition_frontier_majors,
  raft::device_span<size_t const> inclusive_sum_offsets,
  raft::device_span<size_t const> frontier_indices,
  raft::device_span<edge_t> inclusive_sums)
{
  static_assert(per_v_random_select_transform_outgoing_e_block_size % raft::warp_size() == 0);

  auto const tid     = threadIdx.x + blockIdx.x * blockDim.x;
  auto const lane_id = tid % raft::warp_size();

  auto idx = static_cast<size_t>(tid / raft::warp_size());

  using WarpScan = cub::WarpScan<edge_t, raft::warp_size()>;
  __shared__ typename WarpScan::TempStorage temp_storage;

  while (idx < frontier_indices.size()) {
    auto frontier_idx = frontier_indices[idx];
    auto major        = edge_partition_frontier_majors[frontier_idx];
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
  raft::device_span<vertex_t const> edge_partition_frontier_majors,
  raft::device_span<size_t const> inclusive_sum_offsets,
  raft::device_span<size_t const> frontier_indices,
  raft::device_span<edge_t> inclusive_sums)
{
  static_assert(per_v_random_select_transform_outgoing_e_block_size % raft::warp_size() == 0);

  auto idx = static_cast<size_t>(blockIdx.x);

  using BlockScan = cub::BlockScan<edge_t, per_v_random_select_transform_outgoing_e_block_size>;
  __shared__ typename BlockScan::TempStorage temp_storage;

  __shared__ edge_t sum;

  while (idx < frontier_indices.size()) {
    auto frontier_idx = frontier_indices[idx];
    auto major        = edge_partition_frontier_majors[frontier_idx];
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
      ((num_inclusive_sums + per_v_random_select_transform_outgoing_e_block_size - 1) /
       per_v_random_select_transform_outgoing_e_block_size) *
      per_v_random_select_transform_outgoing_e_block_size;
    if (threadIdx.x == per_v_random_select_transform_outgoing_e_block_size - 1) { sum = 0; }
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
      if (threadIdx.x == per_v_random_select_transform_outgoing_e_block_size - 1) { sum += inc; }
    }

    idx += gridDim.x;
  }
}

// divide the frontier to three partitions, the low_degree partition has vertices with degreee in
// [min_low_partition_degree_threshold, min_mid_partition_degree_threshold), the medium degree
// partition has vertices with degree in [min_mid_partition_degree_threshold,
// min_high_partition_degree_threshold), and the high degree partition has vertices with degree in
// [min_high_partition_degree_threshold, infinite).
template <typename edge_t>
std::tuple<rmm::device_uvector<size_t>, std::vector<size_t> /* size = 3 (# partitions) + 1 */>
partition_frontier(raft::handle_t const& handle,
                   raft::device_span<edge_t const> frontier_degrees,
                   edge_t min_low_partition_degree_threshold,
                   edge_t min_mid_partition_degree_threshold,
                   edge_t min_high_partition_degree_threshold)
{
  size_t constexpr num_partitions = 3;  // low, mid, high
  std::vector<size_t> offsets(num_partitions + 1);
  offsets[0] = size_t{0};

  rmm::device_uvector<size_t> indices(frontier_degrees.size(), handle.get_stream());
  indices.resize(
    thrust::distance(indices.begin(),
                     thrust::copy_if(handle.get_thrust_policy(),
                                     thrust::make_counting_iterator(size_t{0}),
                                     thrust::make_counting_iterator(frontier_degrees.size()),
                                     frontier_degrees.begin(),
                                     indices.begin(),
                                     [threshold = min_low_partition_degree_threshold] __device__(
                                       edge_t degree) { return degree >= threshold; })),
    handle.get_stream());

  auto mid_first =
    thrust::partition(handle.get_thrust_policy(),
                      indices.begin(),
                      indices.end(),
                      [frontier_degrees, threshold = min_mid_partition_degree_threshold] __device__(
                        auto idx) { return frontier_degrees[idx] < threshold; });
  offsets[1]      = static_cast<size_t>(thrust::distance(indices.begin(), mid_first));
  auto high_first = thrust::partition(
    handle.get_thrust_policy(),
    mid_first,
    indices.end(),
    [frontier_degrees, threshold = min_high_partition_degree_threshold] __device__(auto idx) {
      return frontier_degrees[idx] < threshold;
    });
  offsets[2] = static_cast<size_t>(thrust::distance(indices.begin(), high_first));
  offsets[3] = indices.size();

  return std::make_tuple(std::move(indices), std::move(offsets));
}

template <typename vertex_t, typename edge_t, bool multi_gpu>
std::tuple<rmm::device_uvector<size_t>, rmm::device_uvector<edge_t>>
compute_valid_local_nbr_count_inclusive_sums(
  raft::handle_t const& handle,
  edge_partition_device_view_t<vertex_t, edge_t, multi_gpu> const& edge_partition,
  edge_partition_edge_property_device_view_t<edge_t, uint32_t const*, bool> const&
    edge_partition_e_mask,
  raft::device_span<vertex_t const> edge_partition_frontier_majors)
{
  auto edge_partition_local_degrees =
    edge_partition.compute_local_degrees(edge_partition_frontier_majors.begin(),
                                         edge_partition_frontier_majors.end(),
                                         handle.get_stream());
  auto inclusive_sum_offsets =
    rmm::device_uvector<size_t>(edge_partition_frontier_majors.size() + 1, handle.get_stream());
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

  auto [edge_partition_frontier_indices, frontier_partition_offsets] = partition_frontier(
    handle,
    raft::device_span<edge_t const>(edge_partition_local_degrees.data(),
                                    edge_partition_local_degrees.size()),
    static_cast<edge_t>(compute_valid_local_nbr_count_inclusive_sum_local_degree_threshold),
    static_cast<edge_t>(compute_valid_local_nbr_count_inclusive_sum_mid_local_degree_threshold),
    static_cast<edge_t>(compute_valid_local_nbr_count_inclusive_sum_high_local_degree_threshold));

  rmm::device_uvector<edge_t> inclusive_sums(
    inclusive_sum_offsets.back_element(handle.get_stream()), handle.get_stream());

  thrust::for_each(
    handle.get_thrust_policy(),
    edge_partition_frontier_indices.begin(),
    edge_partition_frontier_indices.begin() + frontier_partition_offsets[1],
    [edge_partition,
     edge_partition_e_mask,
     edge_partition_frontier_majors,
     inclusive_sum_offsets =
       raft::device_span<size_t const>(inclusive_sum_offsets.data(), inclusive_sum_offsets.size()),
     inclusive_sums = raft::device_span<edge_t>(inclusive_sums.data(),
                                                inclusive_sums.size())] __device__(size_t i) {
      auto major = edge_partition_frontier_majors[i];
      vertex_t major_idx{};
      if constexpr (multi_gpu) {
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
          edge_partition_e_mask.value_first(),
          edge_offset + packed_bools_per_word() * j,
          cuda::std::min(packed_bools_per_word(), local_degree - packed_bools_per_word() * j));
        inclusive_sums[start_offset + j] = sum;
      }
    });

  auto mid_partition_size = frontier_partition_offsets[2] - frontier_partition_offsets[1];
  if (mid_partition_size > 0) {
    raft::grid_1d_warp_t update_grid(mid_partition_size,
                                     per_v_random_select_transform_outgoing_e_block_size,
                                     handle.get_device_properties().maxGridSize[0]);
    compute_valid_local_nbr_count_inclusive_sums_mid_local_degree<<<update_grid.num_blocks,
                                                                    update_grid.block_size,
                                                                    0,
                                                                    handle.get_stream()>>>(
      edge_partition,
      edge_partition_e_mask,
      edge_partition_frontier_majors,
      raft::device_span<size_t const>(inclusive_sum_offsets.data(), inclusive_sum_offsets.size()),
      raft::device_span<size_t const>(
        edge_partition_frontier_indices.data() + frontier_partition_offsets[1],
        frontier_partition_offsets[2] - frontier_partition_offsets[1]),
      raft::device_span<edge_t>(inclusive_sums.data(), inclusive_sums.size()));
  }

  auto high_partition_size = frontier_partition_offsets[3] - frontier_partition_offsets[2];
  if (high_partition_size > 0) {
    raft::grid_1d_block_t update_grid(high_partition_size,
                                      per_v_random_select_transform_outgoing_e_block_size,
                                      handle.get_device_properties().maxGridSize[0]);
    compute_valid_local_nbr_count_inclusive_sums_high_local_degree<<<update_grid.num_blocks,
                                                                     update_grid.block_size,
                                                                     0,
                                                                     handle.get_stream()>>>(
      edge_partition,
      edge_partition_e_mask,
      edge_partition_frontier_majors,
      raft::device_span<size_t const>(inclusive_sum_offsets.data(), inclusive_sum_offsets.size()),
      raft::device_span<size_t const>(
        edge_partition_frontier_indices.data() + frontier_partition_offsets[2],
        frontier_partition_offsets[3] - frontier_partition_offsets[2]),
      raft::device_span<edge_t>(inclusive_sums.data(), inclusive_sums.size()));
  }

  return std::make_tuple(std::move(inclusive_sum_offsets), std::move(inclusive_sums));
}

template <typename edge_t>
rmm::device_uvector<edge_t> get_sampling_index_without_replacement(
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

  auto [frontier_indices, frontier_partition_offsets] = partition_frontier(
    handle,
    raft::device_span<edge_t const>(frontier_degrees.data(), frontier_degrees.size()),
    edge_t{0},
    static_cast<edge_t>(K + 1),
    mid_partition_degree_range_last + 1);

  rmm::device_uvector<edge_t> sample_nbr_indices(frontier_degrees.size() * K, handle.get_stream());

  auto low_partition_size = frontier_partition_offsets[1];
  if (low_partition_size > 0) {
    thrust::for_each(handle.get_thrust_policy(),
                     thrust::make_counting_iterator(size_t{0}),
                     thrust::make_counting_iterator(low_partition_size * K),
                     [K,
                      frontier_index_first = frontier_indices.begin(),
                      frontier_degrees   = raft::device_span<edge_t const>(frontier_degrees.data(),
                                                                         frontier_degrees.size()),
                      sample_nbr_indices = raft::device_span<edge_t>(sample_nbr_indices.data(),
                                                                     sample_nbr_indices.size()),
                      invalid_idx = cugraph::ops::graph::INVALID_ID<edge_t>] __device__(size_t i) {
                       auto frontier_idx = *(frontier_index_first + i);
                       auto degree       = frontier_degrees[frontier_idx];
                       auto sample_idx   = static_cast<edge_t>(i % K);
                       sample_nbr_indices[frontier_idx * K + sample_idx] =
                         (sample_idx < degree) ? sample_idx : invalid_idx;
                     });
  }

  auto mid_partition_size = frontier_partition_offsets[2] - frontier_partition_offsets[1];
  if (mid_partition_size > 0) {
    // FIXME: tmp_degrees & tmp_sample_nbr_indices can be avoided if we customize
    // cugraph::ops::get_sampling_index
    rmm::device_uvector<edge_t> tmp_degrees(mid_partition_size, handle.get_stream());
    rmm::device_uvector<edge_t> tmp_sample_nbr_indices(mid_partition_size * K, handle.get_stream());
    thrust::gather(handle.get_thrust_policy(),
                   frontier_indices.begin() + frontier_partition_offsets[1],
                   frontier_indices.begin() + frontier_partition_offsets[2],
                   frontier_degrees.begin(),
                   tmp_degrees.begin());
    cugraph::ops::graph::get_sampling_index(tmp_sample_nbr_indices.data(),
                                            rng_state,
                                            tmp_degrees.data(),
                                            mid_partition_size,
                                            static_cast<int32_t>(K),
                                            false,
                                            handle.get_stream());
    thrust::for_each(handle.get_thrust_policy(),
                     thrust::make_counting_iterator(size_t{0}),
                     thrust::make_counting_iterator(mid_partition_size * K),
                     [K,
                      seed_index_first = frontier_indices.begin() + frontier_partition_offsets[1],
                      tmp_sample_nbr_indices = tmp_sample_nbr_indices.data(),
                      sample_nbr_indices     = sample_nbr_indices.data()] __device__(size_t i) {
                       auto seed_idx                                 = *(seed_index_first + i / K);
                       auto sample_idx                               = static_cast<edge_t>(i % K);
                       sample_nbr_indices[seed_idx * K + sample_idx] = tmp_sample_nbr_indices[i];
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

    rmm::device_uvector<edge_t> tmp_sample_nbr_indices(
      seeds_to_sort_per_iteration * high_partition_oversampling_K, handle.get_stream());
    assert(high_partition_oversampling_K * 2 <=
           static_cast<size_t>(std::numeric_limits<int32_t>::max()));
    rmm::device_uvector<int32_t> tmp_sample_indices(
      tmp_sample_nbr_indices.size(),
      handle.get_stream());  // sample indices ([0, high_partition_oversampling_K)) within a segment
                             // (one segment per seed)

    rmm::device_uvector<edge_t> segment_sorted_tmp_sample_nbr_indices(tmp_sample_nbr_indices.size(),
                                                                      handle.get_stream());
    rmm::device_uvector<int32_t> segment_sorted_tmp_sample_indices(tmp_sample_nbr_indices.size(),
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
      std::optional<rmm::device_uvector<edge_t>> retry_sample_nbr_indices{std::nullopt};
      std::optional<rmm::device_uvector<int32_t>> retry_sample_indices{std::nullopt};
      std::optional<rmm::device_uvector<edge_t>> retry_segment_sorted_sample_nbr_indices{
        std::nullopt};
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
          retry_sample_nbr_indices = rmm::device_uvector<edge_t>(
            (*retry_segment_indices).size() * high_partition_oversampling_K, handle.get_stream());
          retry_sample_indices =
            rmm::device_uvector<int32_t>((*retry_sample_nbr_indices).size(), handle.get_stream());
          retry_segment_sorted_sample_nbr_indices =
            rmm::device_uvector<edge_t>((*retry_sample_nbr_indices).size(), handle.get_stream());
          retry_segment_sorted_sample_indices =
            rmm::device_uvector<int32_t>((*retry_sample_nbr_indices).size(), handle.get_stream());
        }

        if (retry_segment_indices) {
          cugraph::ops::graph::get_sampling_index(
            (*retry_sample_nbr_indices).data(),
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
            tmp_sample_nbr_indices.data(),
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
             unique_counts                         = unique_counts.data(),
             segment_sorted_tmp_sample_nbr_indices = segment_sorted_tmp_sample_nbr_indices.data(),
             retry_segment_indices                 = (*retry_segment_indices).data(),
             retry_sample_nbr_indices              = (*retry_sample_nbr_indices).data(),
             retry_sample_indices = (*retry_sample_indices).data()] __device__(size_t i) {
              auto segment_idx  = retry_segment_indices[i / high_partition_oversampling_K];
              auto sample_idx   = static_cast<edge_t>(i % high_partition_oversampling_K);
              auto unique_count = unique_counts[segment_idx];
              auto output_first = thrust::make_zip_iterator(
                thrust::make_tuple(retry_sample_nbr_indices, retry_sample_indices));
              // sample index for the previously selected neighbor indices should be smaller than
              // the new candidates to ensure that the previously selected neighbor indices will be
              // selected again
              if (sample_idx < unique_count) {
                *(output_first + i) =
                  thrust::make_tuple(segment_sorted_tmp_sample_nbr_indices
                                       [segment_idx * high_partition_oversampling_K + sample_idx],
                                     static_cast<int32_t>(sample_idx));
              } else {
                *(output_first + i) =
                  thrust::make_tuple(retry_sample_nbr_indices[i],
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
          retry_segment_indices ? (*retry_sample_nbr_indices).data()
                                : tmp_sample_nbr_indices.data(),
          retry_segment_indices ? (*retry_segment_sorted_sample_nbr_indices).data()
                                : segment_sorted_tmp_sample_nbr_indices.data(),
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
          retry_segment_indices ? (*retry_sample_nbr_indices).data()
                                : tmp_sample_nbr_indices.data(),
          retry_segment_indices ? (*retry_segment_sorted_sample_nbr_indices).data()
                                : segment_sorted_tmp_sample_nbr_indices.data(),
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
               thrust::make_tuple((*retry_segment_sorted_sample_nbr_indices).begin(),
                                  (*retry_segment_sorted_sample_indices).begin())),
             segment_sorted_pair_first = thrust::make_zip_iterator(thrust::make_tuple(
               segment_sorted_tmp_sample_nbr_indices.begin(),
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
               segment_sorted_tmp_sample_nbr_indices.begin(),
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
        segment_sorted_tmp_sample_nbr_indices.data(),
        tmp_sample_nbr_indices.data(),
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
        segment_sorted_tmp_sample_nbr_indices.data(),
        tmp_sample_nbr_indices.data(),
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

      // copy the neighbor indices back to sample_nbr_indices

      thrust::for_each(
        handle.get_thrust_policy(),
        thrust::make_counting_iterator(size_t{0}),
        thrust::make_counting_iterator(num_segments * K),
        [K,
         high_partition_oversampling_K,
         frontier_indices = frontier_indices.begin() + frontier_partition_offsets[2] +
                            seeds_to_sort_per_iteration * i,
         tmp_sample_nbr_indices = tmp_sample_nbr_indices.data(),
         sample_nbr_indices     = sample_nbr_indices.data()] __device__(size_t i) {
          auto seed_idx   = *(frontier_indices + i / K);
          auto sample_idx = static_cast<edge_t>(i % K);
          *(sample_nbr_indices + seed_idx * K + sample_idx) =
            *(tmp_sample_nbr_indices + (i / K) * high_partition_oversampling_K + sample_idx);
        });
    }
  }

  frontier_degrees.resize(0, handle.get_stream());
  frontier_degrees.shrink_to_fit(handle.get_stream());

  return sample_nbr_indices;
#else
  CUGRAPH_FAIL("unimplemented.");
#endif
}

template <typename GraphViewType,
          typename VertexFrontierBucketType,
          typename AggregateLocalFrontierBuffer>
std::tuple<rmm::device_uvector<typename GraphViewType::edge_type>,
           std::optional<rmm::device_uvector<size_t>>,
           std::vector<size_t>>
uniform_sample_and_compute_local_nbr_indices(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  VertexFrontierBucketType const& frontier,
  AggregateLocalFrontierBuffer const& aggregate_local_frontier,
  std::vector<size_t> const& local_frontier_displacements,
  std::vector<size_t> const& local_frontier_sizes,
  raft::random::RngState& rng_state,
  size_t K,
  bool with_replacement)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using key_t    = typename VertexFrontierBucketType::key_type;

  auto minor_comm_size =
    GraphViewType::is_multi_gpu
      ? handle.get_subcomm(cugraph::partition_manager::minor_comm_name()).get_size()
      : int{1};
  assert(minor_comm_size == graph_view.number_of_local_edge_partitions());

  auto edge_mask_view = graph_view.edge_mask_view();

  // 1. compute degrees

  rmm::device_uvector<edge_t> frontier_degrees(0, handle.get_stream());
  auto frontier_partitioned_local_degree_displacements =
    (minor_comm_size > 1)
      ? std::make_optional<rmm::device_uvector<edge_t>>(size_t{0}, handle.get_stream())
      : std::nullopt;  // one partition per gpu in the same minor_comm

  std::optional<std::vector<std::tuple<rmm::device_uvector<size_t>, rmm::device_uvector<edge_t>>>>
    local_frontier_valid_local_nbr_count_inclusive_sums{};  // to avoid searching the entire
                                                            // neighbor list K times for high degree
                                                            // vertices with edge masking
  if (edge_mask_view) {
    local_frontier_valid_local_nbr_count_inclusive_sums =
      std::vector<std::tuple<rmm::device_uvector<size_t>, rmm::device_uvector<edge_t>>>{};
    (*local_frontier_valid_local_nbr_count_inclusive_sums)
      .reserve(graph_view.number_of_local_edge_partitions());
  }

  {
    auto aggregate_local_frontier_local_degrees =
      (minor_comm_size > 1)
        ? std::make_optional<rmm::device_uvector<edge_t>>(
            local_frontier_displacements.back() + local_frontier_sizes.back(), handle.get_stream())
        : std::nullopt;

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

      vertex_t const* edge_partition_frontier_major_first{nullptr};

      auto edge_partition_frontier_key_first =
        ((minor_comm_size > 1) ? get_dataframe_buffer_begin(*aggregate_local_frontier)
                               : frontier.begin()) +
        local_frontier_displacements[i];
      if constexpr (std::is_same_v<key_t, vertex_t>) {
        edge_partition_frontier_major_first = edge_partition_frontier_key_first;
      } else {
        edge_partition_frontier_major_first = thrust::get<0>(edge_partition_frontier_key_first);
      }

      auto edge_partition_frontier_local_degrees =
        edge_partition_e_mask ? edge_partition.compute_local_degrees_with_mask(
                                  (*edge_partition_e_mask).value_first(),
                                  edge_partition_frontier_major_first,
                                  edge_partition_frontier_major_first + local_frontier_sizes[i],
                                  handle.get_stream())
                              : edge_partition.compute_local_degrees(
                                  edge_partition_frontier_major_first,
                                  edge_partition_frontier_major_first + local_frontier_sizes[i],
                                  handle.get_stream());

      if (minor_comm_size > 1) {
        // FIXME: this copy is unnecessary if edge_partition.compute_local_degrees() takes a pointer
        // to the output array
        thrust::copy(
          handle.get_thrust_policy(),
          edge_partition_frontier_local_degrees.begin(),
          edge_partition_frontier_local_degrees.end(),
          (*aggregate_local_frontier_local_degrees).begin() + local_frontier_displacements[i]);
      } else {
        frontier_degrees = std::move(edge_partition_frontier_local_degrees);
      }

      if (edge_partition_e_mask) {
        (*local_frontier_valid_local_nbr_count_inclusive_sums)
          .push_back(compute_valid_local_nbr_count_inclusive_sums(
            handle,
            edge_partition,
            *edge_partition_e_mask,
            raft::device_span<vertex_t const>(edge_partition_frontier_major_first,
                                              local_frontier_sizes[i])));
      }
    }

    if (minor_comm_size > 1) {
      auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());

      rmm::device_uvector<edge_t> frontier_gathered_local_degrees(0, handle.get_stream());
      std::tie(frontier_gathered_local_degrees, std::ignore) =
        shuffle_values(minor_comm,
                       (*aggregate_local_frontier_local_degrees).begin(),
                       local_frontier_sizes,
                       handle.get_stream());
      aggregate_local_frontier_local_degrees = std::nullopt;

      frontier_degrees.resize(frontier.size(), handle.get_stream());
      frontier_partitioned_local_degree_displacements =
        rmm::device_uvector<edge_t>(frontier_degrees.size() * minor_comm_size, handle.get_stream());
      thrust::for_each(
        handle.get_thrust_policy(),
        thrust::make_counting_iterator(size_t{0}),
        thrust::make_counting_iterator(frontier_degrees.size()),
        compute_local_degree_displacements_and_global_degree_t<edge_t>{
          raft::device_span<edge_t const>(frontier_gathered_local_degrees.data(),
                                          frontier_gathered_local_degrees.size()),
          raft::device_span<edge_t>((*frontier_partitioned_local_degree_displacements).data(),
                                    (*frontier_partitioned_local_degree_displacements).size()),
          raft::device_span<edge_t>(frontier_degrees.data(), frontier_degrees.size()),
          minor_comm_size});
    }
  }

  // 2. sample neighbor indices

  rmm::device_uvector<edge_t> sample_nbr_indices(0, handle.get_stream());

  if (with_replacement) {
    if (frontier_degrees.size() > 0) {
      sample_nbr_indices.resize(frontier.size() * K, handle.get_stream());
      cugraph::ops::graph::get_sampling_index(sample_nbr_indices.data(),
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
    sample_nbr_indices =
      get_sampling_index_without_replacement(handle, std::move(frontier_degrees), rng_state, K);
  }

  // 3. shuffle neighbor indices

  auto sample_local_nbr_indices = std::move(
    sample_nbr_indices);  // neighbor index within an edge partition (note that each vertex's
                          // neighbors are distributed in minor_comm_size partitions)
  std::optional<rmm::device_uvector<size_t>> sample_key_indices{
    std::nullopt};  // relevant only when (minor_comm_size > 1)
  std::vector<size_t> local_frontier_sample_offsets{};
  if (minor_comm_size > 1) {
    auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());

    sample_key_indices =
      rmm::device_uvector<size_t>(sample_local_nbr_indices.size(), handle.get_stream());
    auto minor_comm_ranks =
      rmm::device_uvector<int>(sample_local_nbr_indices.size(), handle.get_stream());
    auto intra_partition_displacements =
      rmm::device_uvector<size_t>(sample_local_nbr_indices.size(), handle.get_stream());
    rmm::device_uvector<size_t> d_tx_counts(minor_comm_size, handle.get_stream());
    thrust::fill(handle.get_thrust_policy(), d_tx_counts.begin(), d_tx_counts.end(), size_t{0});
    auto input_pair_first = thrust::make_zip_iterator(
      thrust::make_tuple(sample_local_nbr_indices.begin(),
                         thrust::make_transform_iterator(thrust::make_counting_iterator(size_t{0}),
                                                         divider_t<size_t>{K})));
    thrust::transform(
      handle.get_thrust_policy(),
      input_pair_first,
      input_pair_first + sample_local_nbr_indices.size(),
      thrust::make_zip_iterator(thrust::make_tuple(minor_comm_ranks.begin(),
                                                   intra_partition_displacements.begin(),
                                                   sample_local_nbr_indices.begin(),
                                                   (*sample_key_indices).begin())),
      convert_pair_to_quadruplet_t<edge_t>{
        raft::device_span<edge_t const>((*frontier_partitioned_local_degree_displacements).data(),
                                        (*frontier_partitioned_local_degree_displacements).size()),
        raft::device_span<size_t>(d_tx_counts.data(), d_tx_counts.size()),
        frontier.size(),
        minor_comm_size,
        cugraph::ops::graph::INVALID_ID<edge_t>});
    rmm::device_uvector<size_t> tx_displacements(minor_comm_size, handle.get_stream());
    thrust::exclusive_scan(
      handle.get_thrust_policy(), d_tx_counts.begin(), d_tx_counts.end(), tx_displacements.begin());
    auto tmp_sample_local_nbr_indices =
      rmm::device_uvector<edge_t>(tx_displacements.back_element(handle.get_stream()) +
                                    d_tx_counts.back_element(handle.get_stream()),
                                  handle.get_stream());
    auto tmp_sample_key_indices =
      rmm::device_uvector<size_t>(tmp_sample_local_nbr_indices.size(), handle.get_stream());
    auto pair_first = thrust::make_zip_iterator(
      thrust::make_tuple(sample_local_nbr_indices.begin(), (*sample_key_indices).begin()));
    thrust::scatter_if(
      handle.get_thrust_policy(),
      pair_first,
      pair_first + sample_local_nbr_indices.size(),
      thrust::make_transform_iterator(
        thrust::make_counting_iterator(size_t{0}),
        shuffle_index_compute_offset_t{
          raft::device_span<int const>(minor_comm_ranks.data(), minor_comm_ranks.size()),
          raft::device_span<size_t const>(intra_partition_displacements.data(),
                                          intra_partition_displacements.size()),
          raft::device_span<size_t const>(tx_displacements.data(), tx_displacements.size())}),
      minor_comm_ranks.begin(),
      thrust::make_zip_iterator(
        thrust::make_tuple(tmp_sample_local_nbr_indices.begin(), tmp_sample_key_indices.begin())),
      is_not_equal_t<int>{-1});

    sample_local_nbr_indices = std::move(tmp_sample_local_nbr_indices);
    sample_key_indices       = std::move(tmp_sample_key_indices);

    std::vector<size_t> h_tx_counts(d_tx_counts.size());
    raft::update_host(
      h_tx_counts.data(), d_tx_counts.data(), d_tx_counts.size(), handle.get_stream());
    handle.sync_stream();

    pair_first = thrust::make_zip_iterator(
      thrust::make_tuple(sample_local_nbr_indices.begin(), (*sample_key_indices).begin()));
    auto [rx_value_buffer, rx_counts] =
      shuffle_values(minor_comm, pair_first, h_tx_counts, handle.get_stream());

    sample_local_nbr_indices         = std::move(std::get<0>(rx_value_buffer));
    sample_key_indices               = std::move(std::get<1>(rx_value_buffer));
    local_frontier_sample_offsets    = std::vector<size_t>(rx_counts.size() + 1);
    local_frontier_sample_offsets[0] = size_t{0};
    std::inclusive_scan(
      rx_counts.begin(), rx_counts.end(), local_frontier_sample_offsets.begin() + 1);
  } else {
    local_frontier_sample_offsets = std::vector<size_t>{size_t{0}, frontier.size() * K};
  }

  // 4. convert neighbor indices in the neighbor list considering edge mask to neighbor indices in
  // the neighbor list ignoring edge mask

  if (edge_mask_view) {
    auto sample_key_idx_first = thrust::make_transform_iterator(
      thrust::make_counting_iterator(size_t{0}),
      cuda::proclaim_return_type<size_t>([K,
       sample_key_indices = sample_key_indices
                              ? thrust::make_optional<raft::device_span<size_t const>>(
                                  (*sample_key_indices).data(), (*sample_key_indices).size())
                              : thrust::nullopt] __device__(size_t i) {
        return sample_key_indices ? (*sample_key_indices)[i] : i / K;
      }));
    auto pair_first = thrust::make_zip_iterator(sample_local_nbr_indices.begin(), sample_key_idx_first);
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

      auto edge_partition_frontier_key_first =
        ((minor_comm_size > 1) ? get_dataframe_buffer_begin(*aggregate_local_frontier)
                               : frontier.begin()) +
        local_frontier_displacements[i];
      thrust::transform(
        handle.get_thrust_policy(),
        pair_first + local_frontier_sample_offsets[i],
        pair_first + local_frontier_sample_offsets[i + 1],
        sample_local_nbr_indices.begin() + local_frontier_sample_offsets[i],
        find_nth_valid_nbr_idx_t<GraphViewType,
                                 decltype(edge_partition_e_mask),
                                 decltype(edge_partition_frontier_key_first)>{
          edge_partition,
          edge_partition_e_mask,
          edge_partition_frontier_key_first,
          thrust::make_tuple(
            raft::device_span<size_t const>(
              std::get<0>((*local_frontier_valid_local_nbr_count_inclusive_sums)[i]).data(),
              std::get<0>((*local_frontier_valid_local_nbr_count_inclusive_sums)[i]).size()),
            raft::device_span<edge_t const>(
              std::get<1>((*local_frontier_valid_local_nbr_count_inclusive_sums)[i]).data(),
              std::get<1>((*local_frontier_valid_local_nbr_count_inclusive_sums)[i]).size()))});
    }
  }

  return std::make_tuple(std::move(sample_local_nbr_indices),
                         std::move(sample_key_indices),
                         std::move(local_frontier_sample_offsets));
}

#if 0
biased_sampling_nbr_indices(raft::handle_t const& handle GraphViewType const& graph_view,
                            VertexFrontierBucketType const& frontier
                              std::vector<size_t> const& local_frontier_displacements,
                            std::vector<size_t> const& local_frontier_sizes,
                            raft::random::RngState& rng_state,
                            size_t K,
                            bool with_replacement)
{
  using bias_t = typename detail::edge_op_result_type<key_t,
                                                      vertex_t,
                                                      typename EdgeSrcValueInputWrapper::value_type,
                                                      typename EdgeDstValueInputWrapper::value_type,
                                                      typename EdgeValueInputWrapper::value_type,
                                                      EdgeBiasOp>::type;

  auto minor_comm_size =
    GraphViewType::is_multi_gpu
      ? handle.get_subcomm(cugraph::partition_manager::minor_comm_name()).get_size()
      : int{1};
  assert(minor_comm_size == graph_view.number_of_local_edge_partitions());

  auto edge_mask_view = graph_view.edge_mask_view();

  // 1. compute degrees

  rmm::device_uvector<edge_t> frontier_degrees(0, handle.get_stream());
  auto frontier_partitioned_local_bias_sum_displacements =
    (minor_comm_size > 1)
      ? std::make_optional<rmm::device_uvector<bias_t>>(size_t{0}, handle.get_stream())
      : std::nullopt;  // one partition per gpu in the same minor_comm

  {
    auto aggregate_local_frontier_local_bias_sums =
      (minor_comm_size > 1)
        ? std::make_optional<rmm::device_uvector<edge_t>>(
            local_frontier_displacements.back() + local_frontier_sizes.back(), handle.get_stream())
        : std::nullopt;
    std::vector<rmm::device_uvector<bias_t>>;

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

      vertex_t const* edge_partition_frontier_major_first{nullptr};

      auto edge_partition_frontier_key_first =
        ((minor_comm_size > 1) ? get_dataframe_buffer_begin(*aggregate_local_frontier)
                               : frontier.begin()) +
        local_frontier_displacements[i];
      if constexpr (std::is_same_v<key_t, vertex_t>) {
        edge_partition_frontier_major_first = edge_partition_frontier_key_first;
      } else {
        edge_partition_frontier_major_first = thrust::get<0>(edge_partition_frontier_key_first);
      }

      auto edge_partition_frontier_local_degrees = edge_partition.compute_local_degrees_with_mask(
        (*edge_partition_e_mask).value_first(),
        edge_partition_frontier_major_first,
        edge_partition_frontier_major_first + local_frontier_sizes[i],
        handle.get_stream());
      auto edge_partition_frontier_local_degree_inclusive_sums;
      // if wiht_replacment = false && degree <= K skip bias computing? how should I handle bias ==
      // 0 update documentation for this.

      auto edge_partition_frontier_e_biases;

      if (minor_comm_size > 1) {
        // FIXME: this copy is unnecessary if edge_partition.compute_local_degrees() takes a pointer
        // to the output array
        thrust::copy(
          handle.get_thrust_policy(),
          edge_partition_frontier_local_degrees.begin(),
          edge_partition_frontier_local_degrees.end(),
          (*aggregate_local_frontier_local_degrees).begin() + local_frontier_displacements[i]);
      } else {
        frontier_degrees = std::move(edge_partition_frontier_local_degrees);
      }
    }

    if (minor_comm_size > 1) {
      auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());

      rmm::device_uvector<edge_t> frontier_gathered_local_degrees(0, handle.get_stream());
      std::tie(frontier_gathered_local_degrees, std::ignore) =
        shuffle_values(minor_comm,
                       (*aggregate_local_frontier_local_degrees).begin(),
                       local_frontier_sizes,
                       handle.get_stream());
      aggregate_local_frontier_local_degrees = std::nullopt;

      frontier_dgrees.resize(frontier.size(), handle.get_stream());
      frontier_partitioned_local_degree_displacements =
        rmm::device_uvector<edge_t>(frontier_degrees.size() * minor_comm_size, handle.get_stream());
      thrust::for_each(
        handle.get_thrust_policy(),
        thrust::make_counting_iterator(size_t{0}),
        thrust::make_counting_iterator(frontier_degrees.size()),
        compute_local_degree_displacements_and_global_degree_t<edge_t>{
          raft::device_span<edge_t const>(frontier_gathered_local_degrees.data(),
                                          frontier_gathered_local_degrees.size()),
          raft::device_span<edge_t>((*frontier_partitioned_local_degree_displacements).data(),
                                    (*frontier_partitioned_local_degree_displacements).size()),
          raft::device_span<edge_t>(frontier_degrees.data(), frontier_degrees.size()),
          minor_comm_size});
    }
  }

  // 2. sample neighbor indices

  rmm::device_uvector<edge_t> sample_nbr_indices(0, handle.get_stream());

  if (with_replacement) {
    // generate random numbers in [0.0, 1.0);
    // scale by bias_sum;
    // find minor_comm_rank and compute local random number;
  } else {
    if (degree <= K) {
      // thrust::sequence
    }
    else if (degree < K * minor_comm_size * (status_size_per_K/weight_size) {
      // auto bias_first = ();
      // gather biases;
      // generate indices
    }
    else {
      // locally generatei incdices; gather states;
      // generate indices
    }
  }

  // 3. shuffle neighbor indices

  auto sample_local_nbr_indices = std::move(
    sample_nbr_indices);  // neighbor index within an edge partition (note that each vertex's
                          // neighbors are distributed in minor_comm_size partitions)
  std::optional<rmm::device_uvector<size_t>> sample_key_indices{
    std::nullopt};  // relevant only when (minor_comm_size > 1)
  std::vector<size_t> local_frontier_sample_offsets{};
  if (minor_comm_size > 1) {
    auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());

    sample_key_indices =
      rmm::device_uvector<size_t>(sample_local_nbr_indices.size(), handle.get_stream());
    auto minor_comm_ranks =
      rmm::device_uvector<int>(sample_local_nbr_indices.size(), handle.get_stream());
    auto intra_partition_displacements =
      rmm::device_uvector<size_t>(sample_local_nbr_indices.size(), handle.get_stream());
    rmm::device_uvector<size_t> d_tx_counts(minor_comm_size, handle.get_stream());
    thrust::fill(handle.get_thrust_policy(), d_tx_counts.begin(), d_tx_counts.end(), size_t{0});
    auto input_pair_first = thrust::make_zip_iterator(
      thrust::make_tuple(sample_local_nbr_indices.begin(),
                         thrust::make_transform_iterator(thrust::make_counting_iterator(size_t{0}),
                                                         divider_t<size_t>{K})));
    thrust::transform(
      handle.get_thrust_policy(),
      input_pair_first,
      input_pair_first + sample_local_nbr_indices.size(),
      thrust::make_zip_iterator(thrust::make_tuple(minor_comm_ranks.begin(),
                                                   intra_partition_displacements.begin(),
                                                   sample_local_nbr_indices.begin(),
                                                   (*sample_key_indices).begin())),
      convert_pair_to_quadruplet_t<edge_t>{
        raft::device_span<edge_t const>((*frontier_partitioned_local_degree_displacements).data(),
                                        (*frontier_partitioned_local_degree_displacements).size()),
        raft::device_span<size_t>(d_tx_counts.data(), d_tx_counts.size()),
        frontier.size(),
        minor_comm_size,
        cugraph::ops::graph::INVALID_ID<edge_t>});
    rmm::device_uvector<size_t> tx_displacements(minor_comm_size, handle.get_stream());
    thrust::exclusive_scan(
      handle.get_thrust_policy(), d_tx_counts.begin(), d_tx_counts.end(), tx_displacements.begin());
    auto tmp_sample_local_nbr_indices =
      rmm::device_uvector<edge_t>(tx_displacements.back_element(handle.get_stream()) +
                                    d_tx_counts.back_element(handle.get_stream()),
                                  handle.get_stream());
    auto tmp_sample_key_indices =
      rmm::device_uvector<size_t>(tmp_sample_local_nbr_indices.size(), handle.get_stream());
    auto pair_first = thrust::make_zip_iterator(
      thrust::make_tuple(sample_local_nbr_indices.begin(), (*sample_key_indices).begin()));
    thrust::scatter_if(
      handle.get_thrust_policy(),
      pair_first,
      pair_first + sample_local_nbr_indices.size(),
      thrust::make_transform_iterator(
        thrust::make_counting_iterator(size_t{0}),
        shuffle_index_compute_offset_t{
          raft::device_span<int const>(minor_comm_ranks.data(), minor_comm_ranks.size()),
          raft::device_span<size_t const>(intra_partition_displacements.data(),
                                          intra_partition_displacements.size()),
          raft::device_span<size_t const>(tx_displacements.data(), tx_displacements.size())}),
      minor_comm_ranks.begin(),
      thrust::make_zip_iterator(
        thrust::make_tuple(tmp_sample_local_nbr_indices.begin(), tmp_sample_key_indices.begin())),
      is_not_equal_t<int>{-1});

    sample_local_nbr_indices = std::move(tmp_sample_local_nbr_indices);
    sample_key_indices       = std::move(tmp_sample_key_indices);

    std::vector<size_t> h_tx_counts(d_tx_counts.size());
    raft::update_host(
      h_tx_counts.data(), d_tx_counts.data(), d_tx_counts.size(), handle.get_stream());
    handle.sync_stream();

    pair_first = thrust::make_zip_iterator(
      thrust::make_tuple(sample_local_nbr_indices.begin(), (*sample_key_indices).begin()));
    auto [rx_value_buffer, rx_counts] =
      shuffle_values(minor_comm, pair_first, h_tx_counts, handle.get_stream());

    sample_local_nbr_indices         = std::move(std::get<0>(rx_value_buffer));
    sample_key_indices               = std::move(std::get<1>(rx_value_buffer));
    local_frontier_sample_offsets    = std::vector<size_t>(rx_counts.size() + 1);
    local_frontier_sample_offsets[0] = size_t{0};
    std::inclusive_scan(
      rx_counts.begin(), rx_counts.end(), local_frontier_sample_offsets.begin() + 1);
  } else {
    local_frontier_sample_offsets = std::vector<size_t>{size_t{0}, frontier.size() * K};
  }
}

template <bool incoming,
          typename GraphViewType,
          typename VertexFrontierBucketType,
          typename AggregateLocalFrontierBuffer,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename EdgeBiasOp>
std::tuple<rmm::device_uvector<typename GraphViewType::edge_type>,
           std::optional<rmm::device_uvector<size_t>>,
           std::vector<size_t>>  // (neighbor indices, key indices, local frontier sample offsets)
sample_and_compute_local_nbr_indices(raft::handle_t const& handle,
                                     GraphViewType const& graph_view,
                                     VertexFrontierBucketType const& frontier,
                                     AggregateLocalFrontierBuffer const& aggregate_local_frontier,
                                     EdgeSrcValueInputWrapper edge_src_value_input,
                                     EdgeDstValueInputWrapper edge_dst_value_input,
                                     EdgeValueInputWrapper edge_value_input,
                                     EdgeBiasOp e_bias_op,
                                     std::vector<size_t> const& local_frontier_displacements,
                                     std::vector<size_t> const& local_frontier_sizes,
                                     raft::random::RngState& rng_state,
                                     size_t K,
                                     bool with_replacement,
                                     bool do_expensive_check)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using key_t    = typename VertexFrontierBucketType::key_type;

#ifndef NO_CUGRAPH_OPS

  bool constexpr use_bias = !std::is_same_v<EdgeBiasOp,
                                            constant_e_bias_op_t<GraphViewType,
                                                                 EdgeSrcValueInputWrapper,
                                                                 EdgeDstValueInputWrapper,
                                                                 EdgeValueInputWrapper,
                                                                 key_t>>;

  using edge_partition_src_input_device_view_t = std::conditional_t<
    std::is_same_v<typename EdgeSrcValueInputWrapper::value_type, thrust::nullopt_t>,
    edge_partition_endpoint_dummy_property_device_view_t<vertex_t>,
    edge_partition_endpoint_property_device_view_t<
      vertex_t,
      typename EdgeSrcValueInputWrapper::value_iterator,
      typename EdgeSrcValueInputWrapper::value_type>>;
  using edge_partition_dst_input_device_view_t = std::conditional_t<
    std::is_same_v<typename EdgeDstValueInputWrapper::value_type, thrust::nullopt_t>,
    edge_partition_endpoint_dummy_property_device_view_t<vertex_t>,
    edge_partition_endpoint_property_device_view_t<
      vertex_t,
      typename EdgeDstValueInputWrapper::value_iterator,
      typename EdgeDstValueInputWrapper::value_type>>;
  using edge_partition_e_input_device_view_t = std::conditional_t<
    std::is_same_v<typename EdgeValueInputWrapper::value_type, thrust::nullopt_t>,
    detail::edge_partition_edge_dummy_property_device_view_t<vertex_t>,
    detail::edge_partition_edge_property_device_view_t<
      edge_t,
      typename EdgeValueInputWrapper::value_iterator,
      typename EdgeValueInputWrapper::value_type>>;

  static_assert(GraphViewType::is_storage_transposed == incoming);

  CUGRAPH_EXPECTS(K >= size_t{1},
                  "Invalid input argument: invalid K, K should be a positive integer.");
  CUGRAPH_EXPECTS(K <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
                  "Invalid input argument: the current implementation expects K to be no larger "
                  "than std::numeric_limits<int32_t>::max().");

  auto minor_comm_size =
    GraphViewType::is_multi_gpu
      ? handle.get_subcomm(cugraph::partition_manager::minor_comm_name()).get_size()
      : int{1};
  assert(minor_comm_size == graph_view.number_of_local_edge_partitions());

  if (do_expensive_check) {
    // FIXME: should I check frontier & aggregate_local_frontier?
  }

  // 1. compute degrees

  auto edge_mask_view = graph_view.edge_mask_view();

  rmm::device_uvector<edge_t> frontier_degrees(frontier.size(), handle.get_stream());
  auto frontier_partitioned_local_degree_displacements =
    ((minor_comm_size > 1) && (!use_bias || with_replacement))
      ? std::make_optional<rmm::device_uvector<edge_t>>(size_t{0}, handle.get_stream())
      : std::nullopt;  // one partition per gpu in the same minor_comm

  // BIASED: unnecessary for biased as we can just set bias to 0 for masked out edges?
  std::optional<std::vector<std::tuple<rmm::device_uvector<size_t>, rmm::device_uvector<edge_t>>>>
    local_frontier_valid_local_nbr_count_inclusive_sums{};  // to avoid searching the entire
                                                            // neighbor list K times for high degree
                                                            // vertices with edge masking
  if (edge_mask_view) {
    local_frontier_valid_local_nbr_count_inclusive_sums =
      std::vector<std::tuple<rmm::device_uvector<size_t>, rmm::device_uvector<edge_t>>>{};
    (*local_frontier_valid_local_nbr_count_inclusive_sums)
      .reserve(graph_view.number_of_local_edge_partitions());
  }

  {
    auto aggregate_local_frontier_local_degrees =
      (minor_comm_size > 1)
        ? std::make_optional<rmm::device_uvector<edge_t>>(
            local_frontier_displacements.back() + local_frontier_sizes.back(), handle.get_stream())
        : std::nullopt;

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

      vertex_t const* edge_partition_frontier_major_first{nullptr};

      auto edge_partition_frontier_key_first =
        ((minor_comm_size > 1) ? get_dataframe_buffer_begin(*aggregate_local_frontier)
                               : frontier.begin()) +
        local_frontier_displacements[i];
      if constexpr (std::is_same_v<key_t, vertex_t>) {
        edge_partition_frontier_major_first = edge_partition_frontier_key_first;
      } else {
        edge_partition_frontier_major_first = thrust::get<0>(edge_partition_frontier_key_first);
      }

      auto edge_partition_frontier_local_degrees =
        edge_partition_e_mask ? edge_partition.compute_local_degrees_with_mask(
                                  (*edge_partition_e_mask).value_first(),
                                  edge_partition_frontier_major_first,
                                  edge_partition_frontier_major_first + local_frontier_sizes[i],
                                  handle.get_stream())
                              : edge_partition.compute_local_degrees(
                                  edge_partition_frontier_major_first,
                                  edge_partition_frontier_major_first + local_frontier_sizes[i],
                                  handle.get_stream());

      if (minor_comm_size > 1) {
        // FIXME: this copy is unnecessary if edge_partition.compute_local_degrees() takes a pointer
        // to the output array
        thrust::copy(
          handle.get_thrust_policy(),
          edge_partition_frontier_local_degrees.begin(),
          edge_partition_frontier_local_degrees.end(),
          (*aggregate_local_frontier_local_degrees).begin() + local_frontier_displacements[i]);
      } else {
        frontier_degrees = std::move(edge_partition_frontier_local_degrees);
      }

      if (edge_partition_e_mask) {
        (*local_frontier_valid_local_nbr_count_inclusive_sums)
          .push_back(compute_valid_local_nbr_count_inclusive_sums(
            handle,
            edge_partition,
            *edge_partition_e_mask,
            raft::device_span<vertex_t const>(edge_partition_frontier_major_first,
                                              local_frontier_sizes[i])));
      }
    }

    if (minor_comm_size > 1) {
      auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());

      rmm::device_uvector<edge_t> frontier_gathered_local_degrees(0, handle.get_stream());
      std::tie(frontier_gathered_local_degrees, std::ignore) =
        shuffle_values(minor_comm,
                       (*aggregate_local_frontier_local_degrees).begin(),
                       local_frontier_sizes,
                       handle.get_stream());
      aggregate_local_frontier_local_degrees = std::nullopt;
      frontier_partitioned_local_degree_displacements =
        rmm::device_uvector<edge_t>(frontier_degrees.size() * minor_comm_size, handle.get_stream());
      thrust::for_each(
        handle.get_thrust_policy(),
        thrust::make_counting_iterator(size_t{0}),
        thrust::make_counting_iterator(frontier_degrees.size()),
        compute_local_degree_displacements_and_global_degree_t<edge_t>{
          raft::device_span<edge_t const>(frontier_gathered_local_degrees.data(),
                                          frontier_gathered_local_degrees.size()),
          raft::device_span<edge_t>((*frontier_partitioned_local_degree_displacements).data(),
                                    (*frontier_partitioned_local_degree_displacements).size()),
          raft::device_span<edge_t>(frontier_degrees.data(), frontier_degrees.size()),
          minor_comm_size});
    }
  }

  rmm::device_uvector<edge_t> sample_nbr_indices(0, handle.get_stream());

  // BIAS: indices or random number
  // 4. shuffle randomly selected indices

  auto sample_local_nbr_indices = std::move(
    sample_nbr_indices);  // neighbor index within an edge partition (note that each vertex's
                          // neighbors are distributed in minor_comm_size partitions)
  std::optional<rmm::device_uvector<size_t>> sample_key_indices{
    std::nullopt};  // relevant only when (minor_comm_size > 1)
  std::vector<size_t> local_frontier_sample_offsets{};
  if (minor_comm_size > 1) {
    auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());

    sample_key_indices =
      rmm::device_uvector<size_t>(sample_local_nbr_indices.size(), handle.get_stream());
    auto minor_comm_ranks =
      rmm::device_uvector<int>(sample_local_nbr_indices.size(), handle.get_stream());
    auto intra_partition_displacements =
      rmm::device_uvector<size_t>(sample_local_nbr_indices.size(), handle.get_stream());
    rmm::device_uvector<size_t> d_tx_counts(minor_comm_size, handle.get_stream());
    thrust::fill(handle.get_thrust_policy(), d_tx_counts.begin(), d_tx_counts.end(), size_t{0});
    auto input_pair_first = thrust::make_zip_iterator(
      thrust::make_tuple(sample_local_nbr_indices.begin(),
                         thrust::make_transform_iterator(thrust::make_counting_iterator(size_t{0}),
                                                         divider_t<size_t>{K})));
    thrust::transform(
      handle.get_thrust_policy(),
      input_pair_first,
      input_pair_first + sample_local_nbr_indices.size(),
      thrust::make_zip_iterator(thrust::make_tuple(minor_comm_ranks.begin(),
                                                   intra_partition_displacements.begin(),
                                                   sample_local_nbr_indices.begin(),
                                                   (*sample_key_indices).begin())),
      convert_pair_to_quadruplet_t<edge_t>{
        raft::device_span<edge_t const>((*frontier_partitioned_local_degree_displacements).data(),
                                        (*frontier_partitioned_local_degree_displacements).size()),
        raft::device_span<size_t>(d_tx_counts.data(), d_tx_counts.size()),
        frontier.size(),
        minor_comm_size,
        cugraph::ops::graph::INVALID_ID<edge_t>});
    rmm::device_uvector<size_t> tx_displacements(minor_comm_size, handle.get_stream());
    thrust::exclusive_scan(
      handle.get_thrust_policy(), d_tx_counts.begin(), d_tx_counts.end(), tx_displacements.begin());
    auto tmp_sample_local_nbr_indices =
      rmm::device_uvector<edge_t>(tx_displacements.back_element(handle.get_stream()) +
                                    d_tx_counts.back_element(handle.get_stream()),
                                  handle.get_stream());
    auto tmp_sample_key_indices =
      rmm::device_uvector<size_t>(tmp_sample_local_nbr_indices.size(), handle.get_stream());
    auto pair_first = thrust::make_zip_iterator(
      thrust::make_tuple(sample_local_nbr_indices.begin(), (*sample_key_indices).begin()));
    thrust::scatter_if(
      handle.get_thrust_policy(),
      pair_first,
      pair_first + sample_local_nbr_indices.size(),
      thrust::make_transform_iterator(
        thrust::make_counting_iterator(size_t{0}),
        shuffle_index_compute_offset_t{
          raft::device_span<int const>(minor_comm_ranks.data(), minor_comm_ranks.size()),
          raft::device_span<size_t const>(intra_partition_displacements.data(),
                                          intra_partition_displacements.size()),
          raft::device_span<size_t const>(tx_displacements.data(), tx_displacements.size())}),
      minor_comm_ranks.begin(),
      thrust::make_zip_iterator(
        thrust::make_tuple(tmp_sample_local_nbr_indices.begin(), tmp_sample_key_indices.begin())),
      is_not_equal_t<int>{-1});

    sample_local_nbr_indices = std::move(tmp_sample_local_nbr_indices);
    sample_key_indices       = std::move(tmp_sample_key_indices);

    std::vector<size_t> h_tx_counts(d_tx_counts.size());
    raft::update_host(
      h_tx_counts.data(), d_tx_counts.data(), d_tx_counts.size(), handle.get_stream());
    handle.sync_stream();

    pair_first = thrust::make_zip_iterator(
      thrust::make_tuple(sample_local_nbr_indices.begin(), (*sample_key_indices).begin()));
    auto [rx_value_buffer, rx_counts] =
      shuffle_values(minor_comm, pair_first, h_tx_counts, handle.get_stream());

    sample_local_nbr_indices         = std::move(std::get<0>(rx_value_buffer));
    sample_key_indices               = std::move(std::get<1>(rx_value_buffer));
    local_frontier_sample_offsets    = std::vector<size_t>(rx_counts.size() + 1);
    local_frontier_sample_offsets[0] = size_t{0};
    std::inclusive_scan(
      rx_counts.begin(), rx_counts.end(), local_frontier_sample_offsets.begin() + 1);
  } else {
    local_frontier_sample_offsets = std::vector<size_t>{size_t{0}, frontier.size() * K};
  }

  return std::make_tuple(std::move(sample_local_nbr_indices),
                         std::move(sample_key_indices),
                         std::move(local_frontier_sample_offsets));
#else
  CUGRAPH_FAIL("unimplemented.");
  return std::make_tuple(
    rmm::device_uvector<edge_t>(0, handle.get_stream()), std::nullopt, std::vector<size_t>{});
#endif
}
#endif

}  // namespace detail

}  // namespace cugraph
