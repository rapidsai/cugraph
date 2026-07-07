/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/edge_partition_edge_property_device_view.cuh>
#include <cugraph/edge_partition_endpoint_property_device_view.cuh>
#include <cugraph/export.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/prims/detail/heterogeneous_biased_sample.cuh>
#include <cugraph/prims/detail/homogeneous_biased_sample.cuh>
#include <cugraph/prims/detail/partition_v_frontier.cuh>
#include <cugraph/prims/detail/sampling_helpers.cuh>
#include <cugraph/prims/detail/transform_v_frontier_e.cuh>
#include <cugraph/prims/detail/uniform_and_biased_replacement_sample.cuh>
#include <cugraph/prims/detail/uniform_sampling_index.cuh>
#include <cugraph/prims/property_op_utils.cuh>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/device_functors.cuh>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/iterator_utils.hpp>
#include <cugraph/utilities/mask_utils.cuh>
#include <cugraph/utilities/misc_utils.cuh>
#include <cugraph/utilities/shuffle_comm.cuh>
#include <cugraph/utilities/thrust_wrappers/fill.hpp>
#include <cugraph/utilities/thrust_wrappers/gather.hpp>
#include <cugraph/utilities/thrust_wrappers/scan.hpp>
#include <cugraph/utilities/thrust_wrappers/sequence.hpp>
#include <cugraph/utilities/thrust_wrappers/sort.hpp>
#include <cugraph/utilities/thrust_wrappers/unique.hpp>
#include <cugraph/vertex_partition_device_view.cuh>

#include <raft/random/rng.cuh>

#include <cub/cub.cuh>
#include <cuda/atomic>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/cmath>
#include <cuda/std/optional>
#include <cuda/std/tuple>
#include <thrust/adjacent_difference.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>
#include <thrust/unique.h>

#include <optional>
#include <tuple>
#include <type_traits>

namespace CUGRAPH_EXPORT cugraph {

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
    auto nbr_value       = cuda::std::get<0>(pair);
    auto key_idx         = cuda::std::get<1>(pair);
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
      cuda::std::make_tuple(minor_comm_rank, intra_partition_offset, local_nbr_value, key_idx);
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
    auto per_type_nbr_value       = cuda::std::get<0>(pair);
    auto idx                      = cuda::std::get<1>(pair);
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
    *(output_tuple_first + i) = cuda::std::make_tuple(
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
  cuda::std::tuple<raft::device_span<size_t const>, raft::device_span<edge_t const>>
    unique_major_valid_local_nbr_count_inclusive_sums{};

  __device__ edge_t operator()(cuda::std::tuple<edge_t, size_t> pair) const
  {
    edge_t local_nbr_idx    = cuda::std::get<0>(pair);
    size_t major_idx        = cuda::std::get<1>(pair);
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
          cuda::std::tie(indices, edge_offset, local_degree) = edge_partition.local_edges(
            edge_partition.major_offset_from_major_nocheck(*major_hypersparse_first) +
            *major_hypersparse_idx);
        }
      } else {
        cuda::std::tie(indices, edge_offset, local_degree) =
          edge_partition.local_edges(major_offset);
      }
    } else {
      cuda::std::tie(indices, edge_offset, local_degree) = edge_partition.local_edges(major_offset);
    }

    if (local_degree < compute_valid_local_nbr_count_inclusive_sum_local_degree_threshold) {
      local_nbr_idx = find_nth_set_bits(
        (*edge_partition_e_mask).value_first(), edge_offset, local_degree, local_nbr_idx + 1);
    } else {
      auto inclusive_sum_first =
        cuda::std::get<1>(unique_major_valid_local_nbr_count_inclusive_sums).begin();
      auto start_offset =
        cuda::std::get<0>(unique_major_valid_local_nbr_count_inclusive_sums)[unique_major_idx];
      auto end_offset =
        cuda::std::get<0>(unique_major_valid_local_nbr_count_inclusive_sums)[unique_major_idx + 1];
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
    cugraph::sort(handle.get_thrust_policy(),
                  get_dataframe_buffer_begin(tmp_keys) + local_frontier_offsets[i],
                  get_dataframe_buffer_begin(tmp_keys) + local_frontier_offsets[i + 1]);
    local_frontier_unique_key_sizes[i] = cuda::std::distance(
      get_dataframe_buffer_begin(tmp_keys) + local_frontier_offsets[i],
      cugraph::unique(handle.get_thrust_policy(),
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

    auto const frontier_partition_size = local_frontier_offsets[i + 1] - local_frontier_offsets[i];
    auto edge_partition_major_first =
      aggregate_local_frontier_major_first + local_frontier_offsets[i];

    auto edge_partition_local_degrees = [&]() {
      if constexpr (is_arithmetic_pointer_v<std::decay_t<decltype(edge_partition_major_first)>>) {
        auto const majors_span = raft::device_span<vertex_t const>(
          edge_partition_major_first, static_cast<size_t>(frontier_partition_size));
        return edge_partition.compute_local_degrees(majors_span, handle.get_stream());
      } else {
        return edge_partition.compute_local_degrees(
          edge_partition_major_first,
          edge_partition_major_first + frontier_partition_size,
          handle.get_stream());
      }
    }();
    auto inclusive_sum_offsets = rmm::device_uvector<size_t>(
      (local_frontier_offsets[i + 1] - local_frontier_offsets[i]) + 1, handle.get_stream());
    inclusive_sum_offsets.set_element_to_zero_async(0, handle.get_stream());
    auto size_first = cuda::make_transform_iterator(
      edge_partition_local_degrees.begin(),
      cuda::proclaim_return_type<size_t>([] __device__(edge_t local_degree) {
        return static_cast<size_t>((local_degree + packed_bools_per_word() - 1) /
                                   packed_bools_per_word());
      }));
    cugraph::inclusive_scan(handle.get_thrust_policy(),
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

    auto const frontier_partition_size = local_frontier_offsets[i + 1] - local_frontier_offsets[i];
    auto edge_partition_major_first =
      aggregate_local_frontier_major_first + local_frontier_offsets[i];

    auto edge_partition_frontier_local_degrees = [&]() {
      if constexpr (is_arithmetic_pointer_v<std::decay_t<decltype(edge_partition_major_first)>>) {
        auto const majors_span = raft::device_span<vertex_t const>(
          edge_partition_major_first, static_cast<size_t>(frontier_partition_size));
        return !edge_partition_e_mask
                 ? edge_partition.compute_local_degrees(majors_span, handle.get_stream())
                 : edge_partition.compute_local_degrees_with_mask(
                     raft::device_span<uint32_t const>(
                       (*edge_partition_e_mask).value_first(),
                       packed_bool_size(static_cast<size_t>(edge_partition.number_of_edges()))),
                     majors_span,
                     handle.get_stream());
      } else {
        return !edge_partition_e_mask
                 ? edge_partition.compute_local_degrees(
                     edge_partition_major_first,
                     edge_partition_major_first + frontier_partition_size,
                     handle.get_stream())
                 : edge_partition.compute_local_degrees_with_mask(
                     raft::device_span<uint32_t const>(
                       (*edge_partition_e_mask).value_first(),
                       packed_bool_size(static_cast<size_t>(edge_partition.number_of_edges()))),
                     edge_partition_major_first,
                     edge_partition_major_first + frontier_partition_size,
                     handle.get_stream());
      }
    }();

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
             typename edge_op_result_type<GraphViewType,
                                          typename thrust::iterator_traits<KeyIterator>::value_type,
                                          EdgeSrcValueInputWrapper,
                                          EdgeDstValueInputWrapper,
                                          EdgeValueInputWrapper,
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

  using bias_t = typename edge_op_result_type<GraphViewType,
                                              key_t,
                                              EdgeSrcValueInputWrapper,
                                              EdgeDstValueInputWrapper,
                                              EdgeValueInputWrapper,
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

  rmm::device_uvector<size_t> aggregate_local_frontier_local_degrees(
    local_frontier_offsets.back(),
    handle.get_stream());  // excluding 0 bias neighbors
  {
    thrust::adjacent_difference(handle.get_thrust_policy(),
                                aggregate_local_frontier_local_degree_offsets.begin() + 1,
                                aggregate_local_frontier_local_degree_offsets.end(),
                                aggregate_local_frontier_local_degrees.begin());

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
                       auto bias = cuda::std::get<0>(pair);
                       if (bias == 0.0) {
                         auto i   = cuda::std::get<1>(pair);
                         auto idx = cuda::std::distance(
                           offsets.begin() + 1,
                           thrust::upper_bound(thrust::seq, offsets.begin() + 1, offsets.end(), i));
                         cuda::atomic_ref<size_t, cuda::thread_scope_device> degree(degrees[idx]);
                         degree.fetch_sub(size_t{1}, cuda::std::memory_order_relaxed);
                       }
                     });
  }

  auto num_nz_bias_nbrs = thrust::reduce(handle.get_thrust_policy(),
                                         aggregate_local_frontier_local_degrees.begin(),
                                         aggregate_local_frontier_local_degrees.end());

  rmm::device_uvector<edge_t> aggregate_local_frontier_nz_bias_indices(num_nz_bias_nbrs,
                                                                       handle.get_stream());
  {
    auto nz_biases  = rmm::device_uvector<bias_t>(num_nz_bias_nbrs, handle.get_stream());
    auto pair_first = thrust::make_zip_iterator(
      aggregate_local_frontier_biases.begin(),
      cuda::make_transform_iterator(
        thrust::make_counting_iterator(size_t{0}),
        cuda::proclaim_return_type<edge_t>(
          [offsets = raft::device_span<size_t const>(
             aggregate_local_frontier_local_degree_offsets.data(),
             aggregate_local_frontier_local_degree_offsets.size())] __device__(size_t i) {
            auto idx = cuda::std::distance(
              offsets.begin() + 1,
              thrust::upper_bound(thrust::seq, offsets.begin() + 1, offsets.end(), i));
            return static_cast<edge_t>(i - offsets[idx]);
          })));
    thrust::copy_if(handle.get_thrust_policy(),
                    pair_first,
                    pair_first + aggregate_local_frontier_biases.size(),
                    aggregate_local_frontier_biases.begin(),
                    thrust::make_zip_iterator(nz_biases.begin(),
                                              aggregate_local_frontier_nz_bias_indices.begin()),
                    cuda::proclaim_return_type<bool>([] __device__(bias_t b) { return b != 0.0; }));
    aggregate_local_frontier_biases = std::move(nz_biases);
  }

  cugraph::inclusive_scan(handle.get_thrust_policy(),
                          aggregate_local_frontier_local_degrees.begin(),
                          aggregate_local_frontier_local_degrees.end(),
                          aggregate_local_frontier_local_degree_offsets.begin() + 1);

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
             typename edge_op_result_type<GraphViewType,
                                          typename thrust::iterator_traits<KeyIterator>::value_type,
                                          EdgeSrcValueInputWrapper,
                                          EdgeDstValueInputWrapper,
                                          EdgeValueInputWrapper,
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
  using bias_t       = typename edge_op_result_type<GraphViewType,
                                                    key_t,
                                                    EdgeSrcValueInputWrapper,
                                                    EdgeDstValueInputWrapper,
                                                    EdgeValueInputWrapper,
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
        cuda::proclaim_return_type<cuda::std::tuple<bias_t, edge_type_t>>(
          [bias_e_op] __device__(auto src, auto dst, auto src_val, auto dst_val, auto e_val) {
            return cuda::std::make_tuple(bias_e_op(src, dst, src_val, dst_val, cuda::std::nullopt),
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
        cuda::proclaim_return_type<cuda::std::tuple<bias_t, edge_type_t>>(
          [bias_e_op] __device__(auto src, auto dst, auto src_val, auto dst_val, auto e_val) {
            using tuple_type          = decltype(e_val);
            auto constexpr tuple_size = cuda::std::tuple_size<tuple_type>::value;
            edge_value_t bias_e_op_e_val{};
            if constexpr (std::is_arithmetic_v<edge_value_t>) {
              bias_e_op_e_val = cuda::std::get<0>(e_val);
            } else {
              bias_e_op_e_val = thrust_tuple_slice<tuple_type, size_t{0}, tuple_size - 1>(e_val);
            }
            return cuda::std::make_tuple(bias_e_op(src, dst, src_val, dst_val, bias_e_op_e_val),
                                         cuda::std::get<tuple_size - 1>(e_val));
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
                       auto bias = cuda::std::get<0>(pair);
                       if (bias == 0.0) {
                         auto i = cuda::std::get<1>(pair);
                         auto it =
                           thrust::upper_bound(thrust::seq, offsets.begin() + 1, offsets.end(), i);
                         auto idx = cuda::std::distance(offsets.begin() + 1, it);
                         cuda::atomic_ref<size_t, cuda::thread_scope_device> degree(degrees[idx]);
                         degree.fetch_sub(size_t{1}, cuda::std::memory_order_relaxed);
                       }
                     });
  }

  cugraph::inclusive_scan(handle.get_thrust_policy(),
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
                        [] __device__(auto triplet) { return cuda::std::get<0>(triplet) == 0.0; });
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
  cugraph::fill(handle.get_thrust_policy(), d_tx_counts.begin(), d_tx_counts.end(), size_t{0});
  auto input_pair_first = thrust::make_zip_iterator(
    sample_local_nbr_values.begin(),
    cuda::make_transform_iterator(thrust::make_counting_iterator(size_t{0}), divider_t<size_t>{K}));
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
  cugraph::exclusive_scan(
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
    cuda::make_transform_iterator(
      thrust::make_counting_iterator(size_t{0}),
      shuffle_index_compute_offset_t{
        raft::device_span<int const>(minor_comm_ranks.data(), minor_comm_ranks.size()),
        raft::device_span<size_t const>(intra_partition_displacements.data(),
                                        intra_partition_displacements.size()),
        raft::device_span<size_t const>(tx_displacements.data(), tx_displacements.size())}),
    minor_comm_ranks.begin(),
    thrust::make_zip_iterator(
      cuda::std::make_tuple(tmp_sample_local_nbr_values.begin(), tmp_key_indices.begin())),
    is_not_equal_t<int>{-1});

  sample_local_nbr_values = std::move(tmp_sample_local_nbr_values);
  key_indices             = std::move(tmp_key_indices);

  std::vector<size_t> h_tx_counts(d_tx_counts.size());
  raft::update_host(
    h_tx_counts.data(), d_tx_counts.data(), d_tx_counts.size(), handle.get_stream());
  handle.sync_stream();

  pair_first = thrust::make_zip_iterator(
    cuda::std::make_tuple(sample_local_nbr_values.begin(), key_indices.begin()));
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
  cugraph::fill(handle.get_thrust_policy(), d_tx_counts.begin(), d_tx_counts.end(), size_t{0});
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
  cugraph::exclusive_scan(
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
    cuda::make_transform_iterator(
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

  auto sample_major_idx_first = cuda::make_transform_iterator(
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
        cuda::std::make_tuple(
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

  int minor_comm_size{1};
  if constexpr (GraphViewType::is_multi_gpu) {
    auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
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

      auto offset_first = cuda::make_transform_iterator(
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
        cugraph::sequence(handle.get_thrust_policy(),
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
      cugraph::inclusive_scan(
        handle.get_thrust_policy(),
        aggregate_local_frontier_unique_key_per_type_local_degrees.begin(),
        aggregate_local_frontier_unique_key_per_type_local_degrees.end(),
        aggregate_local_frontier_unique_key_per_type_local_degree_offsets.begin() + 1,
        size_t{0},
        converting_plus_t<edge_t, size_t>{});
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

  using bias_t      = typename edge_op_result_type<GraphViewType,
                                                   key_t,
                                                   EdgeSrcValueInputWrapper,
                                                   EdgeDstValueInputWrapper,
                                                   EdgeValueInputWrapper,
                                                   BiasEdgeOp>::type;
  using edge_type_t = int32_t;  // dummy

  int minor_comm_size{1};
  if constexpr (GraphViewType::is_multi_gpu) {
    auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
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
            auto nz_bias_idx    = cuda::std::get<0>(pair);
            auto key_idx        = cuda::std::get<1>(pair);
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
            auto nz_bias_idx    = cuda::std::get<0>(pair);
            auto key_idx        = cuda::std::get<1>(pair) / K;
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

  using bias_t      = typename edge_op_result_type<GraphViewType,
                                                   key_t,
                                                   EdgeSrcValueInputWrapper,
                                                   EdgeDstValueInputWrapper,
                                                   EdgeValueInputWrapper,
                                                   BiasEdgeOp>::type;
  using edge_type_t = typename EdgeTypeInputWrapper::value_type;

  int minor_comm_size{1};
  if constexpr (GraphViewType::is_multi_gpu) {
    auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
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
      cugraph::sequence(handle.get_thrust_policy(), sequences.begin(), sequences.end(), size_t{0});
      rmm::device_uvector<size_t> segment_sorted_sequences(h_nbr_offsets[i + 1] - h_nbr_offsets[i],
                                                           handle.get_stream());

      auto offset_first = cuda::make_transform_iterator(
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
      cugraph::gather(
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
        cugraph::sequence(handle.get_thrust_policy(),
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
      cugraph::inclusive_scan(
        handle.get_thrust_policy(),
        aggregate_local_frontier_unique_key_per_type_local_degrees.begin(),
        aggregate_local_frontier_unique_key_per_type_local_degrees.end(),
        aggregate_local_frontier_unique_key_per_type_local_degree_offsets.begin() + 1,
        size_t{0},
        converting_plus_t<edge_t, size_t>{});
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

}  // namespace CUGRAPH_EXPORT cugraph
