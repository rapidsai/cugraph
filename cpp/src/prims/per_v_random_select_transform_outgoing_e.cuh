/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <prims/property_op_utils.cuh>

#include <cugraph/edge_partition_device_view.cuh>
#include <cugraph/edge_partition_edge_property_device_view.cuh>
#include <cugraph/edge_partition_endpoint_property_device_view.cuh>
#include <cugraph/partition_manager.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/host_scalar_comm.hpp>
#include <cugraph/utilities/misc_utils.cuh>
#include <cugraph/utilities/shuffle_comm.cuh>

#include <raft/random/rng.cuh>
#ifndef NO_CUGRAPH_OPS
#include <cugraph-ops/graph/sampling.hpp>
#endif

#include <cub/cub.cuh>
#include <cuda/atomic>
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

template <typename edge_t>
struct compute_local_degree_displacements_and_global_degree_t {
  raft::device_span<edge_t const> gathered_local_degrees{};
  raft::device_span<edge_t> segmented_local_degree_displacements{};
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
        segmented_local_degree_displacements.begin() + i * minor_comm_size + round * buffer_size);
    }
    global_degrees[i] = sum;
  }
};

// convert a (neighbor index, key index) pair  to a (minor_comm_rank, intra-partition offset,
// neighbor index, key index) quadruplet, minor_comm_rank is set to -1 if an neighbor index is
// invalid
template <typename edge_t>
struct convert_pair_to_quadruplet_t {
  raft::device_span<edge_t const> segmented_local_degree_displacements{};
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
        segmented_local_degree_displacements.begin() + key_idx * minor_comm_size;
      minor_comm_rank =
        static_cast<int>(thrust::distance(
          displacement_first,
          thrust::upper_bound(
            thrust::seq, displacement_first, displacement_first + minor_comm_size, nbr_idx))) -
        1;
      local_nbr_idx -= *(displacement_first + minor_comm_rank);
      cuda::std::atomic_ref<size_t> counter(tx_counts[minor_comm_rank]);
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

template <typename edge_t, typename T>
struct check_invalid_t {
  edge_t invalid_idx{};

  __device__ bool operator()(thrust::tuple<edge_t, T> pair) const
  {
    return thrust::get<0>(pair) == invalid_idx;
  }
};

template <typename edge_t>
struct invalid_minor_comm_rank_t {
  int invalid_minor_comm_rank{};
  __device__ bool operator()(thrust::tuple<edge_t, int, size_t> triplet) const
  {
    return thrust::get<1>(triplet) == invalid_minor_comm_rank;
  }
};

template <typename GraphViewType,
          typename KeyIterator,
          typename LocalNbrIdxIterator,
          typename OutputValueIterator,
          typename EdgePartitionSrcValueInputWrapper,
          typename EdgePartitionDstValueInputWrapper,
          typename EdgePartitionEdgeValueInputWrapper,
          typename EdgeOp,
          typename T>
struct transform_local_nbr_indices_t {
  using key_t    = typename thrust::iterator_traits<KeyIterator>::value_type;
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu> edge_partition{};
  thrust::optional<size_t const*> local_key_indices{thrust::nullopt};
  KeyIterator key_first{};
  LocalNbrIdxIterator local_nbr_idx_first{};
  OutputValueIterator output_value_first{};
  EdgePartitionSrcValueInputWrapper edge_partition_src_value_input;
  EdgePartitionDstValueInputWrapper edge_partition_dst_value_input;
  EdgePartitionEdgeValueInputWrapper edge_partition_e_value_input;
  EdgeOp e_op{};
  edge_t invalid_idx{};
  thrust::optional<T> invalid_value{thrust::nullopt};
  size_t K{};

  __device__ void operator()(size_t i) const
  {
    auto key_idx = local_key_indices ? (*local_key_indices)[i] : (i / K);
    auto key     = *(key_first + key_idx);
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
    auto local_nbr_idx = *(local_nbr_idx_first + i);
    if (local_nbr_idx != invalid_idx) {
      auto minor        = indices[local_nbr_idx];
      auto minor_offset = edge_partition.minor_offset_from_minor_nocheck(minor);

      std::conditional_t<GraphViewType::is_storage_transposed, vertex_t, key_t>
        key_or_src{};  // key if major
      std::conditional_t<GraphViewType::is_storage_transposed, key_t, vertex_t>
        key_or_dst{};  // key if major
      if constexpr (GraphViewType::is_storage_transposed) {
        key_or_src = minor;
        key_or_dst = key;
      } else {
        key_or_src = key;
        key_or_dst = minor;
      }
      auto src_offset = GraphViewType::is_storage_transposed ? minor_offset : major_offset;
      auto dst_offset = GraphViewType::is_storage_transposed ? major_offset : minor_offset;
      *(output_value_first + i) =
        e_op(key_or_src,
             key_or_dst,
             edge_partition_src_value_input.get(src_offset),
             edge_partition_dst_value_input.get(dst_offset),
             edge_partition_e_value_input.get(edge_offset + local_nbr_idx));
    } else if (invalid_value) {
      *(output_value_first + i) = *invalid_value;
    }
  }
};

template <typename edge_t>
struct count_valids_t {
  raft::device_span<edge_t const> sample_local_nbr_indices{};
  size_t K{};
  edge_t invalid_idx{};

  __device__ int32_t operator()(size_t i) const
  {
    auto first = sample_local_nbr_indices.begin() + i * K;
    return static_cast<int32_t>(
      thrust::distance(first, thrust::find(thrust::seq, first, first + K, invalid_idx)));
  }
};

struct count_t {
  raft::device_span<int32_t> sample_counts{};

  __device__ size_t operator()(size_t key_idx) const
  {
    cuda::std::atomic_ref<int32_t> counter(sample_counts[key_idx]);
    return counter.fetch_add(int32_t{1}, cuda::std::memory_order_relaxed);
  }
};

template <bool use_invalid_value>
struct return_value_compute_offset_t {
  raft::device_span<size_t const> sample_key_indices{};
  raft::device_span<int32_t const> sample_intra_partition_displacements{};
  std::conditional_t<use_invalid_value, size_t, raft::device_span<size_t const>>
    K_or_sample_offsets{};

  __device__ size_t operator()(size_t i) const
  {
    auto key_idx = sample_key_indices[i];
    size_t key_start_offset{};
    if constexpr (use_invalid_value) {
      key_start_offset = key_idx * K_or_sample_offsets;
    } else {
      key_start_offset = K_or_sample_offsets[key_idx];
    }
    return key_start_offset + static_cast<size_t>(sample_intra_partition_displacements[i]);
  }
};

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
  size_t high_partition_over_sampling_K = K * 2;                         // tuning parameter
  assert(high_partition_over_sampling_K > K);

  rmm::device_uvector<edge_t> sample_nbr_indices(frontier_degrees.size() * K, handle.get_stream());

  rmm::device_uvector<size_t> seed_indices(frontier_degrees.size(), handle.get_stream());
  thrust::sequence(handle.get_thrust_policy(), seed_indices.begin(), seed_indices.end(), size_t{0});
  auto low_first =
    thrust::make_zip_iterator(thrust::make_tuple(frontier_degrees.begin(), seed_indices.begin()));
  auto mid_first = thrust::partition(
    handle.get_thrust_policy(),
    low_first,
    low_first + frontier_degrees.size(),
    [K] __device__(auto pair) { return thrust::get<0>(pair) <= static_cast<edge_t>(K); });
  auto low_partition_size = static_cast<size_t>(thrust::distance(low_first, mid_first));
  auto high_first =
    thrust::partition(handle.get_thrust_policy(),
                      mid_first,
                      mid_first + (frontier_degrees.size() - low_partition_size),
                      [mid_partition_degree_range_last] __device__(auto pair) {
                        return thrust::get<0>(pair) < mid_partition_degree_range_last;
                      });
  auto mid_partition_size  = static_cast<size_t>(thrust::distance(mid_first, high_first));
  auto high_partition_size = frontier_degrees.size() - (low_partition_size + mid_partition_size);

  if (low_partition_size > 0) {
    thrust::for_each(handle.get_thrust_policy(),
                     thrust::make_counting_iterator(size_t{0}),
                     thrust::make_counting_iterator(low_partition_size * K),
                     [K,
                      low_first,
                      sample_nbr_indices = sample_nbr_indices.data(),
                      invalid_idx = cugraph::ops::graph::INVALID_ID<edge_t>] __device__(size_t i) {
                       auto pair       = *(low_first + (i / K));
                       auto degree     = thrust::get<0>(pair);
                       auto seed_idx   = thrust::get<1>(pair);
                       auto sample_idx = static_cast<edge_t>(i % K);
                       sample_nbr_indices[seed_idx * K + sample_idx] =
                         (sample_idx < degree) ? sample_idx : invalid_idx;
                     });
  }

  if (mid_partition_size > 0) {
    rmm::device_uvector<edge_t> tmp_sample_nbr_indices(mid_partition_size * K, handle.get_stream());
    // FIXME: we can avoid the follow-up copy if get_sampling_index takes output offsets for
    // sampling output
    cugraph::ops::graph::get_sampling_index(tmp_sample_nbr_indices.data(),
                                            rng_state,
                                            thrust::get<0>(mid_first.get_iterator_tuple()),
                                            mid_partition_size,
                                            static_cast<int32_t>(K),
                                            false,
                                            handle.get_stream());
    thrust::for_each(handle.get_thrust_policy(),
                     thrust::make_counting_iterator(size_t{0}),
                     thrust::make_counting_iterator(mid_partition_size * K),
                     [K,
                      seed_index_first       = thrust::get<1>(mid_first.get_iterator_tuple()),
                      tmp_sample_nbr_indices = tmp_sample_nbr_indices.data(),
                      sample_nbr_indices     = sample_nbr_indices.data()] __device__(size_t i) {
                       auto seed_idx                                 = *(seed_index_first + i / K);
                       auto sample_idx                               = static_cast<edge_t>(i % K);
                       sample_nbr_indices[seed_idx * K + sample_idx] = tmp_sample_nbr_indices[i];
                     });
  }

  if (high_partition_size > 0) {
    // to limit memory footprint ((1 << 20) is a tuning parameter), std::max for forward progress
    // guarantee when high_partition_over_sampling_K is exorbitantly large
    auto seeds_to_sort_per_iteration =
      std::max(static_cast<size_t>(handle.get_device_properties().multiProcessorCount * (1 << 20)) /
                 high_partition_over_sampling_K,
               size_t{1});

    rmm::device_uvector<edge_t> tmp_sample_nbr_indices(
      seeds_to_sort_per_iteration * high_partition_over_sampling_K, handle.get_stream());
    assert(high_partition_over_sampling_K * 2 <=
           static_cast<size_t>(std::numeric_limits<int32_t>::max()));
    rmm::device_uvector<int32_t> tmp_sample_indices(
      seeds_to_sort_per_iteration * high_partition_over_sampling_K,
      handle.get_stream());  // sample indices within a segment (one segment per seed)

    rmm::device_uvector<edge_t> segment_sorted_tmp_sample_nbr_indices(
      seeds_to_sort_per_iteration * high_partition_over_sampling_K, handle.get_stream());
    rmm::device_uvector<int32_t> segment_sorted_tmp_sample_indices(
      seeds_to_sort_per_iteration * high_partition_over_sampling_K, handle.get_stream());

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
        auto segment_degree_first =
          thrust::get<0>(high_first.get_iterator_tuple()) + seeds_to_sort_per_iteration * i;

        if (retry_segment_indices) {
          retry_degrees =
            rmm::device_uvector<edge_t>((*retry_segment_indices).size(), handle.get_stream());
          thrust::transform(
            handle.get_thrust_policy(),
            (*retry_segment_indices).begin(),
            (*retry_segment_indices).end(),
            (*retry_degrees).begin(),
            indirection_t<size_t, decltype(segment_degree_first)>{segment_degree_first});
          retry_sample_nbr_indices = rmm::device_uvector<edge_t>(
            (*retry_segment_indices).size() * high_partition_over_sampling_K, handle.get_stream());
          retry_sample_indices = rmm::device_uvector<int32_t>(
            (*retry_segment_indices).size() * high_partition_over_sampling_K, handle.get_stream());
          retry_segment_sorted_sample_nbr_indices = rmm::device_uvector<edge_t>(
            (*retry_segment_indices).size() * high_partition_over_sampling_K, handle.get_stream());
          retry_segment_sorted_sample_indices = rmm::device_uvector<int32_t>(
            (*retry_segment_indices).size() * high_partition_over_sampling_K, handle.get_stream());
        }

        cugraph::ops::graph::get_sampling_index(
          retry_segment_indices ? (*retry_sample_nbr_indices).data()
                                : tmp_sample_nbr_indices.data(),
          rng_state,
          retry_segment_indices ? (*retry_degrees).begin() : segment_degree_first,
          retry_segment_indices ? (*retry_degrees).size() : num_segments,
          static_cast<int32_t>(high_partition_over_sampling_K),
          true,
          handle.get_stream());

        if (retry_segment_indices) {
          thrust::for_each(
            handle.get_thrust_policy(),
            thrust::make_counting_iterator(size_t{0}),
            thrust::make_counting_iterator((*retry_segment_indices).size() *
                                           high_partition_over_sampling_K),
            [high_partition_over_sampling_K,
             unique_counts                         = unique_counts.data(),
             segment_sorted_tmp_sample_nbr_indices = segment_sorted_tmp_sample_nbr_indices.data(),
             retry_segment_indices                 = (*retry_segment_indices).data(),
             retry_sample_nbr_indices              = (*retry_sample_nbr_indices).data(),
             retry_sample_indices = (*retry_sample_indices).data()] __device__(size_t i) {
              auto segment_idx  = retry_segment_indices[i / high_partition_over_sampling_K];
              auto sample_idx   = static_cast<edge_t>(i % high_partition_over_sampling_K);
              auto unique_count = unique_counts[segment_idx];
              auto output_first = thrust::make_zip_iterator(
                thrust::make_tuple(retry_sample_nbr_indices, retry_sample_indices));
              // sample index for the previously selected neighbor indices should be smaller than
              // the new candidates to ensure that the previously selected neighbor indices will be
              // selected again
              if (sample_idx < unique_count) {
                *(output_first + i) =
                  thrust::make_tuple(segment_sorted_tmp_sample_nbr_indices
                                       [segment_idx * high_partition_over_sampling_K + sample_idx],
                                     static_cast<int32_t>(sample_idx));
              } else {
                *(output_first + i) =
                  thrust::make_tuple(retry_sample_nbr_indices[i],
                                     high_partition_over_sampling_K + (sample_idx - unique_count));
              }
            });
        } else {
          thrust::tabulate(
            handle.get_thrust_policy(),
            tmp_sample_indices.begin(),
            tmp_sample_indices.begin() + num_segments * high_partition_over_sampling_K,
            [high_partition_over_sampling_K] __device__(size_t i) {
              return static_cast<int32_t>(i % high_partition_over_sampling_K);
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
            high_partition_over_sampling_K,
          retry_segment_indices ? (*retry_segment_indices).size() : num_segments,
          thrust::make_transform_iterator(thrust::make_counting_iterator(size_t{0}),
                                          multiplier_t<size_t>{high_partition_over_sampling_K}),
          thrust::make_transform_iterator(thrust::make_counting_iterator(size_t{1}),
                                          multiplier_t<size_t>{high_partition_over_sampling_K}),
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
            high_partition_over_sampling_K,
          retry_segment_indices ? (*retry_segment_indices).size() : num_segments,
          thrust::make_transform_iterator(thrust::make_counting_iterator(size_t{0}),
                                          multiplier_t<size_t>{high_partition_over_sampling_K}),
          thrust::make_transform_iterator(thrust::make_counting_iterator(size_t{1}),
                                          multiplier_t<size_t>{high_partition_over_sampling_K}),
          handle.get_stream());

        // count the number of unique neighbor indices

        if (retry_segment_indices) {
          thrust::for_each(
            handle.get_thrust_policy(),
            thrust::make_counting_iterator(size_t{0}),
            thrust::make_counting_iterator((*retry_segment_indices).size()),
            [high_partition_over_sampling_K,
             unique_counts                   = unique_counts.data(),
             retry_segment_indices           = (*retry_segment_indices).data(),
             retry_segment_sorted_pair_first = thrust::make_zip_iterator(
               thrust::make_tuple((*retry_segment_sorted_sample_nbr_indices).begin(),
                                  (*retry_segment_sorted_sample_indices).begin())),
             segment_sorted_pair_first = thrust::make_zip_iterator(thrust::make_tuple(
               segment_sorted_tmp_sample_nbr_indices.begin(),
               segment_sorted_tmp_sample_indices.begin()))] __device__(size_t i) {
              auto unique_count          = static_cast<edge_t>(thrust::distance(
                retry_segment_sorted_pair_first + high_partition_over_sampling_K * i,
                thrust::unique(
                  thrust::seq,
                  retry_segment_sorted_pair_first + high_partition_over_sampling_K * i,
                  retry_segment_sorted_pair_first + high_partition_over_sampling_K * (i + 1),
                  [] __device__(auto lhs, auto rhs) {
                    return thrust::get<0>(lhs) == thrust::get<0>(rhs);
                  })));
              auto segment_idx           = retry_segment_indices[i];
              unique_counts[segment_idx] = unique_count;
              thrust::copy(
                thrust::seq,
                retry_segment_sorted_pair_first + high_partition_over_sampling_K * i,
                retry_segment_sorted_pair_first + high_partition_over_sampling_K * i + unique_count,
                segment_sorted_pair_first + high_partition_over_sampling_K * segment_idx);
            });
        } else {
          thrust::tabulate(
            handle.get_thrust_policy(),
            unique_counts.begin(),
            unique_counts.end(),
            [high_partition_over_sampling_K,
             segment_sorted_pair_first = thrust::make_zip_iterator(thrust::make_tuple(
               segment_sorted_tmp_sample_nbr_indices.begin(),
               segment_sorted_tmp_sample_indices.begin()))] __device__(size_t i) {
              return static_cast<edge_t>(thrust::distance(
                segment_sorted_pair_first + high_partition_over_sampling_K * i,
                thrust::unique(thrust::seq,
                               segment_sorted_pair_first + high_partition_over_sampling_K * i,
                               segment_sorted_pair_first + high_partition_over_sampling_K * (i + 1),
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
        num_segments * high_partition_over_sampling_K,
        num_segments,
        thrust::make_transform_iterator(thrust::make_counting_iterator(size_t{0}),
                                        multiplier_t<size_t>{high_partition_over_sampling_K}),
        thrust::make_transform_iterator(
          thrust::make_counting_iterator(size_t{0}),
          [high_partition_over_sampling_K, unique_counts = unique_counts.data()] __device__(
            size_t i) { return i * high_partition_over_sampling_K + unique_counts[i]; }),
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
        num_segments * high_partition_over_sampling_K,
        num_segments,
        thrust::make_transform_iterator(thrust::make_counting_iterator(size_t{0}),
                                        multiplier_t<size_t>{high_partition_over_sampling_K}),
        thrust::make_transform_iterator(
          thrust::make_counting_iterator(size_t{0}),
          [high_partition_over_sampling_K, unique_counts = unique_counts.data()] __device__(
            size_t i) { return i * high_partition_over_sampling_K + unique_counts[i]; }),
        handle.get_stream());

      // copy the neighbor indices back to sample_nbr_indices

      thrust::for_each(
        handle.get_thrust_policy(),
        thrust::make_counting_iterator(size_t{0}),
        thrust::make_counting_iterator(num_segments * K),
        [K,
         high_partition_over_sampling_K,
         seed_indices =
           thrust::get<1>(high_first.get_iterator_tuple()) + seeds_to_sort_per_iteration * i,
         tmp_sample_nbr_indices = tmp_sample_nbr_indices.data(),
         sample_nbr_indices     = sample_nbr_indices.data()] __device__(size_t i) {
          auto seed_idx   = *(seed_indices + i / K);
          auto sample_idx = static_cast<edge_t>(i % K);
          *(sample_nbr_indices + seed_idx * K + sample_idx) =
            *(tmp_sample_nbr_indices + (i / K) * high_partition_over_sampling_K + sample_idx);
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

template <bool incoming,
          typename GraphViewType,
          typename VertexFrontierBucketType,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename EdgeOp,
          typename T>
std::tuple<std::optional<rmm::device_uvector<size_t>>,
           decltype(allocate_dataframe_buffer<T>(size_t{0}, rmm::cuda_stream_view{}))>
per_v_random_select_transform_e(raft::handle_t const& handle,
                                GraphViewType const& graph_view,
                                VertexFrontierBucketType const& frontier,
                                EdgeSrcValueInputWrapper edge_src_value_input,
                                EdgeDstValueInputWrapper edge_dst_value_input,
                                EdgeValueInputWrapper edge_value_input,
                                EdgeOp e_op,
                                raft::random::RngState& rng_state,
                                size_t K,
                                bool with_replacement,
                                std::optional<T> invalid_value,
                                bool do_expensive_check)
{
#ifndef NO_CUGRAPH_OPS
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;
  using key_t    = typename VertexFrontierBucketType::key_type;
  using key_buffer_t =
    decltype(allocate_dataframe_buffer<key_t>(size_t{0}, rmm::cuda_stream_view{}));

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
  static_assert(std::is_same_v<
                typename detail::edge_op_result_type<key_t,
                                                     vertex_t,
                                                     typename EdgeSrcValueInputWrapper::value_type,
                                                     typename EdgeDstValueInputWrapper::value_type,
                                                     typename EdgeValueInputWrapper::value_type,
                                                     EdgeOp>::type,
                T>);

  CUGRAPH_EXPECTS(K >= size_t{1},
                  "Invalid input argument: invalid K, K should be a positive integer.");
  CUGRAPH_EXPECTS(K <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
                  "Invalid input argument: the current implementation expects K to be no larger "
                  "than std::numeric_limits<int32_t>::max().");

  auto minor_comm_size =
    GraphViewType::is_multi_gpu
      ? handle.get_subcomm(cugraph::partition_manager::minor_comm_name()).get_size()
      : int{1};

  if (do_expensive_check) {
    // FIXME: better re-factor this check function?
    vertex_t const* frontier_vertex_first{nullptr};
    vertex_t const* frontier_vertex_last{nullptr};
    if constexpr (std::is_same_v<key_t, vertex_t>) {
      frontier_vertex_first = frontier.begin();
      frontier_vertex_last  = frontier.end();
    } else {
      frontier_vertex_first = thrust::get<0>(frontier.begin().get_iterator_tuple());
      frontier_vertex_last  = thrust::get<0>(frontier.end().get_iterator_tuple());
    }
    auto num_invalid_keys =
      frontier.size() -
      thrust::count_if(handle.get_thrust_policy(),
                       frontier_vertex_first,
                       frontier_vertex_last,
                       check_in_range_t<vertex_t>{graph_view.local_vertex_partition_range_first(),
                                                  graph_view.local_vertex_partition_range_last()});
    if constexpr (GraphViewType::is_multi_gpu) {
      num_invalid_keys = host_scalar_allreduce(
        handle.get_comms(), num_invalid_keys, raft::comms::op_t::SUM, handle.get_stream());
    }
    CUGRAPH_EXPECTS(num_invalid_keys == size_t{0},
                    "Invalid input argument: frontier includes out-of-range keys.");
  }

  auto frontier_key_first = frontier.begin();
  auto frontier_key_last  = frontier.end();

  std::vector<size_t> local_frontier_sizes{};
  if (minor_comm_size > 1) {
    auto& minor_comm     = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    local_frontier_sizes = host_scalar_allgather(
      minor_comm,
      static_cast<size_t>(thrust::distance(frontier_key_first, frontier_key_last)),
      handle.get_stream());
  } else {
    local_frontier_sizes = std::vector<size_t>{static_cast<size_t>(
      static_cast<vertex_t>(thrust::distance(frontier_key_first, frontier_key_last)))};
  }
  std::vector<size_t> local_frontier_displacements(local_frontier_sizes.size());
  std::exclusive_scan(local_frontier_sizes.begin(),
                      local_frontier_sizes.end(),
                      local_frontier_displacements.begin(),
                      size_t{0});

  // 1. aggregate frontier

  auto aggregate_local_frontier_keys =
    (minor_comm_size > 1)
      ? std::make_optional<key_buffer_t>(
          local_frontier_displacements.back() + local_frontier_sizes.back(), handle.get_stream())
      : std::nullopt;
  if (minor_comm_size > 1) {
    auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    device_allgatherv(minor_comm,
                      frontier_key_first,
                      get_dataframe_buffer_begin(*aggregate_local_frontier_keys),
                      local_frontier_sizes,
                      local_frontier_displacements,
                      handle.get_stream());
  }

  // 2. compute degrees

  auto aggregate_local_frontier_local_degrees =
    (minor_comm_size > 1)
      ? std::make_optional<rmm::device_uvector<edge_t>>(
          local_frontier_displacements.back() + local_frontier_sizes.back(), handle.get_stream())
      : std::nullopt;
  rmm::device_uvector<edge_t> frontier_degrees(frontier.size(), handle.get_stream());
  for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
    auto edge_partition =
      edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
        graph_view.local_edge_partition_view(i));

    vertex_t const* edge_partition_frontier_major_first{nullptr};

    auto edge_partition_frontier_key_first =
      ((minor_comm_size > 1) ? get_dataframe_buffer_begin(*aggregate_local_frontier_keys)
                             : frontier_key_first) +
      local_frontier_displacements[i];
    if constexpr (std::is_same_v<key_t, vertex_t>) {
      edge_partition_frontier_major_first = edge_partition_frontier_key_first;
    } else {
      edge_partition_frontier_major_first = thrust::get<0>(edge_partition_frontier_key_first);
    }

    auto edge_partition_frontier_local_degrees = edge_partition.compute_local_degrees(
      raft::device_span<vertex_t const>(edge_partition_frontier_major_first,
                                        local_frontier_sizes[i]),
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
  }

  auto frontier_segmented_local_degree_displacements =
    (minor_comm_size > 1)
      ? std::make_optional<rmm::device_uvector<edge_t>>(size_t{0}, handle.get_stream())
      : std::nullopt;
  if (minor_comm_size > 1) {
    auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());

    rmm::device_uvector<edge_t> frontier_gathered_local_degrees(0, handle.get_stream());
    std::tie(frontier_gathered_local_degrees, std::ignore) =
      shuffle_values(minor_comm,
                     (*aggregate_local_frontier_local_degrees).begin(),
                     local_frontier_sizes,
                     handle.get_stream());
    aggregate_local_frontier_local_degrees = std::nullopt;
    frontier_segmented_local_degree_displacements =
      rmm::device_uvector<edge_t>(frontier_degrees.size() * minor_comm_size, handle.get_stream());
    thrust::for_each(
      handle.get_thrust_policy(),
      thrust::make_counting_iterator(size_t{0}),
      thrust::make_counting_iterator(frontier_degrees.size()),
      compute_local_degree_displacements_and_global_degree_t<edge_t>{
        raft::device_span<edge_t const>(frontier_gathered_local_degrees.data(),
                                        frontier_gathered_local_degrees.size()),
        raft::device_span<edge_t>((*frontier_segmented_local_degree_displacements).data(),
                                  (*frontier_segmented_local_degree_displacements).size()),
        raft::device_span<edge_t>(frontier_degrees.data(), frontier_degrees.size()),
        minor_comm_size});
  }

  // 3. randomly select neighbor indices

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

  // 4. shuffle randomly selected indices

  auto sample_local_nbr_indices = std::move(
    sample_nbr_indices);  // neighbor index within an edge partition (note that each vertex's
                          // neighbors are distributed in minor_comm_size partitions)
  std::optional<rmm::device_uvector<size_t>> sample_key_indices{
    std::nullopt};        // relevant only when (minor_comm_size > 1)
  auto local_frontier_sample_counts        = std::vector<size_t>{};
  auto local_frontier_sample_displacements = std::vector<size_t>{};
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
        raft::device_span<edge_t const>((*frontier_segmented_local_degree_displacements).data(),
                                        (*frontier_segmented_local_degree_displacements).size()),
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
      not_equal_t<int>{-1});

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

    sample_local_nbr_indices            = std::move(std::get<0>(rx_value_buffer));
    sample_key_indices                  = std::move(std::get<1>(rx_value_buffer));
    local_frontier_sample_displacements = std::vector<size_t>(rx_counts.size());
    std::exclusive_scan(
      rx_counts.begin(), rx_counts.end(), local_frontier_sample_displacements.begin(), size_t{0});
    local_frontier_sample_counts = std::move(rx_counts);
  } else {
    local_frontier_sample_counts.push_back(frontier.size() * K);
    local_frontier_sample_displacements.push_back(size_t{0});
  }

  // 5. transform

  auto sample_e_op_results = allocate_dataframe_buffer<T>(
    local_frontier_sample_displacements.back() + local_frontier_sample_counts.back(),
    handle.get_stream());
  for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
    auto edge_partition =
      edge_partition_device_view_t<vertex_t, edge_t, GraphViewType::is_multi_gpu>(
        graph_view.local_edge_partition_view(i));

    auto edge_partition_frontier_key_first =
      ((minor_comm_size > 1) ? get_dataframe_buffer_begin(*aggregate_local_frontier_keys)
                             : frontier_key_first) +
      local_frontier_displacements[i];
    auto edge_partition_sample_local_nbr_index_first =
      sample_local_nbr_indices.begin() + local_frontier_sample_displacements[i];

    auto edge_partition_sample_e_op_result_first =
      get_dataframe_buffer_begin(sample_e_op_results) + local_frontier_sample_displacements[i];

    edge_partition_src_input_device_view_t edge_partition_src_value_input{};
    edge_partition_dst_input_device_view_t edge_partition_dst_value_input{};
    if constexpr (GraphViewType::is_storage_transposed) {
      edge_partition_src_value_input = edge_partition_src_input_device_view_t(edge_src_value_input);
      edge_partition_dst_value_input =
        edge_partition_dst_input_device_view_t(edge_dst_value_input, i);
    } else {
      edge_partition_src_value_input =
        edge_partition_src_input_device_view_t(edge_src_value_input, i);
      edge_partition_dst_value_input = edge_partition_dst_input_device_view_t(edge_dst_value_input);
    }
    auto edge_partition_e_value_input = edge_partition_e_input_device_view_t(edge_value_input, i);

    if (minor_comm_size > 1) {
      auto edge_partition_sample_key_index_first =
        (*sample_key_indices).begin() + local_frontier_sample_displacements[i];
      thrust::for_each(
        handle.get_thrust_policy(),
        thrust::make_counting_iterator(size_t{0}),
        thrust::make_counting_iterator(local_frontier_sample_counts[i]),
        transform_local_nbr_indices_t<GraphViewType,
                                      decltype(edge_partition_frontier_key_first),
                                      decltype(edge_partition_sample_local_nbr_index_first),
                                      decltype(edge_partition_sample_e_op_result_first),
                                      edge_partition_src_input_device_view_t,
                                      edge_partition_dst_input_device_view_t,
                                      edge_partition_e_input_device_view_t,
                                      EdgeOp,
                                      T>{
          edge_partition,
          thrust::make_optional(edge_partition_sample_key_index_first),
          edge_partition_frontier_key_first,
          edge_partition_sample_local_nbr_index_first,
          edge_partition_sample_e_op_result_first,
          edge_partition_src_value_input,
          edge_partition_dst_value_input,
          edge_partition_e_value_input,
          e_op,
          cugraph::ops::graph::INVALID_ID<edge_t>,
          to_thrust_optional(invalid_value),
          K});
    } else {
      thrust::for_each(
        handle.get_thrust_policy(),
        thrust::make_counting_iterator(size_t{0}),
        thrust::make_counting_iterator(frontier.size() * K),
        transform_local_nbr_indices_t<GraphViewType,
                                      decltype(edge_partition_frontier_key_first),
                                      decltype(edge_partition_sample_local_nbr_index_first),
                                      decltype(edge_partition_sample_e_op_result_first),
                                      edge_partition_src_input_device_view_t,
                                      edge_partition_dst_input_device_view_t,
                                      edge_partition_e_input_device_view_t,
                                      EdgeOp,
                                      T>{edge_partition,
                                         thrust::nullopt,
                                         edge_partition_frontier_key_first,
                                         edge_partition_sample_local_nbr_index_first,
                                         edge_partition_sample_e_op_result_first,
                                         edge_partition_src_value_input,
                                         edge_partition_dst_value_input,
                                         edge_partition_e_value_input,
                                         e_op,
                                         cugraph::ops::graph::INVALID_ID<edge_t>,
                                         to_thrust_optional(invalid_value),
                                         K});
    }
  }
  aggregate_local_frontier_keys = std::nullopt;

  // 6. shuffle randomly selected & transformed results and update sample_offsets

  auto sample_offsets = invalid_value ? std::nullopt
                                      : std::make_optional<rmm::device_uvector<size_t>>(
                                          frontier.size() + 1, handle.get_stream());
  assert(K <= std::numeric_limits<int32_t>::max());
  if (minor_comm_size > 1) {
    sample_local_nbr_indices.resize(0, handle.get_stream());
    sample_local_nbr_indices.shrink_to_fit(handle.get_stream());

    auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());

    std::tie(sample_e_op_results, std::ignore) =
      shuffle_values(minor_comm,
                     get_dataframe_buffer_begin(sample_e_op_results),
                     local_frontier_sample_counts,
                     handle.get_stream());
    std::tie(sample_key_indices, std::ignore) = shuffle_values(
      minor_comm, (*sample_key_indices).begin(), local_frontier_sample_counts, handle.get_stream());

    rmm::device_uvector<int32_t> sample_counts(frontier.size(), handle.get_stream());
    thrust::fill(
      handle.get_thrust_policy(), sample_counts.begin(), sample_counts.end(), int32_t{0});
    auto sample_intra_partition_displacements =
      rmm::device_uvector<int32_t>((*sample_key_indices).size(), handle.get_stream());
    thrust::transform(
      handle.get_thrust_policy(),
      (*sample_key_indices).begin(),
      (*sample_key_indices).end(),
      sample_intra_partition_displacements.begin(),
      count_t{raft::device_span<int32_t>(sample_counts.data(), sample_counts.size())});
    auto tmp_sample_e_op_results = allocate_dataframe_buffer<T>(0, handle.get_stream());
    if (invalid_value) {
      sample_counts.resize(0, handle.get_stream());
      sample_counts.shrink_to_fit(handle.get_stream());

      resize_dataframe_buffer(tmp_sample_e_op_results, frontier.size() * K, handle.get_stream());
      thrust::fill(handle.get_thrust_policy(),
                   get_dataframe_buffer_begin(tmp_sample_e_op_results),
                   get_dataframe_buffer_end(tmp_sample_e_op_results),
                   *invalid_value);
      thrust::scatter(
        handle.get_thrust_policy(),
        get_dataframe_buffer_begin(sample_e_op_results),
        get_dataframe_buffer_end(sample_e_op_results),
        thrust::make_transform_iterator(
          thrust::make_counting_iterator(size_t{0}),
          return_value_compute_offset_t<true>{
            raft::device_span<size_t const>((*sample_key_indices).data(),
                                            (*sample_key_indices).size()),
            raft::device_span<int32_t const>(sample_intra_partition_displacements.data(),
                                             sample_intra_partition_displacements.size()),
            K}),
        get_dataframe_buffer_begin(tmp_sample_e_op_results));
    } else {
      (*sample_offsets).set_element_to_zero_async(size_t{0}, handle.get_stream());
      auto typecasted_sample_count_first =
        thrust::make_transform_iterator(sample_counts.begin(), typecast_t<int32_t, size_t>{});
      thrust::inclusive_scan(handle.get_thrust_policy(),
                             typecasted_sample_count_first,
                             typecasted_sample_count_first + sample_counts.size(),
                             (*sample_offsets).begin() + 1);
      sample_counts.resize(0, handle.get_stream());
      sample_counts.shrink_to_fit(handle.get_stream());

      resize_dataframe_buffer(tmp_sample_e_op_results,
                              (*sample_offsets).back_element(handle.get_stream()),
                              handle.get_stream());
      thrust::scatter(
        handle.get_thrust_policy(),
        get_dataframe_buffer_begin(sample_e_op_results),
        get_dataframe_buffer_end(sample_e_op_results),
        thrust::make_transform_iterator(
          thrust::make_counting_iterator(size_t{0}),
          return_value_compute_offset_t<false>{
            raft::device_span<size_t const>((*sample_key_indices).data(),
                                            (*sample_key_indices).size()),
            raft::device_span<int32_t const>(sample_intra_partition_displacements.data(),
                                             sample_intra_partition_displacements.size()),
            raft::device_span<size_t const>((*sample_offsets).data(), (*sample_offsets).size())}),
        get_dataframe_buffer_begin(tmp_sample_e_op_results));
    }
    sample_e_op_results = std::move(tmp_sample_e_op_results);
  } else {
    if (!invalid_value) {
      rmm::device_uvector<int32_t> sample_counts(frontier.size(), handle.get_stream());
      thrust::tabulate(
        handle.get_thrust_policy(),
        sample_counts.begin(),
        sample_counts.end(),
        count_valids_t<edge_t>{raft::device_span<edge_t const>(sample_local_nbr_indices.data(),
                                                               sample_local_nbr_indices.size()),
                               K,
                               cugraph::ops::graph::INVALID_ID<edge_t>});
      (*sample_offsets).set_element_to_zero_async(size_t{0}, handle.get_stream());
      auto typecasted_sample_count_first =
        thrust::make_transform_iterator(sample_counts.begin(), typecast_t<int32_t, size_t>{});
      thrust::inclusive_scan(handle.get_thrust_policy(),
                             typecasted_sample_count_first,
                             typecasted_sample_count_first + sample_counts.size(),
                             (*sample_offsets).begin() + 1);
      sample_counts.resize(0, handle.get_stream());
      sample_counts.shrink_to_fit(handle.get_stream());

      auto pair_first = thrust::make_zip_iterator(thrust::make_tuple(
        sample_local_nbr_indices.begin(), get_dataframe_buffer_begin(sample_e_op_results)));
      auto pair_last =
        thrust::remove_if(handle.get_thrust_policy(),
                          pair_first,
                          pair_first + sample_local_nbr_indices.size(),
                          check_invalid_t<edge_t, T>{cugraph::ops::graph::INVALID_ID<edge_t>});
      sample_local_nbr_indices.resize(0, handle.get_stream());
      sample_local_nbr_indices.shrink_to_fit(handle.get_stream());

      resize_dataframe_buffer(
        sample_e_op_results, thrust::distance(pair_first, pair_last), handle.get_stream());
      shrink_to_fit_dataframe_buffer(sample_e_op_results, handle.get_stream());
    }
  }

  return std::make_tuple(std::move(sample_offsets), std::move(sample_e_op_results));
#else
  CUGRAPH_FAIL("unimplemented.");
#endif
}

}  // namespace detail

/**
 * @brief Randomly select and transform the input (tagged-)vertices' outgoing edges with biases.
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam VertexFrontierBucketType Type of the vertex frontier bucket class which abstracts the
 * current (tagged-)vertex frontier.
 * @tparam EdgeSrcValueInputWrapper Type of the wrapper for edge source property values.
 * @tparam EdgeDstValueInputWrapper Type of the wrapper for edge destination property values.
 * @tparam EdgeValueInputWrapper Type of the wrapper for edge property values.
 * @tparam EdgeBiasOp Type of the quinary edge operator to set-up selection bias
 * values.
 * @tparam EdgeOp Type of the quinary edge operator.
 * @tparam T Type of the selected and transformed edge output values.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param frontier VertexFrontierBucketType class object to store the (tagged-)vertex list to sample
 * outgoing edges.
 * @param edge_src_value_input Wrapper used to access source input property values (for the edge
 * sources assigned to this process in multi-GPU). Use either cugraph::edge_src_property_t::view()
 * (if @p e_op needs to access source property values) or cugraph::edge_src_dummy_property_t::view()
 * (if @p e_op does not access source property values). Use update_edge_src_property to fill the
 * wrapper.
 * @param edge_dst_value_input Wrapper used to access destination input property values (for the
 * edge destinations assigned to this process in multi-GPU). Use either
 * cugraph::edge_dst_property_t::view() (if @p e_op needs to access destination property values) or
 * cugraph::edge_dst_dummy_property_t::view() (if @p e_op does not access destination property
 * values). Use update_edge_dst_property to fill the wrapper.
 * @param edge_value_input Wrapper used to access edge input property values (for the edges assigned
 * to this process in multi-GPU). Use either cugraph::edge_property_t::view() (if @p e_op needs to
 * access edge property values) or cugraph::edge_dummy_property_t::view() (if @p e_op does not
 * access edge property values).
 * @param e_bias_op Quinary operator takes edge source, edge destination, property values for the
 * source, destination, and edge and returns a floating point bias value to be used in biased random
 * selection.
 * @param e_op Quinary operator takes edge source, edge destination, property values for the source,
 * destination, and edge and returns a value to be collected in the output. This function is called
 * only for the selected edges.
 * @param K Number of outgoing edges to select per (tagged-)vertex.
 * @param with_replacement A flag to specify whether a single outgoing edge can be selected multiple
 * times (if @p with_replacement = true) or can be selected only once (if @p with_replacement =
 * false).
 * @param invalid_value If @p invalid_value.has_value() is true, this value is used to fill the
 * output vector for the zero out-degree vertices (if @p with_replacement = true) or the vertices
 * with their out-degrees smaller than @p K (if @p with_replacement = false). If @p
 * invalid_value.has_value() is false, fewer than @p K values can be returned for the vertices with
 * fewer than @p K selected edges. See the return value section for additional details.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return std::tuple Tuple of an optional offset vector of type
 * std::optional<rmm::device_uvector<size_t>> and a dataframe buffer storing the output values of
 * type @p T from the selected edges. If @p invalid_value is std::nullopt, the offset vector is
 * valid and has the size of @p frontier.size() + 1. If @p invalid_value.has_value() is true,
 * std::nullopt is returned (the dataframe buffer will store @p frontier.size() * @p K elements).
 */
template <typename GraphViewType,
          typename VertexFrontierBucketType,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename EdgeBiasOp,
          typename EdgeOp,
          typename T>
std::tuple<std::optional<rmm::device_uvector<size_t>>,
           decltype(allocate_dataframe_buffer<T>(size_t{0}, rmm::cuda_stream_view{}))>
per_v_random_select_transform_outgoing_e(raft::handle_t const& handle,
                                         GraphViewType const& graph_view,
                                         VertexFrontierBucketType const& frontier,
                                         EdgeSrcValueInputWrapper edge_src_value_input,
                                         EdgeDstValueInputWrapper edge_dst_value_input,
                                         EdgeValueInputWrapper egde_value_input,
                                         EdgeBiasOp e_bias_op,
                                         EdgeOp e_op,
                                         raft::random::RngState& rng_state,
                                         size_t K,
                                         bool with_replacement,
                                         std::optional<T> invalid_value,
                                         bool do_expensive_check = false)
{
  CUGRAPH_EXPECTS(!graph_view.has_edge_mask(), "unimplemented.");

  CUGRAPH_FAIL("unimplemented.");

  return std::make_tuple(std::nullopt,
                         allocate_dataframe_buffer<T>(size_t{0}, rmm::cuda_stream_view{}));
}

/**
 * @brief Randomly select and transform the input (tagged-)vertices' outgoing edges.
 *
 * This function assumes that every outgoing edge of a given vertex has the same odd to be selected
 * (uniform neighbor sampling).
 *
 * @tparam GraphViewType Type of the passed non-owning graph object.
 * @tparam VertexFrontierBucketType Type of the vertex frontier bucket class which abstracts the
 * current (tagged-)vertex frontier.
 * @tparam EdgeSrcValueInputWrapper Type of the wrapper for edge source property values.
 * @tparam EdgeDstValueInputWrapper Type of the wrapper for edge destination property values.
 * @tparam EdgeValueInputWrapper Type of the wrapper for edge property values.
 * @tparam EdgeOp Type of the quinary edge operator.
 * @tparam T Type of the selected and transformed edge output values.
 * @param handle RAFT handle object to encapsulate resources (e.g. CUDA stream, communicator, and
 * handles to various CUDA libraries) to run graph algorithms.
 * @param graph_view Non-owning graph object.
 * @param frontier VertexFrontierBucketType class object to store the (tagged-)vertex list to sample
 * outgoing edges.
 * @param edge_src_value_input Wrapper used to access source input property values (for the edge
 * sources assigned to this process in multi-GPU). Use either cugraph::edge_src_property_t::view()
 * (if @p e_op needs to access source property values) or cugraph::edge_src_dummy_property_t::view()
 * (if @p e_op does not access source property values). Use update_edge_src_property to fill the
 * wrapper.
 * @param edge_dst_value_input Wrapper used to access destination input property values (for the
 * edge destinations assigned to this process in multi-GPU). Use either
 * cugraph::edge_dst_property_t::view() (if @p e_op needs to access destination property values) or
 * cugraph::edge_dst_dummy_property_t::view() (if @p e_op does not access destination property
 * values). Use update_edge_dst_property to fill the wrapper.
 * @param edge_value_input Wrapper used to access edge input property values (for the edges assigned
 * to this process in multi-GPU). Use either cugraph::edge_property_t::view() (if @p e_op needs to
 * access edge property values) or cugraph::edge_dummy_property_t::view() (if @p e_op does not
 * access edge property values).
 * @param e_op Quinary operator takes edge source, edge destination, property values for the source,
 * destination, and edge and returns a value to be collected in the output. This function is called
 * only for the selected edges.
 * @param K Number of outgoing edges to select per (tagged-)vertex.
 * @param with_replacement A flag to specify whether a single outgoing edge can be selected multiple
 * times (if @p with_replacement = true) or can be selected only once (if @p with_replacement =
 * false).
 * @param invalid_value If @p invalid_value.has_value() is true, this value is used to fill the
 * output vector for the zero out-degree vertices (if @p with_replacement = true) or the vertices
 * with their out-degrees smaller than @p K (if @p with_replacement = false). If @p
 * invalid_value.has_value() is false, fewer than @p K values can be returned for the vertices with
 * fewer than @p K selected edges. See the return value section for additional details.
 * @param do_expensive_check A flag to run expensive checks for input arguments (if set to `true`).
 * @return std::tuple Tuple of an optional offset vector of type
 * std::optional<rmm::device_uvector<size_t>> and a dataframe buffer storing the output values of
 * type @p T from the selected edges. If @p invalid_value is std::nullopt, the offset vector is
 * valid and has the size of @p frontier.size() + 1. If @p invalid_value.has_value() is true,
 * std::nullopt is returned (the dataframe buffer will store @p frontier.size() * @p K elements).
 */
template <typename GraphViewType,
          typename VertexFrontierBucketType,
          typename EdgeSrcValueInputWrapper,
          typename EdgeDstValueInputWrapper,
          typename EdgeValueInputWrapper,
          typename EdgeOp,
          typename T>
std::tuple<std::optional<rmm::device_uvector<size_t>>,
           decltype(allocate_dataframe_buffer<T>(size_t{0}, rmm::cuda_stream_view{}))>
per_v_random_select_transform_outgoing_e(raft::handle_t const& handle,
                                         GraphViewType const& graph_view,
                                         VertexFrontierBucketType const& frontier,
                                         EdgeSrcValueInputWrapper edge_src_value_input,
                                         EdgeDstValueInputWrapper edge_dst_value_input,
                                         EdgeValueInputWrapper edge_value_input,
                                         EdgeOp e_op,
                                         raft::random::RngState& rng_state,
                                         size_t K,
                                         bool with_replacement,
                                         std::optional<T> invalid_value,
                                         bool do_expensive_check = false)
{
  CUGRAPH_EXPECTS(!graph_view.has_edge_mask(), "unimplemented.");

  return detail::per_v_random_select_transform_e<false>(handle,
                                                        graph_view,
                                                        frontier,
                                                        edge_src_value_input,
                                                        edge_dst_value_input,
                                                        edge_value_input,
                                                        e_op,
                                                        rng_state,
                                                        K,
                                                        with_replacement,
                                                        invalid_value,
                                                        do_expensive_check);
}

}  // namespace cugraph
