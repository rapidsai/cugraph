/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "sampling_helpers_impl.cuh"
#include "uniform_sampling_index_impl.cuh"

#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/prims/detail/partition_v_frontier.cuh>
#include <cugraph/prims/detail/sample_and_compute_local_nbr_indices.cuh>
#include <cugraph/prims/detail/uniform_and_biased_replacement_sample.cuh>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/thrust_wrappers/gather.hpp>
#include <cugraph/utilities/thrust_wrappers/scan.hpp>

#include <raft/util/cudart_utils.hpp>

#include <cub/cub.cuh>
#include <cuda/functional>
#include <cuda/std/tuple>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>
#include <numeric>
#include <optional>
#include <tuple>
#include <vector>

namespace cugraph {
namespace detail {

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
      auto segment_frontier_degree_first = cuda::make_transform_iterator(
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
          cugraph::gather(handle.get_thrust_policy(),
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
                *(output_first + i) = cuda::std::make_tuple(
                  segment_sorted_tmp_nbr_indices[segment_idx * high_partition_oversampling_K +
                                                 sample_idx],
                  static_cast<int32_t>(sample_idx));
              } else {
                *(output_first + i) =
                  cuda::std::make_tuple(retry_nbr_indices[i], static_cast<int32_t>(sample_idx));
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
          cuda::make_transform_iterator(thrust::make_counting_iterator(size_t{0}),
                                        multiplier_t<size_t>{high_partition_oversampling_K}),
          cuda::make_transform_iterator(thrust::make_counting_iterator(size_t{1}),
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
          cuda::make_transform_iterator(thrust::make_counting_iterator(size_t{0}),
                                        multiplier_t<size_t>{high_partition_oversampling_K}),
          cuda::make_transform_iterator(thrust::make_counting_iterator(size_t{1}),
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
                if (cuda::std::get<0>(cur) ==
                    cuda::std::get<0>(prev)) {  // update the sample index to the minimum
                  cuda::std::get<1>(prev) =
                    cuda::std::min(cuda::std::get<1>(prev), cuda::std::get<1>(cur));
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
              cuda::std::tuple<edge_t, int32_t> prev = *pair_first;
              size_t unique_count                    = 1;
              for (size_t j = 1; j < high_partition_oversampling_K; ++j) {
                auto cur = *(pair_first + j);
                if (cuda::std::get<0>(cur) ==
                    cuda::std::get<0>(prev)) {  // update the sample index to the minimum
                  cuda::std::get<1>(prev) =
                    cuda::std::min(cuda::std::get<1>(prev), cuda::std::get<1>(cur));
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
        cuda::make_transform_iterator(thrust::make_counting_iterator(size_t{0}),
                                      multiplier_t<size_t>{high_partition_oversampling_K}),
        cuda::make_transform_iterator(
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
        cuda::make_transform_iterator(thrust::make_counting_iterator(size_t{0}),
                                      multiplier_t<size_t>{high_partition_oversampling_K}),
        cuda::make_transform_iterator(
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
    auto unique_key_first = cuda::make_transform_iterator(
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

}  // namespace detail
}  // namespace cugraph
