/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "sampling_helpers_impl.cuh"

#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/prims/detail/partition_v_frontier.cuh>
#include <cugraph/prims/detail/uniform_sampling_index.cuh>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/thrust_wrappers/gather.hpp>
#include <cugraph/utilities/thrust_wrappers/scan.hpp>

#include <cub/cub.cuh>
#include <cuda/atomic>
#include <cuda/functional>
#include <cuda/std/optional>
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

#include <cassert>
#include <cstddef>
#include <limits>
#include <optional>
#include <tuple>

namespace cugraph {
namespace detail {

template <typename edge_t, typename bias_t>
void sample_nbr_index_without_replacement(
  raft::handle_t const& handle,
  raft::device_span<edge_t const> frontier_degrees,
  std::optional<raft::device_span<size_t const>> frontier_indices,
  raft::device_span<edge_t> nbr_indices /* [OUT] */,
  raft::random::RngState& rng_state,
  size_t K,
  bool algo_r)
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
      auto count_first = cuda::make_transform_iterator(
        (*frontier_indices).begin(),
        cuda::proclaim_return_type<size_t>([frontier_degrees, K] __device__(size_t i) {
          auto d = static_cast<size_t>(frontier_degrees[i]);
          return d > K ? (d - K) : size_t{0};
        }));
      cugraph::inclusive_scan(handle.get_thrust_policy(),
                              count_first,
                              count_first + num_keys,
                              input_r_offsets.begin() + 1);

    } else {
      auto count_first = cuda::make_transform_iterator(
        frontier_degrees.begin(), cuda::proclaim_return_type<size_t>([K] __device__(auto degree) {
          auto d = static_cast<size_t>(degree);
          return d > K ? (d - K) : size_t{0};
        }));
      cugraph::inclusive_scan(handle.get_thrust_policy(),
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
  bool algo_r)
{
  auto num_edge_types = static_cast<edge_type_t>(K_offsets.size() - 1);

  auto num_keys = frontier_index_type_pairs ? std::get<0>(*frontier_index_type_pairs).size()
                                            : frontier_per_type_degrees.size();
  assert(frontier_index_type_pairs.has_value() || (num_keys % num_edge_types) == 0);

  // initialize reservoirs

  if (frontier_index_type_pairs) {
    rmm::device_uvector<size_t> sample_size_offsets(num_keys + 1, handle.get_stream());
    sample_size_offsets.set_element_to_zero_async(0, handle.get_stream());
    auto k_first = cuda::make_transform_iterator(
      std::get<1>(*frontier_index_type_pairs).begin(),
      cuda::proclaim_return_type<size_t>(
        [K_offsets] __device__(auto type) { return K_offsets[type + 1] - K_offsets[type]; }));
    cugraph::inclusive_scan(
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
      auto count_first = cuda::make_transform_iterator(
        thrust::make_zip_iterator(std::get<0>(*frontier_index_type_pairs).begin(),
                                  std::get<1>(*frontier_index_type_pairs).begin()),
        cuda::proclaim_return_type<size_t>(
          [frontier_per_type_degrees, K_offsets] __device__(auto pair) {
            auto num_edge_types = static_cast<edge_type_t>(K_offsets.size() - 1);
            auto frontier_idx   = cuda::std::get<0>(pair);
            auto type           = cuda::std::get<1>(pair);
            auto d =
              static_cast<size_t>(frontier_per_type_degrees[frontier_idx * num_edge_types + type]);
            auto K = K_offsets[type + 1] - K_offsets[type];
            return d > K ? (d - K) : size_t{0};
          }));
      cugraph::inclusive_scan(handle.get_thrust_policy(),
                              count_first,
                              count_first + num_keys,
                              input_r_offsets.begin() + 1);
    } else {
      auto count_first = cuda::make_transform_iterator(
        thrust::make_counting_iterator(size_t{0}),
        cuda::proclaim_return_type<size_t>(
          [frontier_per_type_degrees, K_offsets] __device__(auto i) {
            auto num_edge_types = static_cast<edge_type_t>(K_offsets.size() - 1);
            auto d              = static_cast<size_t>(frontier_per_type_degrees[i]);
            auto type           = static_cast<edge_type_t>(i % num_edge_types);
            auto K              = K_offsets[type + 1] - K_offsets[type];
            return d > K ? (d - K) : size_t{0};
          }));
      cugraph::inclusive_scan(handle.get_thrust_policy(),
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
      auto segment_frontier_per_type_degree_first = cuda::make_transform_iterator(
        thrust::make_zip_iterator(frontier_indices.begin(), frontier_edge_types.begin()) +
          frontier_partition_offsets[1] + keys_to_sort_per_iteration * i,
        cuda::proclaim_return_type<edge_t>(
          [frontier_per_type_degrees, num_edge_types] __device__(auto pair) {
            return frontier_per_type_degrees[cuda::std::get<0>(pair) * num_edge_types +
                                             cuda::std::get<1>(pair)];
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
          cugraph::gather(handle.get_thrust_policy(),
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
                cuda::std::make_tuple(retry_per_type_nbr_indices, retry_sample_indices));
              // sample index for the previously selected neighbor indices should be smaller than
              // the new candidates to ensure that the previously selected neighbor indices will
              // be selected again
              if (sample_idx < unique_count) {  // re-select the previous ones
                *(output_first + i) = cuda::std::make_tuple(
                  segment_sorted_tmp_per_type_nbr_indices[segment_idx *
                                                            high_partition_oversampling_K +
                                                          sample_idx],
                  static_cast<int32_t>(sample_idx));
              } else {
                *(output_first + i) = cuda::std::make_tuple(retry_per_type_nbr_indices[i],
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
             retry_segment_indices           = (*retry_segment_indices).data(),
             retry_segment_sorted_pair_first = thrust::make_zip_iterator(
               cuda::std::make_tuple((*retry_segment_sorted_per_type_nbr_indices).begin(),
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
               segment_sorted_tmp_per_type_nbr_indices.begin(),
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

        auto pair_first =
          thrust::make_zip_iterator(unique_counts.begin(), segment_frontier_type_first);
        if (retry_segment_indices) {
          auto last =
            thrust::remove_if(handle.get_thrust_policy(),
                              (*retry_segment_indices).begin(),
                              (*retry_segment_indices).end(),
                              [pair_first, K_offsets] __device__(auto segment_idx) {
                                auto pair = *(pair_first + segment_idx);
                                auto type = cuda::std::get<1>(pair);
                                return cuda::std::get<0>(pair) >=
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
              auto count = cuda::std::get<0>(pair);
              auto type  = cuda::std::get<1>(pair);
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
        segment_sorted_tmp_per_type_nbr_indices.data(),
        tmp_per_type_nbr_indices.data(),
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

      rmm::device_uvector<size_t> output_count_offsets(num_segments + 1, handle.get_stream());
      output_count_offsets.set_element_to_zero_async(0, handle.get_stream());
      auto k_first = cuda::make_transform_iterator(
        segment_frontier_type_first,
        cuda::proclaim_return_type<size_t>(
          [K_offsets] __device__(auto type) { return K_offsets[type + 1] - K_offsets[type]; }));
      cugraph::inclusive_scan(handle.get_thrust_policy(),
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

}  // namespace detail
}  // namespace cugraph
