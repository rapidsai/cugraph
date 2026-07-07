/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "sampling_helpers_impl.cuh"

#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/prims/detail/heterogeneous_biased_sample.cuh>
#include <cugraph/prims/detail/partition_v_frontier.cuh>
#include <cugraph/prims/detail/sample_and_compute_local_nbr_indices.cuh>
#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/misc_utils.cuh>
#include <cugraph/utilities/shuffle_comm.cuh>
#include <cugraph/utilities/thrust_wrappers/gather.hpp>
#include <cugraph/utilities/thrust_wrappers/scan.hpp>

#include <raft/util/cudart_utils.hpp>

#include <cub/cub.cuh>
#include <cuda/functional>
#include <cuda/std/cmath>
#include <cuda/std/tuple>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>

#include <algorithm>
#include <cassert>
#include <limits>
#include <optional>
#include <tuple>
#include <vector>

namespace cugraph {
namespace detail {

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
      auto per_type_degree_first = cuda::make_transform_iterator(
        thrust::make_zip_iterator((*input_frontier_indices).begin(),
                                  input_frontier_edge_types.begin()),
        cuda::proclaim_return_type<size_t>(
          [input_per_type_degree_offsets, num_edge_types] __device__(auto pair) {
            auto idx  = cuda::std::get<0>(pair);
            auto type = cuda::std::get<1>(pair);
            return input_per_type_degree_offsets[idx * num_edge_types + type + 1] -
                   input_per_type_degree_offsets[idx * num_edge_types + type];
          }));
      cugraph::inclusive_scan(handle.get_thrust_policy(),
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
        auto bias_first = cuda::make_transform_iterator(
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

      auto offset_first = cuda::make_transform_iterator(
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
          auto idx             = cuda::std::get<0>(pair);
          auto type            = cuda::std::get<1>(pair);
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
        allocate_dataframe_buffer<cuda::std::tuple<size_t, edge_type_t>>(
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
              auto key_idx        = cuda::std::get<0>(pair);
              auto type           = cuda::std::get<1>(pair);
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
        cugraph::inclusive_scan(
          handle.get_thrust_policy(),
          aggregate_mid_local_frontier_per_type_local_degrees.begin(),
          aggregate_mid_local_frontier_per_type_local_degrees.end(),
          aggregate_mid_local_frontier_per_type_local_degree_offsets.begin() + 1,
          size_t{0},
          converting_plus_t<edge_t, size_t>{});
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
        cugraph::gather(handle.get_thrust_policy(),
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
        cugraph::inclusive_scan(handle.get_thrust_policy(),
                                mid_frontier_gathered_per_type_local_degrees.begin(),
                                mid_frontier_gathered_per_type_local_degrees.end(),
                                mid_frontier_gathered_per_type_local_degree_offsets.begin() + 1,
                                size_t{0},
                                converting_plus_t<edge_t, size_t>{});
      }

      rmm::device_uvector<bias_t> mid_frontier_gathered_biases(0, handle.get_stream());
      std::tie(mid_frontier_gathered_biases, std::ignore) =
        shuffle_values(minor_comm,
                       aggregate_mid_local_frontier_biases.data(),
                       raft::host_span<size_t const>(tx_counts.data(), tx_counts.size()),
                       handle.get_stream());
      aggregate_mid_local_frontier_biases.resize(0, handle.get_stream());
      aggregate_mid_local_frontier_biases.shrink_to_fit(handle.get_stream());

      auto mid_frontier_per_type_degree_first = cuda::make_transform_iterator(
        thrust::make_zip_iterator(frontier_indices.begin(), frontier_edge_types.begin()) +
          frontier_partition_offsets[1],
        cuda::proclaim_return_type<edge_t>(
          [frontier_per_type_degrees = raft::device_span<edge_t>(frontier_per_type_degrees.data(),
                                                                 frontier_per_type_degrees.size()),
           num_edge_types] __device__(auto pair) {
            return frontier_per_type_degrees[cuda::std::get<0>(pair) * num_edge_types +
                                             cuda::std::get<1>(pair)];
          }));
      rmm::device_uvector<size_t> mid_frontier_per_type_degree_offsets(mid_frontier_size + 1,
                                                                       handle.get_stream());
      mid_frontier_per_type_degree_offsets.set_element_to_zero_async(0, handle.get_stream());
      cugraph::inclusive_scan(handle.get_thrust_policy(),
                              mid_frontier_per_type_degree_first,
                              mid_frontier_per_type_degree_first + mid_frontier_size,
                              mid_frontier_per_type_degree_offsets.begin() + 1,
                              size_t{0},
                              converting_plus_t<edge_t, size_t>{});
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
            auto idx  = cuda::std::get<0>(pair);
            auto type = cuda::std::get<1>(pair);
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
        allocate_dataframe_buffer<cuda::std::tuple<size_t, edge_type_t>>(
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
        auto K_first = cuda::make_transform_iterator(
          std::get<1>(aggregate_high_local_frontier_index_type_pairs).begin(),
          cuda::proclaim_return_type<size_t>(
            [d_K_offsets = raft::device_span<size_t const>(
               d_K_offsets.data(), d_K_offsets.size())] __device__(auto type) {
              return d_K_offsets[type + 1] - d_K_offsets[type];
            }));
        aggregate_high_local_frontier_output_offsets.set_element_to_zero_async(0,
                                                                               handle.get_stream());
        cugraph::inclusive_scan(
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
        cugraph::gather(
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
        cugraph::gather(handle.get_thrust_policy(),
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
        auto K_first = cuda::make_transform_iterator(
          frontier_edge_types.begin() + frontier_partition_offsets[2],
          cuda::proclaim_return_type<size_t>(
            [K_offsets = raft::device_span<size_t const>(
               d_K_offsets.data(), d_K_offsets.size())] __device__(auto type) {
              return K_offsets[type + 1] - K_offsets[type];
            }));
        high_frontier_output_offsets.set_element_to_zero_async(0, handle.get_stream());
        cugraph::inclusive_scan(handle.get_thrust_policy(),
                                K_first,
                                K_first + high_frontier_size,
                                high_frontier_output_offsets.begin() + 1);
      }

      rmm::device_uvector<edge_t> high_frontier_per_type_nbr_indices(
        high_frontier_output_offsets.back_element(handle.get_stream()) * minor_comm_size,
        handle.get_stream());
      rmm::device_uvector<bias_t> high_frontier_keys(high_frontier_per_type_nbr_indices.size(),
                                                     handle.get_stream());
      auto index_first = cuda::make_transform_iterator(
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
      auto high_frontier_gathered_per_type_nbr_idx_first = cuda::make_transform_iterator(
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
      cugraph::gather(handle.get_thrust_policy(),
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
      auto offset_first = cuda::make_transform_iterator(
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
          auto idx             = cuda::std::get<0>(pair);
          auto type            = cuda::std::get<1>(pair);
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
      cugraph::gather(
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
            auto idx  = cuda::std::get<0>(pair);
            auto type = cuda::std::get<1>(pair);
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

}  // namespace detail
}  // namespace cugraph
