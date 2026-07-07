/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/partition_manager.hpp>
#include <cugraph/prims/detail/sampling_helpers.cuh>
#include <cugraph/utilities/shuffle_comm.cuh>
#include <cugraph/utilities/thrust_wrappers/scan.hpp>

#include <raft/core/host_span.hpp>
#include <raft/random/rng.cuh>

#include <cuda/functional>
#include <cuda/std/optional>
#include <cuda/std/tuple>
#include <thrust/binary_search.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/partition.h>
#include <thrust/transform.h>

#include <optional>
#include <tuple>
#include <vector>

namespace cugraph {
namespace detail {

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
      auto i            = cuda::std::get<0>(pair);
      auto r            = cuda::std::get<1>(pair);
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
    auto k_first = cuda::make_transform_iterator(
      std::get<1>(*frontier_index_type_pairs).begin(),
      cuda::proclaim_return_type<size_t>(
        [K_offsets] __device__(auto type) { return K_offsets[type + 1] - K_offsets[type]; }));
    cugraph::inclusive_scan(
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
        auto i              = cuda::std::get<0>(pair);
        auto r              = cuda::std::get<1>(pair);
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
          auto i              = cuda::std::get<0>(pair);
          auto r              = cuda::std::get<1>(pair);
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
            auto per_type_local_nbr_idx = cuda::std::get<0>(triplet);
            if (per_type_local_nbr_idx != invalid_idx) {
              auto type              = cuda::std::get<1>(triplet);
              auto key_idx           = cuda::std::get<2>(triplet);
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
          auto per_type_local_nbr_idx = cuda::std::get<0>(pair);
          if (per_type_local_nbr_idx != invalid_idx) {
            auto i                 = cuda::std::get<1>(pair);
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
            auto local_nbr_idx = cuda::std::get<0>(pair);
            if (local_nbr_idx != invalid_idx) {
              auto key_idx        = cuda::std::get<1>(pair);
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
            auto local_nbr_idx = cuda::std::get<0>(pair);
            if (local_nbr_idx != invalid_idx) {
              auto key_idx        = cuda::std::get<1>(pair) / K;
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

}  // namespace detail
}  // namespace cugraph
