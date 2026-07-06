/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

namespace cugraph {
namespace detail {

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
      auto degree_first = cuda::make_transform_iterator(
        (*input_frontier_indices).begin(),
        cuda::proclaim_return_type<size_t>([input_degree_offsets] __device__(size_t i) {
          return input_degree_offsets[i + 1] - input_degree_offsets[i];
        }));
      cugraph::inclusive_scan(handle.get_thrust_policy(),
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
        auto bias_first = cuda::make_transform_iterator(
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

      auto offset_first = cuda::make_transform_iterator(
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
        cugraph::inclusive_scan(handle.get_thrust_policy(),
                                aggregate_mid_local_frontier_local_degrees.begin(),
                                aggregate_mid_local_frontier_local_degrees.end(),
                                aggregate_mid_local_frontier_local_degree_offsets.begin() + 1,
                                size_t{0},
                                converting_plus_t<edge_t, size_t>{});
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
        auto map_first = cuda::make_transform_iterator(
          thrust::make_counting_iterator(size_t{0}),
          cuda::proclaim_return_type<size_t>(
            [mid_local_frontier_offsets = raft::device_span<size_t const>(
               d_mid_local_frontier_offsets.data(),
               d_mid_local_frontier_offsets.size())] __device__(size_t i) {
              return mid_local_frontier_offsets[i + 1];
            }));
        cugraph::gather(handle.get_thrust_policy(),
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
        cugraph::inclusive_scan(handle.get_thrust_policy(),
                                mid_frontier_gathered_local_degrees.begin(),
                                mid_frontier_gathered_local_degrees.end(),
                                mid_frontier_gathered_local_degree_offsets.begin() + 1,
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

      auto mid_frontier_degree_first = cuda::make_transform_iterator(
        frontier_indices.begin() + frontier_partition_offsets[1],
        cuda::proclaim_return_type<edge_t>(
          [frontier_degrees = raft::device_span<edge_t>(
             frontier_degrees.data(), frontier_degrees.size())] __device__(size_t i) {
            return frontier_degrees[i];
          }));
      rmm::device_uvector<size_t> mid_frontier_degree_offsets(mid_frontier_size + 1,
                                                              handle.get_stream());
      mid_frontier_degree_offsets.set_element_to_zero_async(0, handle.get_stream());
      cugraph::inclusive_scan(handle.get_thrust_policy(),
                              mid_frontier_degree_first,
                              mid_frontier_degree_first + mid_frontier_size,
                              mid_frontier_degree_offsets.begin() + 1,
                              size_t{0},
                              converting_plus_t<edge_t, size_t>{});
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
        cugraph::gather(
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
      auto index_first = cuda::make_transform_iterator(
        thrust::make_counting_iterator(size_t{0}),
        cuda::proclaim_return_type<size_t>(
          [K, minor_comm_rank, minor_comm_size, high_frontier_size] __device__(size_t i) {
            auto idx             = i / (K * minor_comm_size);
            auto minor_comm_rank = (i % (K * minor_comm_size)) / K;
            return minor_comm_rank * (high_frontier_size * K) + idx * K + (i % K);
          }));
      auto high_frontier_gathered_nbr_idx_first = cuda::make_transform_iterator(
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
      cugraph::gather(
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
        cuda::make_transform_iterator(thrust::make_counting_iterator(size_t{0}),
                                      multiplier_t<size_t>{minor_comm_size * K}),
        cuda::make_transform_iterator(thrust::make_counting_iterator(size_t{1}),
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
        cuda::make_transform_iterator(thrust::make_counting_iterator(size_t{0}),
                                      multiplier_t<size_t>{minor_comm_size * K}),
        cuda::make_transform_iterator(thrust::make_counting_iterator(size_t{1}),
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
      cugraph::gather(
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

}  // namespace detail
}  // namespace cugraph
