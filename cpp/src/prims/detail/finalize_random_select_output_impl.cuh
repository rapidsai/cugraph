/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

namespace cugraph {
namespace detail {

template <typename edge_t, typename T>
std::tuple<std::optional<rmm::device_uvector<size_t>>, dataframe_buffer_type_t<T>>
finalize_random_select_output(raft::handle_t const& handle,
                              int minor_comm_size,
                              rmm::device_uvector<edge_t>& sample_local_nbr_indices,
                              dataframe_buffer_type_t<T>& sample_e_op_results,
                              std::optional<rmm::device_uvector<size_t>>& sample_key_indices,
                              raft::host_span<size_t const> local_key_list_sample_counts,
                              size_t key_list_size,
                              size_t K_sum,
                              std::optional<T> const& invalid_value)
{
  auto sample_offsets = invalid_value ? std::nullopt
                                      : std::make_optional<rmm::device_uvector<size_t>>(
                                          key_list_size + 1, handle.get_stream());
  assert(K_sum <= std::numeric_limits<int32_t>::max());
  if (minor_comm_size > 1) {
    sample_local_nbr_indices.resize(0, handle.get_stream());
    sample_local_nbr_indices.shrink_to_fit(handle.get_stream());

    auto& minor_comm = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());

    std::tie(sample_e_op_results, std::ignore) =
      shuffle_values(minor_comm,
                     get_dataframe_buffer_begin(sample_e_op_results),
                     local_key_list_sample_counts,
                     handle.get_stream());
    std::tie(sample_key_indices, std::ignore) = shuffle_values(
      minor_comm, (*sample_key_indices).begin(), local_key_list_sample_counts, handle.get_stream());

    rmm::device_uvector<int32_t> sample_counts(key_list_size, handle.get_stream());
    cugraph::fill(
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

      resize_dataframe_buffer(tmp_sample_e_op_results, key_list_size * K_sum, handle.get_stream());
      cugraph::fill(handle.get_thrust_policy(),
                    get_dataframe_buffer_begin(tmp_sample_e_op_results),
                    get_dataframe_buffer_end(tmp_sample_e_op_results),
                    *invalid_value);
      cugraph::scatter(
        handle.get_thrust_policy(),
        get_dataframe_buffer_begin(sample_e_op_results),
        get_dataframe_buffer_end(sample_e_op_results),
        cuda::make_transform_iterator(
          thrust::make_counting_iterator(size_t{0}),
          return_value_compute_offset_t<true>{
            raft::device_span<size_t const>((*sample_key_indices).data(),
                                            (*sample_key_indices).size()),
            raft::device_span<int32_t const>(sample_intra_partition_displacements.data(),
                                             sample_intra_partition_displacements.size()),
            K_sum}),
        get_dataframe_buffer_begin(tmp_sample_e_op_results));
    } else {
      (*sample_offsets).set_element_to_zero_async(size_t{0}, handle.get_stream());
      auto typecasted_sample_count_first =
        cuda::make_transform_iterator(sample_counts.begin(), typecast_t<int32_t, size_t>{});
      cugraph::inclusive_scan(handle.get_thrust_policy(),
                              typecasted_sample_count_first,
                              typecasted_sample_count_first + sample_counts.size(),
                              (*sample_offsets).begin() + 1);
      sample_counts.resize(0, handle.get_stream());
      sample_counts.shrink_to_fit(handle.get_stream());

      resize_dataframe_buffer(tmp_sample_e_op_results,
                              (*sample_offsets).back_element(handle.get_stream()),
                              handle.get_stream());
      cugraph::scatter(
        handle.get_thrust_policy(),
        get_dataframe_buffer_begin(sample_e_op_results),
        get_dataframe_buffer_end(sample_e_op_results),
        cuda::make_transform_iterator(
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
      rmm::device_uvector<int32_t> sample_counts(key_list_size, handle.get_stream());
      thrust::tabulate(
        handle.get_thrust_policy(),
        sample_counts.begin(),
        sample_counts.end(),
        count_valids_t<edge_t>{raft::device_span<edge_t const>(sample_local_nbr_indices.data(),
                                                               sample_local_nbr_indices.size()),
                               K_sum,
                               cugraph::invalid_edge_id_v<edge_t>});
      (*sample_offsets).set_element_to_zero_async(size_t{0}, handle.get_stream());
      auto typecasted_sample_count_first =
        cuda::make_transform_iterator(sample_counts.begin(), typecast_t<int32_t, size_t>{});
      cugraph::inclusive_scan(handle.get_thrust_policy(),
                              typecasted_sample_count_first,
                              typecasted_sample_count_first + sample_counts.size(),
                              (*sample_offsets).begin() + 1);
      sample_counts.resize(0, handle.get_stream());
      sample_counts.shrink_to_fit(handle.get_stream());

      auto pair_first = thrust::make_zip_iterator(sample_local_nbr_indices.begin(),
                                                  get_dataframe_buffer_begin(sample_e_op_results));
      auto pair_last =
        thrust::remove_if(handle.get_thrust_policy(),
                          pair_first,
                          pair_first + sample_local_nbr_indices.size(),
                          check_invalid_t<edge_t, T>{cugraph::invalid_edge_id_v<edge_t>});
      sample_local_nbr_indices.resize(0, handle.get_stream());
      sample_local_nbr_indices.shrink_to_fit(handle.get_stream());

      resize_dataframe_buffer(
        sample_e_op_results, cuda::std::distance(pair_first, pair_last), handle.get_stream());
      shrink_to_fit_dataframe_buffer(sample_e_op_results, handle.get_stream());
    }
  }

  return std::make_tuple(std::move(sample_offsets), std::move(sample_e_op_results));
}

}  // namespace detail
}  // namespace cugraph
