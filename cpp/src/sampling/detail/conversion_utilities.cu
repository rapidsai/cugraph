/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "sampling/detail/sampling_utils.hpp"

#include <cugraph/utilities/misc_utils.cuh>

namespace cugraph {
namespace detail {

rmm::device_uvector<int32_t> convert_starting_vertex_label_offsets_to_labels(
  raft::handle_t const& handle, raft::device_span<size_t const> starting_vertex_label_offsets)
{
  return expand_sparse_offsets(starting_vertex_label_offsets, int32_t{0}, handle.get_stream());
}

template <typename label_t>
rmm::device_uvector<int32_t> flatten_label_map(
  raft::handle_t const& handle,
  std::tuple<raft::device_span<label_t const>, raft::device_span<int32_t const>>
    label_to_output_comm_rank)
{
  label_t max_label = thrust::reduce(handle.get_thrust_policy(),
                                     std::get<0>(label_to_output_comm_rank).begin(),
                                     std::get<0>(label_to_output_comm_rank).end(),
                                     label_t{0},
                                     thrust::maximum<label_t>());

  rmm::device_uvector<int32_t> label_map(max_label + 1, handle.get_stream());

  thrust::fill(handle.get_thrust_policy(), label_map.begin(), label_map.end(), int32_t{0});
  thrust::scatter(handle.get_thrust_policy(),
                  std::get<1>(label_to_output_comm_rank).begin(),
                  std::get<1>(label_to_output_comm_rank).end(),
                  std::get<0>(label_to_output_comm_rank).begin(),
                  label_map.begin());

  return label_map;
}

template rmm::device_uvector<int32_t> flatten_label_map(
  raft::handle_t const& handle,
  std::tuple<raft::device_span<int32_t const>, raft::device_span<int32_t const>>
    label_to_output_comm_rank);

}  // namespace detail
}  // namespace cugraph
