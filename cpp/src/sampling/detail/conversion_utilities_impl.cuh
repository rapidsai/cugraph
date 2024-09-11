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

#pragma once

#include "sampling/detail/sampling_utils.hpp"

#include <cugraph/utilities/misc_utils.cuh>

namespace cugraph {
namespace detail {

rmm::device_uvector<int32_t> convert_starting_vertex_offsets_to_labels(
  raft::handle_t const& handle, raft::device_span<size_t const> starting_vertex_offsets)
{
  return expand_sparse_offsets(*starting_vertex_offsets, int32_t{0}, handle.get_stream());
}

rmm::device_uvector<int32_t> flatten_label_map(
  raft::handle_t const& handle,
  std::tuple<raft::device_span<label_t const>, raft::device_span<int32_t const>>
    label_to_output_comm_rank)
{
  rmm::device_uvector<int32_t> label_map(0, handle.get_stream());

  label_t max_label = thrust::reduce(handle.get_thrust_policy(),
                                     std::get<0>(label_to_output_comm_rank).begin(),
                                     std::get<0>(label_to_output_comm_rank).end(),
                                     label_t{0},
                                     thrust::maximum<label_t>());

  label_map.resize(max_label, handle.get_stream());

  thrust::fill(handle.get_thrust_policy(), label_map.begin(), label_map.end(), label_t{0});
  thrust::gather(handle.get_thrust_policy(),
                 std::get<0>(label_to_output_comm_rank).begin(),
                 std::get<0>(label_to_output_comm_rank).end(),
                 std::get<1>(label_to_output_comm_rank).begin(),
                 label_map.begin());

  return label_map;
}

}  // namespace detail
}  // namespace cugraph
