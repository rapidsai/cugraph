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

#include <sampling/detail/shuffle_and_organize_output_impl.cuh>

namespace cugraph {
namespace detail {

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<float>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<size_t>>>
shuffle_and_organize_output(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& majors,
  rmm::device_uvector<int32_t>&& minors,
  std::optional<rmm::device_uvector<float>>&& weights,
  std::optional<rmm::device_uvector<int32_t>>&& edge_ids,
  std::optional<rmm::device_uvector<int32_t>>&& edge_types,
  std::optional<rmm::device_uvector<int32_t>>&& hops,
  std::optional<rmm::device_uvector<int32_t>>&& labels,
  std::optional<std::tuple<raft::device_span<int32_t const>, raft::device_span<int32_t const>>>
    label_to_output_comm_rank);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<float>>,
                    std::optional<rmm::device_uvector<int64_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<size_t>>>
shuffle_and_organize_output(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& majors,
  rmm::device_uvector<int32_t>&& minors,
  std::optional<rmm::device_uvector<float>>&& weights,
  std::optional<rmm::device_uvector<int64_t>>&& edge_ids,
  std::optional<rmm::device_uvector<int32_t>>&& edge_types,
  std::optional<rmm::device_uvector<int32_t>>&& hops,
  std::optional<rmm::device_uvector<int32_t>>&& labels,
  std::optional<std::tuple<raft::device_span<int32_t const>, raft::device_span<int32_t const>>>
    label_to_output_comm_rank);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<float>>,
                    std::optional<rmm::device_uvector<int64_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<size_t>>>
shuffle_and_organize_output(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>&& majors,
  rmm::device_uvector<int64_t>&& minors,
  std::optional<rmm::device_uvector<float>>&& weights,
  std::optional<rmm::device_uvector<int64_t>>&& edge_ids,
  std::optional<rmm::device_uvector<int32_t>>&& edge_types,
  std::optional<rmm::device_uvector<int32_t>>&& hops,
  std::optional<rmm::device_uvector<int32_t>>&& labels,
  std::optional<std::tuple<raft::device_span<int32_t const>, raft::device_span<int32_t const>>>
    label_to_output_comm_rank);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<double>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<size_t>>>
shuffle_and_organize_output(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& majors,
  rmm::device_uvector<int32_t>&& minors,
  std::optional<rmm::device_uvector<double>>&& weights,
  std::optional<rmm::device_uvector<int32_t>>&& edge_ids,
  std::optional<rmm::device_uvector<int32_t>>&& edge_types,
  std::optional<rmm::device_uvector<int32_t>>&& hops,
  std::optional<rmm::device_uvector<int32_t>>&& labels,
  std::optional<std::tuple<raft::device_span<int32_t const>, raft::device_span<int32_t const>>>
    label_to_output_comm_rank);

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<double>>,
                    std::optional<rmm::device_uvector<int64_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<size_t>>>
shuffle_and_organize_output(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& majors,
  rmm::device_uvector<int32_t>&& minors,
  std::optional<rmm::device_uvector<double>>&& weights,
  std::optional<rmm::device_uvector<int64_t>>&& edge_ids,
  std::optional<rmm::device_uvector<int32_t>>&& edge_types,
  std::optional<rmm::device_uvector<int32_t>>&& hops,
  std::optional<rmm::device_uvector<int32_t>>&& labels,
  std::optional<std::tuple<raft::device_span<int32_t const>, raft::device_span<int32_t const>>>
    label_to_output_comm_rank);

template std::tuple<rmm::device_uvector<int64_t>,
                    rmm::device_uvector<int64_t>,
                    std::optional<rmm::device_uvector<double>>,
                    std::optional<rmm::device_uvector<int64_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<int32_t>>,
                    std::optional<rmm::device_uvector<size_t>>>
shuffle_and_organize_output(
  raft::handle_t const& handle,
  rmm::device_uvector<int64_t>&& majors,
  rmm::device_uvector<int64_t>&& minors,
  std::optional<rmm::device_uvector<double>>&& weights,
  std::optional<rmm::device_uvector<int64_t>>&& edge_ids,
  std::optional<rmm::device_uvector<int32_t>>&& edge_types,
  std::optional<rmm::device_uvector<int32_t>>&& hops,
  std::optional<rmm::device_uvector<int32_t>>&& labels,
  std::optional<std::tuple<raft::device_span<int32_t const>, raft::device_span<int32_t const>>>
    label_to_output_comm_rank);

}  // namespace detail
}  // namespace cugraph
