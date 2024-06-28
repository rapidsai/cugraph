/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include "renumber_sampled_edgelist_impl.cuh"

#include <cugraph/sampling_functions.hpp>

// FIXME: deprecated, to be deleted
namespace cugraph {

template std::tuple<rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    rmm::device_uvector<int32_t>,
                    std::optional<rmm::device_uvector<size_t>>>
renumber_sampled_edgelist(
  raft::handle_t const& handle,
  rmm::device_uvector<int32_t>&& edgelist_srcs,
  rmm::device_uvector<int32_t>&& edgelist_dsts,
  std::optional<raft::device_span<int32_t const>> edgelist_hops,
  std::optional<std::tuple<raft::device_span<int32_t const>, raft::device_span<size_t const>>>
    label_offsets,
  bool do_expensive_check);

}  // namespace cugraph
