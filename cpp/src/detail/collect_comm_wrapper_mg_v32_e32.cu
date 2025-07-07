/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

#include "detail/collect_comm_wrapper.cuh"

namespace cugraph {
namespace detail {

template rmm::device_uvector<int32_t> device_allgatherv(raft::handle_t const& handle,
                                                        raft::comms::comms_t const& comm,
                                                        raft::device_span<int32_t const> d_input);

template rmm::device_uvector<float> device_allgatherv(raft::handle_t const& handle,
                                                      raft::comms::comms_t const& comm,
                                                      raft::device_span<float const> d_input);

}  // namespace detail
}  // namespace cugraph
