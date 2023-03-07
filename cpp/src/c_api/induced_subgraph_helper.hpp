/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>

#include <tuple>

namespace cugraph {
namespace c_api {
namespace detail {

template <typename vertex_t>
std::tuple<rmm::device_uvector<vertex_t>, rmm::device_uvector<size_t>>
shuffle_vertex_ids_and_offsets(raft::handle_t const& handle,
                               rmm::device_uvector<vertex_t>&& vertices,
                               raft::device_span<size_t const> offsets);

}  // namespace detail
}  // namespace c_api
}  // namespace cugraph
