/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

namespace cugraph {
namespace test {

template <typename T>
rmm::device_uvector<T> device_gatherv(raft::handle_t const& handle,
                                      raft::device_span<T const> d_input);

template <typename T>
rmm::device_uvector<T> device_gatherv(raft::handle_t const& handle, T const* d_input, size_t size)
{
  return device_gatherv(handle, raft::device_span<T const>{d_input, size});
}

template <typename T>
rmm::device_uvector<T> device_allgatherv(raft::handle_t const& handle,
                                         raft::device_span<T const> d_input);

template <typename T>
rmm::device_uvector<T> device_allgatherv(raft::handle_t const& handle,
                                         T const* d_input,
                                         size_t size)
{
  return device_allgatherv(handle, raft::device_span<T const>{d_input, size});
}

}  // namespace test
}  // namespace cugraph
