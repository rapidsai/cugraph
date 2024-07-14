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

#pragma once

#include <cugraph/mtmg/detail/device_shared_device_span.hpp>

#include <rmm/device_uvector.hpp>

namespace cugraph {
namespace mtmg {
namespace detail {

/**
 * @brief  Manage a device vector on each GPU
 *
 * Uses the device_shared_wrapper to manage an rmm::device_uvector<T> on
 * each GPU.
 */
template <typename T>
class device_shared_device_vector_t : public device_shared_wrapper_t<rmm::device_uvector<T>> {
  using parent_t = detail::device_shared_wrapper_t<rmm::device_uvector<T>>;

 public:
  /**
   * @brief Create a device_shared_device_span (read only view)
   */
  auto view()
  {
    std::lock_guard<std::mutex> lock(parent_t::lock_);

    device_shared_device_span_t<T const> result;

    std::for_each(parent_t::objects_.begin(), parent_t::objects_.end(), [&result](auto& p) {
      result.set(p.first, raft::device_span<T const>{p.second.data(), p.second.size()});
    });

    return result;
  }
};

}  // namespace detail
}  // namespace mtmg
}  // namespace cugraph
