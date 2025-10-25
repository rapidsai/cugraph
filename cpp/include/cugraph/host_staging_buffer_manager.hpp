/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <cugraph/utilities/error.hpp>

#include <raft/core/handle.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/host/pinned_memory_resource.hpp>

#include <cstddef>
#include <memory>
#include <optional>

namespace cugraph {

class host_staging_buffer_manager {
 public:
  static constexpr size_t init_staging_buffer_size = size_t{1024} * size_t{1024};  // 1MB
  static constexpr size_t max_staging_buffer_size =
    size_t{1024} * size_t{1024} * size_t{1024};  // 1 GB

  static void init(raft::handle_t const& handle,
                   std::shared_ptr<rmm::mr::pinned_memory_resource> pinned_mr)
  {
    auto& s = state();
    CUGRAPH_EXPECTS(s.initialized == false, "host_staging_buffer_manager is already initialized.");
    s.initialized    = true;
    s.pinned_pool_mr = rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(
      std::move(pinned_mr), init_staging_buffer_size, max_staging_buffer_size);
  }

  static void clear()
  {
    auto& s = state();
    s.pinned_pool_mr.reset();
    s.initialized = false;
  }

  template <typename T>
  static rmm::device_uvector<T> allocate_staging_buffer(size_t size, rmm::cuda_stream_view stream)
  {
    auto& s = state();
    return rmm::device_uvector<T>(size, stream, s.pinned_pool_mr.get());
  }

 private:
  struct state_t {
    bool initialized = false;
    std::shared_ptr<
      rmm::mr::owning_wrapper<rmm::mr::pool_memory_resource<rmm::mr::pinned_memory_resource>,
                              rmm::mr::pinned_memory_resource>>
      pinned_pool_mr{};
  };

  static state_t& state()
  {
    static state_t s{};
    return s;
  };
};

}  // namespace cugraph
