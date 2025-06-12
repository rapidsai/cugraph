/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

#include <cugraph/utilities/dataframe_buffer.hpp>
#include <cugraph/utilities/error.hpp>

#include <raft/core/handle.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/host/pinned_memory_resource.hpp>

#include <memory>
#include <optional>

namespace cugraph {

namespace detail {

class large_memory_buffer_resource_t {
 public:
  large_memory_buffer_resource_t() = delete;
  large_memory_buffer_resource_t(std::shared_ptr<rmm::mr::pinned_memory_resource> mr) : mr_(mr) {}

  rmm::mr::pinned_memory_resource* get() const { return mr_.get(); }

 private:
  std::shared_ptr<rmm::mr::pinned_memory_resource>
    mr_{};  // currently, large memory buffer is backed by CUDA (rmm) pinned host memory
};

// storage buffer does not support pointer based load & store operations and requries special
// accessors to read/write
class large_storage_buffer_resource_t {  // placeholder for future use
};

}  // namespace detail

template <typename T>
class storage_buffer_t {  // placeholder for future use
};

enum class large_buffer_type_t { MEMORY, STORAGE, NUM_TYPES };

class large_buffer_manager {
 public:
  template <typename T>
  static dataframe_buffer_type_t<T> allocate_memory_buffer(size_t size, rmm::cuda_stream_view stream)
  {
    CUGRAPH_EXPECTS(memory_buffer_initialized(), "large memory buffer resource is not set.");
    return allocate_dataframe_buffer<T>(size, stream, memory_buffer_resource()->get());
  }

  template <typename T>
  static storage_buffer_t<T> allocate_storage_buffer(size_t, rmm::cuda_stream_view)
  {
    CUGRAPH_EXPECTS(storage_buffer_initialized(), "large storage buffer resource is not set.");
    return storage_buffer_t<T>();
  }

  static auto memory_buffer_mr() {
    return memory_buffer_resource()->get();
  }

  static bool memory_buffer_initialized() { return memory_buffer_resource().has_value(); }

  static bool storage_buffer_initialized() { return storage_buffer_resource().has_value(); }

  static void init(raft::handle_t const& handle,
                   std::optional<detail::large_memory_buffer_resource_t> memory_resource,
                   std::optional<detail::large_storage_buffer_resource_t> storage_resource)
  {
    CUGRAPH_EXPECTS(
      !storage_resource,
      "Invalid input argument: storage_resource should be std::nullopt (currently unsupported).");
    memory_buffer_resource()  = memory_resource;
    storage_buffer_resource() = storage_resource;
  }

  static detail::large_memory_buffer_resource_t create_memory_buffer_resource()
  {
    return detail::large_memory_buffer_resource_t(
      std::make_shared<rmm::mr::pinned_memory_resource>());
  }

  static detail::large_storage_buffer_resource_t create_storage_buffer_resource()
  {
    return detail::large_storage_buffer_resource_t();
  }

 private:
  static std::optional<detail::large_memory_buffer_resource_t>& memory_buffer_resource()
  {
    static std::optional<detail::large_memory_buffer_resource_t> memory_resource{std::nullopt};
    return memory_resource;
  }

  static std::optional<detail::large_storage_buffer_resource_t>& storage_buffer_resource()
  {
    static std::optional<detail::large_storage_buffer_resource_t> storage_resource{std::nullopt};
    return storage_resource;
  }
};

}  // namespace cugraph
