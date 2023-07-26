/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cugraph/mtmg/handle.hpp>

namespace cugraph {
namespace mtmg {
namespace detail {

/**
 * @brief  Wrap an object to be available for each GPU
 *
 * In the MTMG environment we need the ability to manage a collection of objects
 * that are associated with a particular GPU, and fetch the objects from an
 * arbitrary GPU thread.  This object will wrap a pointer to any object and
 * allow it to be accessed from different threads.
 */
template <typename T>
class device_shared_wrapper_t {
 public:
  using wrapped_t = T;

  /**
   * @brief Initialize a pointer to an object for a particular thread
   *
   * @param handle  Handle is used to identify the GPU we associated this object with
   * @param args    Parameter pack passed to the constructor of T
   */
  template <class... Args>
  void initialize_pointer(cugraph::mtmg::handle_t const& handle, Args&&... args)
  {
    std::lock_guard<std::mutex> lock(lock_);

    CUGRAPH_EXPECTS(pointers_.find(handle.get_rank()) == pointers_.end(),
                    "Cannot reinitialize pointer");
    pointers_.insert(std::pair(handle.get_rank(), std::make_shared<wrapped_t>(args...)));
  }

  /**
   * @brief Move a pointer to an object for a particular thread into this wrapper
   *
   * @param handle  Handle is used to identify the GPU we associated this object with
   * @param ptr     Pointer to the object to add
   */
  void set_pointer(cugraph::mtmg::handle_t const& handle, std::unique_ptr<wrapped_t>&& ptr)
  {
    std::lock_guard<std::mutex> lock(lock_);

    CUGRAPH_EXPECTS(pointers_.find(handle.get_rank()) == pointers_.end(),
                    "Cannot reinitialize pointer");
    pointers_.insert(std::pair(handle.get_rank(), std::move(ptr)));
  }

  /**
   * @brief Get the pointer to an object for a particular thread from this wrapper
   *
   * @param handle  Handle is used to identify the GPU we associated this object with
   * @return Shared pointer the wrapped object
   */
  std::shared_ptr<wrapped_t> get_pointer(cugraph::mtmg::handle_t const& handle)
  {
    std::lock_guard<std::mutex> lock(lock_);

    return pointers_[handle.get_rank()];
  }

  /**
   * @brief Get the pointer to an object for a particular thread from this wrapper
   *
   * @param handle  Handle is used to identify the GPU we associated this object with
   * @return Shared pointer the wrapped object
   */
  std::shared_ptr<wrapped_t const> const get_pointer(cugraph::mtmg::handle_t const& handle) const
  {
    std::lock_guard<std::mutex> lock(lock_);

    auto pos = pointers_.find(handle.get_rank());

    return std::const_pointer_cast<wrapped_t const>(pos->second);
  }

 private:
  mutable std::mutex lock_{};

  std::map<int, std::shared_ptr<wrapped_t>> pointers_{};
};

}  // namespace detail
}  // namespace mtmg
}  // namespace cugraph
