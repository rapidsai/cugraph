/*
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cugraph/mtmg/handle.hpp>
#include <cugraph/utilities/error.hpp>

#include <map>
#include <mutex>

namespace cugraph {
namespace mtmg {
namespace detail {

/**
 * @brief  Wrap an object to be available for each GPU
 *
 * In the MTMG environment we need the ability to manage a collection of objects
 * that are associated with a particular GPU, and fetch the objects from an
 * arbitrary GPU thread.  This object will wrap any object and allow it to be
 * accessed from different threads.
 */
template <typename T>
class device_shared_wrapper_t {
 public:
  using wrapped_t = T;

  device_shared_wrapper_t() = default;
  device_shared_wrapper_t(device_shared_wrapper_t&& other) : objects_{std::move(other.objects_)} {}
  device_shared_wrapper_t& operator=(device_shared_wrapper_t&& other)
  {
    objects_ = std::move(other.objects_);
    return *this;
  }

  /**
   * @brief Move a wrapped object into the wrapper for this thread
   *
   * @param handle  Handle is used to identify the GPU we associated this object with
   * @param obj     Wrapped object
   */
  void set(cugraph::mtmg::handle_t const& handle, wrapped_t&& obj)
  {
    std::lock_guard<std::mutex> lock(lock_);

    auto pos = objects_.find(handle.get_rank());
    CUGRAPH_EXPECTS(pos == objects_.end(), "Cannot overwrite wrapped object");

    objects_.insert(std::make_pair(handle.get_rank(), std::move(obj)));
  }

  /**
   * @brief Move a wrapped object into the wrapper for this thread
   *
   * @param local_rank  Identify which GPU to associated this object with
   * @param obj         Wrapped object
   */
  void set(int local_rank, wrapped_t&& obj)
  {
    std::lock_guard<std::mutex> lock(lock_);

    auto pos = objects_.find(local_rank);
    CUGRAPH_EXPECTS(pos == objects_.end(), "Cannot overwrite wrapped object");

    objects_.insert(std::make_pair(local_rank, std::move(obj)));
  }

  /**
   * @brief Get reference to an object for a particular thread
   *
   * @param handle  Handle is used to identify the GPU we associated this object with
   * @return Reference to the wrapped object
   */
  wrapped_t& get(cugraph::mtmg::handle_t const& handle)
  {
    std::lock_guard<std::mutex> lock(lock_);

    auto pos = objects_.find(handle.get_rank());
    CUGRAPH_EXPECTS(pos != objects_.end(), "Uninitialized wrapped object");

    return pos->second;
  }

  /**
   * @brief Get the pointer to an object for a particular thread from this wrapper
   *
   * @param handle  Handle is used to identify the GPU we associated this object with
   * @return Shared pointer the wrapped object
   */
  wrapped_t const& get(cugraph::mtmg::handle_t const& handle) const
  {
    std::lock_guard<std::mutex> lock(lock_);

    auto pos = objects_.find(handle.get_rank());

    CUGRAPH_EXPECTS(pos != objects_.end(), "Uninitialized wrapped object");

    return pos->second;
  }

 protected:
  mutable std::mutex lock_{};
  std::map<int, wrapped_t> objects_{};
};

}  // namespace detail
}  // namespace mtmg
}  // namespace cugraph
