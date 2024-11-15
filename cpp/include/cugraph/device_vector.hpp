/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include "cugraph_c/types.h"

#include <cugraph/utilities/cugraph_data_type_id.hpp>
#include <cugraph/utilities/error.hpp>
#include <cugraph/utilities/packed_bool_utils.hpp>

#include <raft/core/handle.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

namespace cugraph {

namespace detail {

inline rmm::device_buffer allocate_buffer(raft::handle_t const& handle,
                                          cugraph_data_type_id_t t,
                                          size_t size)
{
  return (t == BOOL) ? rmm::device_buffer(cugraph::packed_bool_size(size) * data_type_size(t),
                                          handle.get_stream())
                     : rmm::device_buffer(size * data_type_size(t), handle.get_stream());
}

}  // namespace detail

/**
 * Class wrapping a type-erased device vector.
 * */
class device_vector_t {
 public:
  /**
   * Constructor creating a device vector
   *
   * @param data_type    The data type of the vector
   * @param size         The number of elements in the vector
   */
  device_vector_t(raft::handle_t const& handle, cugraph_data_type_id_t data_type, size_t size)
    : data_(detail::allocate_buffer(handle, data_type, size)), type_(data_type), size_(size)
  {
  }

  /**
   * Constructor initializing device vector from an rmm device uvector
   *
   * @tparam T      type for the array/vector
   * @param  vector Vector to
   */
  template <typename T>
  device_vector_t(rmm::device_uvector<T>&& vector) : type_(type_to_id<T>()), size_(vector.size())
  {
    data_ = vector.release();
  }

  template <typename T>
  T* begin()
  {
    return reinterpret_cast<T*>(data_.data());
  }

  template <typename T>
  T const* begin() const
  {
    return reinterpret_cast<T const*>(data_.data());
  }

  template <typename T>
  T* end()
  {
    return reinterpret_cast<T*>(data_.data()) + size_;
  }

  template <typename T>
  T const* end() const
  {
    return reinterpret_cast<T const*>(data_.data()) + size_;
  }

  cugraph_data_type_id_t type() const { return type_; }
  size_t size() const { return size_; }

  void clear(rmm::cuda_stream_view stream_view)
  {
    data_.resize(0, stream_view);
    data_.shrink_to_fit(stream_view);
  }

 private:
  rmm::device_buffer data_{};
  cugraph_data_type_id_t type_{NTYPES};
  size_t size_{0};
};

}  // namespace cugraph
