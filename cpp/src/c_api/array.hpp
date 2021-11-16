/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cugraph_c/array.h>

#include <cugraph/visitors/enum_mapping.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

namespace cugraph {
namespace c_api {

extern cugraph::visitors::DTypes dtypes_mapping[data_type_id_t::NTYPES];

struct cugraph_type_erased_device_array_t {
  rmm::device_buffer data_;
  size_t size_;
  data_type_id_t type_;

  cugraph_type_erased_device_array_t(size_t size,
                                     size_t nbytes,
                                     data_type_id_t type,
                                     rmm::cuda_stream_view const& stream_view)
    : data_(nbytes, stream_view), size_(size), type_(type)
  {
  }

  template <typename T>
  cugraph_type_erased_device_array_t(rmm::device_uvector<T>&& vec, data_type_id_t type)
    : data_(vec.release()), size_(vec.size()), type_(type)
  {
  }

  template <typename T>
  T* as_type()
  {
    return reinterpret_cast<T*>(data_.data());
  }

  template <typename T>
  T const* as_type() const
  {
    return reinterpret_cast<T const*>(data_.data());
  }
};

struct cugraph_type_erased_host_array_t {
  std::byte* data_;
  size_t size_;
  size_t nbytes_;
  data_type_id_t type_;

  template <typename T>
  T* as_type()
  {
    return reinterpret_cast<T*>(data_);
  }
};

}  // namespace c_api
}  // namespace cugraph
