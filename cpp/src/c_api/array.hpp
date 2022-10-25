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

#include <cugraph_c/array.h>

#include <cugraph/visitors/enum_mapping.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

namespace cugraph {
namespace c_api {

extern cugraph::visitors::DTypes dtypes_mapping[data_type_id_t::NTYPES];

struct cugraph_type_erased_device_array_view_t {
  void* data_;
  size_t size_;
  size_t num_bytes_;
  data_type_id_t type_;

  template <typename T>
  T* as_type()
  {
    return reinterpret_cast<T*>(data_);
  }

  template <typename T>
  T const* as_type() const
  {
    return reinterpret_cast<T const*>(data_);
  }

  size_t num_bytes() const { return num_bytes_; }
};

struct cugraph_type_erased_device_array_t {
  // NOTE: size must be first here because the device buffer is released
  size_t size_;
  rmm::device_buffer data_;
  data_type_id_t type_;

  cugraph_type_erased_device_array_t(size_t size,
                                     size_t num_bytes,
                                     data_type_id_t type,
                                     rmm::cuda_stream_view const& stream_view)
    : size_(size), data_(num_bytes, stream_view), type_(type)
  {
  }

  template <typename T>
  cugraph_type_erased_device_array_t(rmm::device_uvector<T>& vec, data_type_id_t type)
    : size_(vec.size()), data_(vec.release()), type_(type)
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

  auto view()
  {
    return new cugraph_type_erased_device_array_view_t{data_.data(), size_, data_.size(), type_};
  }
};

struct cugraph_type_erased_host_array_view_t {
  std::byte* data_;
  size_t size_;
  size_t num_bytes_;
  data_type_id_t type_;

  template <typename T>
  T* as_type()
  {
    return reinterpret_cast<T*>(data_);
  }

  template <typename T>
  T const* as_type() const
  {
    return reinterpret_cast<T const*>(data_);
  }

  size_t num_bytes() const { return num_bytes_; }
};

struct cugraph_type_erased_host_array_t {
  std::unique_ptr<std::byte[]> data_{nullptr};
  size_t size_{0};
  size_t num_bytes_{0};
  data_type_id_t type_;

  cugraph_type_erased_host_array_t(size_t size, size_t num_bytes, data_type_id_t type)
    : data_(std::make_unique<std::byte[]>(num_bytes)),
      size_(size),
      num_bytes_(num_bytes),
      type_(type)
  {
  }

  template <typename T>
  cugraph_type_erased_host_array_t(std::vector<T>& vec, data_type_id_t type)
    : size_(vec.size()), num_bytes_(vec.size() * sizeof(T)), type_(type)
  {
    data_ = std::make_unique<std::byte[]>(num_bytes_);
    std::copy(vec.begin(), vec.end(), reinterpret_cast<T*>(data_.get()));
  }

  auto view()
  {
    return new cugraph_type_erased_host_array_view_t{data_.get(), size_, num_bytes_, type_};
  }
};

}  // namespace c_api
}  // namespace cugraph
