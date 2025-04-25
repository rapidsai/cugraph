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

#include <raft/core/device_span.hpp>

#include <rmm/device_uvector.hpp>

#include <variant>

namespace cugraph {
namespace variant {

using device_uvectors_t    = std::variant<rmm::device_uvector<float>,
                                       rmm::device_uvector<double>,
                                       rmm::device_uvector<int32_t>,
                                       rmm::device_uvector<int64_t>,
                                       rmm::device_uvector<size_t>>;
using device_spans_t       = std::variant<raft::device_span<float>,
                                    raft::device_span<double>,
                                    raft::device_span<int32_t>,
                                    raft::device_span<int64_t>,
                                    raft::device_span<size_t>>;
using const_device_spans_t = std::variant<raft::device_span<float const>,
                                          raft::device_span<double const>,
                                          raft::device_span<int32_t const>,
                                          raft::device_span<int64_t const>,
                                          raft::device_span<size_t const>>;

template <typename func_t>
auto variant_type_dispatch(device_uvectors_t& property, func_t func)
{
  if (std::holds_alternative<rmm::device_uvector<float>>(property)) {
    auto& prop = std::get<rmm::device_uvector<float>>(property);
    return func(prop);
  } else if (std::holds_alternative<rmm::device_uvector<double>>(property)) {
    auto& prop = std::get<rmm::device_uvector<double>>(property);
    return func(prop);
  } else if (std::holds_alternative<rmm::device_uvector<int32_t>>(property)) {
    auto& prop = std::get<rmm::device_uvector<int32_t>>(property);
    return func(prop);
  } else if (std::holds_alternative<rmm::device_uvector<int64_t>>(property)) {
    auto& prop = std::get<rmm::device_uvector<int64_t>>(property);
    return func(prop);
  } else if (std::holds_alternative<rmm::device_uvector<size_t>>(property)) {
    auto& prop = std::get<rmm::device_uvector<size_t>>(property);
    return func(prop);
  }

  CUGRAPH_FAIL("unsupported variant type -- shouldn't happen");
}

template <typename func_t>
auto variant_type_dispatch(device_uvectors_t const& property, func_t func)
{
  if (std::holds_alternative<rmm::device_uvector<float>>(property)) {
    auto& prop = std::get<rmm::device_uvector<float>>(property);
    return func(prop);
  } else if (std::holds_alternative<rmm::device_uvector<double>>(property)) {
    auto& prop = std::get<rmm::device_uvector<double>>(property);
    return func(prop);
  } else if (std::holds_alternative<rmm::device_uvector<int32_t>>(property)) {
    auto& prop = std::get<rmm::device_uvector<int32_t>>(property);
    return func(prop);
  } else if (std::holds_alternative<rmm::device_uvector<int64_t>>(property)) {
    auto& prop = std::get<rmm::device_uvector<int64_t>>(property);
    return func(prop);
  } else if (std::holds_alternative<rmm::device_uvector<size_t>>(property)) {
    auto& prop = std::get<rmm::device_uvector<size_t>>(property);
    return func(prop);
  }

  CUGRAPH_FAIL("unsupported variant type -- shouldn't happen");
}

template <typename func_t>
auto variant_type_dispatch(device_spans_t& property, func_t func)
{
  if (std::holds_alternative<raft::device_span<float>>(property)) {
    auto& prop = std::get<raft::device_span<float>>(property);
    return func(prop);
  } else if (std::holds_alternative<raft::device_span<double>>(property)) {
    auto& prop = std::get<raft::device_span<double>>(property);
    return func(prop);
  } else if (std::holds_alternative<raft::device_span<int32_t>>(property)) {
    auto& prop = std::get<raft::device_span<int32_t>>(property);
    return func(prop);
  } else if (std::holds_alternative<raft::device_span<int64_t>>(property)) {
    auto& prop = std::get<raft::device_span<int64_t>>(property);
    return func(prop);
  } else if (std::holds_alternative<raft::device_span<size_t>>(property)) {
    auto& prop = std::get<raft::device_span<size_t>>(property);
    return func(prop);
  }

  CUGRAPH_FAIL("unsupported variant type -- shouldn't happen");
}

template <typename func_t>
auto variant_type_dispatch(const_device_spans_t& property, func_t func)
{
  if (std::holds_alternative<raft::device_span<float const>>(property)) {
    auto& prop = std::get<raft::device_span<float const>>(property);
    return func(prop);
  } else if (std::holds_alternative<raft::device_span<double const>>(property)) {
    auto& prop = std::get<raft::device_span<double const>>(property);
    return func(prop);
  } else if (std::holds_alternative<raft::device_span<int32_t const>>(property)) {
    auto& prop = std::get<raft::device_span<int32_t const>>(property);
    return func(prop);
  } else if (std::holds_alternative<raft::device_span<int64_t const>>(property)) {
    auto& prop = std::get<raft::device_span<int64_t const>>(property);
    return func(prop);
  } else if (std::holds_alternative<raft::device_span<size_t const>>(property)) {
    auto& prop = std::get<raft::device_span<size_t const>>(property);
    return func(prop);
  }

  CUGRAPH_FAIL("unsupported variant type -- shouldn't happen");
}

struct variant_size {
  template <typename T>
  size_t operator()(rmm::device_uvector<T> const&) const
  {
    return sizeof(T);
  }
  template <typename T>
  size_t operator()(raft::device_span<T> const&) const
  {
    return sizeof(T);
  }
  template <typename T>
  size_t operator()(raft::device_span<T const> const&) const
  {
    return sizeof(T);
  }
};

}  // namespace variant
}  // namespace cugraph
