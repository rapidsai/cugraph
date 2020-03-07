/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/utilities/traits.hpp>
#include <rmm/device_buffer.hpp>

#include <cudf/column/column.hpp>

#include <thrust/distance.h>

namespace detail {

template <typename Element, typename InputIterator>
rmm::device_buffer make_elements(InputIterator begin, InputIterator end) {
  static_assert(cudf::is_fixed_width<Element>(), "Unexpected non-fixed width type.");
  std::vector<Element> elements(begin, end);
  return rmm::device_buffer{elements.data(), elements.size() * sizeof(Element)};
}


template <typename Element, typename iterator_t>
std::unique_ptr<cudf::column> create_column(iterator_t begin, iterator_t end) {

  cudf::size_type size = thrust::distance(begin,end);

  return std::unique_ptr<cudf::column>(new cudf::column{cudf::data_type{cudf::experimental::type_to_id<Element>()}, size, detail::make_elements<Element>(begin, end)});
}

} //namespace detail
